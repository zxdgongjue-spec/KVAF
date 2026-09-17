#!/usr/bin/env python3
"""Focused CPU DRAM -> NPU HBM AIV BatchGet bandwidth comparison."""

import argparse
from contextlib import contextmanager
import ctypes
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peer", required=True, help="CPU service HOST:PORT")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-npu")
    parser.add_argument("--allocator", choices=("torch", "acl-normal"), default="torch")
    parser.add_argument("--blocks", type=int, default=1024)
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--cores", type=int, default=32)
    parser.add_argument("--jettys-per-core", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--pattern", choices=("sequential", "random"), default="sequential")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    if not (
        1 <= args.blocks <= 0x7FFFFFFF
        and 1 <= args.block_bytes <= 65536
        and 1 <= args.cores <= 32
        and args.device_id >= 0
        and args.warmup >= 0
        and args.iterations > 0
        and args.seed >= 0
    ):
        parser.error("invalid blocks, block-bytes, cores, device-id, warmup, iterations or seed")
    return args


@contextmanager
def native_log(path):
    # Capture C/C++ stdout as well as stderr without changing production logging.
    flush_native = ctypes.CDLL(None).fflush
    flush_native.argtypes = [ctypes.c_void_p]
    flush_native.restype = ctypes.c_int
    sys.stdout.flush()
    sys.stderr.flush()
    flush_native(None)
    saved = [os.dup(1), os.dup(2)]
    try:
        with open(path, "a") as log:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                flush_native(None)
    finally:
        for fd, original in zip((1, 2), saved):
            os.dup2(original, fd)
            os.close(original)


def summarize(payload_bytes, samples):
    if not samples or any(sample <= 0 for sample in samples):
        raise ValueError("bandwidth samples must be positive")
    bytes_per_second = payload_bytes * len(samples) * 1e9 / sum(samples)
    return (
        f"GB/s={bytes_per_second / 1e9:.3f} "
        f"GiB/s={bytes_per_second / (1024 ** 3):.3f} "
        f"avg_us={statistics.mean(samples) / 1000:.3f} "
        f"min_us={min(samples) / 1000:.3f} max_us={max(samples) / 1000:.3f}"
    )


def run(args):
    build = args.build_dir.resolve()
    get_kernel = build / "bin/kvaf_execute_batch_get_kernels.o"
    put_kernel = build / "bin/kvaf_execute_batch_put_kernels.o"
    if not get_kernel.is_file() or not put_kernel.is_file():
        raise RuntimeError(f"build AIV kernels first: ./build.sh npu --build-dir {build}")
    sys.path.insert(0, str(build / "python"))
    from backend import detect_torch_backend

    backend = detect_torch_backend()
    if backend.name != "npu":
        raise RuntimeError("this test requires torch_npu and an available NPU")
    import kvaf

    torch = backend.torch
    torch.npu.set_device(args.device_id)
    torch.manual_seed(args.seed)
    device = f"npu:{args.device_id}"
    rows = torch.arange(args.blocks, dtype=torch.int64).unsqueeze(1)
    columns = torch.arange(args.block_bytes, dtype=torch.int64).unsqueeze(0)
    source_host = (
        (rows * 131 + (rows >> 8) * 29 + columns * 17 + (columns >> 3)) & 255
    ).to(torch.uint8)
    source = source_host.to(device)
    output = (
        torch.empty((args.blocks, args.block_bytes), dtype=torch.uint8, device=device)
        if args.allocator == "torch"
        else kvaf.NormalHbmBuffer(args.blocks, args.block_bytes, args.device_id)
    )
    runtime = kvaf.NpuRuntime(
        str(get_kernel),
        batch_get_token_bytes=args.block_bytes,
        batch_get_tokens_per_block=1,
        device_id=args.device_id,
        npu_core_count=args.cores,
        batch_put_kernel_object_path=str(put_kernel),
        jettys_per_core=args.jettys_per_core,
    )
    table = None
    try:
        runtime.link_to(args.peer)
        runtime.initialize(args.blocks * args.block_bytes, args.block_bytes)
        for begin in range(0, args.blocks, 128):
            count = min(128, args.blocks - begin)
            runtime.batch_put(
                [list(range(begin + 1, begin + count + 1))],
                [source[begin:begin + count]],
                [[i * args.block_bytes for i in range(count)]],
                [[args.block_bytes] * count],
            )
        torch.npu.synchronize()
        table = runtime.prepare_access_table(
            torch.arange(1, args.blocks + 1, dtype=torch.int64)
        )
        device_table = runtime.access_table_view(table, device="npu", capacity=args.blocks)
        indices = torch.arange(args.blocks, dtype=torch.int32, device=device)
        output_indices = indices.clone()
        if args.pattern == "random":
            indices = torch.randperm(args.blocks, device=device).to(torch.int32)
            output_indices = torch.randperm(args.blocks, device=device).to(torch.int32)
        offsets = torch.zeros(args.blocks, dtype=torch.int32, device=device)
        mask = torch.ones(args.blocks, dtype=torch.uint8, device=device)
        expected = torch.empty_like(source_host)
        expected[output_indices.cpu().long()] = source_host[indices.cpu().long()]
        torch.npu.synchronize()

        def transfer():
            runtime.batch_get(device_table, indices, offsets, mask, output_indices, output)
            torch.npu.synchronize()

        transfer()
        if not torch.equal(output.cpu(), expected):
            raise AssertionError("preflight data verification failed")
        for _ in range(args.warmup):
            transfer()
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            transfer()
            samples.append(time.perf_counter_ns() - start)
        if not torch.equal(output.cpu(), expected):
            raise AssertionError("final data verification failed")
        return summarize(args.blocks * args.block_bytes, samples)
    finally:
        torch.npu.synchronize()
        if runtime.initialized:
            if table is not None:
                runtime.release_access_table(table)
            runtime.shutdown()
        # Direct-HCCP registration is cached until Runtime shutdown.
        output = None


def main(argv=None):
    args = parse_args(argv)
    fd, log_path = tempfile.mkstemp(prefix="kvaf_aiv_bw_", suffix=".log")
    os.close(fd)
    print(
        f"[AIV_BW] START peer={args.peer} allocator={args.allocator} "
        f"blocks={args.blocks} block_bytes={args.block_bytes} "
        f"payload_bytes={args.blocks * args.block_bytes} cores={args.cores} "
        f"jettys_per_core={args.jettys_per_core} pattern={args.pattern} "
        f"warmup={args.warmup} iterations={args.iterations} seed={args.seed} "
        "timing=call_to_device_completion",
        flush=True,
    )
    try:
        with native_log(log_path):
            try:
                result = run(args)
            except Exception:
                traceback.print_exc()
                raise
    except Exception as error:
        print(f"[AIV_BW] FAIL {error} log={log_path}", flush=True)
        return 1
    print(f"[AIV_BW] READ {result}", flush=True)
    print(f"[AIV_BW] PASS log={log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
