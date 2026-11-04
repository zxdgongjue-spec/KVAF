#!/usr/bin/env python3
"""CPU-initiated URMA WRITE into registered HBM, with no AIV work in the timed loop."""
import argparse
import csv
import ctypes
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import socket
import statistics
import struct
import sys

from aiv_batch_get_test_common import shutdown_runtime
from npu_eids import parse_npu_eids
from scatter_perf_common import percentile

MAGIC, VERSION, COMMAND = 0x4B564350, 6, 15
MAX_SAMPLES, MAX_PAYLOAD = 4096, 256 * 1024**2
HEADER = struct.Struct("<IHHI")
OPTIONS = struct.Struct("<QQQ7I")
META = struct.Struct("<5Q7I")
SAMPLE = struct.Struct("<4Q")
RESPONSE_BYTES = 276 + MAX_SAMPLES * SAMPLE.size
CSV_COLUMNS = ("source_mode", "queues", "payload_bytes", "source_bytes", "message_bytes",
               "fragment_bytes", "post_batch_wrs", "iteration", "comm_us", "bandwidth_GBps",
               "post_us", "cq_wait_us", "data_wrs", "source_sges", "waves", "tail_cqes")


def byte_size(value):
    text = value.strip().upper()
    multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3}.get(text[-1:], 1)
    try:
        result = int(text[:-1] if multiplier != 1 else text) * multiplier
        if result <= 0:
            raise ValueError()
        return result
    except ValueError as error:
        raise argparse.ArgumentTypeError("use positive bytes or a K/M/G suffix") from error


def sizes(value):
    return tuple(byte_size(part) for part in value.split(","))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--sizes", type=sizes, default=sizes("4M,16M,64M"))
    p.add_argument("--message-sizes", type=sizes, default=sizes("64K"))
    p.add_argument("--queue-counts", type=sizes, default=sizes("1,2,4"))
    p.add_argument("--source-modes", default="contiguous")
    p.add_argument("--source-bytes", type=byte_size, default=byte_size("128M"))
    p.add_argument("--fragment-bytes", type=int, default=656)
    p.add_argument("--post-batch-wrs", type=int, default=128)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260903)
    p.add_argument("--cpu-service", default=os.environ.get("KVAF_CPU_SERVICE_ADDR", "127.0.0.1:28990"))
    p.add_argument("--device", type=int, default=int(os.environ.get("DEVICE_ID", "0")))
    p.add_argument("--npu-eids", default=os.environ.get("KVAF_NPU_EIDS", ""))
    p.add_argument("--peer-ub-device", default="")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    a.source_modes = tuple(a.source_modes.split(","))
    try:
        a.npu_eids = parse_npu_eids(a.npu_eids)
    except ValueError as error:
        p.error(str(error))
    if len(a.npu_eids) > 1 or any(x not in ("contiguous", "random-sgl") for x in a.source_modes):
        p.error("use one NPU EID and source-modes contiguous,random-sgl")
    if not 1 <= a.iterations <= MAX_SAMPLES or not 0 <= a.warmup <= MAX_SAMPLES:
        p.error("iterations must be 1..4096 and warmup 0..4096")
    if not 1 <= a.post_batch_wrs <= 8192 or a.timeout <= 0 or a.device < 0 or not 0 <= a.seed < 2**64:
        p.error("invalid submission batch, timeout, device or seed")
    if a.fragment_bytes < 8 or a.fragment_bytes % 8 or a.source_bytes < a.fragment_bytes:
        p.error("fragment-bytes must be a positive multiple of 8 within the source pool")
    if any(x not in (1, 2, 4) for x in a.queue_counts):
        p.error("queue-counts supports 1,2,4")
    if any(x < 8 or x > 65536 or x % 8 for x in a.message_sizes):
        p.error("message-sizes must be multiples of 8 in 8..65536")
    if any(x % 64 or not max(a.queue_counts) * 64 <= x <= MAX_PAYLOAD for x in a.sizes):
        p.error("sizes must be multiples of 64, at least queues*64, and at most 256M")
    a.source_bytes = (max(a.source_bytes, max(a.sizes) if "contiguous" in a.source_modes else 0) + 4095) // 4096 * 4096
    a.build = a.build.expanduser().resolve()
    return a


def receive_exact(connection, size):
    data = bytearray()
    while len(data) < size:
        part = connection.recv(size - len(data))
        if not part:
            raise RuntimeError("CPU benchmark service closed the response")
        data.extend(part)
    return data


def decode_response(data, iterations):
    if len(data) != RESPONSE_BYTES or HEADER.unpack_from(data) != (MAGIC, VERSION, COMMAND | 0x8000, RESPONSE_BYTES):
        raise RuntimeError("unsupported CPU WRITE benchmark response; rebuild/restart CPU service")
    if struct.unpack_from("<i", data, 12)[0]:
        raise RuntimeError(bytes(data[16:208]).split(b"\0", 1)[0].decode("utf-8", errors="replace"))
    names = ("hbm_address", "payload_bytes", "source_bytes", "data_wrs", "source_sges",
             "message_bytes", "max_sge", "queues", "post_batch_wrs", "waves", "optimized", "count")
    result = dict(zip(names, META.unpack_from(data, 208)))
    if result["count"] != iterations:
        raise RuntimeError("CPU WRITE benchmark returned an incomplete sample set")
    result["samples"] = []
    for index in range(iterations):
        comm, post, wait, cqes = SAMPLE.unpack_from(data, 276 + index * SAMPLE.size)
        if not comm or post + wait > comm or cqes < result["queues"]:
            raise RuntimeError("invalid CPU WRITE benchmark timing/completion report")
        result["samples"].append(dict(iteration=index, comm_ns=comm, comm_us=comm / 1000,
            post_us=post / 1000, cq_wait_us=wait / 1000, tail_cqes=cqes,
            bandwidth_GBps=result["payload_bytes"] / comm))
    return result


def run_remote(args, region_base, case, seed, warmup, iterations):
    payload, message, queues, mode = case
    body = struct.pack("<Q", region_base) + OPTIONS.pack(payload, args.source_bytes, seed,
        message, args.fragment_bytes, queues, args.post_batch_wrs, warmup, iterations,
        int(mode == "random-sgl"))
    host, port = args.cpu_service.rsplit(":", 1)
    with socket.create_connection((host.strip("[]"), int(port)), args.timeout) as connection:
        connection.settimeout(args.timeout)
        connection.sendall(HEADER.pack(MAGIC, VERSION, COMMAND, HEADER.size + len(body)) + body)
        header = receive_exact(connection, HEADER.size)
        if HEADER.unpack(header) != (MAGIC, VERSION, COMMAND | 0x8000, RESPONSE_BYTES):
            raise RuntimeError("CPU service lacks WRITE benchmark support; rebuild/restart it")
        data = header + receive_exact(connection, RESPONSE_BYTES - HEADER.size)
    result = decode_response(data, iterations)
    if (result["payload_bytes"], result["source_bytes"], result["queues"], result["post_batch_wrs"]) != \
            (payload, args.source_bytes, queues, args.post_batch_wrs):
        raise RuntimeError("CPU benchmark configuration mismatch")
    if not result["hbm_address"] or not 0 < result["message_bytes"] <= message:
        raise RuntimeError("CPU benchmark returned invalid HBM/message metadata")
    return result


def mix64(values, np):
    with np.errstate(over="ignore"):
        v = np.asarray(values, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        v = (v ^ (v >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        v = (v ^ (v >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return v ^ (v >> np.uint64(31))


def expected_words(np, begin, words, seed, mode, pool, fragment):
    positions = np.arange(begin // 8, begin // 8 + words, dtype=np.uint64)
    if mode == "random-sgl":
        row = positions // np.uint64(fragment // 8)
        slots = mix64(row + np.uint64(seed), np) % np.uint64(pool // fragment)
        positions = slots * np.uint64(fragment // 8) + positions % np.uint64(fragment // 8)
    return mix64(positions + np.uint64(seed), np)


def verify_hbm(result, args, mode, seed, memcpy=None):
    import numpy as np
    if memcpy is None:
        lib = ctypes.CDLL("libascendcl.so")
        memcpy = lib.aclrtMemcpy
        memcpy.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
        memcpy.restype = ctypes.c_int
    for begin in range(0, result["payload_bytes"], 4 * 1024**2):
        size = min(4 * 1024**2, result["payload_bytes"] - begin)
        actual = np.empty(size // 8, dtype=np.uint64)
        error = memcpy(actual.ctypes.data, size, result["hbm_address"] + begin, size, 2)
        if error:
            raise RuntimeError(f"HBM verification D2H failed: acl={error}")
        wanted = expected_words(np, begin, size // 8, seed, mode, args.source_bytes, args.fragment_bytes)
        if not np.array_equal(actual, wanted):
            bad = int(np.flatnonzero(actual != wanted)[0]) * 8 + begin
            raise AssertionError(f"CPU→HBM WRITE payload mismatch at byte {bad}")


def summarize(result):
    times = [sample["comm_ns"] for sample in result["samples"]]
    payload = result["payload_bytes"]
    return dict(mean_us=statistics.mean(times) / 1000, p50_us=percentile(times, .5) / 1000,
        p90_us=percentile(times, .9) / 1000, p99_us=percentile(times, .99) / 1000,
        effective_GBps=payload * len(times) / sum(times),
        p50_GBps=payload / percentile(times, .5), max_observed_GBps=payload / min(times))


def main(argv=None):
    a = arguments(argv)
    cases = list(itertools.product(a.sizes, a.message_sizes, a.queue_counts, a.source_modes))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(measured="CPU WRITE first submit to all payload tail CQEs; no AIV kernel",
        cpu_submit_threads=1,
        source_preparation_in_sample=False, verification_in_sample=False, units="us; decimal GB/s",
        setup="existing CPU-push registration/Jettys and one startup BatchPut probe",
        hbm_capacity_bytes=MAX_PAYLOAD, cases=cases)
    if a.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the NPU host")
        return 77
    sys.path.insert(0, str(a.build / "python"))
    import kvaf
    from backend import detect_torch_backend
    backend = detect_torch_backend()
    if backend.name != "npu":
        raise RuntimeError("real NPU backend required")
    backend.torch.npu.set_device(a.device)
    runtime = kvaf.NpuRuntime(str(a.build / "bin/kvaf_scatter_batch_get_kernels.o"), 4096, 1,
        device_id=a.device, npu_core_count=1, jettys_per_core=2, npu_eids=a.npu_eids,
        peer_ub_device=a.peer_ub_device, aiv_batch_get_mode="cpu_push_scatter",
        batch_put_kernel_object_path=str(a.build / "bin/kvaf_execute_batch_put_kernels.o"))
    runtime.link_to(a.cpu_service)
    root = a.output_dir or Path("results") / f"cpu-write-{datetime.now():%Y%m%d-%H%M%S}-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    report = dict(config=config, cases=[])
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        with (root / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, CSV_COLUMNS)
            writer.writeheader()
            stream.flush()
            print("CPU_WRITE_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
            runtime.initialize(a.source_bytes + 4096, 4096)
            backend.torch.npu.synchronize()
            base = runtime.remote_region["base"]
            for ordinal, case in enumerate(cases):
                seed = (a.seed + ordinal * 2) % 2**64
                print(f"CPU_WRITE_CASE_BEGIN case={case} seed={seed}", flush=True)
                # A full preflight readback precedes timing; formal writes use
                # a changed pattern so stale preflight contents cannot pass.
                preflight = run_remote(a, base, case, seed, 0, 1)
                verify_hbm(preflight, a, case[3], seed)
                seed = (seed + 1) % 2**64
                result = run_remote(a, base, case, seed, a.warmup, a.iterations)
                verify_hbm(result, a, case[3], seed)
                result.update(source_mode=case[3], fragment_bytes=a.fragment_bytes, seed=seed)
                result["summary"] = summarize(result)
                report["cases"].append(result)
                for sample in result["samples"]:
                    writer.writerow({key: sample.get(key, result.get(key, "")) for key in CSV_COLUMNS})
                stream.flush()
                (root / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print("CPU_WRITE_RESULT " + json.dumps({k: v for k, v in result.items()
                                                       if k not in ("samples", "hbm_address")}), flush=True)
        for mode in a.source_modes:
            winner = max((case for case in report["cases"] if case["source_mode"] == mode),
                         key=lambda case: case["summary"]["max_observed_GBps"])
            print("CPU_WRITE_PEAK " + json.dumps(dict(source_mode=mode,
                queues=winner["queues"], payload_bytes=winner["payload_bytes"],
                message_bytes=winner["message_bytes"], **winner["summary"])), flush=True)
        print(f"PASS: CPU URMA WRITE; full HBM payload verified; results={root}", flush=True)
        return 0
    finally:
        shutdown_runtime(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
