#!/usr/bin/env python3
"""Real CPU-push scatter benchmark with separate baseline and profiled passes."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

from aiv_batch_get_test_common import kernel_paths, shutdown_runtime
from aiv_kernel_metadata import check_aiv_kernel_metadata
from scatter_perf_common import profile_metrics, summarize


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=8193)
    parser.add_argument("--token-bytes", type=int, default=656)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=1024)
    parser.add_argument("--tokens-per-block", type=int, default=4)
    parser.add_argument("--plans", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile-warmup", type=int, default=5)
    parser.add_argument("--profile-iterations", type=int, default=20,
                        help="0 disables all optional CPU/AIV instrumentation")
    parser.add_argument("--aiv-clock-mhz", type=float, default=1000.0,
                        help="GetSystemCycle counter: 1000 MHz on Ascend 950; override for other products")
    parser.add_argument("--output-pattern", choices=("dense", "strided"), default="strided")
    parser.add_argument("--source-pattern", choices=("random", "sequential"), default="random")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    for field, lo, hi in (("requests", 1, 65536), ("token_bytes", 1, 65535), ("cores", 1, 8),
                          ("blocks", 1, 65536), ("tokens_per_block", 1, 65536), ("plans", 1, 64)):
        if not lo <= getattr(args, field) <= hi:
            parser.error(f"{field} must be in [{lo}, {hi}]")
    if args.token_bytes * args.tokens_per_block > 0xffffffff:
        parser.error("block byte count exceeds uint32")
    if args.token_bytes * (args.tokens_per_block - 1) > 0x7fffffff:
        parser.error("token offset exceeds nonnegative int32")
    if args.warmup < 0 or args.profile_warmup < 0 or args.iterations < 1 or args.profile_iterations < 0:
        parser.error("iterations must be positive; warmup/profile counts must be nonnegative")
    if not math.isfinite(args.aiv_clock_mhz) or args.aiv_clock_mhz <= 0:
        parser.error("aiv-clock-mhz must be positive and finite")
    if args.json and args.csv and args.json.resolve() == args.csv.resolve():
        parser.error("JSON and CSV outputs must be different files")
    return args


def measure_pass(call, iterations, payload, snapshot=None, clock=None):
    """Time only synchronous calls; optional snapshots are outside each timer."""
    clock = clock or time.perf_counter_ns
    samples = []
    wall_begin = clock()
    for index in range(iterations):
        begin = clock()
        call(index)
        elapsed = clock() - begin
        sample = dict(iteration=index, payload_bytes=payload, end_to_end_ns=elapsed)
        if snapshot:
            snapshot_begin = clock()
            sample.update(snapshot())
            sample["snapshot_ns"] = clock() - snapshot_begin
        samples.append(sample)
    return samples, clock() - wall_begin


def source_data(torch, rows, token_bytes):
    # Bounded temporaries, including the 65535-byte token boundary.
    output = torch.empty((rows, token_bytes), dtype=torch.uint8)
    y = torch.arange(token_bytes, dtype=torch.int64)
    columns = ((y * 13 + (y >> 3)) % 251).to(torch.int16)
    for begin in range(0, rows, 64):
        end = min(begin + 64, rows)
        factors = ((torch.arange(begin, end, dtype=torch.int64) * 37) % 251).to(torch.int16)
        output[begin:end] = ((factors[:, None] + columns[None, :]) % 251).to(torch.uint8)
        row_ids = torch.arange(begin, end, dtype=torch.int64)
        for byte in range(min(token_bytes, 8)):
            output[begin:end, byte] = ((row_ids >> (byte * 8)) & 255).to(torch.uint8)
    return output


def request_plans(torch, args, device):
    plans = []
    for index in range(args.plans):
        generator = torch.Generator().manual_seed(args.seed + index)
        source_rows = args.blocks * args.tokens_per_block
        if args.source_pattern == "random":
            sources = torch.randint(source_rows, (args.requests,), generator=generator)
        else:
            sources = (torch.arange(args.requests, dtype=torch.int64) + index) % source_rows
        row_generator = torch.Generator().manual_seed(args.seed + index + 1000003)
        rows = torch.randperm(args.requests, generator=row_generator)
        if args.output_pattern == "strided":
            rows = rows * 2 + 1
        cpu = ((sources // args.tokens_per_block).to(torch.int32),
               ((sources % args.tokens_per_block) * args.token_bytes).to(torch.int32), rows.to(torch.int32))
        plans.append((cpu, tuple(t.to(device) for t in cpu)))
    return plans


def write_results(args, report):
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [dict(phase=phase, cores=args.cores, requests=args.requests, token_bytes=args.token_bytes,
                     **{k: v for k, v in sample.items() if k != "workers"})
                for phase, samples in report["samples"].items() for sample in samples]
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with args.csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def main(argv=None):
    args = arguments(argv)
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the A5/CANN target")
        return 77
    kernel, put_kernel = kernel_paths("cpu_push_scatter")
    check_aiv_kernel_metadata(kernel, "KvafScatterBatchGetAivKernelV1")
    from backend import detect_torch_backend
    from dsa_keys import build_dsa_keys
    from npu_eids import parse_npu_eids
    import kvaf
    backend = detect_torch_backend()
    if backend.name != "npu" or not hasattr(kvaf, "NpuRuntime"):
        raise RuntimeError("scatter benchmark requires the NPU backend")
    torch = backend.torch
    device_id = int(os.environ.get("DEVICE_ID", "7"))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    source_rows = args.blocks * args.tokens_per_block
    output_rows = (2 * args.requests if args.output_pattern == "strided" else args.requests) + 2
    config = vars(args).copy()
    config.update(device_id=device_id, mode="cpu_push_scatter", units="decimal GB/s, milliseconds",
                  cpu_service=os.environ.get("KVAF_CPU_SERVICE_ADDR"),
                  npu_eids=os.environ.get("KVAF_NPU_EIDS"),
                  torch_version=str(torch.__version__),
                  timing="synchronous BatchGet call through status validation; setup/verification excluded",
                  cpu_comm_window="first WRITE submit to last payload/ready tail CQ; includes posting and inter-wave gaps",
                  source_bytes=source_rows * args.token_bytes,
                  output_bytes=output_rows * args.token_bytes,
                  staging_bytes=65536 * args.token_bytes,
                  get_sha256=hashlib.sha256(Path(kernel).read_bytes()).hexdigest(),
                  put_sha256=hashlib.sha256(Path(put_kernel).read_bytes()).hexdigest())
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    print("SCATTER_PERF_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    block_bytes = args.token_bytes * args.tokens_per_block
    source_cpu = source_data(torch, source_rows, args.token_bytes)
    source = source_cpu.to(device)
    plans = request_plans(torch, args, device)
    count = torch.tensor([args.requests], dtype=torch.int32, device=device)
    output = torch.empty((output_rows, args.token_bytes), dtype=torch.uint8, device=device)
    stream = torch.npu.current_stream(device_id)
    keys = build_dsa_keys(args.blocks, q_capacity=args.blocks)
    runtime = kvaf.NpuRuntime(kernel, args.token_bytes, args.tokens_per_block,
                              device_id=device_id, npu_core_count=args.cores,
                              batch_put_kernel_object_path=put_kernel,
                              npu_eids=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")),
                              aiv_batch_get_mode="cpu_push_scatter")
    if args.profile_iterations and not all(hasattr(runtime, name) for name in (
            "start_scatter_profile", "scatter_profile", "stop_scatter_profile")):
        raise RuntimeError("rebuild the Python binding with scatter profiling support")
    runtime.link_to(os.environ["KVAF_CPU_SERVICE_ADDR"])
    report = dict(config=config, samples={}, summary={}, setup_ms={})
    payload = args.requests * args.token_bytes
    try:
        begin = time.perf_counter_ns()
        runtime.initialize(args.blocks * block_bytes, block_bytes)
        report["setup_ms"]["initialize"] = (time.perf_counter_ns() - begin) / 1e6
        torch.npu.synchronize()
        begin = time.perf_counter_ns()
        runtime.batch_put(keys, source, [i * block_bytes for i in range(args.blocks)], [block_bytes] * args.blocks)
        torch.npu.synchronize()
        report["setup_ms"]["batch_put"] = (time.perf_counter_ns() - begin) / 1e6
        table = runtime.prepare_access_table(keys)
        view = runtime.access_table_view(table, device="npu", capacity=args.blocks)

        def call(index):
            _, (indices, offsets, rows) = plans[index % len(plans)]
            runtime.batch_get(view, indices, offsets, count, rows, output, stream=stream, synchronize=True)

        def check(index):
            cpu, _ = plans[index % len(plans)]
            indices, offsets, rows = cpu
            expected = torch.full((output_rows, args.token_bytes), 0xAD, dtype=torch.uint8)
            source_indices = indices.long() * args.tokens_per_block + offsets.long() // args.token_bytes
            expected[rows.long()] = source_cpu[source_indices]
            if not torch.equal(output.cpu(), expected):
                raise AssertionError(f"scatter output/guard mismatch for plan {index}")

        def verify():
            for index in range(len(plans)):
                output.fill_(0xAD)
                torch.npu.synchronize()
                call(index)
                check(index)

        verify()
        for phase, warmup, iterations in (("baseline", args.warmup, args.iterations),
                                           ("profiled", args.profile_warmup, args.profile_iterations)):
            if not iterations:
                continue
            profiling = phase == "profiled"
            if profiling:
                runtime.start_scatter_profile()
            print(f"SCATTER_PERF_PHASE phase={phase} warmup={warmup} iterations={iterations}", flush=True)
            for index in range(warmup):
                call(index)
            def snapshot():
                records = [dict(r) for r in runtime.scatter_profile()]
                return dict(workers=records, sequence=records[0]["sequence"],
                            **profile_metrics(records, args.cores, args.requests,
                                                              args.token_bytes, args.aiv_clock_mhz))
            samples, wall_ns = measure_pass(call, iterations, payload, snapshot if profiling else None)
            report["samples"][phase] = samples
            metrics = {key: summarize([sample[key] for sample in samples], payload)
                       for key in ("end_to_end_ns", "cpu_comm_ns", "cpu_write_ns", "aiv_body_ns", "aiv_scatter_ns")
                       if key in samples[0]}
            # Non-payload subphases get latency statistics, without misleading bandwidths.
            for key in ("cpu_header_ns", "cpu_descriptor_ns", "cpu_post_ns", "cpu_cq_wait_ns",
                        "aiv_input_ns", "aiv_build_notify_ns", "aiv_notify_to_done_ns", "snapshot_ns"):
                if key in samples[0]:
                    values = [s[key] for s in samples]
                    metrics[key] = dict(mean_ms=sum(values) / len(values) / 1e6,
                                        min_ms=min(values) / 1e6, max_ms=max(values) / 1e6)
            metrics.update(loop_wall_ms=wall_ns / 1e6, loop_effective_gbps=payload * iterations / wall_ns,
                           instrumentation_overhead_included=profiling,
                           profile_snapshot_in_sample=False)
            report["summary"][phase] = metrics
            check(iterations - 1)  # Validate the measured output before any new launch overwrites it.
            if profiling:
                runtime.stop_scatter_profile()
            verify()
    finally:
        shutdown_runtime(runtime)
    write_results(args, report)
    for phase, metrics in report["summary"].items():
        print("SCATTER_PERF_RESULT " + json.dumps(dict(phase=phase, **metrics), sort_keys=True), flush=True)
    print("PASS: scatter performance and full output/guard verification", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
