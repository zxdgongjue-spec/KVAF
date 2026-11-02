#!/usr/bin/env python3
"""Per-call CPU-push scatter benchmark; changing inputs match the AICPU test."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from aiv_batch_get_test_common import kernel_paths, shutdown_runtime
from aiv_kernel_metadata import check_aiv_kernel_metadata
from scatter_perf_common import PAYLOAD_TIMINGS, profile_metrics, request_values, sample_metrics, summarize


# Stable compact layout, including for baseline-only or profiled-only runs.
# Raw ns, worker cycles and less common diagnostics remain in the JSON report.
CSV_COLUMNS = (
    "phase", "iteration", "sequence", "cores", "fused_build", "double_buffer",
    "request_length", "miss_count", "token_bytes", "payload_bytes",
    "end_to_end_us", "end_to_end_gbps", "cpu_comm_us", "cpu_comm_gbps",
    "aiv_build_notify_us", "aiv_scatter_us", "aiv_scatter_gbps",
    "cpu_write_us", "cpu_write_gbps", "aiv_body_us", "aiv_body_gbps",
    "aiv_wall_us", "aiv_wall_gbps", "aiv_input_wall_us", "aiv_notify_to_done_us",
    "aiv_scatter_tail_us", "call_outside_aiv_us",
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("full-blocks", "one-token-per-block", "sampled-tokens"),
                        default="full-blocks", help="all tokens, AICPU reference, or arbitrary token samples")
    parser.add_argument("--request-length", "--requests", dest="request_length", type=int,
                        help="request Tensor capacity; independent of the active miss-count")
    parser.add_argument("--miss-count", type=int, help="active token requests in ONE BatchGet call")
    parser.add_argument("--active-percent", type=int, default=100,
                        help="percentage of workload tokens; use instead of --miss-count")
    parser.add_argument("--token-bytes", type=int, default=656)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--fused-build", type=int, choices=(0, 1),
                        help="parallel UB source-list build (0: serial two-pass); default: KVAF_SCATTER_FUSED_BUILD or 1")
    parser.add_argument("--double-buffer", type=int, choices=(0, 1),
                        help="runtime UB overlap; default: KVAF_SCATTER_DOUBLE_BUFFER or 1")
    parser.add_argument("--blocks", type=int, default=1024)
    parser.add_argument("--tokens-per-block", type=int, default=4)
    parser.add_argument("--plans", type=int, default=2)
    parser.add_argument("--input-mode", choices=("changing", "steady"), default="changing",
                        help="changing refreshes data, metadata and the table before every call like AICPU")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-iterations", "--profile-rounds", type=int, default=20,
                        help="0 disables all optional CPU/AIV instrumentation")
    parser.add_argument("--aiv-clock-mhz", type=float, default=1000.0,
                        help="GetSystemCycle counter: 1000 MHz on Ascend 950; override for other products")
    parser.add_argument("--output-pattern", choices=("dense", "strided"),
                        help="compatibility alias for output-row-step 1 or 2")
    parser.add_argument("--output-row-step", type=int, choices=(1, 2))
    parser.add_argument("--output-order", choices=("random", "sequential"), default="random")
    parser.add_argument("--source-pattern", choices=("random", "sequential"), default="random")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args(argv)
    for field in ("fused_build", "double_buffer"):
        if getattr(args, field) is None:
            name = "KVAF_SCATTER_" + field.upper()
            value = os.environ.get(name, "1")
            if value not in ("0", "1"):
                parser.error(f"{name} must be 0 or 1")
            setattr(args, field, int(value))
    for field, lo, hi in (("token_bytes", 1, 65535), ("cores", 1, 8),
                          ("blocks", 1, 65536), ("tokens_per_block", 1, 65536), ("plans", 1, 64)):
        if not lo <= getattr(args, field) <= hi:
            parser.error(f"{field} must be in [{lo}, {hi}]")
    if not 1 <= args.active_percent <= 100:
        parser.error("active-percent must be in [1, 100] for a nonempty performance call")
    if args.miss_count is not None and args.active_percent != 100:
        parser.error("use either miss-count or active-percent")
    workload_tokens = (args.blocks if args.workload == "one-token-per-block" else
                       args.blocks * args.tokens_per_block)
    if args.workload == "sampled-tokens" and args.request_length is not None:
        workload_tokens = args.request_length
    if args.miss_count is None:
        args.miss_count = workload_tokens * args.active_percent // 100
    if args.request_length is None:
        args.request_length = workload_tokens
    if not 1 <= args.miss_count <= args.request_length <= 65536:
        parser.error("require 1 <= miss-count <= request-length <= 65536")
    if args.workload != "sampled-tokens" and args.miss_count > workload_tokens:
        parser.error("miss-count exceeds this workload; use sampled-tokens for repeated requests")
    pattern_step = {"dense": 1, "strided": 2}.get(args.output_pattern)
    if pattern_step and args.output_row_step and pattern_step != args.output_row_step:
        parser.error("output-pattern and output-row-step disagree")
    args.output_row_step = args.output_row_step or pattern_step or 1
    args.output_pattern = "dense" if args.output_row_step == 1 else "strided"
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


def measure_pass(call, iterations, payload, snapshot=None, clock=None, prepare=None, verify=None):
    """Exactly one synchronous BatchGet per sample; all setup/checks are outside."""
    clock = clock or time.perf_counter_ns
    samples = []
    wall_begin = clock()
    for index in range(iterations):
        if prepare:
            prepare(index)
        begin = clock()
        call(index)
        elapsed = clock() - begin
        sample = dict(iteration=index, payload_bytes=payload, end_to_end_ns=elapsed)
        if snapshot:
            snapshot_begin = clock()
            sample.update(snapshot())
            sample["snapshot_ns"] = clock() - snapshot_begin
        if verify:
            verify(index)
        samples.append(sample_metrics(sample))
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
        cpu = tuple(torch.tensor(values, dtype=torch.int32)
                    for values in request_values(args, args.seed + index))
        plans.append((cpu, tuple(t.to(device) for t in cpu)))
    return plans


def write_results(args, report):
    for samples in report["samples"].values():
        if any(sample["payload_bytes"] != args.miss_count * args.token_bytes for sample in samples):
            raise ValueError("sample payload does not match this call's active miss-count")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for phase, samples in report["samples"].items():
                for sample in samples:
                    values = dict(phase=phase, cores=args.cores, fused_build=args.fused_build,
                                  double_buffer=args.double_buffer, request_length=args.request_length,
                                  miss_count=args.miss_count, token_bytes=args.token_bytes,
                                  **sample_metrics(sample))
                    row = {}
                    for key in CSV_COLUMNS:
                        value = values.get(key, "")
                        if value != "" and key.endswith("_us"):
                            value = f"{value:.3f}"
                        elif value != "" and key.endswith("_gbps"):
                            value = f"{value:.6f}"
                        row[key] = value
                    writer.writerow(row)


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
    output_rows = args.output_row_step * args.request_length + 2
    config = vars(args).copy()
    config.update(device_id=device_id, mode="cpu_push_scatter",
                  units="decimal GB/s; sample durations ns/us; summary durations ms",
                  cpu_service=os.environ.get("KVAF_CPU_SERVICE_ADDR"),
                  npu_eids=os.environ.get("KVAF_NPU_EIDS"),
                  torch_version=str(torch.__version__),
                  timing="synchronous BatchGet call through status validation; setup/verification excluded",
                  calls_per_sample=1, active_tokens_per_call=args.miss_count,
                  payload_bytes_per_call=args.miss_count * args.token_bytes,
                  request_capacity=args.request_length,
                  verification="every_round_full_output_and_npu_consumer",
                  reference="aicpu-urma:access_operator/demo/aicpu_batch_get/python/test_batch_get.py",
                  cpu_comm_window="first WRITE submit to last payload/ready tail CQ; includes posting and inter-wave gaps",
                  source_bytes=source_rows * args.token_bytes,
                  output_bytes=output_rows * args.token_bytes,
                  staging_bytes=65536 * args.token_bytes,
                  get_sha256=hashlib.sha256(Path(kernel).read_bytes()).hexdigest(),
                  put_sha256=hashlib.sha256(Path(put_kernel).read_bytes()).hexdigest())
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    print("SCATTER_PERF_CONFIG " + json.dumps(config, sort_keys=True), flush=True)
    block_bytes = args.token_bytes * args.tokens_per_block
    source_cpu = source_data(torch, source_rows, args.token_bytes) if args.input_mode == "steady" else None
    source = torch.empty((source_rows, args.token_bytes), dtype=torch.uint8, device=device)
    plans = request_plans(torch, args, device) if args.input_mode == "steady" else None
    dynamic_inputs = tuple(torch.empty(args.request_length, dtype=torch.int32, device=device) for _ in range(3))
    count = torch.tensor([args.miss_count], dtype=torch.int32, device=device)
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
    payload = args.miss_count * args.token_bytes
    try:
        # CLI overrides the environment. Host snapshots these during initialize;
        # baseline/profiled phases then use the same algorithms and binary.
        os.environ["KVAF_SCATTER_FUSED_BUILD"] = str(args.fused_build)
        os.environ["KVAF_SCATTER_DOUBLE_BUFFER"] = str(args.double_buffer)
        begin = time.perf_counter_ns()
        runtime.initialize((args.blocks * 2 + 1) * block_bytes, block_bytes)
        report["setup_ms"]["initialize"] = (time.perf_counter_ns() - begin) / 1e6
        input_offsets = [i * block_bytes for i in range(args.blocks)]
        lengths = [block_bytes] * args.blocks
        current = {}
        epoch = 0

        def populate():
            runtime.batch_put(keys, source, input_offsets, lengths, stream)
            current["table"] = runtime.prepare_access_table(keys)
            current["view"] = runtime.access_table_view(current["table"], device="npu", capacity=args.blocks)

        if args.input_mode == "steady":
            source.copy_(source_cpu)
            stream.synchronize()
            begin = time.perf_counter_ns()
            populate()
            stream.synchronize()
            report["setup_ms"]["batch_put"] = (time.perf_counter_ns() - begin) / 1e6

        def prepare(index):
            nonlocal epoch, source_cpu
            if args.input_mode == "changing":
                rng = random.Random(args.seed + epoch)
                data = bytearray(rng.randbytes(source_rows * args.token_bytes))
                source_cpu = torch.frombuffer(data, dtype=torch.uint8).reshape(source_rows, args.token_bytes)
                cpu = tuple(torch.tensor(values, dtype=torch.int32)
                            for values in request_values(args, args.seed + epoch))
                for target, value in zip(dynamic_inputs, cpu):
                    target.copy_(value)
                source.copy_(source_cpu)
                current["cpu"], current["inputs"] = cpu, dynamic_inputs
            else:
                current["cpu"], current["inputs"] = plans[index % len(plans)]
            output.fill_(0xAD)
            stream.synchronize()
            if args.input_mode == "changing":
                populate()
                stream.synchronize()
            epoch += 1

        def call(index):
            indices, offsets, rows = current["inputs"]
            runtime.batch_get(current["view"], indices, offsets, count, rows, output,
                              stream=stream, synchronize=True)

        def check(index):
            indices, offsets, rows = (t[:args.miss_count] for t in current["cpu"])
            expected = torch.full((output_rows, args.token_bytes), 0xAD, dtype=torch.uint8)
            source_indices = indices.long() * args.tokens_per_block + offsets.long() // args.token_bytes
            expected[rows.long()] = source_cpu[source_indices]
            # Like the AICPU reference, consume on the same NPU stream before D2H.
            consumed = output.to(dtype=torch.int32) + 1
            if not torch.equal(consumed.cpu(), expected.to(dtype=torch.int32) + 1):
                raise AssertionError(f"scatter NPU consumer mismatch in round {epoch - 1}")
            if not torch.equal(output.cpu(), expected):
                raise AssertionError(f"scatter output/guard mismatch in round {epoch - 1}")
            if args.input_mode == "changing":
                runtime.release_access_table(current.pop("table"))
                current.pop("view")
                runtime.batch_free(keys)

        for phase, warmup, iterations in (("baseline", args.warmup, args.iterations),
                                           ("profiled", args.profile_warmup, args.profile_iterations)):
            if not iterations:
                continue
            profiling = phase == "profiled"
            if profiling:
                runtime.start_scatter_profile()
            print(f"SCATTER_PERF_PHASE phase={phase} warmup={warmup} iterations={iterations}", flush=True)
            for index in range(warmup):
                prepare(index)
                call(index)
                check(index)
            def snapshot():
                records = [dict(r) for r in runtime.scatter_profile()]
                return dict(workers=records, sequence=records[0]["sequence"],
                            **profile_metrics(records, args.cores, args.miss_count,
                                                              args.token_bytes, args.aiv_clock_mhz))
            samples, wall_ns = measure_pass(call, iterations, payload, snapshot if profiling else None,
                                           prepare=prepare, verify=check)
            report["samples"][phase] = samples
            metrics = {key: summarize([sample[key] for sample in samples], payload)
                       for key in PAYLOAD_TIMINGS
                       if key in samples[0]}
            # Non-payload subphases get latency statistics, without misleading bandwidths.
            for key in ("cpu_header_ns", "cpu_descriptor_ns", "cpu_post_ns", "cpu_cq_wait_ns",
                        "aiv_input_ns", "aiv_build_notify_ns", "aiv_notify_to_done_ns", "snapshot_ns",
                        "aiv_input_wall_ns", "aiv_scatter_tail_ns", "aiv_launch_skew_ns", "call_outside_aiv_ns"):
                if key in samples[0]:
                    values = [s[key] for s in samples]
                    metrics[key] = dict(mean_ms=sum(values) / len(values) / 1e6,
                                        min_ms=min(values) / 1e6, max_ms=max(values) / 1e6)
            metrics.update(loop_wall_ms=wall_ns / 1e6, loop_effective_gbps=payload * iterations / wall_ns,
                           instrumentation_overhead_included=profiling,
                           profile_snapshot_in_sample=False,
                           loop_includes_setup_and_verification=True)
            report["summary"][phase] = metrics
            if profiling:
                runtime.stop_scatter_profile()
        if args.input_mode == "steady":
            runtime.release_access_table(current.pop("table"))
            current.pop("view")
            runtime.batch_free(keys)
    finally:
        shutdown_runtime(runtime)
    write_results(args, report)
    for phase, metrics in report["summary"].items():
        print("SCATTER_PERF_RESULT " + json.dumps(dict(phase=phase, **metrics), sort_keys=True), flush=True)
    print("PASS: scatter performance and full output/guard verification", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
