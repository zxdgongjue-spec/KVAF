#!/usr/bin/env python3
"""Check independent AICPU BatchGet workers through the public Python API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import statistics
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=7)
    parser.add_argument("--peer", default="127.0.0.1:28990")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--modes", nargs="+", choices=["gather_read", "cpu_push_scatter"],
                        default=["gather_read"])
    parser.add_argument("--npu-eid", help="optional single NPU EID for cpu_push_scatter")
    parser.add_argument("--output-row-step", type=int, choices=[1, 2], default=2,
                        help="1 checks adjacent output rows; 2 leaves one guard row between outputs")
    parser.add_argument("--output-order", choices=["random", "sequential"], default="random",
                        help="request order of target slots; sequential matches the V24 probe's output order")
    parser.add_argument("--blocks", type=int, default=4096)
    parser.add_argument("--token-bytes", type=int, default=656)
    parser.add_argument("--tokens-per-block", type=int, default=2)
    parser.add_argument("--active-percent", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile-rounds", type=int, default=0,
                        help="extra instrumented rounds, excluded from bandwidth statistics")
    parser.add_argument("--urma-device", default="")
    parser.add_argument("--eid-index", type=int, default=0)
    cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_CANN_PACKAGE_PATH")
    parser.add_argument(
        "--kernel-json",
        default=str(Path(cann) / "opp/vendors/cust/op_impl/aicpu/config/kvaf_aicpu_urma.json")
        if cann else None,
    )
    args = parser.parse_args()
    if not all(1 <= workers <= 8 for workers in args.workers):
        parser.error("--workers must contain values from 1 to 8")
    if not 1 <= args.blocks <= 65536:
        parser.error("--blocks must be from 1 to 65536")
    if args.token_bytes <= 0 or args.token_bytes * 65536 > 0xFFFFFFFF:
        parser.error("--token-bytes must be positive and 65536 tokens must fit uint32")
    if args.tokens_per_block <= 0 or args.token_bytes * args.tokens_per_block > 0xFFFFFFFF:
        parser.error("block size must be positive and fit uint32")
    if (args.tokens_per_block - 1) * args.token_bytes > 0x7FFFFFFF:
        parser.error("token offsets must fit the int32 input Tensor")
    if (not 0 <= args.active_percent <= 100 or args.warmup < 0 or args.iterations <= 0
            or args.profile_rounds < 0):
        parser.error("invalid active percentage, warmup or iteration count")
    if not 0 <= args.eid_index < 0xFFFFFFFF:
        parser.error("--eid-index must be a nonnegative uint32 index")
    if not args.kernel_json or not Path(args.kernel_json).is_file():
        parser.error("source the CANN environment or pass an installed --kernel-json")
    return args


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def print_profile(samples: list[dict], mode: str) -> None:
    def mean(key: str, rows: list[dict]) -> float:
        return statistics.mean(row[key] for row in rows)

    def fields(rows: list[dict], keys: tuple[str, ...]) -> str:
        return " ".join(f"{key}={mean(key, rows):.1f}" for key in keys)

    print(f"AICPU_BATCH_GET_PROFILE mode={mode} workers={len(samples[0]['workers'])} "
          f"rounds={len(samples)} warmup=1 "
          "aggregation=mean_per_worker units=us "
          "output_payload_flushes=0 "
          "instrumented_rounds_excluded_from_bandwidth=1")
    print("  Python " + fields(samples, ("python_call_us", "binding_prepare_us", "native_call_us",
                                       "binding_total_us", "python_other_us")))
    hosts = [sample["host"] for sample in samples]
    print("  Host " + fields(hosts, ("prepare_us", "launch_us",
                                   "status_enqueue_us", "wait_us", "finish_us", "total_us")))
    print(f"  Device first_entry_to_last_exit_us={mean('device_span_us', samples):.1f}")
    for index in range(len(samples[0]["workers"])):
        rows = [sample["workers"][index] for sample in samples]
        print(f"  Worker {index} " + fields(rows, ("entry_us", "exit_us", "wall_us",
                                                  "request_us", "source_write_us", "notify_us",
                                                  "read_us", "ack_us")))
        print("    Within request " + fields(rows, ("input_cache_us", "descriptor_cache_us")))
        print("    Within READ " + fields(rows, ("read_setup_us", "wr_build_us", "ready_wait_us",
                                                "post_us", "cq_wait_us")))
        print("    Counts " + fields(rows, ("bytes", "spans", "read_wqes", "post_calls",
                                           "ready_queries")))
        if mode == "cpu_push_scatter":
            print("    Push " + fields(rows, ("request_order_us", "scatter_us", "scatter_copies",
                                              "output_cache_us")))
    if mode == "cpu_push_scatter":
        print("  CPU stages: see CPU_PUSH_PROFILE in the CPU service log (units=us).")
    print("  Timing: Host phases sum to Host total; Host wait includes queued device work/D2H. "
          "Worker intervals overlap. Cache times are inside request; READ phases are inside read. "
          "ready_wait includes CPU gather overlap, not a separate CPU gather measurement. "
          "post includes provider encoding/SQ/doorbells. python_other includes binding dispatch "
          "and profile dictionary conversion. Host/device clocks are not subtracted.", flush=True)


def run_case(args: argparse.Namespace, workers: int, torch: object, kvaf: object,
             mode: str = "gather_read") -> None:
    device = f"npu:{args.device_id}"
    n, token = args.blocks, args.token_bytes
    block_bytes = token * args.tokens_per_block
    rng = random.Random(20261021)  # Identical input sequence for every worker count.
    keys = list(range(n))
    input_offsets = [i * block_bytes for i in range(n)]
    lengths = [block_bytes] * n
    stream = torch.npu.current_stream()
    runtime = kvaf.NpuRuntime(
        batch_get_kernel_object_path="",
        batch_get_token_bytes=token,
        batch_get_tokens_per_block=args.tokens_per_block,
        device_id=args.device_id,
        npu_core_count=workers,
        aicpu_urma_kernel_json_path=args.kernel_json,
        urma_device=args.urma_device,
        urma_eid_index=args.eid_index,
        aicpu_batch_get_mode=mode,
        npu_eids=[args.npu_eid] if mode == "cpu_push_scatter" and args.npu_eid else [],
    )
    # Reuse registered allocations while changing their contents each round.
    source = torch.empty((n, block_bytes), dtype=torch.uint8, device=device)
    output = torch.empty((n * 2, token), dtype=torch.uint8, device=device)
    indices = torch.empty((n,), dtype=torch.int32, device=device)
    offsets = torch.empty((n,), dtype=torch.int32, device=device)
    miss_count = torch.empty((1,), dtype=torch.int32, device=device)
    rows = torch.empty((n,), dtype=torch.int32, device=device)
    samples_ns: list[int] = []
    payload_bytes: list[int] = []
    profiles: list[dict] = []
    device_table = None

    def run_round(epoch: int, active_percent: int, raw_tensors: bool = False,
                  synchronize: bool = True, profile: bool = False) -> tuple[int, int]:
        nonlocal device_table
        payload = bytearray(rng.randbytes(n * block_bytes))
        index_values = list(range(n))
        row_values = list(range(n))
        rng.shuffle(index_values)
        rng.shuffle(row_values)
        if args.output_order == "sequential":
            row_values.sort()
        row_values = [args.output_row_step * row for row in row_values]
        offset_values = [rng.randrange(args.tokens_per_block) * token for _ in range(n)]
        active_count = n * active_percent // 100
        expected = bytearray([0xA5]) * (n * 2 * token)
        for i in range(active_count):
            start = index_values[i] * block_bytes + offset_values[i]
            target = row_values[i] * token
            expected[target:target + token] = payload[start:start + token]

        source.copy_(torch.frombuffer(payload, dtype=torch.uint8).reshape(n, block_bytes))
        for target, values, dtype in (
            (indices, index_values, torch.int32),
            (offsets, offset_values, torch.int32),
            (miss_count, [active_count], torch.int32),
            (rows, row_values, torch.int32),
        ):
            target.copy_(torch.tensor(values, dtype=dtype))
        output.fill_(0xA5)
        stream.synchronize()

        # Keys are immutable while an AccessTable is held. Free and reinsert them
        # between rounds so both payload and remote addresses may change.
        runtime.batch_put(keys, source, input_offsets, lengths, stream)
        table = runtime.prepare_access_table(keys)
        device_table = runtime.access_table_view(table, device="npu")
        stream.synchronize()
        table_args = (device_table.addresses, device_table.valid) if raw_tensors else (device_table,)
        begin = time.perf_counter_ns()
        measured = runtime.batch_get(
            *table_args, indices, offsets, miss_count, rows, output,
            stream=stream, synchronize=synchronize, profile=profile,
        )
        elapsed = time.perf_counter_ns() - begin
        if profile:
            measured["python_call_us"] = elapsed / 1000
            measured["python_other_us"] = elapsed / 1000 - measured["binding_total_us"]
            profiles.append(measured)

        # Consume on the NPU before downloading: this also checks same-stream
        # visibility after the asynchronous precheck, including gaps and inactive rows.
        consumed = output.to(dtype=torch.int32) + 1
        expected_tensor = torch.frombuffer(expected, dtype=torch.uint8).reshape(n * 2, token)
        if not torch.equal(consumed.cpu(), expected_tensor.to(dtype=torch.int32) + 1):
            raise AssertionError(f"NPU consumer mismatch: workers={workers}, round={epoch}")
        if not torch.equal(output.cpu(), expected_tensor):
            raise AssertionError(f"output mismatch: workers={workers}, round={epoch}")
        runtime.release_access_table(table)
        runtime.batch_free(keys)
        return elapsed, active_count * token

    try:
        runtime.link_to(args.peer)
        runtime.initialize(ddr_bytes=(n * 2 + 1) * block_bytes, slot_size=block_bytes)
        print(f"AICPU_BATCH_GET_BEGIN mode={mode} workers={workers} blocks={n} token_bytes={token} "
              f"tokens_per_block={args.tokens_per_block} active_percent={args.active_percent} "
              f"output_row_step={args.output_row_step} output_order={args.output_order} warmup={args.warmup} "
              f"iterations={args.iterations}", flush=True)
        # Exercise both Python overloads, partial prefixes and the empty-request path.
        for epoch, (active, raw) in enumerate(((100, False), (75, True), (0, False))):
            run_round(epoch, active, raw)
        run_round(3, 75, synchronize=False)
        for epoch in range(args.warmup):
            run_round(epoch + 4, args.active_percent)
        for epoch in range(args.iterations):
            elapsed, size = run_round(epoch + 4 + args.warmup, args.active_percent)
            samples_ns.append(elapsed)
            payload_bytes.append(size)
        if args.profile_rounds:
            profile_epoch = 4 + args.warmup + args.iterations
            run_round(profile_epoch, args.active_percent, profile=True)
            profiles.clear()
            for epoch in range(args.profile_rounds):
                run_round(profile_epoch + 1 + epoch, args.active_percent, profile=True)
    finally:
        # Registered source/output allocations remain alive until teardown.
        runtime.shutdown()

    samples_us = [value / 1000 for value in samples_ns]
    bandwidth = sum(payload_bytes) / sum(samples_ns)  # bytes/ns == decimal GB/s.
    print(
        f"AICPU_BATCH_GET_PASS mode={mode} workers={workers} iterations={args.iterations} "
        f"mean_us={statistics.mean(samples_us):.1f} "
        f"p50_us={percentile(samples_us, 0.50):.1f} "
        f"p95_us={percentile(samples_us, 0.95):.1f} "
        f"p99_us={percentile(samples_us, 0.99):.1f} GBps={bandwidth:.3f} "
        f"output_order={args.output_order} output_row_step={args.output_row_step} "
        f"active_percent={args.active_percent} "
        "verification=every_round_full_output npu_consumer_checked=1 "
        "padding_checked=1 changing_inputs=1",
        flush=True,
    )
    if profiles:
        print_profile(profiles, mode)


def main() -> None:
    args = parse_args()
    import torch
    import torch_npu  # noqa: F401
    import kvaf

    torch.npu.set_device(args.device_id)
    print(
        "AICPU_BATCH_GET_TIMING metric=python_batch_get_to_stream_completion "
        "input_preparation_batch_put_table_setup_and_verification_excluded=1",
        flush=True,
    )
    for mode in args.modes:
        for workers in args.workers:
            run_case(args, workers, torch, kvaf, mode)


if __name__ == "__main__":
    main()
