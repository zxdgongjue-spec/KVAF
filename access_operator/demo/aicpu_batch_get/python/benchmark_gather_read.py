#!/usr/bin/env python3
"""Benchmark GatherRead with full-pool random sources and random output rows."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from test_batch_get import print_profile, summarize_profiles, timing_stats

DEFAULT_TABLE_CAPACITY = 16_384 * 78
DEFAULT_TOKENS_PER_BLOCK = 128
DEFAULT_REQUEST_LENGTH = 32_768


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device-id", type=int, default=7)
    p.add_argument("--peer", default="127.0.0.1:28990")
    p.add_argument("--urma-device", default="")
    p.add_argument("--eid-index", type=int, default=0)
    cann = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_CANN_PACKAGE_PATH")
    p.add_argument("--kernel-json", default=str(Path(cann) / "opp/vendors/cust/op_impl/aicpu/config/kvaf_aicpu_urma.json") if cann else None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--token-bytes", type=int, default=656)
    p.add_argument("--miss-count", type=int, default=4096)
    p.add_argument("--table-capacity", type=int, default=DEFAULT_TABLE_CAPACITY)
    p.add_argument("--tokens-per-block", type=int, default=DEFAULT_TOKENS_PER_BLOCK)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    a = p.parse_args(argv)
    if not 1 <= a.workers <= 8 or not 0 <= a.miss_count <= 65536:
        p.error("workers must be in 1..8 and miss_count in 0..65536")
    if not 1 <= a.table_capacity <= 0x7FFFFFFF or a.tokens_per_block <= 0:
        p.error("table capacity must fit positive int32; tokens per block must be positive")
    if (a.token_bytes <= 0 or a.token_bytes * 65536 > 0xFFFFFFFF or
            a.token_bytes * a.tokens_per_block > 0xFFFFFFFF or
            a.token_bytes * (a.tokens_per_block - 1) > 0x7FFFFFFF):
        p.error("invalid token/block size for the AICPU input ABI")
    if a.warmup < 0 or a.iterations <= 0:
        p.error("invalid warmup or iterations")
    if a.device_id < 0 or not 0 <= a.eid_index < 0xFFFFFFFF:
        p.error("invalid device or EID index")
    if not a.kernel_json or not Path(a.kernel_json).is_file():
        p.error("source the CANN environment or pass an installed --kernel-json")
    return a


def create_request_plan(torch, a, length, device):
    # Sampling with replacement matches the AIV full-table random workload.
    indices = torch.randint(0, a.table_capacity, (length,), dtype=torch.int32, device=device)
    slots = torch.randint(0, a.tokens_per_block, (length,), dtype=torch.int32, device=device)
    rows = torch.randperm(length, device=device).to(torch.int32)
    return indices, slots * a.token_bytes, rows


def token_pattern(torch, token_ids, byte_ids):
    # Encode the source token ID across eight bytes so wrong-block READs are visible.
    return (((token_ids.unsqueeze(1) >> ((byte_ids % 8) * 8)) ^
             (byte_ids * 17 + (byte_ids >> 3))) & 0xFF).to(torch.uint8)


def summarize_measurements(samples_us, profiles, payload_bytes):
    end_to_end = {key: value for key, value in timing_stats(samples_us).items()
                  if key.endswith("_us")}
    stages = summarize_profiles(profiles, payload_bytes)
    for name, stats in stages.items():
        stats.pop("GBps")
        if name != "kernel_total":
            stats.pop("p50_GBps")
    return end_to_end, stages


def run_case(a, torch, kvaf):
    device = f"npu:{a.device_id}"
    length = max(DEFAULT_REQUEST_LENGTH, a.miss_count)
    block_bytes = a.token_bytes * a.tokens_per_block
    ddr_bytes = a.table_capacity * block_bytes
    payload_bytes = a.miss_count * a.token_bytes
    torch.manual_seed(20260903)
    stream = torch.npu.current_stream()
    runtime = kvaf.NpuRuntime(
        batch_get_kernel_object_path="", batch_get_token_bytes=a.token_bytes,
        batch_get_tokens_per_block=a.tokens_per_block, device_id=a.device_id,
        npu_core_count=a.workers, aicpu_urma_kernel_json_path=a.kernel_json,
        urma_device=a.urma_device, urma_eid_index=a.eid_index,
        aicpu_batch_get_mode="gather_read",
    )
    samples_us, profiles = [], []
    # Source and output allocations stay alive until the session is closed.
    chunk_entries = min(128, a.table_capacity)
    source = torch.empty((chunk_entries, block_bytes), dtype=torch.uint8, device=device)
    output = torch.empty((length, a.token_bytes), dtype=torch.uint8, device=device)
    byte_ids = torch.arange(a.token_bytes, dtype=torch.int64, device=device).unsqueeze(0)
    table = device_table = None
    print(f"AICPU_GATHER_READ_CONFIG table_capacity={a.table_capacity} "
          f"tokens_per_block={a.tokens_per_block} request_length={length} "
          f"token_bytes={a.token_bytes} miss_count={a.miss_count} workers={a.workers} "
          f"ddr_bytes={ddr_bytes} source=random_with_replacement output=random_unique_rows "
          "guard_rows=0 source_initialized_once=1", flush=True)

    def run_round(count, profile=False):
        indices, offsets, rows = create_request_plan(torch, a, length, device)
        miss_count = torch.tensor([count], dtype=torch.int32, device=device)
        expected = torch.full_like(output, 0xA5)
        if count:
            token_ids = indices[:count].to(torch.int64) * a.tokens_per_block + offsets[:count].to(torch.int64) // a.token_bytes
            expected.index_copy_(0, rows[:count].to(torch.int64), token_pattern(torch, token_ids, byte_ids))
        output.fill_(0xA5)
        stream.synchronize()
        begin = time.perf_counter_ns()
        measured = runtime.batch_get(device_table, indices, offsets, miss_count, rows, output,
                                     stream=stream, synchronize=True, profile=profile)
        elapsed_us = (time.perf_counter_ns() - begin) / 1000
        # NPU comparison consumes READ results and checks every row, including untouched rows.
        if not torch.equal(output, expected):
            raise AssertionError(f"GatherRead output mismatch: miss_count={count}, workers={a.workers}")
        if profile:
            measured["python_call_us"] = elapsed_us
            measured["python_other_us"] = elapsed_us - measured["binding_total_us"]
            profiles.append(measured)
        return elapsed_us

    try:
        runtime.link_to(a.peer)
        runtime.initialize(ddr_bytes=ddr_bytes, slot_size=block_bytes)
        print("AICPU_GATHER_READ_POPULATE_BEGIN", flush=True)
        for start in range(0, a.table_capacity, chunk_entries):
            count = min(chunk_entries, a.table_capacity - start)
            ids = torch.arange(start * a.tokens_per_block, (start + count) * a.tokens_per_block,
                               dtype=torch.int64, device=device)
            source[:count].copy_(token_pattern(torch, ids, byte_ids).reshape(count, block_bytes))
            runtime.batch_put(list(range(start, start + count)), source[:count],
                              [i * block_bytes for i in range(count)], [block_bytes] * count, stream)
        table = runtime.prepare_access_table(torch.arange(a.table_capacity, dtype=torch.int64))
        device_table = runtime.access_table_view(table, device="npu", capacity=a.table_capacity)
        stream.synchronize()
        print("AICPU_GATHER_READ_POPULATE_DONE", flush=True)
        run_round(0)
        run_round(a.miss_count)
        for _ in range(a.warmup):
            run_round(a.miss_count, profile=True)
        profiles.clear()
        for _ in range(a.iterations):
            samples_us.append(run_round(a.miss_count, profile=True))
    finally:
        runtime.shutdown()

    stats, stages = summarize_measurements(samples_us, profiles, payload_bytes)
    print(f"AICPU_GATHER_READ_PASS workers={a.workers} iterations={a.iterations} "
          f"mean_us={stats['mean_us']:.1f} p50_us={stats['p50_us']:.1f} "
          f"p90_us={stats['p90_us']:.1f} p99_us={stats['p99_us']:.1f} "
          f"p50_GBps={stages['kernel_total']['p50_GBps']:.3f} "
          "timing=end_to_end bandwidth_scope=kernel_total "
          "verification=every_round_full_output_npu_compare", flush=True)
    if profiles:
        print_profile(profiles, "gather_read", warmup=a.warmup, separate_samples=False)
    result = dict(token_bytes=a.token_bytes, miss_count=a.miss_count, workers=a.workers,
                  table_capacity=a.table_capacity, tokens_per_block=a.tokens_per_block,
                  request_length=length, ddr_bytes=ddr_bytes, payload_bytes=payload_bytes,
                  warmup=a.warmup, iterations=a.iterations)
    quantiles = ("mean_us", "p50_us", "p90_us", "p99_us")
    result.update({key: stages["kernel_total"][key] for key in (*quantiles, "p50_GBps")})
    for name in ("post", "completion_wait"):
        result.update({f"{name}_{key}": stages[name][key] for key in quantiles})
    result.update({f"end_to_end_{key}": stats[key] for key in quantiles})
    for name in ("read", "ready_wait", "wr_build", "request", "source_write", "notify", "ack"):
        result[f"{name}_mean_us"] = stages[name]["mean_us"]
    fields = []
    for key, value in result.items():
        formatted = f"{value:.3f}" if key == "p50_GBps" else f"{value:.1f}" if isinstance(value, float) else str(value)
        fields.append(f"{key}={formatted}")
    print("AICPU_GATHER_READ_RESULT " + " ".join(fields), flush=True)


def main():
    a = parse_args()
    import torch
    import torch_npu  # noqa: F401
    import kvaf
    torch.npu.set_device(a.device_id)
    run_case(a, torch, kvaf)


if __name__ == "__main__":
    main()
