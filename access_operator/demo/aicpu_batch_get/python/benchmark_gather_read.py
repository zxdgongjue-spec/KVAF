#!/usr/bin/env python3
"""Benchmark AICPU BatchGet with full-pool random sources and random output rows."""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

from test_batch_get import timing_stats
from npu_eids import parse_npu_eids

DDR_BUDGET_BYTES = 100 * 1024**3
DEFAULT_TOKENS_PER_BLOCK = 128
DEFAULT_REQUEST_LENGTH = 32_768
PREFIXES = {"gather_read": "AICPU_GATHER_READ", "cpu_push_scatter": "AICPU_CPU_PUSH_SCATTER"}


def parse_args(argv=None, *, mode="gather_read"):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--device-id", type=int, default=7)
    p.add_argument("--peer", default="127.0.0.1:28990")
    p.add_argument("--npu-eids", type=parse_npu_eids,
                   default=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")),
                   help="comma-separated Host EID candidates (environment: KVAF_NPU_EIDS)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--token-bytes", type=int, default=656)
    p.add_argument("--miss-count", type=int, default=4096)
    p.add_argument("--table-capacity", type=int, help="optional smaller source pool; default fills the 100 GiB DDR budget")
    p.add_argument("--tokens-per-block", type=int, default=DEFAULT_TOKENS_PER_BLOCK)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    a = p.parse_args(argv)
    if not 1 <= a.workers <= 8 or not 0 <= a.miss_count <= 65536:
        p.error("workers must be in 1..8 and miss_count in 0..65536")
    if a.tokens_per_block <= 0:
        p.error("tokens per block must be positive")
    if (a.token_bytes <= 0 or a.token_bytes * 65536 > 0xFFFFFFFF or
            a.token_bytes * a.tokens_per_block > 0xFFFFFFFF or
            a.token_bytes * (a.tokens_per_block - 1) > 0x7FFFFFFF):
        p.error("invalid token/block size for the AICPU input ABI")
    max_capacity, _ = pool_layout(a.token_bytes, a.tokens_per_block, mode=mode)
    if a.table_capacity is None:
        a.table_capacity = max_capacity
    if not 1 <= a.table_capacity <= min(max_capacity, 0x7FFFFFFF):
        p.error("source pool plus workspace must fit the 100 GiB DDR budget")
    if a.warmup < 0 or a.iterations <= 0:
        p.error("invalid warmup or iterations")
    if a.device_id < 0:
        p.error("invalid device ID")
    return a


def pool_layout(token_bytes, tokens_per_block, *, mode="gather_read"):
    block_bytes = token_bytes * tokens_per_block
    if mode == "cpu_push_scatter":
        # CPU push staging and source-address workspace live in HBM.
        return DDR_BUDGET_BYTES // block_bytes, 0
    # Match GatherRemoteWorkspaceBytes and PeerDdrAllocator's slot alignment.
    # Reserve the maximum for 1..8 workers so every worker count uses the same source pool.
    reserved = 0
    for workers in range(1, 9):
        lane_capacity = (65536 + workers - 1) // workers
        staging = (lane_capacity * token_bytes + 63) // 64 * 64
        descriptor = (64 + lane_capacity * 16 + 63) // 64 * 64
        workspace = workers * (staging + descriptor + 64) + 128
        reserved = max(reserved, (workspace + block_bytes - 1) // block_bytes * block_bytes)
    capacity = (DDR_BUDGET_BYTES - reserved) // block_bytes
    return capacity, reserved


def create_request_plan(torch, a, length, device):
    # Sampling with replacement matches the AIV full-table random workload.
    indices = torch.randint(0, a.table_capacity, (length,), dtype=torch.int32, device=device)
    slots = torch.randint(0, a.tokens_per_block, (length,), dtype=torch.int32, device=device)
    rows = torch.randperm(length, device=device).to(torch.int32)
    return indices, slots * a.token_bytes, rows


def token_pattern(torch, token_ids, byte_ids):
    # Encode the source token ID across eight bytes so wrong-block READs are visible.
    # torch_npu bitwise ops require equal shapes here; materialize broadcasts.
    shifts = ((byte_ids % 8) * 8).expand(token_ids.numel(), -1).contiguous()
    ids = token_ids.unsqueeze(1).expand_as(shifts).contiguous()
    encoded = ((ids >> shifts) & 0xFF).to(torch.uint8)
    pattern = ((byte_ids * 17 + (byte_ids >> 3)) & 0xFF).to(torch.uint8)
    return encoded ^ pattern.expand_as(encoded).contiguous()


def summarize_measurements(profiles, payload_bytes, *, mode="gather_read"):
    device = timing_stats([sample["device_span_us"] for sample in profiles], payload_bytes)
    summary = dict(device_p50_us=device["p50_us"], p50_GBps=device["p50_GBps"])
    phases = (
        ("lookup_table_us", "request_us"),
        ("address_send_us", "source_write_us"),
        ("notify_cpu_us", "notify_us"),
        ("read_total_us", "read_us"),
        ("build_wqe_us", "wr_build_us"),
        ("cpu_gather_wait_us", "ready_wait_us"),
        ("post_us", "post_us"),
        ("cq_wait_us", "cq_wait_us"),
    ) if mode == "gather_read" else (
        ("lookup_table_us", "request_us"),
        ("notify_cpu_us", "notify_us"),
        ("push_total_us", "read_us"),
        ("ready_wait_us", "ready_wait_us"),
        ("scatter_us", "scatter_us"),
        ("done_wait_us", "ack_us"),
    )
    for name, field in phases:
        summary[name] = statistics.mean(max(worker[field] for worker in sample["workers"])
                                        for sample in profiles)
    return summary


def print_measurements(a, summary, *, mode="gather_read"):
    prefix = PREFIXES[mode]
    source_label = " source_mode=table" if mode == "gather_read" else ""
    print(f"{prefix}_PROFILE device_p50_us={summary['device_p50_us']:.1f} "
          f"p50_GBps={summary['p50_GBps']:.3f}{source_label} phase_aggregation=mean_of_per_round_worker_max")
    if mode == "gather_read":
        request_field = "lookup_table_us"
        print(f"  {request_field}={summary[request_field]:.1f} | "
              f"address_send_us={summary['address_send_us']:.1f} | "
              f"notify_cpu_us={summary['notify_cpu_us']:.1f}")
        print(f"  read_total_us={summary['read_total_us']:.1f}")
        print(f"    build_wqe_us={summary['build_wqe_us']:.1f} | "
              f"cpu_gather_wait_us={summary['cpu_gather_wait_us']:.1f} | "
              f"post_us={summary['post_us']:.1f} | cq_wait_us={summary['cq_wait_us']:.1f}")
    else:
        print(f"  lookup_table_us={summary['lookup_table_us']:.1f} | "
              f"notify_cpu_us={summary['notify_cpu_us']:.1f}")
        print(f"  push_total_us={summary['push_total_us']:.1f}")
        print(f"    ready_wait_us={summary['ready_wait_us']:.1f} | "
              f"scatter_us={summary['scatter_us']:.1f}")
        print(f"  done_wait_us={summary['done_wait_us']:.1f}")
    result = dict(token_bytes=a.token_bytes, miss_count=a.miss_count, workers=a.workers, **summary)
    if mode == "gather_read":
        result = dict(source_mode="table", **result)
    fields = []
    for key, value in result.items():
        formatted = f"{value:.3f}" if key == "p50_GBps" else f"{value:.1f}" if isinstance(value, float) else str(value)
        fields.append(f"{key}={formatted}")
    print(f"{prefix}_RESULT " + " ".join(fields), flush=True)


def run_case(a, torch, kvaf, *, mode="gather_read"):
    prefix = PREFIXES[mode]
    device = f"npu:{a.device_id}"
    length = max(DEFAULT_REQUEST_LENGTH, a.miss_count)
    block_bytes = a.token_bytes * a.tokens_per_block
    ddr_bytes = a.table_capacity * block_bytes
    payload_bytes = a.miss_count * a.token_bytes
    _, reserved = pool_layout(a.token_bytes, a.tokens_per_block, mode=mode)
    torch.manual_seed(20260903)
    stream = torch.npu.current_stream()
    runtime = kvaf.NpuRuntime(
        batch_get_kernel_object_path="", batch_get_token_bytes=a.token_bytes,
        batch_get_tokens_per_block=a.tokens_per_block, device_id=a.device_id,
        npu_core_count=a.workers, aicpu_batch_get_mode=mode, npu_eids=a.npu_eids,
    )
    profiles = []
    # Source and output allocations stay alive until the session is closed.
    chunk_entries = min(128, a.table_capacity)
    source = torch.empty((chunk_entries, block_bytes), dtype=torch.uint8, device=device)
    output = torch.empty((length, a.token_bytes), dtype=torch.uint8, device=device)
    byte_ids = torch.arange(a.token_bytes, dtype=torch.int64, device=device).unsqueeze(0)
    table = device_table = None
    print(f"{prefix}_CONFIG table_capacity={a.table_capacity} "
          f"tokens_per_block={a.tokens_per_block} request_length={length} "
          f"token_bytes={a.token_bytes} miss_count={a.miss_count} workers={a.workers} "
          f"source_pool_GiB={ddr_bytes / 1024**3:.3f} "
          f"workspace_budget_MiB={reserved / 1024**2:.3f} "
          f"total_budget_GiB={DDR_BUDGET_BYTES / 1024**3:.1f} "
          "source_mode=table source=random_with_replacement output=random_unique_rows "
          "guard_rows=0 source_initialized_once=1", flush=True)

    def run_round(count, profile=False):
        indices, offsets, rows = create_request_plan(torch, a, length, device)
        miss_count = torch.tensor([count], dtype=torch.int32, device=device)
        expected = torch.full_like(output, 0xA5)
        if count:
            token_ids = indices[:count].to(torch.int64) * a.tokens_per_block + offsets[:count].to(torch.int64) // a.token_bytes
            expected.index_copy_(0, rows[:count].to(torch.int64), token_pattern(torch, token_ids, byte_ids))
        output.fill_(0xA5)
        inputs = (device_table, indices, offsets, miss_count, rows, output)
        stream.synchronize()
        begin = time.perf_counter_ns()
        measured = runtime.batch_get(*inputs, stream=stream, synchronize=True, profile=profile)
        elapsed_us = (time.perf_counter_ns() - begin) / 1000
        # NPU comparison consumes BatchGet results and checks every row, including untouched rows.
        if not torch.equal(output, expected):
            raise AssertionError(f"{mode} output mismatch: miss_count={count}, workers={a.workers}")
        if profile:
            measured["python_call_us"] = elapsed_us
            measured["python_other_us"] = elapsed_us - measured["binding_total_us"]
            profiles.append(measured)
        return elapsed_us

    try:
        runtime.link_to(a.peer)
        runtime.initialize(ddr_bytes=ddr_bytes, slot_size=block_bytes)
        print(f"{prefix}_POPULATE_BEGIN", flush=True)
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
        print(f"{prefix}_POPULATE_DONE", flush=True)
        run_round(0)
        run_round(a.miss_count)
        for _ in range(a.warmup):
            run_round(a.miss_count, profile=True)
        profiles.clear()
        for _ in range(a.iterations):
            run_round(a.miss_count, profile=True)
    except BaseException:
        try:
            runtime.shutdown()
        except Exception as cleanup_error:
            print(f"{prefix}_CLEANUP_ERROR {cleanup_error}", file=sys.stderr, flush=True)
        raise
    else:
        runtime.shutdown()

    print(f"{prefix}_PASS workers={a.workers} iterations={a.iterations} "
          "verification=every_round_full_output_npu_compare", flush=True)
    summary = summarize_measurements(profiles, payload_bytes, mode=mode)
    print_measurements(a, summary, mode=mode)


def main(*, mode="gather_read"):
    a = parse_args(mode=mode)
    import torch
    import torch_npu  # noqa: F401
    import kvaf
    torch.npu.set_device(a.device_id)
    run_case(a, torch, kvaf, mode=mode)


if __name__ == "__main__":
    main()
