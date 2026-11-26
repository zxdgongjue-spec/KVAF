#!/usr/bin/env python3
"""Benchmark AICPU GatherRead with one append_only Key per tensor."""

import argparse
import csv
import json
import os
from pathlib import Path
import random
import sys

from batch_get_ranges_perf import U32, align_up, host_sample, put_sample, summarize
from npu_eids import parse_npu_eids
from test_npu_batch_get_ranges import pattern

MIB = 1024 * 1024
TOKEN_BYTES = 656
MAX_REQUESTS = 65536


def worker_payloads(lengths, workers):
    count = len(lengths)
    return [sum(lengths[count * i // workers:count * (i + 1) // workers])
            for i in range(workers)]


def slot_count(lengths, slot_size):
    slots, used = 1, 0
    for length in lengths:
        if length > slot_size - used:
            slots, used = slots + 1, 0
        used += length
    return slots


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tensor-bytes", type=int, nargs="+", default=[MIB, 32 * MIB])
    p.add_argument("--repeat", type=int, default=2, help="repeat the size list, one Key per tensor")
    p.add_argument("--workers", type=int, nargs="+", choices=range(1, 9), default=[2])
    p.add_argument("--read-chunk-bytes", type=int, nargs="+", default=[0, 16 * 1024, 32 * 1024],
                   help="READ WR byte limit. 0 uses whole ranges as SGEs; positive values split by bytes")
    p.add_argument("--slot-size", type=int, default=0,
                   help="0 sizes slots to cover the heaviest lane across all requested worker counts")
    p.add_argument("--request-order", choices=["sequential", "reverse", "random"], default="sequential")
    p.add_argument("--seed", type=int, default=20261124)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--device", type=int, default=int(os.environ.get("DEVICE_ID", "0")))
    p.add_argument("--peer", default=os.environ.get("KVAF_CPU_SERVICE_ADDR"))
    p.add_argument("--npu-eids", type=parse_npu_eids,
                   default=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")))
    p.add_argument("--output-csv", type=Path, help="write the timing summaries to this CSV")
    p.add_argument("--dry-run", action="store_true", help="print capacity plans without an NPU")
    a = p.parse_args(argv)
    if a.repeat < 1 or len(a.tensor_bytes) * a.repeat > MAX_REQUESTS:
        p.error("total Tensor count must be in 1..65536")
    if any(not 1 <= n <= U32 for n in a.tensor_bytes):
        p.error("each tensor byte length must be in 1..UINT32_MAX")
    if any(not 0 <= n <= U32 for n in a.read_chunk_bytes):
        p.error("read-chunk-bytes must be in 0..UINT32_MAX")
    if a.slot_size < 0 or a.warmup < 0 or a.iterations < 1 or a.device < 0:
        p.error("invalid slot size, warmup count, iteration count or device ID")
    if not a.dry_run and not a.peer:
        p.error("--peer or KVAF_CPU_SERVICE_ADDR is required")
    return a


def make_plan(a):
    lengths = a.tensor_bytes * a.repeat
    order = list(range(len(lengths)))
    if a.request_order == "reverse":
        order.reverse()
    elif a.request_order == "random":
        random.Random(a.seed).shuffle(order)
    ordered_lengths = [lengths[i] for i in order]
    lane_bytes = {workers: worker_payloads(ordered_lengths, workers) for workers in a.workers}
    peak = max(max(loads) for loads in lane_bytes.values())
    if peak > U32:
        raise ValueError(f"one worker requires {peak} bytes, above the GatherRead uint32 limit")
    slot_size = a.slot_size or align_up(peak, 4096)
    if slot_size < max(lengths):
        raise ValueError("slot-size is smaller than the largest tensor")
    slots = slot_count(lengths, slot_size)
    plans = []
    for workers, loads in lane_bytes.items():
        max_requests_in_lane = (MAX_REQUESTS + workers - 1) // workers
        stride = align_up(max(min(slot_size, U32), TOKEN_BYTES * max_requests_in_lane), 64)
        capacity = min(stride, U32)
        for worker, load in enumerate(loads):
            if load > capacity:
                raise ValueError(f"workers={workers} worker={worker} requires {load} bytes, "
                                 f"staging capacity={capacity}. Increase --slot-size or reduce --repeat")
        descriptor = align_up(64 + max_requests_in_lane * 16, 64)
        workspace = align_up(workers * (stride + descriptor + 64) + 128, slot_size)
        if slots * slot_size + workspace >= (1 << 64):
            raise ValueError("remote DDR allocation exceeds uint64")
        plans.append(dict(workers=workers, worker_payload_bytes=loads, lane_capacity_bytes=capacity,
                          slot_size=slot_size, slots=slots, ddr_bytes=slots * slot_size,
                          workspace_bytes=workspace, requests=len(lengths),
                          payload_bytes=sum(lengths), request_order=a.request_order,
                          tensor_bytes=a.tensor_bytes, repeat=a.repeat, seed=a.seed))
    return lengths, order, plans


def run(a):
    lengths, order, plans = make_plan(a)
    for plan in plans:
        print("AICPU_GATHER_RANGES_PERF_PLAN " + json.dumps(plan, separators=(",", ":")), flush=True)
    if a.dry_run:
        return 0

    import torch
    import torch_npu  # noqa: F401
    import kvaf

    torch.npu.set_device(a.device)
    device = f"npu:{a.device}"
    stream = torch.npu.current_stream()

    payload_bytes = sum(lengths)
    source_host = bytearray(payload_bytes)
    offsets, cursor = [], 0
    for i, length in enumerate(lengths):
        offsets.append(cursor)
        source_host[cursor:cursor + length] = pattern(i + 1, length)
        cursor += length
    source = torch.frombuffer(source_host, dtype=torch.uint8).to(device)
    output = torch.empty_like(source)
    metadata = (
        torch.tensor(order, dtype=torch.int32, device=device),
        torch.zeros(len(lengths), dtype=torch.int64, device=device),
        torch.tensor([lengths[i] for i in order], dtype=torch.int64, device=device),
        torch.tensor([len(lengths)], dtype=torch.int32, device=device),
        torch.tensor([offsets[i] for i in order], dtype=torch.int64, device=device),
    )
    stream.synchronize()
    keys = list(range(len(lengths)))
    rows = []

    def emit(plan, operation, scope, samples, chunk=None):
        row = dict(plan, device=a.device, allocation_mode="append_only", operation=operation,
                   scope=scope, read_chunk_bytes=chunk, effective_read_chunk_bytes=chunk,
                   **summarize(samples, payload_bytes, bandwidth=scope != "submit"))
        rows.append(row)
        print("AICPU_GATHER_RANGES_PERF_RESULT " + json.dumps(row, separators=(",", ":")), flush=True)

    for plan in plans:
        runtime = kvaf.NpuRuntime("", TOKEN_BYTES, 1, device_id=a.device,
                                  npu_core_count=plan["workers"], aicpu_batch_get_mode="gather_read",
                                  allocation_mode="append_only", npu_eids=a.npu_eids)
        handle = None

        def put():
            runtime.batch_put(keys, source, offsets, lengths, stream=stream)

        try:
            runtime.link_to(a.peer)
            runtime.initialize(plan["ddr_bytes"], plan["slot_size"])
            put()
            for _ in range(a.warmup):
                runtime.batch_free(keys)
                put()
            put_times = [put_sample(lambda: runtime.batch_free(keys), put)
                         for _ in range(a.iterations)]
            handle = runtime.prepare_access_table(keys)
            table = runtime.access_table_view(handle, device="npu")

            def check():
                runtime.check_batch_get_ranges_status(stream)

            def get(chunk):
                runtime.batch_get_ranges(table, *metadata, output, stream=stream,
                                         synchronize=False, read_chunk_bytes=chunk)

            def verify(chunk):
                output.fill_(0xa5)
                get(chunk)
                check()
                if output.cpu().numpy().tobytes() != source_host:
                    raise AssertionError(f"payload mismatch for workers={plan['workers']} chunk={chunk}")

            for chunk in a.read_chunk_bytes:
                verify(chunk)
                for _ in range(a.warmup):
                    get(chunk)
                    check()
                samples = [host_sample(lambda: get(chunk), stream.synchronize, check)
                           for _ in range(a.iterations)]
                verify(chunk)
                submits, totals = zip(*samples)
                emit(plan, "get_ranges", "submit", submits, chunk)
                emit(plan, "get_ranges", "call_to_completion", totals, chunk)
            emit(plan, "put", "call_to_completion", put_times)
        finally:
            failed = sys.exc_info()[0] is not None
            try:
                stream.synchronize()
                if handle is not None:
                    runtime.release_access_table(handle)
                runtime.shutdown()
            except Exception as error:
                if not failed:
                    raise
                print(f"AICPU_GATHER_RANGES_PERF_CLEANUP_FAIL {error}", file=sys.stderr)

    if a.output_csv:
        with a.output_csv.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"AICPU_GATHER_RANGES_PERF_PASS configurations={len(plans) * len(a.read_chunk_bytes)}",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as error:
        print(f"AICPU_GATHER_RANGES_PERF_FAIL {type(error).__name__}: {error}", file=sys.stderr)
        raise
