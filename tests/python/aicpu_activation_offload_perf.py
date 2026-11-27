#!/usr/bin/env python3
"""Measure eager AICPU activation offload with variable sequence lengths."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

from aicpu_gather_ranges_perf import MIB, MAX_REQUESTS, TOKEN_BYTES, slot_count, worker_payloads
from batch_get_ranges_perf import U32, align_up, make_layout, summarize
from npu_eids import parse_npu_eids
from test_npu_batch_get_ranges import pattern


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lengths", type=int, nargs="+", action="append",
                        help="one batch of sequence lengths, repeat to cycle through shapes")
    parser.add_argument("--bytes-per-token", type=int, default=4096,
                        help="offloaded activation bytes per token in one layer, per rank")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--token-bytes", type=int, default=TOKEN_BYTES,
                        help="KVAF fixed-token setting used to size GatherRead staging")
    parser.add_argument("--slot-bytes", type=int, default=4 * MIB)
    parser.add_argument("--allocation-mode", choices=("fixed", "append_only"), default="fixed")
    parser.add_argument("--put-group-size", type=int, default=0, help="0 puts all sequences of a layer together")
    parser.add_argument("--get-group-size", type=int, default=0, help="0 reads all sequences of a layer together")
    parser.add_argument("--buffers", choices=("reuse", "per-layer"), default="reuse",
                        help="share registered ranges across layers or use separate ranges per layer")
    parser.add_argument("--warmup", type=int, default=1, help="warmup steps per shape")
    parser.add_argument("--steps", type=int, default=5, help="timed steps per shape")
    parser.add_argument("--peer", default=os.environ.get("KVAF_CPU_SERVICE_ADDR"))
    parser.add_argument("--device", type=int, default=int(os.environ.get("DEVICE_ID", "0")))
    parser.add_argument("--npu-eids", type=parse_npu_eids,
                        default=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")))
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--trace", action="store_true", help="print the phase before each potentially blocking call")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the workload without NPU or KVAF")
    args = parser.parse_args(argv)
    args.seq_lengths = args.seq_lengths or [[128, 384, 768, 1024], [1024, 192, 640, 256]]
    if min(args.layers, args.bytes_per_token, args.slot_bytes, args.steps) < 1:
        parser.error("layers, bytes-per-token, slot-bytes and steps must be positive")
    if min(args.put_group_size, args.get_group_size, args.warmup, args.device) < 0:
        parser.error("group sizes, warmup and device must be nonnegative")
    if not 1 <= args.token_bytes <= U32 // MAX_REQUESTS:
        parser.error("token-bytes must be in 1..65535 for AICPU GatherRead")
    if not args.dry_run and not args.peer:
        parser.error("--peer or KVAF_CPU_SERVICE_ADDR is required")
    return args


def group_indices(order, size):
    size = size or len(order)
    return [tuple(order[first:first + size]) for first in range(0, len(order), size)]


def make_plan(args):
    lane_requests = (MAX_REQUESTS + args.workers - 1) // args.workers
    minimum_staging = min(args.slot_bytes, U32) if args.allocation_mode == "append_only" else 0
    lane_stride = align_up(max(lane_requests * args.token_bytes, minimum_staging), 64)
    lane_capacity = min(lane_stride, U32)
    descriptor_bytes = align_up(64 + lane_requests * 16, 64)
    workspace = align_up(args.workers * (lane_stride + descriptor_bytes + 64) + 128, args.slot_bytes)
    shapes = []
    for sequences in args.seq_lengths:
        lengths = tuple(seq * args.bytes_per_token for seq in sequences)
        if any(n < 1 or n > min(U32, args.slot_bytes) for n in lengths):
            raise ValueError(f"tensor sizes {lengths} must fit slot-bytes and uint32")
        order = list(range(len(lengths)))
        put_groups = group_indices(order, args.put_group_size)
        get_groups = group_indices(order[::-1], args.get_group_size)
        if any(len(group) > MAX_REQUESTS for group in put_groups + get_groups):
            raise ValueError("each Put/Get group must contain at most 65536 tensors")
        loads = [worker_payloads([lengths[i] for i in group], args.workers) for group in get_groups]
        peak = max(map(max, loads))
        if peak > lane_capacity:
            raise ValueError(f"one worker needs {peak} bytes but staging holds {lane_capacity}; "
                             "reduce --get-group-size or increase staging")
        slots = (args.layers * len(lengths) if args.allocation_mode == "fixed"
                 else slot_count(lengths * args.layers, args.slot_bytes))
        shapes.append(dict(sequences=sequences, layout=make_layout(lengths, 1, 1),
                           put_groups=put_groups, get_groups=get_groups, worker_bytes=loads, slots=slots))

    ddr_bytes = max(shape["slots"] for shape in shapes) * args.slot_bytes
    if ddr_bytes + workspace >= 1 << 64:
        raise ValueError("CPU DDR region plus workspace exceeds uint64")
    input_stride = align_up(max(shape["layout"].packet_bytes for shape in shapes), 4096)
    output_stride = align_up(max(sum(shape["layout"].lengths[i] for i in group)
                                 for shape in shapes for group in shape["get_groups"]), 4096)
    output_count = args.layers if args.buffers == "per-layer" else 1
    print("ACTIVATION_OFFLOAD_PLAN " + json.dumps(dict(
        device=args.device, workers=args.workers, layers=args.layers, buffers=args.buffers,
        allocation_mode=args.allocation_mode, token_bytes=args.token_bytes, slot_bytes=args.slot_bytes,
        cpu_ddr_payload_bytes=ddr_bytes, cpu_ddr_workspace_bytes=workspace,
        staging_bytes_per_worker=lane_capacity, input_hbm_bytes=args.layers * input_stride,
        output_hbm_bytes=output_count * output_stride,
        shapes=[dict(seq_lengths=s["sequences"], tensor_bytes=s["layout"].lengths,
                     payload_bytes_per_step=s["layout"].packet_bytes * args.layers,
                     put_calls_per_step=len(s["put_groups"]) * args.layers,
                     get_calls_per_step=len(s["get_groups"]) * args.layers,
                     get_worker_bytes=s["worker_bytes"]) for s in shapes])), flush=True)
    return shapes, ddr_bytes, input_stride, output_stride


def run(args):
    shapes, ddr_bytes, input_stride, output_stride = make_plan(args)
    if args.dry_run:
        return 0

    import torch
    import torch_npu  # noqa: F401
    import kvaf

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    stream = torch.npu.current_stream()
    source_host = bytearray(args.layers * input_stride)
    for layer in range(args.layers):
        start = layer * input_stride
        source_host[start:start + input_stride] = pattern(layer + 1, input_stride)
    source = torch.frombuffer(source_host, dtype=torch.uint8).to(device)
    shared_buffers = args.buffers == "reuse"
    output_count = 1 if shared_buffers else args.layers
    outputs = [torch.empty(output_stride, dtype=torch.uint8, device=device) for _ in range(output_count)]
    sources = ([source] * args.layers if shared_buffers else
               [source.narrow(0, layer * input_stride, input_stride) for layer in range(args.layers)])
    for shape in shapes:
        sizes = shape["layout"].lengths
        metadata = []
        for group in shape["get_groups"]:
            layout = make_layout([sizes[i] for i in group], 1, 1)
            metadata.append((torch.arange(len(group), dtype=torch.int32, device=device),
                             torch.zeros(len(group), dtype=torch.int64, device=device),
                             torch.tensor(layout.lengths, dtype=torch.int64, device=device),
                             torch.tensor([len(group)], dtype=torch.int32, device=device),
                             torch.tensor(layout.offsets, dtype=torch.int64, device=device)))
        shape["metadata"] = metadata
    stream.synchronize()

    runtime = kvaf.NpuRuntime(transport="urma", aicpu_batch_get_mode="gather_read",
                              batch_get_token_bytes=args.token_bytes, batch_get_tokens_per_block=1,
                              device_id=args.device, npu_core_count=args.workers,
                              allocation_mode=args.allocation_mode, npu_eids=args.npu_eids)
    now = time.perf_counter_ns

    def run_step(shape_index, label, verify=False):
        shape = shapes[shape_index]
        layout = shape["layout"]
        count = len(layout.lengths)
        times = {}

        def measure(phase, layer, group, call, *params, **kwargs):
            if args.trace:
                print(f"==============[KVAF_AICPU]=========== ACTIVATION_OFFLOAD_PHASE step={label} shape={shape_index} "
                      f"layer={layer} group={group} phase={phase}", flush=True)
            start = now()
            result = call(*params, **kwargs)
            times[phase] = times.get(phase, 0) + now() - start
            return result

        started = now()
        for layer in range(args.layers):
            base = layer * input_stride if shared_buffers else 0
            for group_index, group in enumerate(shape["put_groups"]):
                measure("put", layer, group_index, runtime.batch_put,
                        [layer * count + i for i in group], sources[layer],
                        [base + layout.offsets[i] for i in group], [layout.lengths[i] for i in group],
                        stream=stream)
        finished_forward = now()
        for layer in reversed(range(args.layers)):
            output = outputs[0 if shared_buffers else layer]
            for group_index, (group, metadata) in enumerate(zip(shape["get_groups"], shape["metadata"])):
                keys = [layer * count + i for i in group]
                if verify:
                    output.fill_(0xa5)
                    stream.synchronize()
                handle = measure("get_prepare_table", layer, group_index, runtime.prepare_access_table, keys)
                table = measure("get_device_table", layer, group_index, runtime.access_table_view, handle, device="npu")
                measure("get_submit", layer, group_index, runtime.batch_get_ranges,
                        table, *metadata, output, stream=stream, synchronize=False)
                measure("get_wait", layer, group_index, stream.synchronize)
                measure("get_status_check", layer, group_index, runtime.check_batch_get_ranges_status, stream)
                if verify:
                    base = layer * input_stride
                    expected = b"".join(source_host[base + layout.offsets[i]:
                                                    base + layout.offsets[i] + layout.lengths[i]] for i in group)
                    if output[:len(expected)].cpu().numpy().tobytes() != expected:
                        raise AssertionError(f"readback mismatch for shape={shape_index} layer={layer} group={group_index}")
                measure("get_release_table", layer, group_index, runtime.release_access_table, handle)
            measure("free", layer, 0, runtime.batch_free, list(range(layer * count, (layer + 1) * count)))
        finished = now()
        times["get_completion"] = times["get_submit"] + times["get_wait"]
        times["get_checked_completion"] = times["get_completion"] + times["get_status_check"]
        times["forward_wall"] = finished_forward - started
        times["backward_wall"] = finished - finished_forward
        times["step_wall"] = finished - started
        return times

    rows = []

    def report(shape_index, stage, samples):
        shape = shapes[shape_index]
        payload = shape["layout"].packet_bytes * args.layers
        metrics = {}
        for phase in samples[0]:
            summary = summarize([sample[phase] for sample in samples], payload,
                                bandwidth=phase in {"put", "get_completion", "get_checked_completion",
                                                    "forward_wall", "backward_wall"})
            rows.append(dict(device=args.device, workers=args.workers, layers=args.layers, buffers=args.buffers,
                             allocation_mode=args.allocation_mode, shape=shape_index,
                             seq_lengths="/".join(map(str, shape["sequences"])), stage=stage,
                             phase=phase, payload_bytes=payload, **summary))
            metrics[phase] = {key: round(summary[key], 3) for key in ("p50_us", "p99_us", "mean_GiBps")
                              if summary[key] is not None}
        print("ACTIVATION_OFFLOAD_RESULT " + json.dumps(dict(
            stage=stage, shape=shape_index, seq_lengths=shape["sequences"],
            payload_bytes=payload, samples=len(samples), phases=metrics)), flush=True)

    try:
        print(f"ACTIVATION_OFFLOAD_INITIALIZE peer={args.peer} native={kvaf._kvaf.__file__}", flush=True)
        start = now()
        runtime.link_to(args.peer)
        runtime.initialize(ddr_bytes, args.slot_bytes)
        print(f"ACTIVATION_OFFLOAD_READY initialize_us={(now() - start) / 1000:.3f}", flush=True)
        report(0, "cold", [run_step(0, "cold")])
        for shape in range(len(shapes)):
            run_step(shape, "verify", verify=True)
        for step in range(args.warmup):
            for shape in range(len(shapes)):
                run_step(shape, f"warmup-{step}")
        samples = [[] for _ in shapes]
        for step in range(args.steps):
            for shape in range(len(shapes)):
                samples[shape].append(run_step(shape, f"steady-{step}"))
        for shape, measurements in enumerate(samples):
            run_step(shape, "verify-final", verify=True)
            report(shape, "steady", measurements)
    finally:
        failed = sys.exc_info()[0] is not None
        try:
            stream.synchronize()
            runtime.shutdown()
        except Exception as error:
            if not failed:
                raise
            print(f"ACTIVATION_OFFLOAD_CLEANUP_FAIL {error}", file=sys.stderr)

    if args.output_csv:
        with args.output_csv.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print("ACTIVATION_OFFLOAD_PASS", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (RuntimeError, ValueError) as error:
        print(f"ACTIVATION_OFFLOAD_FAIL {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
