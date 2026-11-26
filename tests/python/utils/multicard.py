"""Coordinate a sustained Get graph worker with the multi-device controller."""

import json
import time
from functools import partial

from .get_graph import captured_get_nodes
from .get_options import parallelism_field


def run_worker(kvaf, torch, args, runtime, execution, output,
               make_plan, submit, check_output, summarize_profile):
    control = args.contention_control_dir
    device = args.device_id
    stream = torch.npu.current_stream()
    plans = [make_plan() for _ in range(args.contention_graph_calls)]
    submit(plans[0], stream, execution.get_kwargs)
    execution.complete_async_get(runtime)
    nodes = [(output, partial(submit, plan, stream)) for plan in plans]
    print(f"MULTICARD_GRAPH_CAPTURE_BEGIN backend={args.backend} calls={len(nodes)}", flush=True)
    with captured_get_nodes(args, runtime, torch, kvaf, execution, stream, nodes) as (graph, leases):
        print("MULTICARD_GRAPH_CAPTURE_DONE", flush=True)
        graph.replay()
        graph.synchronize()
        check_output(plans[-1], leases[-1])
        for _ in range(args.warmup):
            graph.replay()
            graph.synchronize()

        print(f"MULTICARD_READY device={device} peer={args.peer}", flush=True)
        (control / f"ready_{device}").touch()
        while not (control / "start.json").exists():
            if (control / "abort").exists():
                raise RuntimeError("multi-card run aborted before start")
            time.sleep(0.02)
        window = json.loads((control / "start.json").read_text())
        remaining = window["start"] - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        profiles = []
        begin = time.monotonic()
        batches = 0
        while time.monotonic() < window["stop"]:
            if (control / "abort").exists():
                raise RuntimeError("multi-card run aborted during measurement")
            graph.replay()
            graph.synchronize()
            batches += 1
            if args.kernel_profile:
                profiles.append(runtime.batch_get_kernel_profile(graph_lease=leases[-1]))
        end = time.monotonic()
        if not batches:
            raise RuntimeError("worker missed the measurement window")
        check_output(plans[-1], leases[-1])
        graph.validated = True
        calls = batches * len(plans)
        payload_bytes = calls * args.miss_count * args.token_bytes
        result = {
            "device": device, "peer": args.peer, "backend": args.backend, "graph_api": args.graph_api,
            "start": begin, "end": end, "calls": calls, "payload_bytes": payload_bytes,
            "bandwidth_GBps": payload_bytes / (end - begin) / 1e9,
            "graph_calls": len(plans), "graphs": 1, "batches": batches, "verified": True,
            "request_length": args.request_length, "miss_count": args.miss_count,
            parallelism_field(args.backend): args.parallelism,
            "profiling_enabled": args.kernel_profile,
            "execution_mode": args.execution_mode,
        }
        if args.backend == "aiv":
            result["jettys_per_core"] = args.jettys_per_core
        if args.transport == "urma":
            result["peer_ub_device"] = args.peer_ub_device
    if args.kernel_profile:
        result["kernel_profile"] = summarize_profile(args, profiles)
    args.contention_result.write_text(json.dumps(result, indent=2) + "\n")
    print("MULTICARD_RESULT " + " ".join(f"{k}={v}" for k, v in result.items()), flush=True)
