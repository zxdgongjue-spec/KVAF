"""Coordinate a sustained Get graph worker with the multi-device controller."""

import json
import time
from contextlib import ExitStack

from .get_graph import GraphInputs, captured_get_nodes, validate_graph_inputs
from .get_options import parallelism_field


def run_worker(kvaf, torch, args, runtime, execution, miss_count, output,
               make_plan, submit, check_output):
    control = args.contention_control_dir
    device = args.device_id
    stream = torch.npu.current_stream()
    inputs = GraphInputs(make_plan, miss_count, output, stream)
    inputs.prepare(args.miss_count)
    submit(inputs, stream, execution.get_kwargs)
    execution.complete_async_get(runtime)
    check_output(inputs, args.miss_count)
    execution.release_output(runtime)

    graphs = []
    graph_inputs = []
    leases = []

    def replay_batch():
        for graph in graphs:
            graph.replay()
        stream.synchronize()

    print(f"MULTICARD_GRAPH_CAPTURE_BEGIN backend={args.backend} api={args.graph_api} "
          f"graphs={args.contention_graphs}", flush=True)
    with ExitStack() as captured:
        for index in range(args.contention_graphs):
            current = inputs if index == 0 else GraphInputs(
                make_plan, torch.empty_like(miss_count), torch.empty_like(output), stream)
            current.prepare(args.miss_count)

            def capture(options, current=current):
                submit(current, stream, options)

            graph, node_leases = captured.enter_context(captured_get_nodes(
                args, runtime, torch, kvaf, execution, stream, [(current.output, capture)]))
            lease = node_leases[0]
            graphs.append(graph)
            graph_inputs.append(current)
            leases.append(lease)

            def check(plan, count, current=current, lease=lease):
                check_output(current, count, lease)

            validate_graph_inputs(graph, current, check, args.miss_count, args.graph_replays)
            current.prepare(args.miss_count)

        print(f"MULTICARD_GRAPH_CAPTURE_DONE graphs={len(graphs)}", flush=True)
        replay_batch()
        for current, lease in zip(graph_inputs, leases):
            check_output(current, args.miss_count, lease)
        for _ in range(args.warmup):
            replay_batch()
        # Clear warmup output and fix each graph's measured request plan.
        for current in graph_inputs:
            current.prepare(args.miss_count)

        print(f"MULTICARD_READY device={device}", flush=True)
        (control / f"ready_{device}").touch()
        while not (control / "start.json").exists():
            if (control / "abort").exists():
                raise RuntimeError("multi-card run aborted before start")
            time.sleep(0.02)
        window = json.loads((control / "start.json").read_text())
        remaining = window["start"] - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        begin = time.monotonic()
        batches = 0
        while time.monotonic() < window["stop"]:
            if (control / "abort").exists():
                raise RuntimeError("multi-card run aborted during measurement")
            replay_batch()
            batches += 1
        end = time.monotonic()
        if not batches:
            raise RuntimeError("worker missed the measurement window")
        for graph, current, lease in zip(graphs, graph_inputs, leases):
            check_output(current, args.miss_count, lease)
            graph.validated = True
        replays = batches * len(graphs)
        payload_bytes = replays * args.miss_count * args.token_bytes
        result = {
            "device": device, "backend": args.backend, "graph_api": args.graph_api,
            "start": begin, "end": end, "calls": replays, "payload_bytes": payload_bytes,
            "bandwidth_GBps": payload_bytes / (end - begin) / 1e9,
            "graph_calls": 1, "graphs": len(graphs), "batches": batches, "verified": True,
            "request_length": args.request_length, "miss_count": args.miss_count,
            parallelism_field(args.backend): args.parallelism,
            "profiling_enabled": False,
            "execution_mode": args.execution_mode,
        }
    args.contention_result.write_text(json.dumps(result, indent=2) + "\n")
    print("MULTICARD_RESULT " + " ".join(f"{k}={v}" for k, v in result.items()), flush=True)
