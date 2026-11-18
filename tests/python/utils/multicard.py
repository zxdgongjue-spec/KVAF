"""Coordinate a sustained Get graph worker with the multi-device controller."""

import json
import time

from .get_graph import GraphInputs, captured_get, validate_graph_inputs
from .get_options import parallelism_field


def run_worker(kvaf, torch, args, runtime, execution, miss_count, output,
               make_plan, submit, check_output):
    control = args.contention_control_dir
    device = args.device_id
    stream = torch.npu.current_stream()
    inputs = GraphInputs(make_plan, miss_count, output, stream)
    inputs.prepare(args.miss_count)
    submit(inputs.fixed, stream, execution.get_kwargs)
    execution.complete_async_get(runtime)
    check_output(inputs.fixed, args.miss_count)

    def capture(options):
        submit(inputs.fixed, stream, options)

    print(f"MULTICARD_GRAPH_CAPTURE_BEGIN backend={args.backend} api={args.graph_api}", flush=True)
    with captured_get(args, runtime, torch, kvaf, execution, output, stream, capture) as graph:
        print("MULTICARD_GRAPH_CAPTURE_DONE", flush=True)
        validate_graph_inputs(graph, inputs, check_output, args.miss_count, args.graph_replays)
        inputs.prepare(args.miss_count)
        graph.replay()
        graph.synchronize()
        check_output(inputs.fixed, args.miss_count)
        for _ in range(args.warmup):
            graph.replay()
            graph.synchronize()

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
        replays = 0
        while time.monotonic() < window["stop"]:
            if (control / "abort").exists():
                raise RuntimeError("multi-card run aborted during measurement")
            graph.replay()
            graph.synchronize()
            replays += 1
        end = time.monotonic()
        if not replays:
            raise RuntimeError("worker missed the measurement window")
        check_output(inputs.fixed, args.miss_count)
        graph.validated = True
        payload_bytes = replays * args.miss_count * args.token_bytes
        result = {
            "device": device, "backend": args.backend, "graph_api": args.graph_api,
            "start": begin, "end": end, "calls": replays, "payload_bytes": payload_bytes,
            "bandwidth_GBps": payload_bytes / (end - begin) / 1e9,
            "graph_calls": 1, "verified": True,
            "request_length": args.request_length, "miss_count": args.miss_count,
            parallelism_field(args.backend): args.parallelism,
            "profiling_enabled": False,
            "execution_mode": args.execution_mode,
        }
    args.contention_result.write_text(json.dumps(result, indent=2) + "\n")
    print("MULTICARD_RESULT " + " ".join(f"{k}={v}" for k, v in result.items()), flush=True)
