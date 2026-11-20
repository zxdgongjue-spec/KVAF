"""Time KVAF graph replays on each card's own stream."""

from contextlib import nullcontext
import time

import batch_get_profiler_overlap as profiler_overlap


def measure(graph, stream, replays, *, sync_per_graph=False, profiler=None):
    begin = time.monotonic()
    with profiler.range(0, "graph_replay") if profiler is not None else nullcontext():
        for index in range(replays):
            graph.replay(stream)
            # Both modes use the same final wait and profiler marker.
            if sync_per_graph and index + 1 < replays:
                graph.synchronize(stream)
    with profiler.range(0, "wait_completion") if profiler is not None else nullcontext():
        graph.synchronize(stream)
    end = time.monotonic()
    if profiler is not None:
        profiler.step()
    return begin, end


def print_summary(directory, results, config):
    kernel_times = {}
    overlap = {}
    if config["torch_npu_profiler"]:
        profiles = directory / "concurrent" / "profiles"
        print(f"PROFILE_DIR {profiles}")
        kernel = profiler_overlap.KERNEL_NAMES[config["transport"]]
        traces = {
            result["device"]: profiler_overlap.load_device_rounds(
                profiles / f"device{result['device']}", result["device"], kernel, result["calls"], (0,))
            for result in results
        }
        payload_per_call = config["requests"] * config["token_bytes"]
        overlap = profiler_overlap.overlap_bandwidth(traces, (0,), payload_per_call)
        kernel_times = {device: sum(end - begin for begin, end in rounds[0]) for device, rounds in traces.items()}
    for result in results:
        device = result["device"]
        payload = result["payload_bytes"]
        case = f"case=kvaf-batch-get-{config['transport']} device={device}"
        bandwidth = f"bandwidth_gbps={result['bandwidth_GBps']:.3f}"
        if device in kernel_times:
            overlap_rate = overlap[device]["overlap_mean_GBps"]
            overlap_display = f"{overlap_rate:.3f}" if overlap_rate is not None else "n/a"
            bandwidth += (f" kernel_mean_GBps={payload / kernel_times[device]:.3f}"
                          f" overlap_mean_GBps={overlap_display} profiled_kernel_calls={result['calls']}")
        print(f"GRAPH_BANDWIDTH {case} {bandwidth}")
