"""Time KVAF graph replays on each card's own stream."""

from contextlib import nullcontext
import time

import batch_get_profiler_overlap as profiler_overlap


def measure(graph, stream, record_event, replays, *, event_breakdown=False, sync_per_graph=False, profiler=None):
    samples = []
    begin_ns = time.monotonic_ns()
    if not event_breakdown:
        start = record_event()
    with profiler.range(0, "graph_replay") if profiler is not None else nullcontext():
        submit_begin_ns = time.monotonic_ns()
        intermediate_wait_ns = 0
        for index in range(replays):
            if event_breakdown:
                graph_start = record_event()
                replay_begin_ns = time.monotonic_ns()
                graph.replay()
                replay_ns = time.monotonic_ns() - replay_begin_ns
                samples.append((graph_start, record_event(), replay_ns))
            else:
                graph.replay()
            # The final wait uses the shared path below, after the last timing event.
            if sync_per_graph and index + 1 < replays:
                wait_begin_ns = time.monotonic_ns()
                stream.synchronize()
                intermediate_wait_ns += time.monotonic_ns() - wait_begin_ns
        submit_ns = time.monotonic_ns() - submit_begin_ns - intermediate_wait_ns
    if event_breakdown:
        start, end = samples[0][0], samples[-1][1]
    else:
        end = record_event()
    with profiler.range(0, "wait_completion") if profiler is not None else nullcontext():
        stream.synchronize()
    end_ns = time.monotonic_ns()
    if profiler is not None:
        profiler.step()
    timing = {"device_event_seconds": start.elapsed_time(end) / 1000}
    if event_breakdown:
        timing["graph_event_breakdown"] = {
            "graph_ms": [start.elapsed_time(end) for start, end, _ in samples],
            "gap_ms": [left[1].elapsed_time(right[0]) for left, right in zip(samples, samples[1:])],
            "host_submit_batch_ms": submit_ns / 1e6,
            "host_replay_ms": [replay_ns / 1e6 for _, _, replay_ns in samples],
        }
    return begin_ns / 1e9, end_ns / 1e9, timing


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
        device_seconds = result["device_event_seconds"]
        case = f"case=kvaf-batch-get-{config['transport']} device={device}"
        bandwidth = f"bandwidth_gbps={result['bandwidth_GBps']:.3f} "
        bandwidth += (f"device_event_bandwidth_gbps={payload / (device_seconds * 1e9):.3f}"
                      if device_seconds > 0 else "device_event_bandwidth_gbps=n/a")
        if device in kernel_times:
            overlap_rate = overlap[device]["overlap_mean_GBps"]
            overlap_display = f"{overlap_rate:.3f}" if overlap_rate is not None else "n/a"
            bandwidth += (f" kernel_mean_GBps={payload / kernel_times[device]:.3f}"
                          f" overlap_mean_GBps={overlap_display} profiled_kernel_calls={result['calls']}")
        print(f"GRAPH_BANDWIDTH {case} {bandwidth}")
        if config["graph_event_breakdown"]:
            timings = result["graph_event_breakdown"]
            graph_ms, gap_ms = timings["graph_ms"], timings["gap_ms"]
            timing = (f"graph_span_sum_ms={sum(graph_ms):.3f} graph_gap_sum_ms={sum(gap_ms):.3f} "
                      f"graph_span_ms={format_ms_list(graph_ms)} graph_gap_ms={format_ms_list(gap_ms)}")
            if device in kernel_times:
                timing += f" kernel_sum_ms={kernel_times[device] / 1e6:.3f}"
            replay_ms = timings["host_replay_ms"]
            print(f"GRAPH_TIMING {case} {timing}")
            print(f"GRAPH_SUBMIT {case} host_submit_batch_ms={timings['host_submit_batch_ms']:.3f} "
                  f"host_replay_sum_ms={sum(replay_ms):.3f} host_replay_ms={format_ms_list(replay_ms)}")


def format_ms_list(samples):
    return "[" + ",".join(f"{value:.3f}" for value in samples) + "]"
