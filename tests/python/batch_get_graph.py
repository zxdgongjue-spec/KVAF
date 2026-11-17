"""Caller-owned ACL graph and fixed request ring for the BatchGet benchmark."""

from contextlib import contextmanager
from types import SimpleNamespace


@contextmanager
def capture_batch_get(npu, graph_type, create_plan, enqueue, ring):
    # ACL capture retains raw pointers. Hold every descriptor until replay has
    # completed and the graph has been destroyed; allocate nothing in capture.
    plans = [create_plan() for _ in range(ring)]
    stream = npu.Stream()
    graph = graph_type()
    start = npu.Event(enable_timing=True)
    end = npu.Event(enable_timing=True)
    capturing = False
    try:
        npu.synchronize()
        # Populate transport registration/launch caches before capture, even
        # with --warmup 0. All ring entries use the same caller-owned output.
        enqueue(plans[0], stream)
        stream.synchronize()
        graph.capture_begin(stream)
        capturing = True
        for plan in plans:
            enqueue(plan, stream)
        capturing = False  # capture_end clears capture state even on error.
        graph.capture_end()
        # NPU events initialize lazily. Keep their first use outside timing.
        start.record(stream)
        end.record(stream)
        end.synchronize()
        start.elapsed_time(end)
        yield SimpleNamespace(graph=graph, stream=stream, start=start, end=end, last_plan=plans[-1])
    finally:
        if capturing:
            try:
                graph.capture_end()
            except Exception:
                pass  # Preserve the original capture/launch error.
        try:
            stream.synchronize()
        finally:
            graph.reset()
        # plans, stream and events stay alive through synchronization/reset.
