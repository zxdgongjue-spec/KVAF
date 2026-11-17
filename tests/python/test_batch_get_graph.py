"""Host-only checks of graph ownership, capture/replay ordering and accounting."""

import contextlib
import io
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import test_npu_batch_get_perf as perf


class Tensor:
    def __init__(self, value):
        self.value = value


class GraphFixture(unittest.TestCase):
    def setUp(self):
        self.trace = []
        self.descriptors = []
        self.captured = []
        self.capturing = False
        self.duration_ms = 8.0
        self.fail_capture = self.fail_replay = False
        self.output = SimpleNamespace(last=None)
        fixture = self

        class Stream:
            def synchronize(self):
                fixture.trace.append("stream_sync")

        class Event:
            def __init__(self, enable_timing):
                fixture.assertTrue(enable_timing)
                self.name = "start" if not fixture.events else "end"
                fixture.events.append(self)

            def record(self, stream):
                fixture.assertIs(stream, fixture.stream)
                fixture.trace.append(self.name + "_record")

            def synchronize(self):
                fixture.trace.append(self.name + "_sync")

            def elapsed_time(self, end):
                fixture.assertIs(end, fixture.events[-1])
                fixture.trace.append("elapsed")
                return fixture.duration_ms

        class Graph:
            def capture_begin(self, stream):
                fixture.stream = stream
                fixture.capturing = True
                fixture.trace.append("capture_begin")

            def capture_end(self):
                fixture.capturing = False
                fixture.trace.append("capture_end")

            def replay(self, stream):
                fixture.assertIs(stream, fixture.stream)
                fixture.trace.append("replay")
                if fixture.fail_replay:
                    raise RuntimeError("replay failed")
                for plan in fixture.captured:
                    fixture.assertTrue(all(ref() is not None for ref in plan), "captured tensor freed")
                    fixture.output.last = plan[0]().value

            def reset(self):
                fixture.trace.append("reset")
                fixture.assertTrue(all(ref() is not None for ref in fixture.descriptors))

        self.events = []
        self.graph_type = Graph
        self.torch = SimpleNamespace(npu=SimpleNamespace(
            Stream=Stream, Event=Event, synchronize=lambda: self.trace.append("device_sync")))
        self.runtime = SimpleNamespace(batch_get=self.enqueue, batch_get_kernel_profile=self.profile)

        class Measurement:
            def __init__(self, iterations, payload_bytes, **kwargs):
                fixture.metadata = dict(iterations=iterations, payload_bytes=payload_bytes, **kwargs)

            def __enter__(self):
                fixture.trace.append("measurement_enter")
                return self

            def before(self, iteration):
                fixture.trace.append(("before", iteration))

            def after(self, iteration, begin, end, **kwargs):
                fixture.trace.append(("after", iteration, begin, end, kwargs))

            def __exit__(self, *unused):
                fixture.trace.append("measurement_exit")

        self.measurement_type = Measurement

    def create_plan(self):
        self.trace.append("prepare")
        plan = tuple(Tensor(len(self.descriptors) // 3) for _ in range(3))
        self.descriptors.extend(weakref.ref(tensor) for tensor in plan)
        return plan

    def enqueue(self, table, indices, offsets, miss_count, destinations, output, *, stream, synchronize):
        self.assertFalse(synchronize)
        self.assertIs(output, self.output)
        self.trace.append("capture_get" if self.capturing else "eager_get")
        if self.capturing:
            if self.fail_capture:
                raise RuntimeError("capture launch failed")
            self.captured.append(tuple(weakref.ref(t) for t in (indices, offsets, destinations)))
        else:
            output.last = indices.value

    def profile(self):
        self.trace.append("profile")
        return [dict(last=self.output.last)]

    def validate(self, plan):
        self.assertEqual(self.output.last, plan[0].value)
        self.trace.append("validate")

    def run_benchmark(self, mode="graph", ring=8, warmup=2, validate=True, profiler_dir=None):
        args = perf.parse_args(["--execution-mode", mode, "--graph-ring", str(ring),
                                "--warmup", str(warmup), "--iterations", "2",
                                "--miss-count", "50", "--token-bytes", "64", "--kernel-profile",
                                *(["--torch-npu-profiler", "--profiler-dir", str(profiler_dir),
                                   "--profiler-warmup", "0", "--profiler-active", "2"]
                                  if profiler_dir else [])])
        with patch.object(perf, "ConcurrentMeasurement", self.measurement_type), \
                patch.object(perf.time, "monotonic_ns", side_effect=[
                    1_000_000, 2_000_000, 10_000_000, 11_000_000, 12_000_000, 20_000_000]), \
                contextlib.redirect_stdout(io.StringIO()):
            results = perf.measure_batch_get(
                args, self.torch, self.runtime, object(), object(), self.output, self.create_plan,
                graph_type=self.graph_type, validate_graph_output=self.validate if validate else None)
        return args, results


class GraphTests(GraphFixture):
    def test_fixed_ring_capture_events_barriers_and_byte_accounting(self):
        args, results = self.run_benchmark()
        self.assertEqual(self.metadata, dict(iterations=2, payload_bytes=50 * 64 * 8,
                                            execution_mode="graph", calls_per_round=8,
                                            torch_npu_profiler=False))
        self.assertEqual(self.trace.count("prepare"), 8)
        self.assertEqual(self.trace.count("capture_get"), 8)
        self.assertEqual(self.trace.count("replay"), 5)  # validation, warmup x2, measured x2
        self.assertEqual(self.trace.count("eager_get"), 1)  # registration priming
        self.assertLess(self.trace.index("validate"), self.trace.index("measurement_enter"))
        for iteration in range(2):
            index = self.trace.index(("before", iteration))
            self.assertEqual(self.trace[index + 1:index + 6],
                             ["start_record", "replay", "end_record", "end_sync", "elapsed"])
            self.assertEqual(self.trace[index + 6][0:2], ("after", iteration))
            self.assertEqual(self.trace[index + 7], "profile")
        self.assertEqual(results["total"], [9.0, 9.0])
        self.assertEqual(results["device"], [8.0, 8.0])
        self.assertEqual(results["profiles"], [[dict(last=7)], [dict(last=7)]])
        self.assertEqual(self.trace[-2:], ["stream_sync", "reset"])
        self.assertTrue(all(ref() is None for ref in self.descriptors))
        with contextlib.redirect_stdout(io.StringIO()) as output:
            perf.print_graph_samples(results, args)
        self.assertIn("payload_bytes_per_replay=25600", output.getvalue())
        self.assertIn("mean_us_per_get=1000.000000 useful_GBps=0.003200", output.getvalue())

    def test_ring_one_and_no_warmup_still_capture_and_validate(self):
        _, results = self.run_benchmark(ring=1, warmup=0)
        self.assertEqual(self.metadata["payload_bytes"], 50 * 64)
        self.assertEqual(self.trace.count("replay"), 3)
        self.assertEqual(len(results["device"]), 2)

    def test_eager_mode_generates_each_request_and_preserves_per_call_timing(self):
        _, results = self.run_benchmark(mode="eager")
        self.assertEqual(self.trace.count("prepare"), 4)
        self.assertEqual(self.trace.count("eager_get"), 4)
        self.assertNotIn("capture_begin", self.trace)
        self.assertEqual(self.metadata["calls_per_round"], 1)
        self.assertEqual(self.metadata["payload_bytes"], 3200)
        self.assertEqual(results["device"], [])

    def test_capture_error_ends_capture_and_destroys_graph(self):
        self.fail_capture = True
        with self.assertRaisesRegex(RuntimeError, "capture launch failed"):
            self.run_benchmark()
        self.assertEqual(self.trace[-3:], ["capture_end", "stream_sync", "reset"])
        self.assertNotIn("measurement_enter", self.trace)

    def test_replay_error_destroys_graph_before_releasing_inputs(self):
        self.fail_replay = True
        with self.assertRaisesRegex(RuntimeError, "replay failed"):
            self.run_benchmark()
        self.assertEqual(self.trace[-2:], ["stream_sync", "reset"])

    def test_invalid_device_durations_fail_instead_of_reporting_bandwidth(self):
        for duration in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(duration=duration):
                self.setUp()
                self.duration_ms = duration
                with self.assertRaisesRegex(RuntimeError, "positive finite"):
                    self.run_benchmark()
                self.assertEqual(self.trace[-2:], ["stream_sync", "reset"])

    def test_ring_must_be_positive(self):
        for ring in (0, -1):
            args = perf.parse_args(["--graph-ring", str(ring)])
            self.assertEqual(perf.validate_args(args), "graph-ring must be positive")

    def test_graph_requires_npu_and_graph_binding_instead_of_silent_skip(self):
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            with patch.object(perf, "detect_torch_backend", return_value=SimpleNamespace(
                    name="cpu", reason="host-only test")):
                self.assertEqual(perf.main(["--execution-mode", "graph"]), 2)
            with patch.object(perf, "detect_torch_backend", return_value=SimpleNamespace(name="npu")), \
                    patch.dict(sys.modules, kvaf=SimpleNamespace(NpuRuntime=object)):
                self.assertEqual(perf.main(["--execution-mode", "graph"]), 2)
                self.assertIn("kvaf.AclGraph", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
