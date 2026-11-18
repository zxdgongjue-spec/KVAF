"""Host-only profiler lifecycle and graph/eager annotation checks."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from batch_get_profiler import BatchGetProfiler
import test_batch_get_graph as graph_tests
import test_npu_batch_get_perf as perf


class FakeProfilerApi:
    """Fake the vendor collector, leaving benchmark orchestration real."""

    def __init__(self, trace):
        self.trace = trace
        self.profiles = []
        self.parse_calls = []
        self.write_trace = True
        self.api = ModuleType("torch_npu.profiler")
        self.api.ProfilerActivity = SimpleNamespace(CPU="CPU", NPU="NPU")
        self.api.ProfilerLevel = SimpleNamespace(Level0=0, Level1=1, Level2=2)
        self.api.AiCMetrics = SimpleNamespace(AiCoreNone="none")
        self.api.ExportType = SimpleNamespace(Text="text")
        self.api._ExperimentalConfig = lambda **kwargs: kwargs
        self.api.schedule = lambda **kwargs: kwargs
        self.api.profile = self.profile
        self.api.tensorboard_trace_handler = self.handler
        self.implementation = ModuleType("torch_npu.profiler.profiler")
        self.implementation.analyse = self.analyse
        self.package = ModuleType("torch_npu")
        self.package.profiler = self.api

    def installed(self):
        return patch.dict(sys.modules, {"torch_npu": self.package,
                                       "torch_npu.profiler": self.api,
                                       "torch_npu.profiler.profiler": self.implementation})

    def handler(self, path, **kwargs):
        self.output = Path(path)
        self.handler_options = kwargs
        return lambda unused: self.trace.append("save_raw")

    def profile(self, **kwargs):
        fixture = self

        class Profile:
            def __init__(self):
                self.options, self.steps = kwargs, 0

            def __enter__(self):
                fixture.trace.append("profiler_enter")
                (fixture.output / "fake_ascend_pt").mkdir()
                return self

            def step(self):
                self.steps += 1
                fixture.trace.append("profiler_step")
                schedule = self.options["schedule"]
                if self.steps == schedule["wait"] + schedule["warmup"] + schedule["active"]:
                    self.options["on_trace_ready"](self)

            def __exit__(self, *unused):
                fixture.trace.append("profiler_exit")

        profiler = Profile()
        self.profiles.append(profiler)
        return profiler

    def analyse(self, path, **kwargs):
        self.trace.append("analyse")
        self.parse_calls.append((path, kwargs))
        if self.write_trace:
            output = Path(path) / "fake_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
            output.mkdir()
            (output / "trace_view.json").write_text(json.dumps({"traceEvents": [
                {"name": "fake_kernel", "cat": "kernel", "ph": "X", "ts": 1, "dur": 2}]}))

    @contextlib.contextmanager
    def record_function(self, name):
        self.trace.append(("range_enter", name))
        try:
            yield
        finally:
            self.trace.append(("range_exit", name))


class ProfilerTests(unittest.TestCase):
    def test_disabled_profiler_never_imports_vendor_package(self):
        args = perf.parse_args([])
        with patch.dict(sys.modules, {"torch_npu": None}), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            with BatchGetProfiler(args, None) as profiler:
                with profiler.range("graph_replay", 0):
                    profiler.step()
        self.assertEqual(output.getvalue(), "")

    def test_validates_output_and_complete_schedule(self):
        for options, message in (
                (["--torch-npu-profiler"], "requires profiler-dir"),
                (["--torch-npu-profiler", "--profiler-dir", "trace", "--iterations", "3"], "iterations"),
                (["--profiler-wait", "-1"], "non-negative"),
                (["--profiler-warmup", "-1"], "non-negative"),
                (["--profiler-active", "0"], "positive")):
            with self.subTest(options=options):
                self.assertIn(message, perf.validate_args(perf.parse_args(options)))
        self.assertIsNone(perf.validate_args(perf.parse_args([
            "--torch-npu-profiler", "--profiler-dir", "trace", "--iterations", "4"])))

    def test_collects_bounded_rounds_and_parses_only_after_context_exit(self):
        trace = []
        api = FakeProfilerApi(trace)
        with tempfile.TemporaryDirectory() as folder, api.installed(), \
                patch.dict(os.environ, DEVICE_ID="7", KVAF_BATCH_GET_FLOW_NAME="flow1"), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            args = perf.parse_args(["--torch-npu-profiler", "--profiler-dir", folder,
                                    "--execution-mode", "graph", "--iterations", "6",
                                    "--profiler-wait", "1", "--profiler-level", "Level2"])
            torch = SimpleNamespace(profiler=SimpleNamespace(record_function=api.record_function))
            with BatchGetProfiler(args, torch) as collector:
                for iteration in range(args.iterations):
                    with collector.range("graph_replay", iteration):
                        trace.append("replay")
                    collector.step()
                self.assertEqual(api.parse_calls, [])
                trace.append("all_rounds_done")
            options = api.profiles[0].options
            self.assertEqual(options["activities"], ["CPU", "NPU"])
            self.assertEqual(options["schedule"], dict(wait=1, warmup=1, active=3, repeat=1))
            self.assertEqual(options["experimental_config"]["export_type"], "text")
            self.assertEqual(options["experimental_config"]["profiler_level"], 2)
            self.assertFalse(api.handler_options["analyse_flag"])
            self.assertIn("flow1_device7_pid", api.handler_options["worker_name"])
            self.assertEqual(trace.count("save_raw"), 1)
            self.assertEqual(trace[-3:], ["all_rounds_done", "profiler_exit", "analyse"])
            self.assertEqual(api.parse_calls[0][1], dict(max_process_number=1))
            metadata = json.loads((api.output / "collection.json").read_text())
            self.assertEqual(metadata["first_active_round"], 2)
            self.assertEqual(metadata["calls_per_round"], 100)
            self.assertTrue(metadata["performance_numbers_include_profiler_overhead"])
            self.assertIn("BATCH_GET_TORCH_NPU_TRACE_WRITTEN", output.getvalue())
            # Rerunning the same shape cannot masquerade as a fresh collection.
            with self.assertRaises(FileExistsError):
                with BatchGetProfiler(args, torch):
                    pass

    def test_missing_parsed_trace_is_failure_and_preserves_raw_data(self):
        api = FakeProfilerApi([])
        api.write_trace = False
        with tempfile.TemporaryDirectory() as folder, api.installed(), \
                contextlib.redirect_stdout(io.StringIO()):
            args = perf.parse_args(["--torch-npu-profiler", "--profiler-dir", folder])
            with self.assertRaisesRegex(RuntimeError, "no trace_view.json"):
                with BatchGetProfiler(args, None):
                    pass
            self.assertTrue((api.output / "fake_ascend_pt").is_dir())

    def test_error_stops_profiler_without_parsing_or_suppressing_original_error(self):
        trace = []
        api = FakeProfilerApi(trace)
        with tempfile.TemporaryDirectory() as folder, api.installed(), \
                contextlib.redirect_stdout(io.StringIO()):
            args = perf.parse_args(["--torch-npu-profiler", "--profiler-dir", folder])
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                with BatchGetProfiler(args, None):
                    raise RuntimeError("kernel failed")
            self.assertEqual(trace[-1], "profiler_exit")
            self.assertEqual(api.parse_calls, [])


class ProfilerIntegrationTests(graph_tests.GraphFixture):
    def test_graph_and_eager_measurements_annotate_each_round_and_parse_after_barriers(self):
        for mode in ("graph", "eager"):
            with self.subTest(mode=mode):
                self.setUp()
                api = FakeProfilerApi(self.trace)
                self.torch.profiler = SimpleNamespace(record_function=api.record_function)
                with tempfile.TemporaryDirectory() as folder, api.installed():
                    _, results = self.run_benchmark(mode=mode, profiler_dir=folder)
                self.assertTrue(self.metadata["torch_npu_profiler"])
                self.assertEqual(api.profiles[0].steps, 2)
                self.assertEqual(len(results["total"]), 2)
                first = self.trace.index("profiler_enter")
                if mode == "graph":
                    self.assertLess(self.trace.index("capture_end"), first)
                    self.assertLess(self.trace.index("validate"), first)
                for iteration in range(2):
                    labels = [item[1].rsplit("/", 1)[-1] for item in self.trace
                              if isinstance(item, tuple) and item[0] == "range_enter"
                              and f"/round{iteration}/" in item[1]]
                    self.assertEqual(labels, (["prepare_request"] if mode == "eager" else []) +
                                     ["barrier_before", "graph_replay" if mode == "graph" else "batch_get",
                                      "wait_completion", "barrier_after", "kernel_profile_snapshot"])
                self.assertLess(self.trace.index("measurement_exit"), self.trace.index("analyse"))
                self.assertEqual(self.trace.count("save_raw"), 1)
                after_indices = [index for index, item in enumerate(self.trace)
                                 if isinstance(item, tuple) and item[0] == "after"]
                step_indices = [index for index, item in enumerate(self.trace) if item == "profiler_step"]
                self.assertTrue(all(after < step for after, step in zip(after_indices, step_indices)))


if __name__ == "__main__":
    unittest.main()
