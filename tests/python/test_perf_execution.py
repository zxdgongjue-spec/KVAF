"""Portable coverage for API selection, timing labels and sweep failures."""

import argparse
import contextlib
import csv
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from perf_execution import BenchmarkExecution, add_execution_mode
import test_npu_batch_get_perf as direct
from batch_get_host_sweep import parse_measurements

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "access_operator/demo/aicpu_batch_get/python"
with patch.object(sys, "path", [str(DEMO), *sys.path]):
    spec = importlib.util.spec_from_file_location("gather_benchmark", DEMO / "benchmark_gather_read.py")
    gather = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gather)


class ExecutionTests(unittest.TestCase):
    def test_legacy_does_not_require_new_binding_symbols(self):
        execution = BenchmarkExecution(object(), "legacy")
        self.assertEqual(execution.runtime_kwargs, {})
        self.assertEqual(execution.put_kwargs, {})
        self.assertEqual(execution.get_kwargs, {})
        execution.drain_get(object())
        parser = argparse.ArgumentParser()
        add_execution_mode(parser)
        self.assertEqual(parser.parse_args([]).execution_mode, "legacy")

    def test_per_call_enables_only_the_requested_providers_and_capacities(self):
        api = SimpleNamespace(PutExecutionConfig=lambda *args: args,
                              GetExecutionConfig=lambda *args: args,
                              ExecutionResources=lambda **args: args)
        for transport, executor, algorithm in (("urma", "aicpu", "gather"),
                                               ("urma", "aicpu", "scatter"),
                                               ("urma", "aiv", "direct"),
                                               ("ubmem", "aiv", "direct")):
            with self.subTest(transport=transport, executor=executor, algorithm=algorithm):
                execution = BenchmarkExecution(api, "per-call", transport=transport,
                    executor=executor, algorithm=algorithm, put_capacity=17, get_capacity=32768)
                options = execution.runtime_kwargs
                self.assertTrue(options["per_call_execution"])
                self.assertEqual(options["enabled_put_configs"], [(transport, executor)])
                self.assertEqual(options["enabled_get_configs"], [(transport, executor, algorithm)])
                self.assertEqual(execution.put_kwargs["execution"], options["default_put"])
                self.assertEqual(execution.get_kwargs["execution"], options["default_get"])
                self.assertEqual(options["put_resources"][0][1]["request_capacity"], 17)
                self.assertEqual(options["get_resources"][0][1]["request_capacity"], 32768)
                runtime = Mock()
                execution.drain_get(runtime)
                runtime.drain_get.assert_called_once_with()

    def test_reusable_output_and_release_failure(self):
        api = SimpleNamespace(PutExecutionConfig=lambda *a: a, GetExecutionConfig=lambda *a: a,
                              ExecutionResources=lambda **a: a)
        for executor, algorithm in (("aiv", "direct"), ("aicpu", "gather")):
            execution = BenchmarkExecution(api, "per-call", executor=executor, algorithm=algorithm,
                                           get_capacity=1024)
            runtime = Mock(); runtime.begin_get_output.return_value = 17
            execution.bind_output(runtime, "output", "stream")
            runtime.begin_get_output.assert_called_once_with(execution.get_config, "output", 1024, "stream")
            self.assertEqual(execution.get_kwargs["output_lease"], 17)
            execution.drain_get(runtime)
            runtime.end_get_output.side_effect = RuntimeError("unresolved")
            with self.assertRaisesRegex(RuntimeError, "unresolved"):
                execution.release_output(runtime)
            self.assertEqual(execution.get_kwargs["output_lease"], 17)
            runtime.end_get_output.side_effect = None
            execution.release_output(runtime)
            self.assertNotIn("output_lease", execution.get_kwargs)
        for mode, transport, executor, algorithm in (("legacy", "urma", "aiv", "direct"),
                ("per-call", "ubmem", "aiv", "direct"), ("per-call", "urma", "aicpu", "scatter")):
            execution = BenchmarkExecution(api, mode, transport=transport, executor=executor, algorithm=algorithm)
            execution.bind_output(object(), "output")
            execution.release_output(object())
        with patch.dict(os.environ, KVAF_PERF_REUSE_GET_OUTPUT="0"):
            execution = BenchmarkExecution(api, "per-call")
            execution.bind_output(object(), "output")
            self.assertEqual(execution.output_binding, "transient")

    def test_unsupported_profiling_and_graph_shapes_fail_before_running(self):
        with patch.dict(os.environ, {}, clear=True):
            for options, error in ((["--kernel-profile"], "no kernel cycle profiling"),
                                   (["--adaptive-ub-device"], "adaptive-ub-device"),
                                   (["--contention-control-dir", "/tmp/control",
                                     "--contention-result", "/tmp/result"], "one Get")):
                args = direct.parse_args(["--execution-mode", "per-call", *options])
                self.assertIn(error, direct.validate_args(args))
            args = direct.parse_args(["--execution-mode", "per-call", "--no-kernel-profile"])
            self.assertIsNone(direct.validate_args(args))

    def test_aicpu_host_and_device_statistics_remain_separate(self):
        args = gather.parse_args(["--execution-mode", "per-call", "--table-capacity", "4"])
        phases = dict(request_us=1, source_write_us=2, notify_us=3, read_us=4,
                      wr_build_us=5, ready_wait_us=6, post_us=7, cq_wait_us=8,
                      scatter_us=9, ack_us=10)
        samples = [dict(device_span_us=100, python_call_us=1000, workers=[phases]),
                   dict(device_span_us=300, python_call_us=3000, workers=[phases])]
        for mode in ("gather_read", "cpu_push_scatter"):
            summary = gather.summarize_measurements(samples, 200000, mode=mode)
            self.assertEqual(summary["device_p50_us"], 200)
            self.assertEqual(summary["host_p50_us"], 2000)
            self.assertEqual(summary["p50_GBps"], 1)
            self.assertEqual(summary["host_p50_GBps"], 0.1)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                gather.print_measurements(args, summary, mode=mode)
            self.assertIn("execution_mode=per-call", output.getvalue())
            self.assertIn("host_p50_GBps=0.100", output.getvalue())

    def test_per_call_report_refuses_legacy_or_missing_measurements(self):
        with self.assertRaisesRegex(ValueError, "missing per-call"):
            parse_measurements("BATCH_GET_PERF_CONFIG timing=synchronous_batch_get\n"
                               "PERF_RESULT operation=batch_get execution_mode=legacy p50_ms=1", "per-call")
        with self.assertRaisesRegex(ValueError, "missing synchronous BatchGet"):
            parse_measurements("PERF_RESULT operation=batch_get execution_mode=per-call p50_ms=1", "per-call")

    def test_synchronous_get_timing_requires_host_mode(self):
        args = direct.parse_args(["--get-timing", "sync-call", "--no-kernel-profile"])
        self.assertIsNone(direct.validate_args(args))
        args = direct.parse_args(["--get-timing", "sync-call", "--kernel-profile"])
        self.assertIn("--no-kernel-profile", direct.validate_args(args))

    def test_per_call_sweep_forwards_options_and_labels_host_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runner = directory / "runner"
            runner.write_text('''#!/usr/bin/env python3
import os, sys
def option(name): return sys.argv[sys.argv.index(name) + 1]
mode = option("--execution-mode")
assert mode in ("per-call", "legacy")
assert option("--token-bytes") == "656"
assert option("--miss-count") == "32"
assert option("--table-capacity") == "64"
assert "--no-kernel-profile" in sys.argv and "--kernel-profile" not in sys.argv
assert option("--get-timing") == "sync-call"
if os.environ.get("FAKE_FAIL"): sys.exit(3)
print("BATCH_GET_PERF_CONFIG timing=synchronous_batch_get")
print(f"PERF_RESULT operation=batch_get execution_mode={mode} mean_ms=0.1 p50_ms=0.1 p90_ms=0.2 p99_ms=0.3")
''')
            runner.chmod(0o755)
            for mode, transport, fail, count in (("per-call", "urma", False, 2), ("per-call", "ubmem", False, 1),
                                                  ("per-call", "urma", True, 0), ("legacy", "urma", False, 2)):
                logs = directory / f"{mode}-{transport}-{fail}"
                env = dict(os.environ, PYTHON=sys.executable, KVAF_BATCH_GET_PERF_RUNNER=str(runner),
                           KVAF_BATCH_GET_TOKEN_BYTES="656", KVAF_BATCH_GET_MISS_COUNTS="32",
                           KVAF_BATCH_GET_AIV_CORES="2", KVAF_BATCH_GET_JETTYS_PER_CORE="1 2",
                           KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR=str(logs),
                           KVAF_BATCH_GET_MISS_SWEEP_CSV=str(logs / "summary.csv"))
                env.pop("FAKE_FAIL", None)
                if fail:
                    env["FAKE_FAIL"] = "1"
                result = subprocess.run(["bash", str(ROOT / "tests/run_remote_cpu_batch_get_miss_count_perf.sh"),
                    "build", f"--execution-mode={mode}", "--timing", "host",
                    "--transport", transport, "--table-capacity", "64"],
                    env=env, text=True, capture_output=True, timeout=20)
                if fail:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("SWEEP_PASS", result.stdout)
                    continue
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                with (logs / "summary.csv").open() as source:
                    rows = list(csv.DictReader(source))
                self.assertEqual(len(rows), count)
                for row in rows:
                    self.assertEqual(row["execution_mode"], mode)
                    self.assertEqual(row["timing"], "synchronous_batch_get")
                    self.assertEqual(float(row["host_p50_us"]), 100)
                    self.assertAlmostEqual(float(row["host_p50_GBps"]), 656 * 32 / 100 / 1000)
                    self.assertNotIn("device_p50_us", row)
                    self.assertNotIn("submit_p50_us", row)


if __name__ == "__main__":
    unittest.main()
