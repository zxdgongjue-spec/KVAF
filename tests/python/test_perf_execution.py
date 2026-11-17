"""Portable coverage for API selection, timing labels and sweep failures."""

import argparse
import contextlib
import csv
import io
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from perf_execution import BenchmarkExecution, add_execution_metadata
import test_npu_batch_get_perf as direct
import test_npu_batch_put as put
from batch_get_host_sweep import parse_measurements
import benchmark_aicpu_get as gather

ROOT = Path(__file__).resolve().parents[2]


class ExecutionTests(unittest.TestCase):
    def test_runtime_benchmark_waits_once_in_measured_calls(self):
        events = []
        runtime = Mock(initialized=True)
        runtime.begin_get_output.return_value = 17
        runtime.batch_put.side_effect = lambda *a, **k: events.append("put")
        runtime.batch_get.side_effect = lambda *a, **k: events.append("get")
        runtime.drain_get.side_effect = lambda *a, **k: events.append("drain")
        api = SimpleNamespace(NpuRuntime=Mock(return_value=runtime),
            PutExecutionConfig=lambda *a: a, GetExecutionConfig=lambda *a: a,
            ExecutionResources=lambda **a: a)
        torch = MagicMock()
        torch.equal.return_value = True
        torch.npu.synchronize.side_effect = lambda: events.append("wait")
        torch.rand.return_value.__lt__.return_value = MagicMock()
        counts = torch.stack.return_value.cpu.return_value
        counts.__ge__.return_value = MagicMock()
        counts.__ge__.return_value.any.return_value.item.return_value = False
        counts.sum.return_value.item.return_value = 3

        def clock():
            events.append("clock")
            return len(events) / 1000

        with tempfile.TemporaryDirectory() as temporary:
            kernel = Path(temporary) / "get.o"
            kernel.write_bytes(b"kernel")
            put_kernel = kernel.with_name("kvaf_execute_batch_put_kernels.o")
            put_kernel.write_bytes(b"kernel")
            env = dict(KVAF_RUN_NPU_TESTS="1", KVAF_CPU_SERVICE_ADDR="127.0.0.1:38999",
                KVAF_BATCH_GET_KERNEL=str(kernel), KVAF_RUNTIME_PERF_BATCH="1",
                KVAF_RUNTIME_PERF_AIV_CORES="2", KVAF_RUNTIME_PERF_TOKEN_BYTES="8",
                KVAF_RUNTIME_PERF_TOKENS_PER_BLOCK="1", KVAF_RUNTIME_PERF_TABLE_CAPACITY="384",
                KVAF_RUNTIME_PERF_REQUESTS="2048", KVAF_RUNTIME_PERF_WARMUP="1",
                KVAF_RUNTIME_PERF_ITERATIONS="2")
            spec = importlib.util.spec_from_file_location("runtime_perf_contract",
                ROOT / "tests/python/test_npu_runtime_perf.py")
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {"kvaf": api}):
                spec.loader.exec_module(module)
            with patch.dict(os.environ, env, clear=True), patch.object(module,
                    "detect_torch_backend", return_value=SimpleNamespace(name="npu", torch=torch)), \
                    patch.object(module.time, "perf_counter", side_effect=clock), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), 0)
        # Timed Put is synchronous; timed Get includes one runtime-owned drain.
        self.assertEqual(sum(events[i:i + 3] == ["clock", "put", "clock"]
            for i in range(len(events))), 2)
        self.assertEqual(sum(events[i:i + 5] == ["clock", "get", "clock", "drain", "clock"]
            for i in range(len(events))), 2)
        # Input preparation waits happen before starting each timed Get.
        self.assertEqual(sum(events[i:i + 6] == ["wait", "clock", "get", "clock", "drain", "clock"]
            for i in range(len(events))), 2)

    def test_put_backend_and_aicpu_only_kernel_resolution(self):
        for transport, executor in (("urma", "aiv"), ("urma", "aicpu"), ("ubmem", "aiv")):
            self.assertEqual(put.put_backend(transport, executor), f"{transport}_{executor}")
        with self.assertRaisesRegex(ValueError, "UBMEM Put requires"):
            put.put_backend("ubmem", "aicpu")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(put.kernel_paths("aicpu"), ("", ""))
            with self.assertRaisesRegex(ValueError, "KVAF_BATCH_PUT_KERNEL"):
                put.kernel_paths("aiv")

    def test_put_only_requires_its_own_kernel(self):
        with tempfile.TemporaryDirectory() as temporary:
            put_kernel = Path(temporary) / "put.o"
            put_kernel.write_bytes(b"kernel")
            with patch.dict(os.environ, {"KVAF_BATCH_PUT_KERNEL": str(put_kernel)}, clear=True):
                self.assertEqual(put.kernel_paths("aiv"), ("", str(put_kernel)))

    def test_provider_budgets_only_contain_applicable_fields(self):
        api = SimpleNamespace(PutExecutionConfig=lambda *a: a, GetExecutionConfig=lambda *a: a,
                              ExecutionResources=lambda **a: a)
        for transport, executor, algorithm in (("urma", "aiv", "direct"),
                ("ubmem", "aiv", "direct"), ("urma", "aicpu", "gather"),
                ("urma", "aicpu", "scatter")):
            e = BenchmarkExecution(api, transport=transport, executor=executor, algorithm=algorithm,
                cores=4, jettys=8, get_kernel="get.o", put_kernel="put.o", max_inflight_total=0)
            put = e.runtime_kwargs["put_resources"][0][1]
            get = e.runtime_kwargs["get_resources"][0][1]
            self.assertEqual((put["cores"], get["cores"]), (4, 4))
            if executor == "aicpu":
                for budget in (put, get):
                    self.assertNotIn("kernel_path", budget)
                    self.assertNotIn("jettys", budget)
                    self.assertNotIn("max_inflight_total", budget)
            elif transport == "urma":
                self.assertEqual((put["jettys"], get["jettys"], get["max_inflight_total"]), (2, 8, 0))
            else:
                self.assertEqual((put["kernel_path"], get["kernel_path"]), ("put.o", "get.o"))
                self.assertNotIn("jettys", get)
                self.assertNotIn("max_inflight_total", get)

    def test_retired_cli_and_multiple_get_graph_are_rejected(self):
        with patch.dict(os.environ, {}, clear=True):
            for options in (["--execution-mode", "legacy"], ["--execution-mode", "per-call"],
                            ["--adaptive-ub-device"]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    direct.parse_args(options)
            args = direct.parse_args(["--contention-control-dir", "/tmp/control",
                "--contention-result", "/tmp/result", "--contention-graph-calls", "2"])
            self.assertIn("one Get", direct.validate_args(args))

    def test_put_result_distinguishes_block_payload_and_bandwidth_statistic(self):
        result = put.format_measurements([1.0, 1.0, 4.0], execution_mode="unified",
            transport="ubmem", executor="aiv", warmup=10, rows=1000, block_bytes=1000, cores=2)
        fields = dict(item.split("=", 1) for item in result.split()[1:])
        self.assertEqual(fields["metric"], "runtime_batch_put_sync_call_to_return")
        self.assertEqual(fields["timing"], "synchronous_batch_put")
        self.assertEqual(fields["backend"], "ubmem_aiv")
        self.assertEqual(fields["execution_mode"], "unified")
        self.assertEqual(fields["operation"], "batch_put")
        self.assertEqual(int(fields["payload_bytes_per_call"]), 1000000)
        self.assertEqual(float(fields["useful_GBps"]), 0.5)
        self.assertEqual(float(fields["p50_GBps"]), 1.0)
        self.assertEqual(int(fields["samples"]), 3)

    def test_put_benchmark_selects_each_backend_and_frees_between_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for direction in ("get", "put"):
                (directory / f"kvaf_execute_batch_{direction}_kernels.o").write_bytes(b"kernel")
            for transport, executor in (("urma", "aiv"), ("urma", "aicpu"), ("ubmem", "aiv")):
                for mode in ("unified",):
                    with self.subTest(transport=transport, executor=executor, mode=mode):
                        committed = False

                        def write(*args, **kwargs):
                            nonlocal committed
                            self.assertFalse(committed, "duplicate Put without Free")
                            committed = True

                        def free(*args):
                            nonlocal committed
                            self.assertTrue(committed, "Free without a committed Put")
                            committed = False

                        runtime = Mock(initialized=True, remote_region={"base": 4096, "slot_size": 4096})
                        runtime.begin_put_input.return_value = 19
                        runtime.batch_put.side_effect = write
                        runtime.batch_free.side_effect = free
                        runtime.access_table_view.return_value = dict(
                            lengths=[8] * 4, valid=[1] * 4, addresses=[4096 * (i + 1) for i in range(4)])
                        api = SimpleNamespace(NpuRuntime=Mock(return_value=runtime),
                            PutExecutionConfig=lambda *a: a, GetExecutionConfig=lambda *a: a,
                            ExecutionResources=lambda **a: a)
                        env = dict(KVAF_RUN_NPU_TESTS="1", KVAF_CPU_SERVICE_ADDR="127.0.0.1:38999",
                            KVAF_NPU_REQUESTS="4", KVAF_NPU_TOKEN_BYTES="8", KVAF_NPU_TOKENS_PER_BLOCK="1",
                            KVAF_NPU_CORE_COUNT="2", KVAF_NPU_WARMUP="1", KVAF_NPU_ITERATIONS="2")
                        if executor == "aiv":
                            env["KVAF_BATCH_PUT_KERNEL"] = str(directory / "kvaf_execute_batch_put_kernels.o")
                        output = io.StringIO()
                        with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, {"kvaf": api}), \
                             patch.object(put, "detect_torch_backend", return_value=SimpleNamespace(
                                 name="npu", torch=MagicMock())), contextlib.redirect_stdout(output):
                            self.assertEqual(put.main(transport, mode, executor), 0)
                        self.assertFalse(committed)
                        self.assertEqual(runtime.batch_put.call_count, 4)  # preflight + warmup + two samples
                        self.assertEqual(runtime.batch_free.call_count, 4)
                        runtime.batch_get.assert_not_called()
                        runtime.shutdown.assert_called_once()
                        constructor = api.NpuRuntime.call_args.kwargs
                        self.assertEqual(constructor["transport_mode"], transport)
                        self.assertEqual(constructor["default_put"], (transport, executor))
                        expected = {"execution": (transport, executor)}
                        if transport == "urma":
                            expected["input_lease"] = 19
                            runtime.begin_put_input.assert_called_once()
                            runtime.end_put_input.assert_called_once_with(19)
                            self.assertIn("input_binding=reused", output.getvalue())
                        else:
                            runtime.begin_put_input.assert_not_called()
                        self.assertEqual(runtime.batch_put.call_args.kwargs, expected)
                        self.assertIn("samples=2 warmup_excluded=1", output.getvalue())


    def test_unified_sets_defaults_and_capacities_without_an_enable_list(self):
        api = SimpleNamespace(PutExecutionConfig=lambda *args: args,
                              GetExecutionConfig=lambda *args: args,
                              ExecutionResources=lambda **args: args)
        for transport, executor, algorithm in (("urma", "aicpu", "gather"),
                                               ("urma", "aicpu", "scatter"),
                                               ("urma", "aiv", "direct"),
                                               ("ubmem", "aiv", "direct")):
            with self.subTest(transport=transport, executor=executor, algorithm=algorithm):
                execution = BenchmarkExecution(api, transport=transport,
                    executor=executor, algorithm=algorithm, put_capacity=17, get_capacity=32768)
                options = execution.runtime_kwargs
                self.assertNotIn("per_call_execution", options)
                self.assertNotIn("enabled_put_configs", options)
                self.assertNotIn("enabled_get_configs", options)
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
            execution = BenchmarkExecution(api, executor=executor, algorithm=algorithm,
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
        for mode, transport, executor, algorithm in (("unified", "ubmem", "aiv", "direct"), ("unified", "urma", "aicpu", "scatter")):
            execution = BenchmarkExecution(api, transport=transport, executor=executor, algorithm=algorithm)
            execution.bind_output(object(), "output")
            execution.release_output(object())
        with patch.dict(os.environ, KVAF_PERF_REUSE_GET_OUTPUT="0"):
            execution = BenchmarkExecution(api)
            execution.bind_output(object(), "output")
            self.assertEqual(execution.output_binding, "transient")

    def test_reusable_put_input_keeps_lease_on_release_failure(self):
        api = SimpleNamespace(PutExecutionConfig=lambda *a: a, GetExecutionConfig=lambda *a: a,
                              ExecutionResources=lambda **a: a)
        for executor in ("aiv", "aicpu"):
            with self.subTest(executor=executor):
                execution = BenchmarkExecution(api, executor=executor,
                    algorithm="gather" if executor == "aicpu" else "direct", put_capacity=512)
                runtime = Mock()
                runtime.begin_put_input.return_value = 23
                execution.bind_input(runtime, "source", "stream")
                runtime.begin_put_input.assert_called_once_with(("urma", executor), "source", 512, "stream")
                self.assertEqual(execution.put_kwargs["input_lease"], 23)
                with self.assertRaisesRegex(RuntimeError, "already bound"):
                    execution.bind_input(runtime, "source", "stream")
                runtime.end_put_input.side_effect = RuntimeError("transaction unresolved")
                with self.assertRaisesRegex(RuntimeError, "transaction unresolved"):
                    execution.release_input(runtime)
                self.assertEqual(execution.input_lease, 23)
                self.assertEqual(execution.put_kwargs["input_lease"], 23)
                runtime.end_put_input.side_effect = None
                execution.release_input(runtime)
                self.assertEqual(execution.input_lease, 0)
                self.assertNotIn("input_lease", execution.put_kwargs)
        for mode, transport, executor, algorithm in (("unified", "ubmem", "aiv", "direct"),):
            execution = BenchmarkExecution(api, transport=transport, executor=executor, algorithm=algorithm)
            execution.bind_input(object(), "source")
            execution.release_input(object())
            self.assertNotIn("input_lease", execution.put_kwargs)
        with patch.dict(os.environ, KVAF_PERF_REUSE_PUT_INPUT="0"):
            for executor in ("aiv", "aicpu"):
                execution = BenchmarkExecution(api, executor=executor,
                    algorithm="gather" if executor == "aicpu" else "direct")
                execution.bind_input(object(), "source")
                execution.release_input(object())
                self.assertEqual(execution.input_binding, "transient")
                self.assertNotIn("input_lease", execution.put_kwargs)


    def test_aicpu_host_and_device_statistics_remain_separate(self):
        args = gather.parse_args(["--table-capacity", "4"])
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
            self.assertIn("execution_mode=unified", output.getvalue())
            self.assertIn("host_p50_GBps=0.100", output.getvalue())

    def test_unified_report_refuses_legacy_or_missing_measurements(self):
        with self.assertRaisesRegex(ValueError, "missing unified"):
            parse_measurements("BATCH_GET_PERF_CONFIG timing=synchronous_batch_get\n"
                               "PERF_RESULT operation=batch_get execution_mode=legacy p50_ms=1", "unified")
        with self.assertRaisesRegex(ValueError, "missing synchronous BatchGet"):
            parse_measurements("PERF_RESULT operation=batch_get execution_mode=unified p50_ms=1", "unified")

    def test_synchronous_get_timing_accepts_profile_outside_call(self):
        args = direct.parse_args(["--get-timing", "sync-call", "--no-kernel-profile"])
        self.assertIsNone(direct.validate_args(args))
        args = direct.parse_args(["--get-timing", "sync-call", "--kernel-profile"])
        self.assertIsNone(direct.validate_args(args))
        args = direct.parse_args(["--get-timing", "sync-call", "--kernel-profile",
                                  ])
        self.assertIsNone(direct.validate_args(args))

    def test_unified_sweep_forwards_options_and_labels_host_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runner = directory / "runner"
            runner.write_text('''#!/usr/bin/env python3
import os, sys
def option(name): return sys.argv[sys.argv.index(name) + 1]
mode = "unified"
assert "--execution-mode" not in sys.argv
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
            for mode, transport, fail, count in (("unified", "urma", False, 2), ("unified", "ubmem", False, 1),
                                                  ("unified", "urma", True, 0)):
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
                    "build", "--timing", "host",
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
