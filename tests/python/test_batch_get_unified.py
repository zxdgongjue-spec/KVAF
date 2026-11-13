"""Host regressions for backend selection, timing scopes and mutable graph inputs."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_npu_batch_get_perf as perf


def records(output):
    return [dict(word.split("=", 1) for word in line.split()[1:])
            for line in output.splitlines() if line.startswith("BATCH_GET_RESULT ")]


class UnifiedConfigTests(unittest.TestCase):
    def parse(self, *values):
        with patch.dict(os.environ, {}, clear=True):
            return perf.parse_args(list(values))

    def test_all_backends_share_shapes_and_select_runtime(self):
        for backend in ("aiv", "ubmem", "gather_read", "cpu_push_scatter"):
            args = self.parse("--backend", backend, "--table-capacity", "1000000",
                              "--request-length", "32768", "--miss-count", "4096", "--workers", "4")
            self.assertIsNone(perf.validate_args(args))
            options = perf.runtime_options(args, "put.o")
            self.assertEqual((args.table_capacity, args.request_length, args.miss_count), (1000000, 32768, 4096))
            self.assertEqual(options["transport"], "ubmem" if backend == "ubmem" else "urma")
            self.assertEqual(options["npu_core_count"], 4)
            self.assertEqual(options.get("aicpu_batch_get_mode", ""), backend if perf.uses_aicpu(args) else "")

    def test_legacy_defaults_and_explicit_cli_overrides(self):
        args = self.parse()
        self.assertEqual((args.backend, args.transport, args.execution), ("aiv", "urma", "eager"))
        self.assertFalse(args.kernel_profile)
        self.assertFalse(args.backend_explicit)
        self.assertEqual(self.parse("--transport", "ubmem").backend, "ubmem")
        self.assertEqual(self.parse("--graph").execution, "graph")
        with patch.dict(os.environ, DEVICE_ID="1", KVAF_CPU_SERVICE_ADDR="old:1"):
            args = perf.parse_args(["--device-id", "7", "--peer", "new:2"])
        self.assertEqual((args.device_id, args.peer), (7, "new:2"))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.parse("--backend", "gather_read", "--transport", "ubmem")

    def test_backend_specific_limits_do_not_restrict_aiv(self):
        self.assertIsNone(perf.validate_args(self.parse("--request-length", "100000", "--workers", "16")))
        for backend in ("gather_read", "cpu_push_scatter"):
            for options in (("--request-length", "65537"), ("--workers", "9"), ("--adaptive-ub-device",)):
                self.assertIsNotNone(perf.validate_args(self.parse("--backend", backend, *options)))
            self.assertIsNone(perf.validate_args(self.parse("--backend", backend, "--peer-ub-device", "udma0")))
        self.assertIn("graph-replays", perf.validate_args(self.parse("--graph-replays", "0")))

    def test_timing_is_default_and_diagnostics_are_explicit(self):
        calls = []
        runtime = SimpleNamespace(
            start_batch_get_kernel_profile=lambda detailed=False: calls.append(detailed))
        for backend in ("aiv", "ubmem", "gather_read", "cpu_push_scatter"):
            args = self.parse("--backend", backend)
            perf.start_kernel_measurement(args, runtime)
            self.assertFalse(calls[-1])
            args = self.parse("--backend", backend, "--kernel-profile")
            perf.start_kernel_measurement(args, runtime)
            self.assertFalse(calls[-1])
        args = self.parse("--backend", "ubmem", "--kernel-profile", "--ubmem-profile-detail")
        perf.start_kernel_measurement(args, runtime)
        self.assertTrue(calls[-1])

    def test_wrapper_accepts_aicpu_build_without_aiv_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "aicpu-only"
            (build / "python/kvaf").mkdir(parents=True)
            runner = root / "fake-python"
            runner.write_text("#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
            runner.chmod(0o755)
            timeout = root / "timeout"
            timeout.write_text("#!/usr/bin/env python3\nimport os, sys\nos.execvp(sys.argv[2], sys.argv[2:])\n")
            timeout.chmod(0o755)
            env = dict(os.environ, PYTHON=str(runner), PATH=str(root) + os.pathsep + os.environ["PATH"])
            result = subprocess.run([str(Path(__file__).resolve().parents[1] / "run_remote_cpu_batch_get_perf.sh"),
                                     str(build), "--backend", "cpu_push_scatter"],
                                    env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            argv = json.loads(result.stdout)
            self.assertEqual(argv[argv.index("--build-dir") + 1], str(build))
            self.assertEqual(argv[-2:], ["--backend", "cpu_push_scatter"])


class TimingTests(unittest.TestCase):
    def test_operator_time_is_parallel_wall_span(self):
        ns = [{"device_start_ns": 1000000, "device_finish_ns": 1040000},
              {"device_start_ns": 1010000, "device_finish_ns": 1060000}]
        self.assertAlmostEqual(perf.device_profile_ms(ns, 2000), .06)
        cycles = [{"kernel_start_cycle": 100, "kernel_finish_cycle": 40100},
                  {"kernel_start_cycle": 10100, "kernel_finish_cycle": 60100}]
        self.assertAlmostEqual(perf.device_profile_ms(cycles, 1000), .06)
        with self.assertRaises(AssertionError):
            perf.device_profile_ms([ns[0], cycles[0]], 1000)
        with self.assertRaises(AssertionError):
            perf.device_profile_ms([{"device_start_ns": 20, "device_finish_ns": 10}], 1000)

    def test_two_scopes_and_latency_corresponding_bandwidth(self):
        with patch.dict(os.environ, {}, clear=True):
            args = perf.parse_args(["--backend", "gather_read", "--workers", "2",
                                    "--miss-count", "10", "--token-bytes", "1000"])
        sample = [{"device_start_ns": 1000000, "device_finish_ns": 1040000},
                  {"device_start_ns": 1010000, "device_finish_ns": 1060000}]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            perf.print_measurements(args, "graph", [1, 3], [sample, sample])
        host, kernel = records(output.getvalue())
        self.assertEqual((host["scope"], kernel["scope"]), ("end_to_end", "kernel"))
        self.assertEqual(host["mean_GBps"], "0.005000")  # payload / mean latency, not mean of rates.
        self.assertEqual(kernel["mean_us"], "60.000000")
        self.assertEqual(kernel["mean_GBps"], "0.166667")
        self.assertEqual(kernel["timing_source"], "device_ns")
        self.assertEqual(kernel["profiling_enabled"], "1")
        self.assertEqual(kernel["detail_output_enabled"], "0")
        self.assertNotIn("BATCH_GET_PROFILE", output.getvalue())
        args.miss_count = 0
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            perf.print_measurements(args, "graph", [1, 3], [sample, sample])
        self.assertTrue(all(row["mean_GBps"] == "0.000000" for row in records(output.getvalue())))


class Tensor:
    def __init__(self, values):
        self.values = list(values)

    def copy_(self, other):
        self.values[:] = other.values

    def fill_(self, value):
        self.values[:] = [value] * len(self.values)

    def zero_(self):
        self.fill_(0)


class Engine:
    def __init__(self):
        self.capturing = False
        self.nodes, self.pending = [], []
        self.calls, self.resets = 0, 0

    def synchronize(self):
        if self.capturing:
            raise AssertionError("synchronization inside capture")
        while self.pending:
            self.pending.pop(0)()


class Graph:
    def __init__(self, engine):
        self.engine = engine
        self.nodes = []

    def capture_begin(self, stream):
        self.engine.capturing = True
        self.engine.nodes = []

    def capture_end(self):
        self.nodes = self.engine.nodes[:]
        self.engine.capturing = False

    def replay(self, stream=None):
        self.engine.pending.extend(self.nodes)

    def reset(self):
        self.engine.resets += 1


class Runtime:
    def __init__(self, engine, ns, fail_on=0, detailed=False):
        self.engine, self.ns, self.fail_on, self.detailed = engine, ns, fail_on, detailed
        self.profiles = []

    def batch_get(self, table, indices, offsets, count, rows, output, stream):
        def kernel():
            self.engine.calls += 1
            for i in range(count.values[0]):
                output.values[rows.values[i]] = indices.values[i] * 10 + offsets.values[i]
            start, finish = (("device_start_ns", "device_finish_ns") if self.ns else
                              ("kernel_start_cycle", "kernel_finish_cycle"))
            self.profiles = [{start: self.engine.calls * 1000 + w,
                              finish: self.engine.calls * 1000 + w + 100 + self.engine.calls}
                             for w in range(2)]
            if self.detailed:
                for w, p in enumerate(self.profiles):
                    p["stages"] = dict.fromkeys(("request_us", "descriptor_cache_us", "source_write_us", "notify_us",
                        "read_us", "read_setup_us", "wr_build_us", "ready_wait_us", "post_us", "cq_wait_us",
                        "scatter_us", "ack_us"), 0.0)
                    p["stages"]["scatter_us"] = (w + 1) * self.engine.calls / 1000
                    p["stages"]["read_us"] = p["stages"]["scatter_us"] * 1.5
        if self.engine.capturing:
            self.engine.nodes.append(kernel)
        else:
            self.engine.pending.append(kernel)

    def check_batch_get_status(self, stream):
        if self.engine.capturing:
            raise AssertionError("status D2H inside capture")
        if self.engine.calls == self.fail_on:
            raise RuntimeError("injected protocol error")

    def batch_get_kernel_profile(self):
        if self.engine.capturing or self.engine.pending:
            raise AssertionError("profile read before stream completion")
        return [dict(p) for p in self.profiles]


class GraphRunnerTests(unittest.TestCase):
    def run_graph(self, backend, api, fail_on=0, detailed=False):
        with patch.dict(os.environ, {}, clear=True):
            args = perf.parse_args(["--backend", backend, "--workers", "2",
                                    "--request-length", "4", "--miss-count", "4", "--warmup", "2",
                                    "--iterations", "3", "--graph-api", api])
            args.kernel_profile = detailed
        engine = Engine()
        runtime = Runtime(engine, perf.uses_aicpu(args), fail_on, detailed)
        output, count = Tensor([0] * 4), Tensor([4])
        generated, checked = [], []

        def make_plan():
            epoch = len(generated) + 1
            plan = Tensor([epoch + i for i in range(4)]), Tensor([epoch % 2] * 4), Tensor([3, 1, 0, 2])
            generated.append(plan)
            return plan

        def check(plan, active):
            expected = [0] * 4
            for i in range(active):
                expected[plan[2].values[i]] = plan[0].values[i] * 10 + plan[1].values[i]
            self.assertEqual(output.values, expected)
            checked.append(active)

        @contextlib.contextmanager
        def torch_capture(graph, stream):
            graph.capture_begin(stream)
            try:
                yield
            finally:
                graph.capture_end()

        stream = SimpleNamespace(synchronize=engine.synchronize)
        torch = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: stream,
                                NPUGraph=lambda: Graph(engine), graph=torch_capture))
        kvaf = SimpleNamespace(AclGraph=lambda: Graph(engine))
        captured_output = io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            if fail_on:
                with self.assertRaisesRegex(RuntimeError, "protocol error"):
                    perf.run_graph_benchmark(args, runtime, torch, kvaf, make_plan, None, count, output, check)
            else:
                perf.run_graph_benchmark(args, runtime, torch, kvaf, make_plan, None, count, output, check)
        self.assertEqual(engine.resets, 1)
        self.assertFalse(engine.pending)
        if fail_on:
            self.assertFalse(records(captured_output.getvalue()))
        else:
            self.assertEqual(engine.calls, 11)  # warmup Get + 5 checks + 2 warmups + 3 measurements.
            self.assertIn(0, checked)
            self.assertIn(1, checked)
            self.assertEqual(count.values, [4])
            results = records(captured_output.getvalue())
            self.assertEqual([r["samples"] for r in results], ["3", "3"])
            self.assertEqual(results[1]["mean_us"], "0.111000")  # Actual measured replays 9, 10, 11.
            if detailed:
                self.assertIn("stage=scatter mean_us=0.020000", captured_output.getvalue())
                self.assertEqual(captured_output.getvalue().count("BATCH_GET_PROFILE_WORKER "), 2)
            else:
                self.assertNotIn("BATCH_GET_PROFILE", captured_output.getvalue())
                self.assertNotIn("KERNEL_PROFILE_CORE", captured_output.getvalue())

    def test_all_backends_both_graph_apis(self):
        for backend in ("aiv", "ubmem", "gather_read", "cpu_push_scatter"):
            for api in ("acl", "torch"):
                with self.subTest(backend=backend, api=api):
                    self.run_graph(backend, api)

    def test_protocol_failure_does_not_emit_success_and_releases_graph(self):
        self.run_graph("cpu_push_scatter", "torch", fail_on=4)

    def test_explicit_aicpu_profile_reports_stages_and_workers(self):
        self.run_graph("cpu_push_scatter", "torch", detailed=True)


if __name__ == "__main__":
    unittest.main()
