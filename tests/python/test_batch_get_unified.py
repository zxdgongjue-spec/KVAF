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
from unittest.mock import MagicMock, Mock, patch

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
            options = perf.runtime_options(args)
            self.assertEqual((args.table_capacity, args.request_length, args.miss_count), (1000000, 32768, 4096))
            self.assertEqual(options["batch_get_token_bytes"], args.token_bytes)
            self.assertNotIn("aicpu_batch_get_mode", options)
            self.assertNotIn("npu_core_count", options)

    def test_defaults_and_explicit_cli_overrides(self):
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

    def test_execution_axes_are_independent(self):
        for mode in ("unified",):
            for execution in ("eager", "graph", "both"):
                args = self.parse("--backend", "ubmem", "--execution", execution)
                self.assertEqual((args.execution_mode, args.execution), (mode, execution))
                self.assertIsNone(perf.validate_args(args))
        for backend in ("gather_read", "cpu_push_scatter"):
            args = self.parse("--backend", backend, "--graph")
            self.assertIsNone(perf.validate_args(args))

    def test_backend_specific_limits_do_not_restrict_aiv(self):
        self.assertIsNone(perf.validate_args(self.parse("--request-length", "65536", "--workers", "16")))
        for backend in ("gather_read", "cpu_push_scatter"):
            for options in (("--request-length", "65537"), ("--workers", "9")):
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
        self.assertEqual(host["profile_readback_excluded"], "1")
        self.assertNotIn("status_readback_excluded", host)
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


class EagerCompletionTests(unittest.TestCase):
    def run_eager(self, backend, mode, timing="async-drain", fail_drain=False):
        with patch.dict(os.environ, {}, clear=True):
            args = perf.parse_args(["--backend", backend, "--get-timing", timing, "--workers", "2",
                                   "--table-capacity", "4", "--request-length", "4",
                                   "--miss-count", "4", "--warmup", "2", "--iterations", "3"])
        events, pending = [], []
        runtime, torch = Mock(initialized=True), MagicMock()
        runtime.begin_get_output.return_value = 17
        runtime.batch_get_kernel_profile.return_value = [
            {"device_start_ns": 1000, "device_finish_ns": 1100},
            {"device_start_ns": 1001, "device_finish_ns": 1101},
        ]

        def submit(*unused, synchronize=False, **kwargs):
            self.assertFalse(pending, "the preceding Get must have completed")
            events.append("sync_submit" if synchronize else "submit")
            if not synchronize:
                pending.append("Get")

        def wait():
            events.append("wait")
            pending.clear()

        def drain():
            events.append("drain")
            if fail_drain:
                raise RuntimeError("injected drain failure")
            pending.clear()

        def validate(*unused):
            self.assertFalse(pending, "validation must follow completed Get")
            return True

        def clock():
            events.append("clock")
            return len(events) / 1000

        runtime.batch_get.side_effect = submit
        runtime.drain_get.side_effect = drain
        torch.npu.synchronize.side_effect = wait
        torch.equal.side_effect = validate
        kvaf = SimpleNamespace(NpuRuntime=lambda *a, **k: runtime,
                               PutExecutionConfig=lambda *a: a, GetExecutionConfig=lambda *a: a,
                               ExecutionResources=lambda **k: k)
        output = io.StringIO()
        with patch.object(perf, "token_pattern", return_value=MagicMock()), \
             patch.object(perf.time, "perf_counter", side_effect=clock), \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            if fail_drain:
                with self.assertRaisesRegex(RuntimeError, "injected drain failure"):
                    perf.run_case(args, torch, kvaf, "get.o", "put.o", "127.0.0.1:1")
            else:
                self.assertEqual(perf.run_case(args, torch, kvaf, "get.o", "put.o", "127.0.0.1:1"), 0)
        return events, runtime, output.getvalue()

    def test_async_completion_is_single_wait_for_all_backends(self):
        for backend in ("aiv", "ubmem", "gather_read", "cpu_push_scatter"):
            for mode in ("unified",):
                with self.subTest(backend=backend, mode=mode):
                    events, runtime, output = self.run_eager(backend, mode)
                    completion = "drain" if mode == "unified" else "wait"
                    submissions = [i for i, event in enumerate(events) if event == "submit"]
                    self.assertEqual(len(submissions), 6)  # Preflight + warmup + measured calls.
                    for i in submissions[:3]:
                        self.assertEqual(events[i + 1], completion)
                    for i in submissions[3:]:
                        self.assertEqual(events[i - 1:i + 4],
                                         ["clock", "submit", "clock", completion, "clock"])
                    self.assertEqual(runtime.drain_get.call_count, 6 if mode == "unified" else 0)
                    self.assertEqual([row["samples"] for row in records(output)], ["3", "3"])

    def test_sync_call_measurement_has_no_external_wait_or_drain(self):
        for mode in ("unified",):
            with self.subTest(mode=mode):
                events, _, _ = self.run_eager("ubmem", mode, "sync-call")
                submissions = [i for i, event in enumerate(events) if event == "sync_submit"]
                self.assertEqual(len(submissions), 5)
                self.assertEqual(events[submissions[2] - 1:],
                                 ["clock", "sync_submit", "clock"] +
                                 ["wait", "clock", "sync_submit", "clock"] * 2)

    def test_drain_failure_is_not_hidden_by_external_wait(self):
        events, runtime, output = self.run_eager("ubmem", "unified", fail_drain=True)
        i = events.index("submit")
        self.assertEqual(events[i + 1:], ["drain"])
        self.assertNotIn("BATCH_GET_RESULT ", output)
        runtime.shutdown.assert_called_once()


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
        self.graph_leases = []
        self.released_leases = []
        self.drains = 0

    def batch_get(self, table, indices, offsets, count, rows, output, stream, **kwargs):
        if self.engine.capturing and "execution" in kwargs:
            if kwargs.get("graph_lease") != 23:
                raise AssertionError("graph capture must own a graph lease")
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

    def begin_get_graph(self, config, output, capacity, stream):
        if self.engine.capturing or self.engine.pending:
            raise AssertionError("resources must be acquired before capture")
        self.graph_leases.append((config, output, capacity))
        return 23

    def end_get_graph(self, lease, stream, discard_result=False):
        if self.engine.resets != 1 or self.engine.pending:
            raise AssertionError("graph must be reset after completion before releasing resources")
        self.released_leases.append((lease, discard_result))

    def drain_get(self):
        self.engine.synchronize()
        self.drains += 1


class GraphRunnerTests(unittest.TestCase):
    def run_graph(self, backend, api, fail_on=0, detailed=False, execution_mode="unified"):
        with patch.dict(os.environ, {}, clear=True):
            args = perf.parse_args(["--backend", backend, "--workers", "2",
                                    "--request-length", "4", "--miss-count", "4", "--warmup", "2",
                                    "--iterations", "3", "--graph-api", api,
                                    ])
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
        kvaf = SimpleNamespace(AclGraph=lambda: Graph(engine),
                               PutExecutionConfig=lambda *a: a, GetExecutionConfig=lambda *a: a,
                               ExecutionResources=lambda **k: k)
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
            if execution_mode == "unified":
                self.assertEqual(runtime.released_leases, [(23, True)])
        else:
            self.assertEqual(engine.calls, 11)  # warmup Get + 5 checks + 2 warmups + 3 measurements.
            self.assertIn(0, checked)
            self.assertIn(1, checked)
            self.assertEqual(count.values, [4])
            results = records(captured_output.getvalue())
            self.assertEqual([r["samples"] for r in results], ["3", "3"])
            self.assertEqual(results[1]["mean_us"], "0.111000")  # Actual measured replays 9, 10, 11.
            if execution_mode == "unified":
                self.assertEqual(len(runtime.graph_leases), 1)
                self.assertEqual(runtime.released_leases, [(23, False)])
                self.assertEqual(runtime.drains, 1)
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
        self.run_graph("cpu_push_scatter", "acl", fail_on=4, execution_mode="unified")

    def test_explicit_aicpu_profile_reports_stages_and_workers(self):
        self.run_graph("cpu_push_scatter", "torch", detailed=True)

    def test_unified_all_graphs_keep_resources_until_graph_reset(self):
        for backend in ("aiv", "ubmem", "gather_read", "cpu_push_scatter"):
            for api in ("acl", "torch"):
                with self.subTest(backend=backend, api=api):
                    self.run_graph(backend, api, execution_mode="unified")


if __name__ == "__main__":
    unittest.main()
