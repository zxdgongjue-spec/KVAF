"""Host-only checks for the same-device AIV/Scatter benchmark controller."""

from contextlib import contextmanager, nullcontext
import csv
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import batch_get_aiv_scatter_perf as bench


class FakeGraph:
    def __init__(self, rendezvous=None):
        self.rendezvous = rendezvous
        self.calls = 0
        self.completed = 0

    def replay(self):
        self.calls += 1

    def synchronize(self):
        if self.calls == 1 and self.rendezvous is not None:
            # If the host waits on the first lane before submitting the second,
            # this rendezvous fails instead of accidentally testing serial work.
            self.rendezvous.wait(timeout=2)
        time.sleep(0.001)
        self.completed += 1

    def replay_timed(self):
        self.replay()
        self.synchronize()
        return 1.0


class FakeLane:
    def __init__(self, backend, rendezvous=None):
        self.backend, self.device_id = backend, 2
        self.stream = object()
        self.torch = SimpleNamespace(npu=SimpleNamespace(
            set_device=lambda device: None, stream=lambda stream: nullcontext()))
        self.graph = FakeGraph(rendezvous)
        self.payload_per_graph = 1000000
        self.resources = dict(cores=2 if backend == "aiv" else 5, jettys=2)
        self.reset_count = self.check_count = 0

    def reset_output(self):
        self.reset_count += 1

    def verify(self):
        if self.graph.calls != self.graph.completed:
            raise AssertionError("verified while device work was pending")
        self.check_count += 1


class ContentionTests(unittest.TestCase):
    def test_two_lanes_submit_before_either_completes(self):
        rendezvous = threading.Barrier(2)
        lanes = [FakeLane(name, rendezvous) for name in bench.BACKENDS]
        result = bench.measure(lanes, 0.03)
        for lane, row in zip(lanes, result["lanes"]):
            self.assertTrue(row["verified"])
            self.assertGreater(row["batches"], 0)
            self.assertEqual(row["batches"], lane.graph.completed)
            self.assertEqual(row["payload_bytes"], row["batches"] * lane.payload_per_graph)
            self.assertEqual(row["device_event_ms"], row["batches"])
            self.assertAlmostEqual(row["bandwidth_GBps"], row["payload_bytes"] / (row["end_ns"] - row["start_ns"]))
            self.assertEqual((lane.reset_count, lane.check_count), (1, 1))
        a, b = result["lanes"]
        self.assertLess(max(a["start_ns"], b["start_ns"]), min(a["end_ns"], b["end_ns"]))

    def test_slow_validation_is_outside_timer(self):
        lane = FakeLane("aiv")
        checked = []
        lane.verify = lambda: checked.append(time.perf_counter_ns())
        result = bench.measure([lane], 0.01)
        self.assertGreaterEqual(checked[0], result["lanes"][0]["end_ns"])

    def test_failure_stops_and_joins_other_lane_before_return(self):
        lanes = [FakeLane(name) for name in bench.BACKENDS]
        original_sync = lanes[0].graph.synchronize

        def fail():
            original_sync()
            raise RuntimeError("device replay failed")

        lanes[0].graph.synchronize = fail
        with self.assertRaisesRegex(RuntimeError, "device replay failed"):
            bench.measure(lanes, 10)
        completed = lanes[1].graph.completed
        time.sleep(0.01)
        self.assertEqual(lanes[1].graph.completed, completed)
        self.assertEqual(lanes[1].check_count, 0)

    def test_startup_failure_aborts_barrier(self):
        lanes = [FakeLane(name) for name in bench.BACKENDS]

        def fail(device):
            raise RuntimeError("set_device failed")

        lanes[1].torch.npu.set_device = fail
        with self.assertRaisesRegex(RuntimeError, "set_device failed"):
            bench.measure(lanes, 10)

    def test_post_measurement_payload_failure_is_not_reported_successful(self):
        lane = FakeLane("aiv")

        def fail():
            raise AssertionError("payload mismatch")

        lane.verify = fail
        with self.assertRaisesRegex(AssertionError, "payload mismatch"):
            bench.measure([lane], 0.01)

    def test_cli_checks_both_resource_budgets(self):
        with patch.dict("os.environ", {}, clear=True):
            args = bench.parse_args(["--request-length", "16", "--miss-count", "8",
                                     "--aiv-cores", "4", "--aicpu-workers", "5"])
            self.assertEqual(bench.lane_options(args, "aiv").parallelism, 4)
            self.assertEqual(bench.lane_options(args, "cpu_push_scatter").parallelism, 5)
            for options in (["--seconds", "nan"], ["--seconds", "0"], ["--seconds", "inf"],
                            ["--aiv-cores", "0"], ["--aicpu-workers", "9"], ["--miss-count", "0"],
                            ["--repeats", "0"], ["--graph-calls", "0"], ["--max-inflight-total", "1"]):
                with self.subTest(options=options), patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        bench.parse_args(options)
                    self.assertEqual(error.exception.code, 2)

    def test_reports_use_outer_window_not_sum_of_lane_bandwidths(self):
        def row(name, start, end, payload):
            return dict(backend=name, start_ns=start, end_ns=end, payload_bytes=payload,
                        bandwidth_GBps=payload / (end - start), verified=True)

        aiv, scatter = bench.BACKENDS
        phases = {aiv: dict(lanes=[row(aiv, 0, 10, 100)]),
                  scatter: dict(lanes=[row(scatter, 0, 10, 200)]),
                  "concurrent": dict(lanes=[row(aiv, 0, 20, 100), row(scatter, 5, 15, 100)])}
        rows, totals = bench.summarize([phases])
        self.assertEqual([row["retained_percent"] for row in rows], [50, 50])
        self.assertEqual(totals[0]["concurrent_wall_GBps"], 10)
        self.assertEqual(totals[0]["concurrent_sum_GBps"], 15)
        self.assertEqual(totals[0]["host_window_overlap_percent"], 50)
        self.assertAlmostEqual(totals[0]["retained_percent"], 100 / 3)
        with tempfile.TemporaryDirectory() as temporary, patch("sys.stdout", io.StringIO()):
            directory = Path(temporary)
            bench.write_reports(directory, [phases])
            with (directory / "summary.csv").open() as source:
                self.assertEqual(len(list(csv.DictReader(source))), 2)
            self.assertIn("不能证明设备 kernel 同时执行", (directory / "report.md").read_text())

    def test_controller_runs_all_phases_and_retires_both_lanes(self):
        lanes, lifecycle = {}, []

        @contextmanager
        def prepare(args, backend, torch, kvaf):
            lane = lanes[backend] = FakeLane(backend)
            lifecycle.append(("prepare", backend))
            try:
                yield lane
            finally:
                self.assertEqual(lane.graph.calls, lane.graph.completed)
                lifecycle.append(("close", backend))

        torch = SimpleNamespace(npu=SimpleNamespace(set_device=lambda device: None))
        with tempfile.TemporaryDirectory() as temporary, patch.dict("os.environ", {}, clear=True):
            root = Path(temporary)
            (root / "bin").mkdir()
            for name in ("get", "put"):
                (root / f"bin/kvaf_execute_batch_{name}_kernels.o").touch()
            output = root / "result"
            with patch.object(bench, "prepared_lane", prepare), patch.dict("sys.modules", kvaf=SimpleNamespace()), \
                 patch.object(bench, "detect_torch_backend", return_value=SimpleNamespace(name="npu", torch=torch)), \
                 patch("sys.stdout", io.StringIO()):
                self.assertEqual(bench.main(["--build-dir", str(root), "--output", str(output),
                                             "--seconds", "0.01", "--repeats", "2"]), 0)
            self.assertEqual(len(list(output.glob("repeat_*.json"))), 6)
            for path in output.glob("repeat_*.json"):
                self.assertTrue(all(row["verified"] for row in json.loads(path.read_text())["lanes"]))
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["streams"], 2)
            self.assertFalse(config["profiling_enabled"])
            with (output / "summary.csv").open() as source:
                self.assertEqual(len(list(csv.DictReader(source))), 4)
        self.assertEqual(lifecycle, [("prepare", "aiv"), ("prepare", "cpu_push_scatter"),
                                     ("close", "cpu_push_scatter"), ("close", "aiv")])
        self.assertEqual([lane.check_count for lane in lanes.values()], [4, 4])


class RequestPlanTests(unittest.TestCase):
    def test_same_active_source_set_and_destination_mapping(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional CPU torch needed for request-plan checks")
        with patch.dict("os.environ", {}, clear=True):
            for source in ("random", "contiguous", "shuffled-contiguous"):
                for output in ("random", "contiguous"):
                    with self.subTest(source=source, output=output):
                        args = bench.parse_args(["--request-length", "16", "--miss-count", "9",
                                                 "--table-capacity", "4", "--tokens-per-block", "8",
                                                 "--token-bytes", "16", "--source-access-pattern", source,
                                                 "--output-access-pattern", output, "--no-sort-input-indices"])
                        addresses = torch.arange(4, dtype=torch.int64) * 128 + 4096
                        first = bench.make_plan(args, torch, "cpu", addresses)
                        second = bench.make_plan(args, torch, "cpu", addresses)
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
                        self.assertEqual(sorted(first[2].tolist()), list(range(16)))
                        args.sort_input_indices = True
                        ordered = bench.make_plan(args, torch, "cpu", addresses)

                        def mapping(plan):
                            return sorted(zip(*(values[:9].tolist() for values in plan)))

                        self.assertEqual(mapping(first), mapping(ordered))
                        sources = ordered[0][:9] * 128 + ordered[1][:9]
                        self.assertEqual(sources.tolist(), sorted(sources.tolist()))
                        if source != "random":
                            self.assertEqual(sorted(sources.tolist()), list(range(0, 9 * 16, 16)))


if __name__ == "__main__":
    unittest.main()
