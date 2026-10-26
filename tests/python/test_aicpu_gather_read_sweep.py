#!/usr/bin/env python3
"""Offline checks for GatherRead sweep coverage, timing scopes and failure handling."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load("gather_sweep", ROOT / "tests/run_aicpu_gather_read_sweep.py")
demo = load("gather_demo", ROOT / "access_operator/demo/aicpu_batch_get/python/test_batch_get.py")
sys.path.insert(0, str(ROOT / "access_operator/demo/aicpu_batch_get/python"))
bench = load("gather_benchmark", sweep.RUNNER)


def sample(span, post_a, post_b):
    workers = []
    for post in (post_a, post_b):
        workers.append(dict(post_us=post, cq_wait_us=8, ready_wait_us=12, read_us=130,
                            wr_build_us=10, request_us=5, source_write_us=2, notify_us=1, ack_us=3))
    return dict(device_span_us=span, workers=workers)


def result():
    return dict(mode="gather_read", token_bytes=656, miss_count=32, workers=2,
                table_capacity=128, tokens_per_block=128, request_length=32768,
                ddr_bytes=128 * 128 * 656, source_access_pattern="random_with_replacement",
                output_access_pattern="random_unique_rows", payload_bytes=20992,
                warmup=1, iterations=2, profile_rounds=2,
                end_to_end=demo.timing_stats([200, 300], 20992),
                stages=demo.summarize_profiles([sample(150, 100, 1), sample(200, 1, 100)], 20992))


class GatherSweepTests(unittest.TestCase):
    def test_profile_max_is_per_round_and_not_sum(self):
        summary = result()
        self.assertEqual(summary["stages"]["post"]["mean_us"], 100)
        self.assertEqual(summary["stages"]["kernel_total"]["mean_us"], 175)
        self.assertEqual(summary["stages"]["kernel_total"]["p90_us"], 195)
        row = sweep.summary_row(summary)
        self.assertEqual(row["end_to_end_mean_us"], 250)
        self.assertEqual(row["mean_us"], 175)
        self.assertAlmostEqual(row["p50_GBps"], 20992 / 175 / 1000)
        self.assertAlmostEqual(row["end_to_end_GBps"], 20992 / 250 / 1000)
        self.assertEqual(demo.timing_stats([1, 2], 0)["GBps"], 0)

    def test_exact_miss_count_and_empty_prefix(self):
        with tempfile.NamedTemporaryFile() as kernel:
            for count in (0, 17, 64):
                a = demo.parse_args(["--kernel-json", kernel.name, "--blocks", "64",
                                     "--miss-count", str(count), "--active-percent", "75"])
                self.assertEqual(a.miss_count, count)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                demo.parse_args(["--kernel-json", kernel.name, "--blocks", "64", "--miss-count", "65"])

    def test_full_pool_random_sampling_and_unspaced_output(self):
        class Values(list):
            def to(self, dtype):
                return self
            def __mul__(self, n):
                return Values(value * n for value in self)
        calls = []
        def randint(low, high, shape, **kwargs):
            calls.append((low, high, shape))
            return Values([high - 1] * shape[0])
        torch = SimpleNamespace(int32="int32", randint=randint,
                                randperm=lambda n, **kw: Values(reversed(range(n))))
        args = SimpleNamespace(table_capacity=1277952, tokens_per_block=128, token_bytes=656)
        indices, offsets, rows = bench.create_request_plan(torch, args, 4, "npu:7")
        self.assertEqual(calls, [(0, 1277952, (4,)), (0, 128, (4,))])
        self.assertEqual(indices, [1277951] * 4)  # Repeated sources are retained.
        self.assertEqual(offsets, [127 * 656] * 4)
        self.assertEqual(rows, [3, 2, 1, 0])  # No factor-of-two spacing or guard rows.
        with tempfile.NamedTemporaryFile() as kernel:
            a = bench.parse_args(["--kernel-json", kernel.name, "--table-capacity", "1", "--miss-count", "64"])
            self.assertEqual(a.miss_count, 64)  # Pool size does not constrain request count.

    def test_default_plan_and_override_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            a = sweep.parse_args(["--dry-run", "--output-dir", directory + "/plan"])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sweep.run_sweep(a), 0)
            plan = json.loads((a.output_dir / "plan.json").read_text())
            self.assertEqual(len(plan["cases"]), 252)
            for case in plan["cases"]:
                command = case["command"]
                self.assertTrue(command[1].endswith("benchmark_gather_read.py"))
                self.assertNotIn("--request-capacity", command)
                self.assertNotIn("--output-row-step", command)
                self.assertEqual(command[command.index("--table-capacity") + 1], "1277952")
                self.assertEqual(command[command.index("--tokens-per-block") + 1], "128")
                self.assertNotIn("--aiv-cores", command)
                self.assertNotIn("--jettys-per-core", command)
                self.assertEqual(int(command[command.index("--miss-count") + 1]), case["miss_count"])
            self.assertFalse((a.output_dir / "summary.csv").exists())
        for args in (["--workers", "9"], ["--profile-rounds", "0"],
                     ["--request-capacity", "65536"], ["--table-capacity", "0"],
                     ["--miss-counts", "65537"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sweep.parse_args(args)

    def test_success_then_failure_keeps_completed_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            build = base / "build"
            (build / "python/kvaf").mkdir(parents=True)
            a = sweep.parse_args(["--build-dir", str(build), "--output-dir", str(base / "out"),
                                  "--token-bytes", "656", "--miss-counts", "32", "64", "128",
                                  "--workers", "2", "--table-capacity", "128"])
            def execute(command, **kwargs):
                count = int(command[command.index("--miss-count") + 1])
                self.assertEqual(command[command.index("--table-capacity") + 1], "128")
                self.assertTrue(kwargs["env"]["PYTHONPATH"].startswith(str((build / "python").resolve())))
                kwargs["stdout"].write(f"case {count}\n")
                if count == 64:
                    return subprocess.CompletedProcess(command, 1)
                data = result()
                Path(command[command.index("--result-json") + 1]).write_text(json.dumps([data]))
                return subprocess.CompletedProcess(command, 0)
            with patch.object(sweep.subprocess, "run", side_effect=execute) as run, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(sweep.run_sweep(a), 1)
                self.assertEqual(run.call_count, 2)
            csv = (a.output_dir / "summary.csv").read_text().splitlines()
            self.assertEqual(len(csv), 2)
            self.assertIn("p90_us", csv[0])
            self.assertIn("ready_wait_mean_us", csv[0])
            self.assertTrue((a.output_dir / "token_656_miss_64_workers_2.log").exists())
            self.assertFalse((a.output_dir / "token_656_miss_128_workers_2.log").exists())

            a.output_dir = base / "success"
            a.miss_counts = [32]
            output = io.StringIO()
            with patch.object(sweep.subprocess, "run", side_effect=execute), contextlib.redirect_stdout(output):
                self.assertEqual(sweep.run_sweep(a), 0)
            self.assertIn("AICPU_GATHER_READ_SWEEP_PASS cases=1", output.getvalue())
            self.assertIn("stage=end_to_end", output.getvalue())
            self.assertIn("p50_GBps", output.getvalue())

    def test_timeout_stops_without_result(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "build/python/kvaf").mkdir(parents=True)
            a = sweep.parse_args(["--build-dir", str(base / "build"), "--output-dir", str(base / "out"),
                                  "--token-bytes", "656", "--miss-counts", "0", "32", "--workers", "1"])
            with patch.object(sweep.subprocess, "run", side_effect=subprocess.TimeoutExpired("test", 1)) as run, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(sweep.run_sweep(a), 1)
                self.assertEqual(run.call_count, 1)
            self.assertFalse((a.output_dir / "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
