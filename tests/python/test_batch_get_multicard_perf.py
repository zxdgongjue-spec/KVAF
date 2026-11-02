#!/usr/bin/env python3
"""Host checks for multi-card coordination, accounting and graph lifecycle."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import batch_get_multicard_perf as multi
import test_npu_batch_get_perf as perf


FAKE_RUNNER = r'''#!/usr/bin/env python3
import json, os, pathlib, sys, time
def option(name):
    return sys.argv[sys.argv.index(name) + 1]
directory = pathlib.Path(option("--contention-control-dir"))
device = int(os.environ["DEVICE_ID"])
assert option("--token-bytes") == "656"
assert option("--table-capacity") == "4096"
assert "--no-kernel-profile" in sys.argv
assert os.environ["KVAF_TRANSPORT"] == "ubmem"
if directory.name == "concurrent" and device == int(os.environ.get("FAIL_DEVICE", "-1")):
    print("FAKE_UNDERLYING_DEVICE_ERROR", flush=True)
    sys.exit(3)
time.sleep(device * 0.01)
(directory / f"ready_{device}").touch()
while not (directory / "start.json").exists():
    if (directory / "abort").exists():
        (directory / f"cleaned_{device}").touch()
        sys.exit(1)
    time.sleep(0.005)
assert all((directory / f"ready_{d}").exists() for d in ([0, 1] if directory.name == "concurrent" else [device]))
window = json.loads((directory / "start.json").read_text())
time.sleep(max(0, window["stop"] - time.monotonic()))
rate = 15 if directory.name == "concurrent" else 30
elapsed = window["stop"] - window["start"]
result = dict(device=device, start=window["start"], end=window["stop"],
              bandwidth_GBps=rate, payload_bytes=round(elapsed * rate * 1e9),
              p50_call_us=90, calls=100, verified=True)
pathlib.Path(option("--contention-result")).write_text(json.dumps(result))
'''


class MulticardTests(unittest.TestCase):
    def run_controller(self, root, fail=None):
        runner = root / "runner"
        runner.write_text(FAKE_RUNNER)
        runner.chmod(0o755)
        env = dict(os.environ, KVAF_BATCH_GET_PERF_RUNNER=str(runner), KVAF_TRANSPORT="ubmem")
        if fail is not None:
            env["FAIL_DEVICE"] = str(fail)
        return subprocess.run([
            str(Path(__file__).resolve().parents[1] / "run_remote_cpu_batch_get_multicard_perf.sh"),
            str(root / "build"), "--devices", "0", "1", "--seconds", "0.05",
            "--ready-timeout", "3", "--output", str(root / "results"),
        ], env=env, capture_output=True, text=True, timeout=15)

    def test_coordinated_workers_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_controller(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = root / "results"
            with (output / "summary.csv").open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([row["device"] for row in rows], ["0", "1"])
            self.assertEqual([float(row["retained_percent"]) for row in rows], [50, 50])
            aggregate = json.loads((output / "aggregate.json").read_text())
            self.assertEqual(aggregate["isolated_sum_GBps"], 60)
            self.assertAlmostEqual(aggregate["concurrent_wall_GBps"], 30, places=5)
            self.assertEqual(aggregate["overlap_percent"], 100)
            self.assertTrue((output / "concurrent/device_0.log").exists())

    def test_failed_worker_aborts_peers_without_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_controller(root, fail=1)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("device 1", result.stderr)
            self.assertIn("exit_status=3", result.stderr)
            self.assertIn("FAKE_UNDERLYING_DEVICE_ERROR", result.stderr)
            self.assertTrue((root / "results/concurrent/abort").exists())
            self.assertTrue((root / "results/concurrent/cleaned_0").exists())
            self.assertFalse((root / "results/summary.csv").exists())

    def test_aggregate_uses_shared_wall_time_and_checks_overlap(self):
        def record(device, begin, end):
            return dict(device=device, start=begin, end=end, payload_bytes=10_000_000_000,
                        bandwidth_GBps=10, p50_call_us=80)

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            solo = [record(0, 0, 1), record(1, 0, 1)]
            shared = [record(0, 0, 1), record(1, 0.01, 1.01)]
            multi.write_summary(root, solo, shared)
            aggregate = json.loads((root / "aggregate.json").read_text())
            self.assertAlmostEqual(aggregate["concurrent_wall_GBps"], 20 / 1.01)
            self.assertAlmostEqual(aggregate["overlap_percent"], 0.99 / 1.01 * 100)
            self.assertAlmostEqual(aggregate["retained_percent"], 100 / 1.01)
            with self.assertRaisesRegex(RuntimeError, "overlapped"):
                multi.write_summary(root, solo, [record(0, 0, 1), record(1, 2, 3)])

    def test_worker_captures_warms_validates_and_releases_graph(self):
        submitted, validated = [], []

        class Graph:
            def capture_begin(self, stream):
                submitted.append("capture")

            def capture_end(self):
                submitted.append("end_capture")

            def replay(self, stream):
                time.sleep(0.001)

            def synchronize(self, stream):
                pass

            def reset(self):
                submitted.append("reset")

        torch = SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: 1, synchronize=lambda: None))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.joinpath("start.json").write_text(json.dumps({"start": time.monotonic(), "stop": time.monotonic() + 0.1}))
            args = SimpleNamespace(contention_control_dir=root, contention_result=root / "result.json",
                                   contention_graph_calls=4, warmup=3, miss_count=4096, token_bytes=656)
            plans = iter(range(4))
            with patch.dict(os.environ, DEVICE_ID="6"), contextlib.redirect_stdout(io.StringIO()):
                multi.run_worker(SimpleNamespace(AclGraph=Graph), torch, args, lambda: next(plans),
                                 lambda plan, stream: submitted.append(plan), validated.append)
            result = json.loads(args.contention_result.read_text())
            self.assertEqual(submitted, [0, "capture", 0, 1, 2, 3, "end_capture", "reset"])
            self.assertEqual(validated, [3, 3])
            self.assertEqual(result["calls"] % 4, 0)
            self.assertEqual(result["payload_bytes"], result["calls"] * 4096 * 656)
            self.assertTrue(result["verified"])

    def test_existing_benchmark_defaults_and_contention_config(self):
        with patch.dict(os.environ, {}, clear=True):
            args = perf.parse_args([])
            self.assertIsNone(args.contention_control_dir)
            self.assertIsNone(perf.validate_args(args))
            args = perf.parse_args(["--contention-control-dir", "/tmp/run"])
            self.assertIn("result path", perf.validate_args(args))
            args = perf.parse_args(["--contention-control-dir", "/tmp/run",
                                    "--contention-result", "/tmp/result.json", "--kernel-profile"])
            self.assertIn("--no-kernel-profile", perf.validate_args(args))


if __name__ == "__main__":
    unittest.main()
