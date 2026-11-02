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
transport = os.environ["KVAF_TRANSPORT"]
assert transport == os.environ["EXPECTED_TRANSPORT"]
assert option("--aiv-cores") in os.environ["EXPECTED_CORES"].split()
assert option("--jettys-per-core") == os.environ["EXPECTED_JETTYS"]
assert option("--max-inflight-total") == os.environ["EXPECTED_INFLIGHT"]
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
# Synthetic results use the coordinator's timestamps; no worker clock is needed.
time.sleep(window["stop"] - window["start"])
rate = 15 if directory.name == "concurrent" else 30
elapsed = window["stop"] - window["start"]
result = dict(device=device, start=window["start"], end=window["stop"],
              bandwidth_GBps=rate, payload_bytes=round(elapsed * rate * 1e9),
              p50_call_us=90, calls=100, verified=True)
pathlib.Path(option("--contention-result")).write_text(json.dumps(result))
'''


class MulticardTests(unittest.TestCase):
    def run_controller(self, root, fail=None, transport="ubmem", options=(), cores=16,
                       jettys=2, inflight=2048, expected_transport=None, core_env="16"):
        runner = root / "runner"
        runner.write_text(FAKE_RUNNER)
        runner.chmod(0o755)
        env = dict(os.environ, KVAF_BATCH_GET_PERF_RUNNER=str(runner), KVAF_TRANSPORT=transport,
                   KVAF_BATCH_GET_MAX_INFLIGHT_TOTAL="2048",
                   KVAF_BATCH_GET_AIV_CORES=core_env,
                   EXPECTED_TRANSPORT=expected_transport or transport, EXPECTED_CORES=str(cores),
                   EXPECTED_JETTYS=str(jettys), EXPECTED_INFLIGHT=str(inflight))
        if fail is not None:
            env["FAIL_DEVICE"] = str(fail)
        return subprocess.run([
            str(Path(__file__).resolve().parents[1] / "run_remote_cpu_batch_get_multicard_perf.sh"),
            str(root / "build"), "--devices", "0", "1", "--seconds", "0.05",
            "--ready-timeout", "10", "--output", str(root / "results"),
            *options,
        ], env=env, capture_output=True, text=True, timeout=30)

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
            self.assertEqual(aggregate["concurrent_sum_GBps"], 30)
            self.assertAlmostEqual(aggregate["concurrent_wall_GBps"], 30, places=5)
            self.assertAlmostEqual(aggregate["drop_percent"], 50, places=5)
            self.assertAlmostEqual(aggregate["lost_GBps"], 30, places=5)
            self.assertEqual(aggregate["overlap_percent"], 100)
            self.assertTrue((output / "concurrent/device_0.log").exists())
            with (output / "aggregate.csv").open() as source:
                totals = list(csv.DictReader(source))
            self.assertEqual(len(totals), 1)
            self.assertEqual(totals[0]["device_order"], "0 1")
            self.assertEqual(float(totals[0]["isolated_sum_GBps"]), 60)
            report = (output / "report.md").read_text()
            self.assertIn("每卡独立 328 MiB", report)
            self.assertIn("| 各卡相加 | 60.000 | 30.000 |", report)
            self.assertIn("| 并发总吞吐（实际窗口） | 30.000 | GB/s |", report)

            # Reporting works from saved results even with no runnable worker.
            (root / "runner").unlink()
            regenerated = subprocess.run([
                str(Path(__file__).resolve().parents[1] / "run_remote_cpu_batch_get_multicard_perf.sh"),
                str(root / "missing-build"), "--summarize", str(output),
            ], capture_output=True, text=True, timeout=5)
            self.assertEqual(regenerated.returncode, 0, regenerated.stdout + regenerated.stderr)
            self.assertIn("MULTICARD_REPORT", regenerated.stdout)
            self.assertNotIn("MULTICARD_MEASURE", regenerated.stdout)
            self.assertEqual((output / "report.md").read_text(), report)

    def test_urma_core_sweep_and_offline_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_controller(root, transport="urma", cores="1 2",
                                         options=("--aiv-cores", "1", "2"))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = root / "results"
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["transport"], "urma")
            self.assertEqual(config["aiv_cores"], [1, 2])
            self.assertIn("jettys_per_core=2 max_inflight_total=2048", result.stdout)
            report = (output / "cores_2/report.md").read_text()
            self.assertIn("| 协议 | urma |", report)
            self.assertIn("| Jetty 数 / AIV 核 | 2 |", report)
            self.assertIn("| 未完成 READ 上限 / 卡 | 2048 |", report)
            with (output / "aggregate.csv").open() as source:
                totals = list(csv.DictReader(source))
            self.assertEqual([row["aiv_cores"] for row in totals], ["1", "2"])
            self.assertTrue(all(row["transport"] == "urma" for row in totals))
            self.assertTrue(all(row["max_inflight_total"] == "2048" for row in totals))
            with (output / "summary.csv").open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([(row["aiv_cores"], row["device"]) for row in rows],
                             [("1", "0"), ("1", "1"), ("2", "0"), ("2", "1")])
            sweep_report = (output / "report.md").read_text()
            (root / "runner").unlink()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(multi.main([str(root / "missing-build"), "--summarize", str(output)]), 0)
            self.assertEqual((output / "report.md").read_text(), sweep_report)

    def test_core_sweep_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_controller(root, core_env="8 16", cores="8 16")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config = json.loads((root / "results/config.json").read_text())
            self.assertEqual(config["aiv_cores"], [8, 16])

    def test_explicit_urma_configuration_overrides_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # The command-line protocol must override KVAF_TRANSPORT too.
            runner_result = self.run_controller(
                root, options=("--transport", "urma", "--aiv-cores", "4",
                               "--jettys-per-core", "1", "--max-inflight-total", "1024"),
                cores=4, jettys=1, inflight=1024, expected_transport="urma",
            )
            self.assertEqual(runner_result.returncode, 0, runner_result.stdout + runner_result.stderr)

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
            self.assertEqual(aggregate["concurrent_sum_GBps"], 20)
            self.assertAlmostEqual(aggregate["overlap_percent"], 0.99 / 1.01 * 100)
            self.assertAlmostEqual(aggregate["retained_percent"], 100 / 1.01)
            self.assertAlmostEqual(aggregate["drop_percent"], 100 - 100 / 1.01)
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
