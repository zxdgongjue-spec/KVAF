"""Portable runner contract tests for the AIV/UBMEM kernel profile sweep."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "tests/run_remote_cpu_batch_get_miss_count_perf.sh"
RUNNER = '''#!/usr/bin/env python3
import json, os, pathlib, sys
def option(name): return sys.argv[sys.argv.index(name) + 1]
with pathlib.Path(os.environ["CALL_LOG"]).open("a") as output:
    output.write(json.dumps(sys.argv[1:]) + "\\n")
mode = os.environ.get("REPORTED_MODE", "unified")
if not os.environ.get("MISSING_CONFIG"):
    print(f"BATCH_GET_PERF_CONFIG execution_mode={mode} timing=synchronous_batch_get")
if "--adaptive-calibration-only" in sys.argv:
    print("[KVAF_AIV] UB_DEVICE_ADAPTIVE_SKIPPED reason=k_side_device_filter")
if "--kernel-profile" in sys.argv:
    stages = ["total", "post", "completion_wait"]
    if os.environ.get("DETAIL"):
        stages += ["setup", "address_prepare", "mte_submit", "final_drain", "other"]
    for stage in stages:
        if stage == os.environ.get("MISSING_STAGE"): continue
        # Field order is intentionally different from the legacy profiler.
        print(f"PERF_RESULT execution_mode={mode} operation=batch_get_kernel_{stage} "
              f"detail={int(bool(os.environ.get('DETAIL')))} "
              "mean_ms=0.004 p50_ms=0.002 p90_ms=0.005 p99_ms=0.006 "
              "mean_cycles=4000 p50_cycles=2000 p90_cycles=5000 p99_cycles=6000")
else:
    print(f"PERF_RESULT execution_mode={mode} operation=batch_get "
          "mean_ms=0.1 p50_ms=0.1 p90_ms=0.2 p99_ms=0.3")
sys.exit(int(os.environ.get("RUNNER_EXIT", "0")))
'''


class KernelProfileSweepTests(unittest.TestCase):
    def run_sweep(self, options, **overrides):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        runner = directory / "runner"
        runner.write_text(RUNNER)
        runner.chmod(0o755)
        calls = directory / "calls.jsonl"
        summary = directory / "summary.csv"
        environment = dict(os.environ, PYTHON=sys.executable,
            KVAF_BATCH_GET_PERF_RUNNER=str(runner), CALL_LOG=str(calls),
            KVAF_BATCH_GET_TOKEN_BYTES="656", KVAF_BATCH_GET_MISS_COUNTS="32",
            KVAF_BATCH_GET_AIV_CORES="2", KVAF_BATCH_GET_JETTYS_PER_CORE="1 2",
            KVAF_BATCH_GET_KERNEL_CLOCK_MHZ="1000",
            KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR=str(directory / "logs"),
            KVAF_BATCH_GET_MISS_SWEEP_CSV=str(summary))
        for name in ("REPORTED_MODE", "MISSING_CONFIG", "DETAIL", "MISSING_STAGE", "RUNNER_EXIT"):
            environment.pop(name, None)
        environment.update(overrides)
        result = subprocess.run(["bash", str(SWEEP), str(directory / "build"), *options],
            env=environment, capture_output=True, text=True, timeout=20)
        records = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        rows = []
        if summary.exists():
            with summary.open() as source:
                rows = list(csv.DictReader(source))
        return result, records, rows, summary

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_explicit_kernel_modes_forward_and_label_results(self):
        for mode in ("unified",):
            for transport in ("urma", "ubmem"):
                with self.subTest(mode=mode, transport=transport):
                    result, calls, rows, _ = self.run_sweep([
                        "--timing", "kernel", "--transport", transport,
                        "--table-capacity", "64"])
                    self.assert_success(result)
                    self.assertEqual(len(calls), 2 if transport == "urma" else 1)
                    for call in calls:
                        self.assertNotIn("--execution-mode", call)
                        self.assertEqual(call[call.index("--transport") + 1], transport)
                        self.assertEqual(call[call.index("--table-capacity") + 1], "64")
                        self.assertIn("--kernel-profile", call)
                        self.assertNotIn("--no-kernel-profile", call)
                    self.assertEqual(len(rows), len(calls))
                    for row in rows:
                        self.assertEqual(row["execution_mode"], mode)
                        self.assertEqual(row["timing"], "kernel_cycles")
                        self.assertEqual(float(row["p50_us"]), 2)
                        self.assertAlmostEqual(float(row["p50_GBps"]), 10.496)
                        self.assertEqual(row["jettys_per_core"] == "n/a", transport == "ubmem")
                    for tag in ("PLAN", "RESULT", "SUMMARY"):
                        self.assertIn(f"BATCH_GET_MISS_SWEEP_{tag} execution_mode={mode} timing=kernel_cycles", result.stdout)

    def test_host_default_and_explicit_kernel_timing(self):
        for options, timing in (([], "synchronous_batch_get"), (["--timing", "kernel"], "kernel_cycles")):
            result, calls, rows, _ = self.run_sweep(options)
            self.assert_success(result)
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual((row["execution_mode"], row["timing"]), ("unified", timing))
            self.assertEqual("--kernel-profile" in calls[0], timing == "kernel_cycles")
            self.assertEqual("--no-kernel-profile" in calls[0], timing == "synchronous_batch_get")

    def test_missing_or_wrong_reported_mode_is_not_accepted(self):
        for overrides in ({"REPORTED_MODE": "legacy"}, {"MISSING_CONFIG": "1"}):
            result, _, rows, _ = self.run_sweep(["--timing=kernel"], **overrides)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("expected execution_mode=unified", result.stderr)
            self.assertFalse(rows)

    def test_missing_profile_stages_fail(self):
        for transport, missing in (("urma", "total"), ("urma", "post"),
                ("urma", "completion_wait"), ("ubmem", "total"), ("ubmem", "address_prepare")):
            with self.subTest(transport=transport, missing=missing):
                result, _, rows, _ = self.run_sweep(["--timing", "kernel", "--transport", transport], MISSING_STAGE=missing, DETAIL="1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("profile", result.stderr)
                self.assertFalse(rows)

    def test_child_failure_propagates_even_after_profiles(self):
        result, calls, rows, _ = self.run_sweep(["--timing", "kernel"], RUNNER_EXIT="37")
        self.assertEqual(result.returncode, 37)
        self.assertEqual(len(calls), 1)
        self.assertIn("status=37", result.stderr)
        self.assertFalse(rows)

    def test_ubmem_detail_labels_use_selected_mode(self):
        result, _, _, summary = self.run_sweep(["--timing", "kernel", "--transport", "ubmem"], DETAIL="1")
        self.assert_success(result)
        with summary.with_name("summary_ubmem_stages.csv").open() as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 5)
        self.assertEqual({(row["execution_mode"], row["timing"]) for row in rows},
                         {("unified", "kernel_cycles")})

    def test_retired_switches_rejected(self):
        for options in (["--adaptive-ub-device"], ["--execution-mode", "legacy"]):
            result, calls, rows, _ = self.run_sweep(options)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(calls)
            self.assertFalse(rows)



if __name__ == "__main__":
    unittest.main()
