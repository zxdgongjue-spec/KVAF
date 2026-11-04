#!/usr/bin/env python3
"""Host-only recovery checks for interrupted BatchGet sweeps."""

import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summarize_batch_get_miss_sweep as summary


PROFILE = """KERNEL_PROFILE_CONTRACT clock_mhz=1000 unit=ms samples=5
PERF_RESULT operation=batch_get_kernel_total samples=5 mean_cycles=10000.000 p50_cycles=8000.000 p90_cycles=12000.000 p99_cycles=16000.000
PERF_RESULT operation=batch_get_kernel_post samples=5 mean_cycles=4000.000 p50_cycles=3000.000 p90_cycles=5000.000 p99_cycles=6000.000
PERF_RESULT operation=batch_get_kernel_completion_wait samples=5 mean_cycles=6000.000 p50_cycles=5000.000 p90_cycles=7000.000 p99_cycles=10000.000
"""


class SweepRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_log(self, cores=1, profile=PROFILE):
        path = self.root / (
            f"token_bytes_656_miss_count_32_aiv_cores_{cores}_jettys_per_core_2.log"
        )
        path.write_text(
            f"BATCH_GET_PERF_CONFIG token_bytes=656 miss_count=32 aiv_cores={cores} "
            "jettys_per_core=2 kernel_profile_clock_mhz=1000\n"
            "[KVAF_AIV] UB_DEVICE_SELECTED ub_device=udmac0d1e2\n" + profile,
            encoding="utf-8",
        )
        return path

    def test_four_complete_cases_survive_truncated_fifth(self):
        logs = [self.write_log(cores) for cores in (1, 2, 4, 8)]
        incomplete = self.write_log(16, PROFILE.split(
            "PERF_RESULT operation=batch_get_kernel_completion_wait"
        )[0])
        snapshots = {path: path.read_bytes() for path in [*logs, incomplete]}
        (self.root / "ub_device_selection.log").write_text(PROFILE, encoding="utf-8")
        output = io.StringIO()
        errors = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            status = summary.main([str(self.root)])
        self.assertEqual(status, 0)
        self.assertIn("rows=4 skipped=1", output.getvalue())
        self.assertIn(str(incomplete), errors.getvalue())
        csv_path = self.root / "batch_get_miss_sweep_summary.csv"
        with csv_path.open(newline="") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, [
                "token_bytes", "miss_count", "aiv_cores", "jettys_per_core",
                "mean_us", "p50_us", "p90_us", "p99_us", "p50_GBps",
                "post_mean_us", "post_p50_us", "post_p90_us", "post_p99_us",
                "completion_wait_mean_us", "completion_wait_p50_us",
                "completion_wait_p90_us", "completion_wait_p99_us", "selected_ub_device", "metric",
            ])
        self.assertEqual([row["aiv_cores"] for row in rows], ["1", "2", "4", "8"])
        self.assertEqual(rows[0]["p50_us"], "8.000")
        self.assertEqual(rows[0]["p50_GBps"], "2.624")
        self.assertEqual(rows[0]["post_mean_us"], "4.000")
        self.assertEqual(rows[0]["completion_wait_p99_us"], "10.000")
        self.assertEqual(rows[0]["selected_ub_device"], "udmac0d1e2")
        self.assertEqual(rows[0]["metric"], "batch_get_kernel_total")
        for path, original in snapshots.items():
            self.assertEqual(path.read_bytes(), original)

    def test_recorded_clock_and_explicit_override(self):
        path = self.write_log(profile=PROFILE.replace("clock_mhz=1000", "clock_mhz=500"))
        row = summary.parse_log(path)
        self.assertEqual(row["p50_us"], "16.000")
        self.assertEqual(row["p50_GBps"], "1.312")
        row = summary.parse_log(path, clock_override=2000)
        self.assertEqual(row["p50_us"], "4.000")
        self.assertEqual(row["p50_GBps"], "5.248")

    def test_graph_microseconds_need_no_kernel_profile_or_clock(self):
        profile = (
            "PERF_RESULT operation=batch_get_graph_event samples=5 "
            "mean_us=250.125 p50_us=200.000 p90_us=300.000 p99_us=400.000 unit=us\n"
        )
        path = self.write_log(profile=profile)
        path.write_text(path.read_text().replace("clock_mhz=1000", "clock_mhz=invalid"))
        for override in (None, 500):
            row = summary.parse_log(path, clock_override=override)
            self.assertEqual(row["mean_us"], "250.125")
            self.assertEqual(row["p50_GBps"], "0.105")
            self.assertEqual(row["metric"], "batch_get_graph_event")
            self.assertEqual(row["post_mean_us"], "")
            self.assertEqual(row["completion_wait_p99_us"], "")
        path.write_text(path.read_text().replace("p50_us=200.000", "p50_us=nan"))
        with self.assertRaises(ValueError):
            summary.parse_log(path)

    def test_microsecond_kernel_results_and_clock_override(self):
        profile = PROFILE.replace("unit=ms", "unit=us").replace(
            "p50_cycles=8000.000", "p50_cycles=8000.000 p50_us=8.000"
        )
        path = self.write_log(profile=profile)
        self.assertEqual(summary.parse_log(path)["p50_us"], "8.000")
        self.assertEqual(summary.parse_log(path, clock_override=500)["p50_us"], "16.000")

    def test_config_clock_and_legacy_filename_fallback(self):
        profile = PROFILE.split("\n", 1)[1]
        path = self.write_log(profile=profile)
        path.write_text(path.read_text().replace("clock_mhz=1000", "clock_mhz=500"))
        self.assertEqual(summary.parse_log(path)["p50_us"], "16.000")
        path.write_text(profile)
        row = summary.parse_log(path)
        self.assertEqual(row["token_bytes"], 656)
        self.assertEqual(row["p50_us"], "8.000")
        self.assertEqual(row["selected_ub_device"], "")

    def test_corrupt_values_and_failed_run_are_rejected(self):
        for value in ("nan", "inf", "-1", "bad"):
            with self.subTest(value=value):
                path = self.write_log(profile=PROFILE.replace("p50_cycles=8000.000", f"p50_cycles={value}"))
                with self.assertRaises(ValueError):
                    summary.parse_log(path)
        path = self.write_log(profile=PROFILE + "BATCH_GET_PERF_UNHANDLED_ERROR type=RuntimeError\n")
        with self.assertRaisesRegex(ValueError, "benchmark error"):
            summary.parse_log(path)

    def test_no_valid_results_preserves_existing_csv(self):
        self.write_log(profile="connection lost\n")
        output = self.root / "batch_get_miss_sweep_summary.csv"
        output.write_text("previous results\n")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(summary.main([str(self.root)]), 1)
        self.assertEqual(output.read_text(), "previous results\n")

    def test_mixed_or_misnamed_cases_are_rejected(self):
        path = self.write_log()
        path.write_text(path.read_text().replace("aiv_cores=1", "aiv_cores=2"))
        with self.assertRaisesRegex(ValueError, "differs between filename and log"):
            summary.parse_log(path)
        path = self.write_log()
        path.write_text(path.read_text() * 2)
        with self.assertRaisesRegex(ValueError, "multiple runs"):
            summary.parse_log(path)


if __name__ == "__main__":
    unittest.main()
