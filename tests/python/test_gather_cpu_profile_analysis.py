"""Verify batch CPU summary extraction, grouping, units and statistics."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

import analyze_gather_cpu_profile as analysis


def line(sequence=1, token=656, lane=0, total=100000, **extra):
    row = dict(session=17, lane=lane, sequence=sequence, rows=4096, bytes=4096 * token,
               active_threads=16, prepare_ns=1000, copy_ns=20000, ack_ns=500,
               total_ns=total, observed_ns=1000000000, done_ns=1000100000,
               notify_poll_window_ns=0, notify_poll_window_valid=0, result="ok")
    row.update(extra)
    return "CPU_GATHER_PROFILE " + " ".join(f"{k}={v}" for k, v in row.items())


class GatherCpuSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def log(self, name, lines):
        path = self.root / name
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_units_variance_and_percentiles_preserve_slow_samples(self):
        path = self.log("656-sgl6.log", [line(i + 1, total=value * 1000) for i, value in
                                        enumerate([100, 100, 100, 100, 1000])])
        groups, count = analysis.collect([path])
        total = next(r for r in analysis.summarize(groups) if r["field"] == "total_us")
        self.assertEqual(count, 5)
        self.assertEqual(total["mean_us"], 280)
        self.assertEqual(total["variance_us2"], 162000)
        self.assertAlmostEqual(total["p95_us"], 820)
        self.assertAlmostEqual(total["p99_us"], 964)
        self.assertEqual(total["max_us"], 1000)
        fields = {r["field"] for r in analysis.summarize(groups)}
        self.assertNotIn("observed_us", fields)
        self.assertNotIn("done_us", fields)
        self.assertNotIn("notify_poll_window_us", fields)

    def test_cases_files_lanes_and_shapes_stay_separate(self):
        first = self.log("gather-cpu-656-sgl6.log", [line(), line(lane=1), line(token=4096)])
        second = self.log("gather-cpu-4096-sgl1.log", [line(token=4096)])
        groups, _ = analysis.collect([first, second], fields=["total"])
        rows = analysis.summarize(groups)
        self.assertEqual(len(rows), 4)
        self.assertEqual({(r["token_bytes"], r["sgl"]) for r in rows}, {(656, 6), (4096, 6), (4096, 1)})
        self.assertEqual({r["lane"] for r in rows}, {0, 1})
        self.assertEqual(analysis.input_files([str(self.root), str(first)]), sorted([first, second]))

    def test_filters_ignore_threads_failures_and_invalid_optional_measurements(self):
        path = self.log("4096_sgl=1.log", [line(1), line(2), line(3, result="error"),
                                          "CPU_GATHER_THREAD_PROFILE copy_ns=999999999",
                                          "CPU_GATHER_PROFILE_BUFFER samples=3 dropped=0"])
        groups, count = analysis.collect([path], fields=["copy_us", "prepare_ns"], first_sequence=2, last_sequence=3)
        self.assertEqual(count, 1)
        rows = analysis.summarize(groups)
        self.assertEqual({r["field"] for r in rows}, {"copy_us", "prepare_us"})
        self.assertTrue(all(r["samples"] == 1 and r["stddev_us"] is None for r in rows))

    def test_missing_sgl_requires_filename_or_explicit_option(self):
        path = self.log("cpu.log", [line()])
        with self.assertRaisesRegex(ValueError, "SGL|sgl"):
            analysis.collect([path])
        groups, _ = analysis.collect([path], sgl=6)
        self.assertTrue(all(r["sgl"] == 6 for r in analysis.summarize(groups)))
        named = self.log("cpu-sgl1.log", [line()])
        with self.assertRaisesRegex(ValueError, "不一致"):
            analysis.collect([named], sgl=6)

    def test_cli_writes_one_csv_and_keeps_input_unchanged(self):
        path = self.log("cpu-656-sgl6.log", [line(), line(2)])
        before = path.read_bytes()
        output = self.root / "summary.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(analysis.main([str(self.root), "-o", str(output), "--fields", "total", "copy"]), 0)
        with output.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["samples"] == "2" for r in rows))
        self.assertTrue(all(not column.endswith("_ns") for column in rows[0]))
        self.assertEqual(path.read_bytes(), before)

    def test_malformed_summary_and_empty_selection_fail_before_output(self):
        path = self.log("cpu-sgl6.log", [line(total_ns=-1)])
        output = self.root / "summary.csv"
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(analysis.main([str(path), "-o", str(output)]), 2)
            path.write_text(line() + "\n")
            self.assertEqual(analysis.main([str(path), "-o", str(output), "--first-sequence", "100"]), 2)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
