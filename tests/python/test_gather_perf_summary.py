"""Validate benchmark shape and timeline handling without importing torch/ACL."""
import contextlib
import copy
import csv
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_aiv_gather_batch_get_perf as perf


def snapshot(sequence=1, cores=3, requests=1):
    lanes = []
    names = [perf.STAGE_BOUNDARIES[0][1]] + [s[2] for s in perf.STAGE_BOUNDARIES[:-1]]
    for core in range(cores):
        count = requests * (core + 1) // cores - requests * core // cores
        # Empty lanes are legal; retain their overhead instead of dropping them.
        lane = {name: core * 100 + step * (core + 1) for step, name in enumerate(names)}
        lane.update(core=core, sequence=sequence, result=0, requests=count, data_wrs=count)
        lanes.append(lane)
    return {"sequence": sequence, "host": {name: 100 for name in perf.HOST_FIELDS}, "cores": lanes}


class GatherPerfSummaryTest(unittest.TestCase):
    def csv_metadata(self, clock=None):
        with patch.object(perf, "REQUEST_COUNT", 1), patch.object(perf, "PROFILE_CLOCK_MHZ", clock), \
             patch.object(perf.subprocess, "run", side_effect=FileNotFoundError):
            return perf.csv_metadata(3, 7, "gather.o", "put.o")

    @staticmethod
    def read_csv(path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            assert len(reader.fieldnames) == len(set(reader.fieldnames)), reader.fieldnames
            return list(reader)

    def test_default_config_and_supported_shapes(self):
        for cores in (1, 3, 8):
            for count in (1024, 8193, 65536):
                with patch.object(perf, "REQUEST_COUNT", count):
                    perf.validate_config(cores)

    def test_invalid_shapes_and_counts_fail_before_hardware(self):
        for field, value in (
            ("TOKEN_BYTES", 0), ("TOKEN_BYTES", 65536),
            ("REQUEST_COUNT", 65537), ("REQUEST_COUNT", True),
            ("WARMUP", -1), ("ITERATIONS", -1), ("PROFILE_SAMPLES", -1),
            ("TABLE_CAPACITY", 1 << 31), ("TOKENS_PER_BLOCK", 1 << 31),
            ("GATHER_SGL_SIZE", 7), ("SEED", -1),
        ):
            with self.subTest(field=field, value=value), patch.object(perf, field, value):
                with self.assertRaises(ValueError):
                    perf.validate_config(1)
        for cores in (0, 9):
            with self.assertRaises(ValueError):
                perf.validate_config(cores)

    def test_profile_only_debugging_and_empty_sampling_rejection(self):
        with patch.object(perf, "WARMUP", 0), patch.object(perf, "ITERATIONS", 0):
            perf.validate_config(1)
            with patch.object(perf, "PROFILE_SAMPLES", 0):
                with self.assertRaisesRegex(ValueError, "both be zero"):
                    perf.validate_config(1)

    def test_clock_requires_explicit_positive_verified_value(self):
        for clock in (None, 50, 1000.0):
            with patch.object(perf, "PROFILE_CLOCK_MHZ", clock):
                perf.validate_config(1)
        for clock in (0, -1, True, "1000", float("nan"), float("inf")):
            with self.subTest(clock=clock), patch.object(perf, "PROFILE_CLOCK_MHZ", clock):
                with self.assertRaises(ValueError):
                    perf.validate_config(1)

    def test_percentile_uses_samples_and_interpolates(self):
        self.assertEqual(perf.percentile([10.0], 0.99), 10.0)
        self.assertEqual(perf.percentile([30.0, 10.0, 20.0], 0.5), 20.0)
        self.assertAlmostEqual(perf.percentile([0.0, 10.0], 0.95), 9.5)
        for values, fraction in (([], 0.5), ([1.0], -1), ([1.0], 2)):
            with self.assertRaises(ValueError):
                perf.percentile(values, fraction)

    def test_empty_lanes_and_per_core_duration_are_preserved(self):
        host, lanes = perf.summarize_profiles([snapshot(), snapshot(2)], 3, 1)
        self.assertEqual(host["total_ns"], 100)
        self.assertEqual([lane["requests"] for lane in lanes], [0, 0, 1])
        self.assertEqual([lane["total"] for lane in lanes], [6, 12, 18])
        for lane in lanes:
            self.assertEqual(sum(lane[name] for name, _, _ in perf.STAGE_BOUNDARIES[:-1]), lane["total"])

    def test_profiles_reject_missing_stale_failed_or_wrong_lane_data(self):
        original = snapshot()
        for case in ("missing_core", "duplicate", "wrong_partition", "stale_lane", "error", "time_reversal", "negative_counter", "negative_host"):
            invalid = copy.deepcopy(original)
            if case == "missing_core":
                invalid["cores"].pop()
            elif case == "duplicate":
                invalid["cores"][1]["core"] = 0
            elif case == "wrong_partition":
                invalid["cores"][0]["requests"] = 1
            elif case == "stale_lane":
                invalid["cores"][0]["sequence"] = 0
            elif case == "error":
                invalid["cores"][0]["result"] = 4
            elif case == "time_reversal":
                invalid["cores"][0]["ack_done_cycle"] = 0
            elif case == "negative_counter":
                invalid["cores"][0]["data_wrs"] = -1
            else:
                invalid["host"]["wait_ns"] = -1
            with self.subTest(case=case), self.assertRaises(ValueError):
                perf.summarize_profiles([invalid], 3, 1)
        with self.assertRaisesRegex(ValueError, "sequence"):
            perf.summarize_profiles([original, original], 3, 1)
        with self.assertRaises(ValueError):
            perf.summarize_profiles([], 3, 1)

    def test_unverified_clock_prints_cycles_without_guessed_microseconds(self):
        out = io.StringIO()
        with patch.object(perf, "REQUEST_COUNT", 1), patch.object(perf, "PROFILE_CLOCK_MHZ", None):
            with contextlib.redirect_stdout(out):
                perf.print_profiles([snapshot()], 3)
        self.assertIn("clock_mhz=unverified", out.getvalue())
        lines = [line for line in out.getvalue().splitlines() if line.startswith("GATHER_PROFILE_CORE")]
        self.assertEqual(len(lines), 3)
        self.assertTrue(all("total_cycles=" in line and "total_us=" not in line for line in lines))

    def test_hardware_opt_out_does_not_import_runtime(self):
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(perf.main(), 77)

    def test_csv_round_trip_raw_samples_and_multi_core_summary(self):
        metadata = self.csv_metadata()
        samples = [snapshot(123), snapshot(124)]
        # Large absolute counters must remain exact integers in raw CSV rows.
        for sample in samples:
            for lane in sample["cores"]:
                for name in [perf.STAGE_BOUNDARIES[0][1]] + [s[2] for s in perf.STAGE_BOUNDARIES[:-1]]:
                    lane[name] += 97320160017685
        with tempfile.TemporaryDirectory() as directory:
            paths = perf.export_csv_results(directory, metadata, [1000.0, 2000.0], samples, [2100.0, 2200.0])
            summary, e2e, host, cores = map(self.read_csv, paths)
            self.assertEqual([len(e2e), len(host), len(cores)], [2, 2, 6])
            self.assertEqual([float(row["e2e_us"]) for row in e2e], [1000.0, 2000.0])
            self.assertEqual([float(row["call_e2e_us"]) for row in host], [2100.0, 2200.0])
            self.assertEqual([row["sequence"] for row in host], ["123", "124"])
            self.assertEqual(int(cores[0]["kernel_start_cycle"]), 97320160017685)
            self.assertEqual([int(row["requests"]) for row in cores[:3]], [0, 0, 1])
            self.assertTrue(all(row["total_us"] == "" for row in cores))
            self.assertTrue(all(row["clock_source"] == "unverified" for row in cores))
            results = {row["row_kind"]: row for row in summary if row["row_kind"] != "profile_core_mean"}
            self.assertEqual(results["sampling_status"]["sampling_status"], "complete")
            self.assertEqual(float(results["e2e_summary"]["mean_us"]), 1500.0)
            self.assertEqual(float(results["e2e_summary"]["p95_us"]), 1950.0)
            self.assertEqual(float(results["profile_e2e_summary"]["mean_us"]), 2150.0)
            self.assertEqual(results["profile_host_mean"]["first_sequence"], "123")
            core_means = [row for row in summary if row["row_kind"] == "profile_core_mean"]
            self.assertEqual(len(core_means), 3)
            self.assertEqual([float(row["total_cycles"]) for row in core_means], [6, 12, 18])
            self.assertTrue(all(row["run_id"] == metadata["run_id"] for row in summary + e2e + host + cores))

    def test_csv_clock_conversion_uses_captured_configuration(self):
        metadata = self.csv_metadata(clock=50)
        with tempfile.TemporaryDirectory() as directory:
            paths = perf.export_csv_results(directory, metadata, [], [snapshot()], [100.0])
            cores = self.read_csv(paths[3])
            self.assertAlmostEqual(float(cores[0]["total_us"]), 6 / 50)
            self.assertEqual(self.read_csv(paths[1]), [])

    def test_csv_failure_preserves_completed_samples_and_quotes_errors(self):
        metadata = self.csv_metadata()
        error = RuntimeError('failed, "profile"\nreason=valid_mask')
        with tempfile.TemporaryDirectory() as directory:
            paths = perf.export_csv_results(directory, metadata, [1000.0], [snapshot()], [2000.0], error)
            summary = self.read_csv(paths[0])
            self.assertEqual(summary[0]["sampling_status"], "failed")
            self.assertEqual(summary[0]["error_message"], str(error))
            self.assertEqual(len(self.read_csv(paths[1])), 1)
            self.assertEqual(len(self.read_csv(paths[2])), 1)
            self.assertEqual(len(self.read_csv(paths[3])), 3)

    def test_csv_unique_runs_and_no_overwrite(self):
        first, second = self.csv_metadata(), self.csv_metadata()
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["git_dirty"], "unknown")
        with tempfile.TemporaryDirectory() as directory:
            paths = perf.export_csv_results(Path(directory) / "nested", first, [1000.0], [], [])
            before = paths[0].read_bytes()
            second_paths = perf.export_csv_results(Path(directory) / "nested", second, [], [snapshot()], [100.0])
            self.assertNotEqual(paths[0], second_paths[0])
            with self.assertRaises(FileExistsError):
                perf.export_csv_results(paths[0].parent, first, [2000.0], [], [])
            self.assertEqual(paths[0].read_bytes(), before)

    def test_csv_rejects_partial_or_invalid_profile_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must match"):
                perf.export_csv_results(directory, self.csv_metadata(), [], [snapshot()], [])
            invalid = snapshot()
            invalid["cores"][0]["sequence"] = 0
            with self.assertRaisesRegex(ValueError, "stale"):
                perf.export_csv_results(directory, self.csv_metadata(), [], [invalid], [100.0])


if __name__ == "__main__":
    unittest.main()
