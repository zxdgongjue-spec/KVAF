"""Validate benchmark shape and timeline handling without importing torch/ACL."""
import contextlib
import copy
import csv
from datetime import datetime, timezone
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
            assert not any(name.endswith("_ns") for name in reader.fieldnames), reader.fieldnames
            return list(reader)

    def test_default_config_and_supported_shapes(self):
        for cores in (1, 3, 8):
            for count in (1024, 8192, 8193, 65536):
                for sgl in range(1, 14):
                    with patch.object(perf, "REQUEST_COUNT", count), \
                         patch.object(perf, "GATHER_SGL_SIZE", sgl):
                        perf.validate_config(cores)

    def test_invalid_shapes_and_counts_fail_before_hardware(self):
        for field, value in (
            ("TOKEN_BYTES", 0), ("TOKEN_BYTES", 65536),
            ("REQUEST_COUNT", 65537), ("REQUEST_COUNT", True),
            ("WARMUP", -1), ("ITERATIONS", -1), ("PROFILE_SAMPLES", -1),
            ("TABLE_CAPACITY", 1 << 31), ("TOKENS_PER_BLOCK", 1 << 31),
            ("GATHER_SGL_SIZE", 0), ("GATHER_SGL_SIZE", 14), ("SEED", -1),
        ):
            with self.subTest(field=field, value=value), patch.object(perf, field, value):
                with self.assertRaises(ValueError):
                    perf.validate_config(1)
        for cores in (0, 9):
            with self.assertRaises(ValueError):
                perf.validate_config(cores)

    def test_output_layout_validation_and_random_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.csv_metadata()["output_layout"], "random")
            self.assertEqual(self.csv_metadata()["coalesce_sges"], 0)
        for layout in ("random", "segmented", "contiguous"):
            with patch.dict(os.environ, {"KVAF_GATHER_OUTPUT_LAYOUT": layout}):
                perf.validate_config(2)
                self.assertEqual(self.csv_metadata()["output_layout"], layout)
        with patch.dict(os.environ, {"KVAF_GATHER_OUTPUT_LAYOUT": "unknown"}):
            with self.assertRaisesRegex(ValueError, "KVAF_GATHER_OUTPUT_LAYOUT"):
                perf.validate_config(2)
        with patch.dict(os.environ, {"KVAF_GATHER_OUTPUT_LAYOUT": "segmented"}), \
             patch.object(perf, "TOKEN_BYTES", 16384):
            with self.assertRaisesRegex(ValueError, "65536"):
                perf.validate_config(2)

    def test_coalescing_requires_consecutive_rows_and_preserves_logical_sgl(self):
        for layout in ("segmented", "contiguous"):
            with patch.dict(os.environ, {"KVAF_GATHER_OUTPUT_LAYOUT": layout,
                                         "KVAF_AIV_GATHER_COALESCE_SGES": "1"}):
                perf.validate_config(2)
                metadata = self.csv_metadata()
                self.assertEqual(metadata["coalesce_sges"], 1)
                self.assertEqual(metadata["sgl"], perf.GATHER_SGL_SIZE)
        for value in ("2", "true"):
            with patch.dict(os.environ, {"KVAF_AIV_GATHER_COALESCE_SGES": value}):
                with self.assertRaisesRegex(ValueError, "COALESCE_SGES"):
                    perf.validate_config(2)
        with patch.dict(os.environ, {"KVAF_AIV_GATHER_COALESCE_SGES": "1",
                                     "KVAF_GATHER_OUTPUT_LAYOUT": "random"}):
            with self.assertRaisesRegex(ValueError, "segmented/contiguous"):
                perf.validate_config(2)
        with patch.dict(os.environ, {"KVAF_AIV_GATHER_COALESCE_SGES": "1",
                                     "KVAF_GATHER_OUTPUT_LAYOUT": "contiguous"}), \
             patch.object(perf, "TOKEN_BYTES", 16384):
            with self.assertRaisesRegex(ValueError, "65536"):
                perf.validate_config(2)

    def test_coalesced_preflight_detects_ignored_flag_and_changed_read_granularity(self):
        for wrs, blocks in ((683, 683), (683, 2048), (4096, 4096)):
            profile = {"cores": [{"requests": 4096, "data_wrs": wrs, "data_wqebbs": blocks},
                                 {"requests": 0, "data_wrs": 0, "data_wqebbs": 0}]}
            if blocks == 683:
                perf.check_coalesced_profile(profile)
            else:
                with self.assertRaisesRegex(RuntimeError, "rebuild and deploy"):
                    perf.check_coalesced_profile(profile)

    def test_output_rows_preserve_random_mapping_and_contiguous_guards(self):
        calls = []
        def permutation(count):
            calls.append(count)
            return [2, 0, 3, 1]
        self.assertEqual(perf.make_output_rows(4, 2, 6, "random", permutation), [3, 1, 4, 2])
        self.assertEqual(calls, [4])
        self.assertEqual(perf.make_output_rows(4, 2, 6, "contiguous", permutation), [1, 2, 3, 4])
        self.assertEqual(calls, [4])  # Contiguous output consumes no random numbers.

    def test_segmented_rows_follow_per_core_wqe_boundaries_and_keep_a_permutation(self):
        for requests in (1, 7, 19, 8192, 8193):
            for cores in (1, 2, 3, 8):
                for sgl in (1, 6, 13):
                    rows = perf.make_output_rows(requests, cores, sgl, "segmented", lambda n: reversed(range(n)))
                    self.assertEqual(sorted(rows), list(range(1, requests + 1)))
                    for core in range(cores):
                        first, last = requests * core // cores, requests * (core + 1) // cores
                        for begin in range(first, last, sgl):
                            segment = rows[begin:min(begin + sgl, last)]
                            self.assertEqual(segment, list(range(segment[0], segment[0] + len(segment))))
                        tail = first + (last - first) // sgl * sgl
                        self.assertEqual(rows[tail:last], list(range(tail + 1, last + 1)))
        rows = perf.make_output_rows(8192, 2, 6, "segmented", lambda n: reversed(range(n)))
        self.assertGreater(rows[0], 4096)  # Full segments also shuffle across lanes.
        self.assertNotEqual(rows[6], rows[5] + 1)

    def test_output_layout_is_recorded_in_all_csv_files(self):
        with patch.dict(os.environ, {"KVAF_GATHER_OUTPUT_LAYOUT": "segmented",
                                     "KVAF_AIV_GATHER_COALESCE_SGES": "1"}):
            metadata = self.csv_metadata()
        with tempfile.TemporaryDirectory() as directory:
            paths = perf.export_csv_results(directory, metadata, [1000.0], [snapshot()], [1100.0])
            for path in paths:
                self.assertTrue(all(row["output_layout"] == "segmented" for row in self.read_csv(path)))
                self.assertTrue(all(row["coalesce_sges"] == "1" for row in self.read_csv(path)))

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

    def test_chunk_configuration_matches_host_uint32_contract(self):
        for value, expected in (("", 131072), ("0", 0), ("64", 64), ("131072", 131072)):
            with patch.dict(os.environ, {"KVAF_AIV_GATHER_CHUNK_BYTES": value}):
                perf.validate_config(1)
                self.assertEqual(perf.configured_chunk_bytes(), expected)
        for value in ("-64", "+64", "63", "65", "64K", "4294967296", " 64", "６４"):
            with self.subTest(value=value), patch.dict(os.environ, {"KVAF_AIV_GATHER_CHUNK_BYTES": value}):
                with self.assertRaisesRegex(ValueError, "CHUNK_BYTES"):
                    perf.validate_config(1)

    def test_chunk_profile_accepts_ack_after_read_and_exports_overlap(self):
        sample = snapshot(1, cores=1, requests=1)
        lane = sample["cores"][0]
        lane.update(pipeline_chunk_bytes=64, first_ready_cycle=4, first_read_submit_cycle=5,
                    read_done_cycle=9, ack_done_cycle=10, kernel_finish_cycle=11,
                    ready_queries=3, ready_wait_cycles=2, first_read_ready_bytes=656)
        _, lanes = perf.summarize_profiles([sample], 1, 1)
        self.assertEqual(lanes[0]["wait_cpu"], 7)
        self.assertEqual(lanes[0]["read"], 4)
        self.assertEqual(lanes[0]["finalize"], 1)
        self.assertEqual(lanes[0]["total"], 11)
        metadata = {**self.csv_metadata(clock=50), "cores": 1}
        with tempfile.TemporaryDirectory() as directory:
            paths = perf.export_csv_results(directory, metadata, [], [sample], [100.0])
            row = self.read_csv(paths[3])[0]
            self.assertEqual(int(row["first_read_submit_cycle"]), 5)
            self.assertEqual(int(row["first_read_ready_bytes"]), 656)
            self.assertEqual(float(row["read_cycles"]), 4)
            self.assertEqual(float(row["ready_wait_us"]), 2 / 50)
        for key, value in (("first_ready_cycle", 2), ("first_read_submit_cycle", 10),
                           ("ack_done_cycle", 12), ("first_read_ready_bytes", 657)):
            bad = copy.deepcopy(sample)
            bad["cores"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                perf.summarize_profiles([bad], 1, 1)

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
            self.assertEqual(results["e2e_summary"]["timing_contract"], "async_call_to_stream_completion")
            self.assertTrue(all(row["timing_contract"] == "synchronous_call_with_status_check"
                                for row in host + cores))
            # Host wait includes current-batch device/D2H completion. It has no
            # separately measured READ stage, so never fill AIV cells in it.
            self.assertEqual(results["profile_host_mean"]["read_cycles"], "")
            self.assertEqual(results["profile_host_mean"]["read_us"], "")
            core_means = [row for row in summary if row["row_kind"] == "profile_core_mean"]
            self.assertEqual(len(core_means), 3)
            self.assertEqual([float(row["total_cycles"]) for row in core_means], [6, 12, 18])
            self.assertTrue(all(row["prepare_us"] == "" and row["wait_us"] == "" for row in core_means))
            self.assertTrue(all(row["run_id"] == metadata["run_id"] for row in summary + e2e + host + cores))

    def test_async_baseline_rejects_forced_status_readback(self):
        with patch.dict(os.environ, {"KVAF_RUN_NPU_TESTS": "1", "KVAF_AIV_GATHER_CHECK_STATUS": "1"}, clear=True), \
             contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(perf.main(), 2)
        self.assertIn("unset KVAF_AIV_GATHER_CHECK_STATUS", output.getvalue())

    def test_csv_summary_bandwidth_uses_p50_despite_slow_outliers(self):
        metadata = self.csv_metadata()
        baseline = [1000.0, 1000.0, 100000.0]
        profiled = [2000.0, 2000.0, 200000.0]
        samples = [snapshot(sequence) for sequence in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as directory:
            paths = perf.export_csv_results(directory, metadata, baseline, samples, profiled)
            results = {row["row_kind"]: row for row in self.read_csv(paths[0])}
            for kind, mean_us, p50_us in (
                ("e2e_summary", 34000.0, 1000.0),
                ("profile_e2e_summary", 68000.0, 2000.0),
            ):
                with self.subTest(row_kind=kind):
                    self.assertEqual(float(results[kind]["mean_us"]), mean_us)
                    self.assertEqual(float(results[kind]["p50_us"]), p50_us)
                    self.assertAlmostEqual(
                        float(results[kind]["useful_GiBps"]),
                        metadata["payload_bytes"] / (p50_us / 1e6) / 1024**3,
                    )
            # Raw sample bandwidth still describes that sample's own latency.
            raw = self.read_csv(paths[1])
            self.assertAlmostEqual(float(raw[-1]["useful_GiBps"]),
                                   metadata["payload_bytes"] / (baseline[-1] / 1e6) / 1024**3)

    def test_csv_clock_conversion_uses_captured_configuration(self):
        metadata = self.csv_metadata(clock=50)
        sample = snapshot()
        sample["cores"][0]["read_build_cycles"] = 1
        with tempfile.TemporaryDirectory() as directory:
            paths = perf.export_csv_results(directory, metadata, [], [sample], [100.0])
            cores = self.read_csv(paths[3])
            self.assertAlmostEqual(float(cores[0]["total_us"]), 6 / 50)
            self.assertAlmostEqual(float(cores[0]["read_build_us"]), 1 / 50)
            host = self.read_csv(paths[2])
            self.assertAlmostEqual(float(host[0]["prepare_us"]), 0.1)
            self.assertAlmostEqual(float(host[0]["total_us"]), 0.1)
            summary = self.read_csv(paths[0])
            core_mean = next(row for row in summary if row["row_kind"] == "profile_core_mean" and row["core"] == "0")
            self.assertAlmostEqual(float(core_mean["read_us"]), 1 / 50)
            self.assertEqual(self.read_csv(paths[1]), [])

    def test_csv_metadata_and_error_columns_follow_requested_order(self):
        metadata = self.csv_metadata()
        with tempfile.TemporaryDirectory() as directory:
            paths = perf.export_csv_results(directory, metadata, [1000.0], [snapshot()], [1100.0])
            moved = ["clock_mhz", "clock_source", "cpu_profile", "kernel_path", "put_kernel_path",
                     "request_plan", "timing_contract"]
            for path in paths:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    columns = next(csv.reader(stream))
                mode = columns.index("mode")
                self.assertEqual(columns[mode - len(moved):mode], moved)
                self.assertLess(mode, columns.index("cores"))
                if path.name == "summary.csv":
                    total = columns.index("total_us")
                    times = [name + "_us" for name, _, _ in perf.STAGE_BOUNDARIES if name != "total"]
                    cycles = [name + "_cycles" for name, _, _ in perf.STAGE_BOUNDARIES]
                    self.assertEqual(columns[total + 1:total + 1 + len(times) + len(cycles)], times + cycles)
                    error = columns.index("error_type")
                    self.assertEqual(columns[error - 2:error], ["sampling_status", "core"])
                    self.assertEqual(error, total + 1 + len(times) + len(cycles) + 2)
                    self.assertEqual(columns[error + 1], "error_message")
                    self.assertLess(columns.index("profile"), total)

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
        second["run_directory"] = first["run_directory"]  # Same second/config.
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.assertEqual(first["git_dirty"], "unknown")
        with tempfile.TemporaryDirectory() as directory:
            paths = perf.export_csv_results(Path(directory) / "nested", first, [1000.0], [], [])
            self.assertEqual(len({path.parent for path in paths}), 1)
            self.assertEqual({path.name for path in paths}, {"summary.csv", "e2e.csv", "profile_host.csv", "profile_core.csv"})
            before = paths[0].read_bytes()
            second_paths = perf.export_csv_results(Path(directory) / "nested", second, [], [snapshot()], [100.0])
            self.assertNotEqual(paths[0], second_paths[0])
            self.assertEqual(second_paths[0].parent.name, first["run_directory"] + "_2")
            self.assertEqual(paths[0].read_bytes(), before)
            self.assertEqual(self.read_csv(second_paths[0])[0]["run_directory"], second_paths[0].parent.name)

    def test_csv_directory_name_contains_beijing_time_and_shape(self):
        with patch.object(perf, "datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 31, 1, 32, 35, tzinfo=timezone.utc)
            metadata = self.csv_metadata()
        self.assertEqual(metadata["run_directory"], "20261031_093235-cores3-token-bytes656-request-count1")
        self.assertEqual(metadata["started_at_local"], "2026-10-31T09:32:35+08:00")
        self.assertEqual(metadata["timezone"], "Asia/Shanghai")
        self.assertEqual(metadata["schema_version"], 4)

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
