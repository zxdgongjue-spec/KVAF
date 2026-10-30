"""Validate benchmark shape and timeline handling without importing torch/ACL."""
import contextlib
import copy
import io
import os
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
    def test_default_config_and_supported_shapes(self):
        for cores in (1, 3, 8):
            for count in (1024, 8193, 65536):
                with patch.object(perf, "REQUEST_COUNT", count):
                    perf.validate_config(cores)

    def test_invalid_shapes_and_counts_fail_before_hardware(self):
        for field, value in (
            ("TOKEN_BYTES", 0), ("TOKEN_BYTES", 65536),
            ("REQUEST_COUNT", 65537), ("REQUEST_COUNT", True),
            ("WARMUP", -1), ("ITERATIONS", 0), ("PROFILE_SAMPLES", -1),
            ("TABLE_CAPACITY", 1 << 31), ("TOKENS_PER_BLOCK", 1 << 31),
            ("GATHER_SGL_SIZE", 7), ("SEED", -1),
        ):
            with self.subTest(field=field, value=value), patch.object(perf, field, value):
                with self.assertRaises(ValueError):
                    perf.validate_config(1)
        for cores in (0, 9):
            with self.assertRaises(ValueError):
                perf.validate_config(cores)

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


if __name__ == "__main__":
    unittest.main()
