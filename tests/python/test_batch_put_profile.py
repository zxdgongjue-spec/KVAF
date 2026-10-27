#!/usr/bin/env python3
"""Portable checks for BatchPut profiling units and aggregation."""

import contextlib
import copy
import io
import unittest

from batch_put_profile import (
    AGGREGATE_FIELDS, HOST_FIELDS, print_profile, profile_options, summarize_sample,
)


def snapshot():
    result = {field: 100 for field in HOST_FIELDS + AGGREGATE_FIELDS}
    result.update(row_count=3, payload_bytes=1000, metadata_bytes=72, status_bytes=1152)
    result["cores"] = [
        dict(core=0, active_requests=1, kernel_start_cycle=100, setup_done_cycle=110,
             first_doorbell_cycle=120, post_done_cycle=140,
             completion_done_cycle=180, kernel_finish_cycle=190),
        dict(core=1, active_requests=2, kernel_start_cycle=105, setup_done_cycle=115,
             first_doorbell_cycle=125, post_done_cycle=155,
             completion_done_cycle=200, kernel_finish_cycle=210),
        dict(core=2, active_requests=0, kernel_start_cycle=90, setup_done_cycle=95,
             first_doorbell_cycle=0, post_done_cycle=95,
             completion_done_cycle=95, kernel_finish_cycle=100),
    ]
    return result


class BatchPutProfileTests(unittest.TestCase):
    def test_default_off_and_explicit_counter_clock(self):
        self.assertEqual(profile_options({}), (False, None, None))
        self.assertEqual(profile_options({
            "KVAF_BATCH_PUT_PROFILE": "1", "KVAF_BATCH_PUT_PROFILE_CLOCK_MHZ": "50",
        }), (True, 50.0, "batch_put_profile.jsonl"))
        for clock in (None, "0", "-1", "nan", "inf", "wrong"):
            env = {"KVAF_BATCH_PUT_PROFILE": "1"}
            if clock is not None:
                env["KVAF_BATCH_PUT_PROFILE_CLOCK_MHZ"] = clock
            with self.subTest(clock=clock), self.assertRaises(ValueError):
                profile_options(env)

    def test_windows_exclude_idle_core_doorbells_and_do_not_sum_cores(self):
        result = summarize_sample(snapshot(), 2.0, 50)
        self.assertEqual(result["kernel_wall_cycles"], 120)
        self.assertEqual(result["communication_window_cycles"], 80)
        self.assertAlmostEqual(result["communication_window_ms"], .0016)
        self.assertAlmostEqual(result["communication_useful_GBps"], .625)
        self.assertAlmostEqual(result["e2e_useful_GBps"], .0005)
        self.assertAlmostEqual(result["kernel_post_max_core_ms"], .0008)
        self.assertAlmostEqual(result["kernel_cq_wait_max_core_ms"], .0009)
        a5 = summarize_sample(snapshot(), 2.0, 1000)
        self.assertAlmostEqual(a5["communication_window_ms"], .00008)
        self.assertAlmostEqual(a5["communication_useful_GBps"], 12.5)

    def test_invalid_or_stale_timelines_are_rejected(self):
        for field, value in (("post_done_cycle", 100), ("first_doorbell_cycle", 0),
                             ("completion_done_cycle", 130), ("kernel_start_cycle", 0)):
            profile = snapshot()
            profile["cores"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize_sample(profile, 2, 50)
        profile = snapshot()
        profile["row_count"] = 4
        with self.assertRaises(ValueError):
            summarize_sample(profile, 2, 50)

    def test_summary_uses_total_bytes_divided_by_total_time(self):
        first = snapshot()
        second = copy.deepcopy(first)
        second["payload_bytes"] = 2000
        # A longer communication window makes averaging per-call bandwidth wrong.
        for core in second["cores"]:
            if core["active_requests"]:
                core["completion_done_cycle"] += 160
                core["kernel_finish_cycle"] += 160
        records = [{"profile": p, "derived": summarize_sample(p, 2, 50)}
                   for p in (first, second)]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_profile(records, 20, 50)
        line = next(line for line in output.getvalue().splitlines()
                    if "operation=batch_put_communication_window " in line)
        self.assertIn("useful_GBps=0.468750", line)
        self.assertIn("scope=global_window", line)
        self.assertIn("scope=aggregate", output.getvalue())


if __name__ == "__main__":
    unittest.main()
