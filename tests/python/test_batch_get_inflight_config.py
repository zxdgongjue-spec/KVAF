#!/usr/bin/env python3
"""Host-only checks for the inflight CLI and accumulated profile reporting."""

import contextlib
import io
import os
import unittest
from unittest.mock import patch

import test_npu_batch_get_perf as perf


class InflightConfigTests(unittest.TestCase):
    def parse(self, *arguments):
        with patch.dict(os.environ, {}, clear=True):
            return perf.parse_args(list(arguments))

    def test_default_baseline_and_minimum(self):
        self.assertEqual(self.parse().max_inflight_total, 2048)
        for value in (0, 8, 2048, 2049, 0xFFFFFFFF):
            args = self.parse("--aiv-cores", "4", "--max-inflight-total", str(value))
            self.assertIsNone(perf.validate_args(args))
        for value in (-1, 1, 7, 0x100000000):
            args = self.parse("--aiv-cores", "4", "--max-inflight-total", str(value))
            self.assertIn("max-inflight-total", perf.validate_args(args))

    def test_environment_and_cli_override(self):
        with patch.dict(os.environ, {"KVAF_BATCH_GET_MAX_INFLIGHT_TOTAL": "512"}, clear=True):
            self.assertEqual(perf.parse_args([]).max_inflight_total, 512)
            self.assertEqual(perf.parse_args(["--max-inflight-total", "0"]).max_inflight_total, 0)

    def test_detailed_jetty_profile_requires_profile(self):
        self.assertFalse(self.parse().kernel_profile_jetty_timeline)
        self.assertIn("requires --kernel-profile", perf.validate_args(
            self.parse("--kernel-profile-jetty-timeline")))
        self.assertIsNone(perf.validate_args(
            self.parse("--kernel-profile", "--kernel-profile-jetty-timeline")))

    def test_jetty_timeline_uses_common_origin_and_actual_publication_counts(self):
        profile = dict(
            core=0, kernel_start_cycle=1000, setup_done_cycle=1100,
            compact_done_cycle=1100, first_doorbell_cycle=1200,
            post_done_cycle=1300, completion_done_cycle=2100, kernel_finish_cycle=2200,
            active_requests=16, total_posts=2, submitted_group_count=1,
            completed_group_count=1, completion_count=2, post_active_cycles=200,
            drain_wait_cycles=800, peak_submitted_not_drained=16,
            jetty_completion_wait_cycles=[900, 1] + [0] * 62,
            jetty_completion_polls=[5, 5] + [0] * 62,
            small_post_max_requests=256, selected_post_width=8,
            small_post_group_count=1, fused_group_count=1, wqe_build_calls=1,
            jetty_timeline_enabled=True, profile_version=3,
            schedule_metadata_version=1, lean_schedule_enabled=True,
            configured_jettys=3, managed_jettys=2, single_group_fast_path=True,
            min_tail_reads=32, tail_jetty_reused=True,
        )
        trace = dict(submitted_reads=8, submitted_posts=1, completed_checkpoints=1,
                     first_doorbell_cycle=1200, last_doorbell_cycle=1200,
                     first_cqe_seen_cycle=2000, last_cqe_seen_cycle=2000,
                     probe_active_cycles=5)
        idle = {key: 0 for key in trace}
        profile["jettys"] = [trace, dict(trace, first_doorbell_cycle=1210,
                                       last_doorbell_cycle=1210,
                                       first_cqe_seen_cycle=2001,
                                       last_cqe_seen_cycle=2001)] + [idle] * 62
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            perf.print_kernel_profile([[profile]], 0, 1000, 1, 3, 16, 16, 656)
        text = output.getvalue()
        self.assertIn("completion_poll_mode=round_robin", text)
        self.assertIn("fused_groups_per_call=1.000000 wqe_build_calls_per_call=1.000000", text)
        self.assertIn("lean_schedule_enabled=1 configured_jettys=3", text)
        self.assertIn("active_jettys_per_call=2.000000 managed_jettys_min=2", text)
        self.assertIn("managed_jettys_per_call=2.000000 single_group_fast_path_samples=1", text)
        self.assertIn("min_tail_reads=32 tail_jetty_reuse_samples=1", text)
        self.assertNotIn("jetty0_wait_ms_mean", text)
        records = [dict(word.split("=", 1) for word in line.split()[1:])
                   for line in text.splitlines() if line.startswith("KERNEL_PROFILE_JETTY ")]
        self.assertEqual(records[0]["reads_per_call"], "8.000000")
        self.assertEqual(records[0]["last_db_to_last_cqe_us_mean"], "0.800000")
        self.assertEqual(records[1]["last_db_to_last_cqe_us_mean"], "0.791000")
        self.assertEqual(records[2]["active_samples"], "0")
        self.assertEqual(records[2]["first_doorbell_offset_us_mean"], "n/a")

    def test_profile_excludes_intermediate_waits_from_post(self):
        profile = {
            "core": 0, "kernel_start_cycle": 10, "setup_done_cycle": 20,
            "compact_done_cycle": 20, "first_doorbell_cycle": 30,
            "post_done_cycle": 200, "completion_done_cycle": 220,
            "kernel_finish_cycle": 230, "active_requests": 2049,
            "total_posts": 17, "submitted_group_count": 3,
            "completed_group_count": 3, "completion_count": 3,
            "post_active_cycles": 80, "drain_wait_cycles": 100,
            "peak_submitted_not_drained": 2048,
            "jetty_completion_wait_cycles": [100] + [0] * 63,
            "jetty_completion_polls": [3] + [0] * 63,
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            perf.print_kernel_profile([[profile]], 0, 1000, 1, 1, 2049, 2049, 656)
        records = {}
        for line in output.getvalue().splitlines():
            if line.startswith("PERF_RESULT "):
                fields = dict(field.split("=", 1) for field in line.split()[1:])
                records[fields["operation"]] = fields
        self.assertEqual(records["batch_get_kernel_post"]["mean_cycles"], "80.000")
        self.assertEqual(records["batch_get_kernel_completion_wait"]["mean_cycles"], "100.000")
        self.assertEqual(records["batch_get_kernel_first_doorbell"]["mean_cycles"], "20.000")
        self.assertEqual(records["batch_get_kernel_transfer_window"]["mean_cycles"], "190.000")
        self.assertIn("peak_submitted_not_drained_mean=2048", output.getvalue())

        profile.update(first_doorbell_cycle=0, active_requests=0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            perf.print_kernel_profile([[profile]], 0, 1000, 1, 1, 2049, 0, 656)
        records = {}
        for line in output.getvalue().splitlines():
            if line.startswith("PERF_RESULT "):
                fields = dict(field.split("=", 1) for field in line.split()[1:])
                records[fields["operation"]] = fields
        self.assertEqual(records["batch_get_kernel_first_doorbell"]["mean_cycles"], "0.000")
        self.assertEqual(records["batch_get_kernel_transfer_window"]["mean_cycles"], "0.000")

    def test_independent_fast_path_log_flags_and_counts(self):
        trace = dict(submitted_reads=16, submitted_posts=1, completed_checkpoints=1,
                     first_doorbell_cycle=1200, last_doorbell_cycle=1200,
                     first_cqe_seen_cycle=2000, last_cqe_seen_cycle=2000, probe_active_cycles=0)
        profile = dict(core=0, kernel_start_cycle=1000, setup_done_cycle=1100,
                       compact_done_cycle=1100, first_doorbell_cycle=1200, post_done_cycle=1300,
                       completion_done_cycle=2100, kernel_finish_cycle=2200, active_requests=16,
                       total_posts=1, submitted_group_count=1, completed_group_count=1, completion_count=1,
                       post_active_cycles=200, drain_wait_cycles=800, peak_submitted_not_drained=16,
                       jetty_completion_wait_cycles=[900] + [0] * 63, jetty_completion_polls=[5] + [0] * 63,
                       small_post_max_requests=0, selected_post_width=128, small_post_group_count=0,
                       fused_group_count=0, wqe_build_calls=1, jetty_timeline_enabled=False, profile_version=3,
                       schedule_metadata_version=2, lean_schedule_enabled=True, configured_jettys=1,
                       managed_jettys=1, single_group_fast_path=False, min_tail_reads=0, tail_jetty_reused=False,
                       single_group_fast_path_enabled=False, single_post_fast_path_enabled=True,
                       single_post_fast_path_count=1, jettys=[trace])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            perf.print_kernel_profile([[profile]], 0, 1000, 1, 1, 16, 16, 656)
        self.assertIn("schedule_metadata_version=2", output.getvalue())
        self.assertIn("single_group_fast_path_samples=0", output.getvalue())
        self.assertIn("single_group_fast_path_enabled=0 single_post_fast_path_enabled=1 "
                      "single_post_fast_path_calls_per_call=1.000000", output.getvalue())

    def test_host_timing_keeps_sub_microsecond_precision(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            perf.print_samples("batch_get", [0.007001, 0.007123, 0.007245], 0, 32, 32, 656)
        self.assertIn("p50_us=7.123000", output.getvalue())


if __name__ == "__main__":
    unittest.main()
