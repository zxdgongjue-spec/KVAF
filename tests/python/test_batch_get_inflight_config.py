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

    def test_raw_address_mode(self):
        self.assertFalse(self.parse().raw_addresses)
        self.assertTrue(self.parse("--raw-addresses").raw_addresses)
        with patch.dict(os.environ, {"KVAF_BATCH_GET_RAW_ADDRESSES": "1"}, clear=True):
            self.assertTrue(perf.parse_args([]).raw_addresses)

    def test_operator_modes_and_environment_precedence(self):
        self.assertEqual(self.parse().batch_get_mode, "lookup")
        for mode in ("lookup", "no-pipeline", "raw"):
            with patch.dict(os.environ, {
                "KVAF_BATCH_GET_MODE": mode, "KVAF_BATCH_GET_RAW_ADDRESSES": "1",
            }, clear=True):
                args = perf.parse_args([])
                self.assertEqual(args.batch_get_mode, mode)
                self.assertEqual(args.raw_addresses, mode == "raw")
                self.assertEqual(perf.parse_args(["--batch-get-mode", "lookup"]).batch_get_mode,
                                 "lookup")
                self.assertEqual(perf.parse_args(["--raw-addresses"]).batch_get_mode, "raw")
        with contextlib.redirect_stderr(io.StringIO()):
            for options in (("--batch-get-mode", "bad"), ("--batch-get-mode",),
                            ("--batch-get-mode", "lookup", "--raw-addresses"),
                            ("--batch-get-mode", "no-pipeline", "--raw-addresses")):
                with self.assertRaises(SystemExit) as caught:
                    self.parse(*options)
                self.assertEqual(caught.exception.code, 2)
            with patch.dict(os.environ, {"KVAF_BATCH_GET_MODE": "bad"}, clear=True):
                with self.assertRaises(SystemExit) as caught:
                    perf.parse_args([])
                self.assertEqual(caught.exception.code, 2)

    def test_no_pipeline_kernel_selection(self):
        with patch.dict(os.environ, {
            "KVAF_BATCH_GET_KERNEL": "/build/bin/kvaf_execute_batch_get_kernels.o",
        }, clear=True):
            self.assertEqual(str(perf.batch_get_kernel_path("no-pipeline")),
                             "/build/bin/kvaf_execute_batch_get_no_pipeline_kernels.o")
            for mode in ("lookup", "raw"):
                self.assertEqual(str(perf.batch_get_kernel_path(mode)),
                                 os.environ["KVAF_BATCH_GET_KERNEL"])

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
        self.assertIn("peak_submitted_not_drained_mean=2048", output.getvalue())


if __name__ == "__main__":
    unittest.main()
