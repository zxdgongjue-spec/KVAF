#!/usr/bin/env python3
"""Portable checks for per-stream configuration and multi-kernel timing."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


# The benchmark's parsing/reporting functions do not need native KVAF or
# PyTorch. Keep a fake module scoped to this import only.
spec = importlib.util.spec_from_file_location(
    "batch_get_perf", Path(__file__).with_name("test_npu_batch_get_perf.py")
)
perf = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"kvaf": types.ModuleType("kvaf")}):
    spec.loader.exec_module(perf)


class BatchGetStreamConfigTest(unittest.TestCase):
    def test_per_stream_core_count_and_total_limit(self):
        with patch.dict(os.environ, {}, clear=True):
            for streams, cores, valid in ((1, 48, True), (2, 1, True), (2, 3, True), (2, 24, True), (2, 25, False)):
                with self.subTest(streams=streams, cores=cores):
                    args = perf.parse_args(["--stream-count", str(streams), "--aiv-cores", str(cores)])
                    self.assertEqual(args.aiv_cores, cores)
                    self.assertEqual(perf.validate_args(args) is None, valid)

    def test_default_and_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(perf.parse_args([]).stream_count, 1)
            self.assertEqual(perf.parse_args([]).transport_count, 1)
        with patch.dict(os.environ, {"KVAF_NPU_BATCH_GET_STREAM_COUNT": "2"}, clear=True):
            self.assertEqual(perf.parse_args([]).stream_count, 2)
            self.assertEqual(perf.parse_args(["--stream-count", "1"]).stream_count, 1)
        with patch.dict(os.environ, {"KVAF_NPU_BATCH_GET_TRANSPORT_COUNT": "2"}, clear=True):
            self.assertEqual(perf.parse_args([]).transport_count, 2)
            self.assertEqual(perf.parse_args(["--transport-count", "1"]).transport_count, 1)

    def test_transport_topology_and_eid_capacity(self):
        with patch.dict(os.environ, {}, clear=True):
            for streams, transports, valid in ((1, 1, True), (2, 1, True), (2, 2, True), (1, 2, False)):
                with self.subTest(streams=streams, transports=transports):
                    args = perf.parse_args(["--stream-count", str(streams), "--transport-count", str(transports)])
                    self.assertEqual(perf.validate_args(args) is None, valid)
            args = perf.parse_args([
                "--stream-count", "2", "--transport-count", "2", "--aiv-cores", "1",
                "--jettys-per-core", "1", "--npu-eids", "1" * 32 + "," + "2" * 32,
            ])
            self.assertIsNotNone(perf.validate_args(args))
            args.jettys_per_core = 2
            self.assertIsNone(perf.validate_args(args))

    def test_two_streams_expand_worker_capacity(self):
        with patch.dict(os.environ, {}, clear=True):
            args = perf.parse_args([
                "--stream-count", "2", "--aiv-cores", "1", "--jettys-per-core", "1",
                "--npu-eids", "1" * 32 + "," + "2" * 32,
            ])
            self.assertIsNone(perf.validate_args(args))
            args.stream_count = 1
            self.assertIsNotNone(perf.validate_args(args))

    def test_serialized_kernels_do_not_report_double_bandwidth(self):
        # Two 1 ms kernels run sequentially; their full span is 2 ms. The
        # benchmark must not divide both payloads by max(1 ms, 1 ms).
        profiles = []
        for core, start in enumerate((0, 1_000_000)):
            profiles.append({
                "core": core,
                "kernel_start_cycle": start,
                "setup_done_cycle": start,
                "compact_done_cycle": start,
                "first_doorbell_cycle": start + 1,
                "post_done_cycle": start + 500_000,
                "completion_done_cycle": start + 1_000_000,
                "kernel_finish_cycle": start + 1_000_000,
                "active_requests": 1,
                "total_posts": 1,
                "jetty_completion_wait_cycles": [0] * 64,
                "jetty_completion_polls": [0] * 64,
            })
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            perf.print_kernel_profile([profiles], 0, 1000, 2, 1, 2, 2, 1024 * 1024, 2)
        lines = output.getvalue().splitlines()
        wall = next(line for line in lines if "operation=batch_get_kernel_wall " in line)
        total = next(line for line in lines if "operation=batch_get_kernel_total " in line)
        self.assertIn("mean_ms=2.000", wall)
        self.assertIn("useful_GiBps=0.977", wall)
        self.assertNotIn("useful_GiBps=", total)
        self.assertIn("stream=1 stream_core=0", output.getvalue())


if __name__ == "__main__":
    unittest.main()
