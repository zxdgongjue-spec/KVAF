#!/usr/bin/env python3
"""Portable checks for the focused AIV bandwidth test."""

import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest

import aiv_batch_get_bw as bw


class BandwidthTest(unittest.TestCase):
    def test_exact_payload_without_miss_count_groups(self):
        for blocks in (128, 1024, 2048):
            args = bw.parse_args(["--peer", "host:28999", "--blocks", str(blocks)])
            self.assertEqual(args.blocks * args.block_bytes, blocks * 4096)
        self.assertEqual(bw.parse_args(["--peer", "host:28999"]).allocator, "torch")
        self.assertEqual(
            bw.parse_args(["--peer", "host:28999", "--allocator", "acl-normal"]).allocator,
            "acl-normal",
        )

    def test_invalid_configuration(self):
        for flag, value in (("--blocks", "0"), ("--block-bytes", "65537"),
                            ("--cores", "33"), ("--warmup", "-1"),
                            ("--iterations", "0"), ("--device-id", "-1")):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    bw.parse_args(["--peer", "host:28999", flag, value])

    def test_completed_payload_rate(self):
        result = bw.summarize(4 * 1024 * 1024, [100_000, 200_000])
        self.assertIn("GB/s=27.962", result)
        self.assertIn("avg_us=150.000", result)
        self.assertIn("min_us=100.000 max_us=200.000", result)
        for samples in ([], [0], [-1]):
            with self.assertRaises(ValueError):
                bw.summarize(4096, samples)

    def test_native_log_captures_both_fds_and_restores_after_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "native.log"
            original_stdout = os.fstat(1)
            with self.assertRaisesRegex(RuntimeError, "failure"):
                with bw.native_log(log):
                    os.write(1, b"native stdout\n")
                    os.write(2, b"native stderr\n")
                    raise RuntimeError("failure")
            self.assertEqual(os.fstat(1).st_ino, original_stdout.st_ino)
            self.assertEqual(log.read_text(), "native stdout\nnative stderr\n")


if __name__ == "__main__":
    unittest.main()
