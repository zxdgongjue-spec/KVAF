#!/usr/bin/env python3
"""Check that the Get trace summary keeps provider and execution paths separate."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("summarize_get_dispatch_trace.py")


class GetDispatchTraceTests(unittest.TestCase):
    def test_grouped_provider_phases_and_last_samples(self):
        fields = ("event", "index", "algorithm", "path", "total_ns", "provider_ns",
                  "provider_prepare_ns", "provider_submit_ns", "provider_completion_ns",
                  "provider_cleanup_ns")
        rows = [
            dict(event="submit", index=1, algorithm="gather", path="output",
                 total_ns=50000, provider_ns=40000, provider_prepare_ns=8000,
                 provider_submit_ns=9000, provider_completion_ns=20000, provider_cleanup_ns=1000),
            dict(event="submit", index=2, algorithm="gather", path="output",
                 total_ns=30000, provider_ns=20000, provider_prepare_ns=3000,
                 provider_submit_ns=4000, provider_completion_ns=10000, provider_cleanup_ns=1000),
            dict(event="submit", index=3, algorithm="scatter", path="ordinary",
                 total_ns=70000, provider_ns=60000, provider_prepare_ns=5000,
                 provider_submit_ns=10000, provider_completion_ns=40000, provider_cleanup_ns=0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.csv"
            with trace.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            result = subprocess.run([sys.executable, str(SCRIPT), str(trace), "--last", "1"],
                                    text=True, capture_output=True, check=True)
        self.assertIn("algorithm=gather path=output event=submit samples=1", result.stdout)
        self.assertIn("algorithm=scatter path=ordinary event=submit samples=1", result.stdout)
        self.assertIn("provider_prepare: p50=3.00", result.stdout)
        self.assertIn("provider_other: p50=2.00", result.stdout)
        self.assertNotIn("provider_prepare: p50=8.00", result.stdout)

    def test_legacy_provider_trace_uses_reduced_columns(self):
        fields = ("event", "index", "algorithm", "path", "total_ns", "provider_ns",
                  "provider_prepare_ns", "provider_submit_ns", "provider_completion_ns",
                  "provider_cleanup_ns")
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "legacy.csv"
            with trace.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(event="submit", index=1, algorithm="legacy", path="legacy",
                                     total_ns=40000, provider_ns=30000, provider_prepare_ns=5000,
                                     provider_submit_ns=5000, provider_completion_ns=18000,
                                     provider_cleanup_ns=1000))
            result = subprocess.run([sys.executable, str(SCRIPT), str(trace)],
                                    text=True, capture_output=True, check=True)
        self.assertIn("algorithm=legacy path=legacy event=submit samples=1", result.stdout)
        self.assertIn("provider_other: p50=1.00", result.stdout)
        self.assertIn("outside_provider: p50=10.00", result.stdout)

    def test_aiv_submit_breakdown_and_other_provider_omission(self):
        fields = ("event", "index", "algorithm", "path", "total_ns", "provider_ns",
                  "provider_prepare_ns", "provider_submit_ns", "provider_completion_ns",
                  "provider_cleanup_ns", "aiv_capture_prepare_ns", "aiv_capture_submit_ns",
                  "aiv_status_reset_ns", "aiv_kernel_launch_ns")
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "aiv.csv"
            with trace.open("w", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(event="submit", index=1, algorithm="aiv_direct", path="output",
                                     total_ns=50000, provider_ns=40000, provider_prepare_ns=3000,
                                     provider_submit_ns=30000, provider_completion_ns=6000,
                                     provider_cleanup_ns=0, aiv_capture_prepare_ns=1000,
                                     aiv_capture_submit_ns=5000, aiv_status_reset_ns=15000,
                                     aiv_kernel_launch_ns=8000))
                writer.writerow(dict(event="submit", index=2, algorithm="scatter", path="ordinary",
                                     total_ns=50000, provider_ns=40000, provider_prepare_ns=3000,
                                     provider_submit_ns=30000, provider_completion_ns=6000,
                                     provider_cleanup_ns=0, aiv_capture_prepare_ns=0,
                                     aiv_capture_submit_ns=0, aiv_status_reset_ns=0,
                                     aiv_kernel_launch_ns=0))
            result = subprocess.run([sys.executable, str(SCRIPT), str(trace)],
                                    text=True, capture_output=True, check=True)
        aiv, scatter = result.stdout.split("GET_DISPATCH_TIMING algorithm=scatter")
        self.assertIn("aiv_capture_prepare: p50=1.00", aiv)
        self.assertIn("aiv_capture_submit: p50=5.00", aiv)
        self.assertIn("aiv_status_reset: p50=15.00", aiv)
        self.assertIn("aiv_kernel_launch: p50=8.00", aiv)
        self.assertIn("aiv_submit_other: p50=2.00", aiv)
        self.assertNotIn("aiv_status_reset", scatter)


if __name__ == "__main__":
    unittest.main()
