#!/usr/bin/env python3
"""Host-only checks for the inflight CLI and accumulated profile reporting."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
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

    def test_ubmem_detail_requires_explicit_profiling(self):
        args = self.parse("--transport", "ubmem", "--ubmem-profile-detail")
        self.assertIn("--kernel-profile", perf.validate_args(args))
        args = self.parse("--transport", "ubmem", "--kernel-profile", "--ubmem-profile-detail")
        self.assertIsNone(perf.validate_args(args))
        args = self.parse("--kernel-profile", "--ubmem-profile-detail")
        self.assertIn("--transport ubmem", perf.validate_args(args))

    def test_transport_is_explicit_and_defaults_to_urma(self):
        with patch.dict(os.environ, {"KVAF_TRANSPORT": "ubmem"}, clear=True):
            self.assertEqual(perf.parse_args([]).transport, "urma")
            self.assertEqual(perf.parse_args(["--transport", "ubmem"]).transport, "ubmem")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse("--transport", "invalid")

    def test_sweep_passes_transport_and_skips_unused_jettys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "runner"
            runner.write_text('''#!/usr/bin/env python3
import json, os, pathlib, sys
with pathlib.Path(os.environ["CALL_LOG"]).open("a") as output:
    output.write(json.dumps(sys.argv[1:]) + "\\n")
if "--adaptive-calibration-only" in sys.argv:
    print("[KVAF_AIV] UB_DEVICE_ADAPTIVE_SKIPPED reason=k_side_device_filter")
for stage in ("total", "post", "completion_wait"):
    print(f"PERF_RESULT operation=batch_get_kernel_{stage} detail=0 "
          "mean_ms=0.001 p50_ms=0.001 p90_ms=0.001 p99_ms=0.001 "
          "mean_cycles=1000 p50_cycles=1000 p90_cycles=1000 p99_cycles=1000")
''')
            runner.chmod(0o755)
            for number, (options, transport, calls) in enumerate((
                ([], "urma", 2),
                (["--transport", "ubmem"], "ubmem", 1),
                (["--transport=ubmem"], "ubmem", 1),
                (["--transport", "urma", "--adaptive-ub-device"], "urma", 3),
            )):
                with self.subTest(options=options):
                    log = root / f"calls_{number}.jsonl"
                    env = dict(os.environ, KVAF_TRANSPORT="ubmem", CALL_LOG=str(log),
                               KVAF_BATCH_GET_PERF_RUNNER=str(runner),
                               KVAF_BATCH_GET_TOKEN_BYTES="656", KVAF_BATCH_GET_MISS_COUNTS="4096",
                               KVAF_BATCH_GET_AIV_CORES="2", KVAF_BATCH_GET_JETTYS_PER_CORE="1 2",
                               KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR=str(root / f"results_{number}"),
                               KVAF_BATCH_GET_MISS_SWEEP_CSV=str(root / f"summary_{number}.csv"))
                    result = subprocess.run([
                        str(Path(__file__).resolve().parents[1] / "run_remote_cpu_batch_get_miss_count_perf.sh"),
                        str(root / "build"), *options,
                    ], env=env, capture_output=True, text=True, timeout=15)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    records = [json.loads(line) for line in log.read_text().splitlines()]
                    self.assertEqual(len(records), calls)
                    for arguments in records:
                        self.assertEqual(arguments[arguments.index("--transport") + 1], transport)

    def test_ubmem_stages_follow_slowest_total_core(self):
        def core(start, total, setup, address, submit, drain):
            return {
                "kernel_start_cycle": start, "kernel_finish_cycle": start + total,
                "ubmem_profile_detail": True, "ubmem_setup_cycles": setup,
                "ubmem_address_cycles": address, "ubmem_submit_cycles": submit,
                "ubmem_drain_cycles": drain,
            }

        # The faster core has more address work. Taking each stage's maximum
        # would incorrectly attribute its 150 cycles to the 300-cycle core.
        samples = [[core(10, 300, 10, 50, 150, 70), core(1000, 200, 10, 150, 20, 10)],
                   [core(20, 200, 10, 150, 20, 10), core(2000, 400, 20, 100, 180, 80)]]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            perf.print_kernel_profile(samples, 3, 1000, 2, 1, 17, 17, 656, "ubmem")
        records = {}
        for line in output.getvalue().splitlines():
            if line.startswith("PERF_RESULT "):
                fields = dict(field.split("=", 1) for field in line.split()[1:])
                records[fields["operation"]] = fields
        total = records.pop("batch_get_kernel_total")
        self.assertEqual(total["mean_cycles"], "350.000")
        self.assertEqual(records["batch_get_kernel_address_prepare"]["mean_cycles"], "75.000")
        self.assertEqual(records["batch_get_kernel_other"]["mean_cycles"], "20.000")
        self.assertEqual(sum(float(r["mean_cycles"]) for r in records.values()), 350)
        self.assertIn("per_row_timing_perturbs_pipeline=1", output.getvalue())

    def test_ubmem_total_only_remains_available(self):
        output = io.StringIO()
        samples = [[{"kernel_start_cycle": 10, "kernel_finish_cycle": 110,
                     "ubmem_profile_detail": False}]]
        with contextlib.redirect_stdout(output):
            perf.print_kernel_profile(samples, 0, 1000, 1, 1, 1, 0, 656, "ubmem")
        self.assertEqual(output.getvalue().count("PERF_RESULT "), 1)
        self.assertIn("detail=0", output.getvalue())
        self.assertIn("useful_GiBps=0.000", output.getvalue())


if __name__ == "__main__":
    unittest.main()
