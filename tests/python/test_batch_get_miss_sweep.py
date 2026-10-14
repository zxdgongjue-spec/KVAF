#!/usr/bin/env python3
"""Run both production shell wrappers with synthetic kernel profile samples.

No device timing is claimed here: the fake Python process exercises real CLI
parsing/profile formatting, so forwarding, cycle conversion and CSV aggregation
can be checked without CANN or an NPU.
"""

import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]


class MissSweepTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="kvaf_miss_sweep_test_")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        build = self.directory / "build"
        (build / "bin").mkdir(parents=True)
        (build / "python" / "kvaf").mkdir(parents=True)
        for name in ("get", "get_no_pipeline", "put"):
            (build / "bin" / f"kvaf_execute_batch_{name}_kernels.o").write_bytes(b"stub")
        self.build = build
        # macOS need not have GNU timeout. This stub only forwards its argv;
        # subprocess.run below bounds the whole host-only test.
        timeout = self.directory / "timeout"
        timeout.write_text('#!/bin/sh\nshift\nexec "$@"\n')
        timeout.chmod(0o755)
        runner = self.directory / "fake_python"
        runner.write_text(f"#!{sys.executable}\n" + textwrap.dedent(f"""\
            import os
            import sys
            sys.path.insert(0, {str(ROOT / 'tests' / 'python')!r})
            import test_npu_batch_get_perf as perf

            assert sys.argv[1].endswith("test_npu_batch_get_perf.py")
            args = perf.parse_args(sys.argv[2:])
            assert perf.validate_args(args) is None
            assert args.kernel_profile
            mode = "raw" if args.raw_addresses else "lookup"
            assert mode == os.environ["EXPECTED_ADDRESS_MODE"]
            assert args.batch_get_mode == os.environ["EXPECTED_BATCH_GET_MODE"]
            filename = ("kvaf_execute_batch_get_no_pipeline_kernels.o"
                        if args.batch_get_mode == "no-pipeline"
                        else "kvaf_execute_batch_get_kernels.o")
            assert os.path.basename(os.environ["KVAF_BATCH_GET_KERNEL"]) == filename
            assert perf.batch_get_kernel_path(args.batch_get_mode).name == filename
            assert args.peer_service_addr == "test-peer:28999"
            assert args.max_inflight_total == 8
            print(f"BATCH_GET_PERF_CONFIG address_mode={{mode}} batch_get_mode={{args.batch_get_mode}}")
            # An unrelated host timing must never be used by the sweep.
            print("PERF_RESULT operation=batch_get mean_ms=200 p50_ms=200")
            sample = []
            for core in range(args.aiv_cores):
                start = 1000 + core * 2000
                finish = start + (core + 1) * 4000
                sample.append({{
                    "core": core, "kernel_start_cycle": start,
                    "setup_done_cycle": start + 100,
                    "compact_done_cycle": start + 100,
                    "first_doorbell_cycle": start + 200,
                    "post_done_cycle": finish - 1000,
                    "completion_done_cycle": finish - 100,
                    "kernel_finish_cycle": finish,
                    "active_requests": args.miss_count // args.aiv_cores,
                    "total_posts": 1, "submitted_group_count": 1,
                    "completed_group_count": 1, "completion_count": 1,
                    "post_active_cycles": 1000 + core * 500,
                    "drain_wait_cycles": 2000 + core * 500,
                    "peak_submitted_not_drained": 4,
                    "post_stage_profile_enabled": int(args.batch_get_mode == "no-pipeline"),
                    "wqe_fill_cycles": (200 + core * 100) if args.miss_count else 0,
                    "mte_copy_cycles": (400 - core * 100) if args.miss_count else 0,
                    "doorbell_cycles": 50 if args.miss_count else 0,
                    "jetty_completion_wait_cycles": [0] * 64,
                    "jetty_completion_polls": [0] * 64,
                }})
            if os.environ.get("OMIT_POST_STAGE_PROFILE") == "1":
                for profile in sample:
                    profile.pop("post_stage_profile_enabled")
            perf.print_kernel_profile([sample], args.warmup,
                args.kernel_profile_clock_mhz, args.aiv_cores, args.jettys_per_core,
                args.request_length, args.miss_count, args.token_bytes,
                address_mode=mode, batch_get_mode=args.batch_get_mode)
            """))
        runner.chmod(0o755)
        self.environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith("KVAF_")
        }
        self.environment.update({
            "PATH": f"{self.directory}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "PYTHON": str(runner),
            "KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR": str(self.directory / "logs"),
            "KVAF_BATCH_GET_SWEEP_TOKEN_BYTES": "128",
            "KVAF_BATCH_GET_SWEEP_MISS_COUNTS": "0 64",
            "KVAF_BATCH_GET_SWEEP_AIV_CORES": "2",
            "KVAF_BATCH_GET_SWEEP_JETTYS_PER_CORE": "2",
            "KVAF_BATCH_GET_KERNEL_CLOCK_MHZ": "500",
            "EXPECTED_ADDRESS_MODE": "lookup",
            "EXPECTED_BATCH_GET_MODE": "lookup",
        })

    def run_sweep(self, *options):
        return subprocess.run(
            ["bash", str(ROOT / "tests/run_remote_cpu_batch_get_miss_count_perf.sh"),
             str(self.build), "--request-length", "64", "--table-capacity", "64",
             "--tokens-per-block", "1", "--peer-service-addr", "test-peer:28999",
             "--max-inflight-total", "8", *options],
            env=self.environment, text=True, capture_output=True, timeout=30,
        )

    def test_modes_timing_bandwidth_and_csv(self):
        for batch_get_mode, options, raw_env, mode_env in (
            ("lookup", (), "0", ""),
            ("raw", ("--raw-addresses",), "0", ""),
            ("raw", (), "1", ""),
            ("lookup", ("--batch-get-mode", "lookup"), "1", ""),
            ("no-pipeline", ("--batch-get-mode", "no-pipeline"), "0", ""),
            ("no-pipeline", ("--batch-get-mode=no-pipeline",), "1", "raw"),
            ("raw", ("--batch-get-mode", "raw"), "0", "no-pipeline"),
            ("no-pipeline", (), "1", "no-pipeline"),
            ("lookup", (), "1", "lookup"),
            ("raw", ("--raw-addresses",), "0", "no-pipeline"),
        ):
            mode = "raw" if batch_get_mode == "raw" else "lookup"
            with self.subTest(mode=batch_get_mode, options=options):
                self.environment["EXPECTED_ADDRESS_MODE"] = mode
                self.environment["EXPECTED_BATCH_GET_MODE"] = batch_get_mode
                self.environment["KVAF_BATCH_GET_MODE"] = mode_env
                self.environment["KVAF_BATCH_GET_RAW_ADDRESSES"] = raw_env
                result = self.run_sweep(*options)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(f"address_mode={mode}", result.stdout)
                self.assertIn(f"batch_get_mode={batch_get_mode}", result.stdout)
                self.assertIn("combinations=2", result.stdout)
                self.assertIn("stage=kernel_post", result.stdout)
                self.assertIn("stage=kernel_completion_wait", result.stdout)
                summary = self.directory / "logs/batch_get_miss_sweep_summary.csv"
                with summary.open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 2)
                for row, count in zip(rows, (0, 64)):
                    self.assertEqual(row["address_mode"], mode)
                    self.assertEqual(row["batch_get_mode"], batch_get_mode)
                    for stage, expected in (("wqe_fill", 0.6), ("mte_copy", 0.8), ("doorbell", 0.1)):
                        for statistic in ("mean", "p50", "p90", "p99"):
                            value = row[f"{stage}_{statistic}_us"]
                            if batch_get_mode == "no-pipeline":
                                self.assertEqual(float(value), expected if count else 0)
                            else:
                                self.assertEqual(value, "")
                        self.assertEqual(f"stage=kernel_{stage}" in result.stdout,
                                         batch_get_mode == "no-pipeline")
                    self.assertEqual(int(row["miss_count"]), count)
                    self.assertEqual(float(row["mean_us"]), 16)
                    self.assertEqual(float(row["p50_us"]), 16)
                    self.assertEqual(float(row["post_mean_us"]), 3)
                    self.assertEqual(float(row["completion_wait_mean_us"]), 5)
                    self.assertAlmostEqual(float(row["p50_GBps"]), count * 128 / 16000)
                totals = [line for line in result.stdout.splitlines()
                          if line.startswith("PERF_RESULT operation=batch_get_kernel_total ")]
                self.assertEqual(len(totals), 2)
                for total, count in zip(totals, (0, 64)):
                    fields = dict(field.split("=", 1) for field in total.split()[1:])
                    self.assertEqual(fields["mean_us"], "16.000")
                    self.assertEqual(fields["address_mode"], mode)
                    self.assertEqual(fields["batch_get_mode"], batch_get_mode)
                    expected_gibps = count * 128 / (16 / 1_000_000) / (1024 ** 3)
                    self.assertAlmostEqual(float(fields["useful_GiBps"]), expected_gibps, places=3)

    def test_old_profile_fails_without_zero_stage_fallback(self):
        self.environment["EXPECTED_BATCH_GET_MODE"] = "no-pipeline"
        self.environment["OMIT_POST_STAGE_PROFILE"] = "1"
        result = self.run_sweep("--batch-get-mode", "no-pipeline")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No-pipeline stage profiling is unavailable", result.stdout)
        self.assertFalse((self.directory / "logs/batch_get_miss_sweep_summary.csv").exists())

    def test_invalid_modes_fail_before_launch(self):
        for options in (
            ("--batch-get-mode", "unknown"), ("--batch-get-mode",),
            ("--batch-get-mode=",),
            ("--batch-get-mode", "lookup", "--raw-addresses"),
            ("--raw-addresses", "--batch-get-mode", "no-pipeline"),
        ):
            with self.subTest(options=options):
                result = self.run_sweep(*options)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("ERROR:", result.stderr)
                self.assertNotIn("BATCH_GET_PERF_CONFIG", result.stdout)
        self.environment["KVAF_BATCH_GET_MODE"] = "unknown"
        self.assertEqual(self.run_sweep().returncode, 2)

    def test_missing_no_pipeline_kernel_fails_without_fallback(self):
        (self.build / "bin/kvaf_execute_batch_get_no_pipeline_kernels.o").unlink()
        result = self.run_sweep("--batch-get-mode", "no-pipeline")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("kvaf_execute_batch_get_no_pipeline_kernels.o", result.stdout)
        self.assertNotIn("BATCH_GET_PERF_CONFIG", result.stdout)

    def test_invalid_sweep_lists_fail_before_launch(self):
        for name, value in (
            ("TOKEN_BYTES", "0"), ("MISS_COUNTS", "-1"),
            ("AIV_CORES", "49"), ("JETTYS_PER_CORE", "65"),
            ("MISS_COUNTS", "1 garbage"), ("AIV_CORES", " "),
            ("TOKEN_BYTES", "999999999999999999999"),
        ):
            with self.subTest(name=name, value=value):
                key = f"KVAF_BATCH_GET_SWEEP_{name}"
                previous = self.environment[key]
                self.environment[key] = value
                result = self.run_sweep("--raw-addresses")
                self.environment[key] = previous
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn(f"ERROR: {key}", result.stderr)
                self.assertNotIn("BATCH_GET_PERF_CONFIG", result.stdout)


if __name__ == "__main__":
    unittest.main()
