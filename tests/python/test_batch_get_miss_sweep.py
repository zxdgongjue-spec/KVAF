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
            if sys.argv[1].endswith("batch_get_jetty_profile.py"):
                os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
            sys.path.insert(0, {str(ROOT / 'tests' / 'python')!r})
            import test_npu_batch_get_perf as perf

            assert sys.argv[1].endswith("test_npu_batch_get_perf.py")
            args = perf.parse_args(sys.argv[2:])
            assert perf.validate_args(args) is None
            if args.adaptive_calibration_only:
                if os.environ.get("FIXED_CPU_DEVICE"):
                    print("[KVAF_AIV] UB_DEVICE_ADAPTIVE_SKIPPED reason=k_side_device_filter")
                print("BATCH_GET_PERF_ADAPTIVE_CALIBRATION_DONE")
                raise SystemExit(0)
            if not args.kernel_profile:
                assert args.peer_ub_device in ("udmaA", "udmaB")
                speed = 2 if args.peer_ub_device == "udmaB" else 1
                print(f"PERF_RESULT operation=batch_get useful_GiBps={{speed}}")
                raise SystemExit(0)
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
            for profile in sample:
                active = profile["active_requests"]
                profile["completion_count"] = int(active > 0)
                profile["jettys"] = []
                for slot in range(args.jettys_per_core):
                    requests = active if slot == 0 else 0
                    profile["jettys"].append({{
                        "requests": requests, "posts": int(requests > 0),
                        "completion_count": int(requests > 0),
                        "priority": 3 + slot, "jetty_id": 100 + slot,
                        "eid_path": slot % 2,
                        "first_doorbell_cycle": profile["first_doorbell_cycle"] if requests else 0,
                        "last_doorbell_cycle": profile["post_done_cycle"] if requests else 0,
                        "first_completion_cycle": profile["completion_done_cycle"] if requests else 0,
                        "last_completion_cycle": profile["completion_done_cycle"] if requests else 0,
                    }})
            rows = perf.jetty_profile.summarize([sample],
                clock_mhz=args.kernel_profile_clock_mhz, npu_core_count=args.aiv_cores,
                jettys_per_core=args.jettys_per_core, token_bytes=args.token_bytes,
                miss_count=args.miss_count, request_length=args.request_length,
                max_inflight_total=args.max_inflight_total, warmup=args.warmup,
                peer_device=args.peer_device, batch_get_mode=args.batch_get_mode)
            perf.jetty_profile.write_csv(args.jetty_csv, rows)
            print(f"BATCH_GET_JETTY_CSV_WRITTEN path={{args.jetty_csv}} rows={{len(rows)}}")
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
                with (self.directory / "logs/batch_get_jetty_summary.csv").open() as handle:
                    jetty_rows = list(csv.DictReader(handle))
                self.assertEqual(len(jetty_rows), 8)
                self.assertEqual({row["batch_get_mode"] for row in jetty_rows}, {batch_get_mode})
                self.assertEqual(sum(int(row["requests_total"]) for row in jetty_rows), 64)
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

    def test_full_path_calibration_selects_and_reuses_best_cpu_device(self):
        self.environment["KVAF_ADAPTIVE_UB_DEVICES"] = "udmaA,udmaB"
        result = self.run_sweep("--adaptive-ub-device")
        self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr)
        self.assertIn("selected_ub_device=udmaB", result.stdout)
        selection = (self.directory / "logs/ub_device_selection.log").read_text()
        self.assertIn("mode=full_perf_path", selection)
        self.assertEqual(selection.count("BATCH_GET_UB_DEVICE_CANDIDATE_BEGIN"), 2)
        with (self.directory / "logs/batch_get_miss_sweep_summary.csv").open() as stream:
            self.assertEqual({r["selected_ub_device"] for r in csv.DictReader(stream)}, {"udmaB"})

    def test_fixed_cpu_device_skips_full_path_calibration(self):
        self.environment["FIXED_CPU_DEVICE"] = "1"
        result = self.run_sweep("--adaptive-ub-device")
        self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr)
        self.assertIn("BATCH_GET_UB_DEVICE_CALIBRATION_SKIPPED", result.stdout)
        self.assertNotIn("BATCH_GET_UB_DEVICE_CANDIDATE_BEGIN", result.stdout)

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
