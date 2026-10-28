"""Portable CLI coverage for per-core CPU EID mappings."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from npu_eids import parse_peer_eids_by_core
import test_npu_batch_get_perf as perf
import run_peer_eid_sweep as sweep
from argparse import Namespace
from pathlib import Path


class PeerEidConfigTests(unittest.TestCase):
    def parse(self, *arguments):
        with patch.dict(os.environ, {}, clear=True):
            return perf.parse_args(list(arguments))

    def test_core_counts_and_shared_controls(self):
        for cores in (2, 4, 8):
            for distinct in (False, True):
                eids = [f"{i + 1 if distinct else 1:032x}" for i in range(cores)]
                args = self.parse("--aiv-cores", str(cores), "--peer-eids-by-core", ",".join(eids))
                self.assertEqual(args.peer_eids_by_core, eids)
                self.assertIsNone(perf.validate_args(args))

    def test_invalid_combinations(self):
        mapping = ",".join(["1" * 32] * 4)
        for extra in (("--aiv-cores", "8"), ("--adaptive-ub-device",),
                      ("--npu-eids", "1" * 32 + "," + "2" * 32)):
            args = self.parse("--aiv-cores", "4", "--peer-eids-by-core", mapping, *extra)
            self.assertIn("peer-eids-by-core", perf.validate_args(args))

    def test_encoding_environment_and_override(self):
        self.assertEqual(parse_peer_eids_by_core(None), [])
        self.assertEqual(parse_peer_eids_by_core("A" * 32 + "," + "A" * 32), ["a" * 32] * 2)
        for invalid in ("abc", "0" * 32, "z" * 32, "1" * 32 + ",", "," + "1" * 32):
            with self.assertRaises(ValueError):
                parse_peer_eids_by_core(invalid)
        with patch.dict(os.environ, {"KVAF_PEER_EIDS_BY_CORE": ",".join(["1" * 32] * 2)}, clear=True):
            self.assertEqual(perf.parse_args([]).peer_eids_by_core, ["1" * 32] * 2)
            self.assertEqual(perf.parse_args(["--peer-eids-by-core", ""]).peer_eids_by_core, [])

    def test_sweep_keeps_workload_and_inflight_fixed(self):
        args = Namespace(cores=[2, 4, 8], jettys=[1, 2, 4],
                         peer_eids=[f"{i + 1:032x}" for i in range(8)],
                         npu_eid="a" * 32, max_inflight_total=2048, requests=8192,
                         build_dir=Path("/tmp/build"), table_capacity=64,
                         token_bytes=656, tokens_per_block=128, warmup=1, iterations=5,
                         kernel_profile_clock_mhz=1000)
        cases = sweep.build_cases(args)
        self.assertEqual(len(cases), 18)
        for i in range(0, len(cases), 2):
            baseline, distinct = cases[i][1], cases[i + 1][1]
            location = baseline.index("--peer-eids-by-core") + 1
            self.assertEqual(len(set(baseline[location].split(","))), 1)
            core_count = int(distinct[distinct.index("--aiv-cores") + 1])
            self.assertEqual(len(set(distinct[location].split(","))), core_count)
            baseline[location] = distinct[location]
            self.assertEqual(baseline, distinct)
        args.peer_eids.pop()
        with self.assertRaises(ValueError):
            sweep.build_cases(args)


class PeerEidSweepSummaryTests(unittest.TestCase):
    # Keep fractional-millisecond values to catch accidental use of rounded *_ms fields.
    LOG = """[KVAF_AIV] PEER_ENDPOINT_READ_VALIDATED round=0 cores=2 jettys_per_core=1
[KVAF_AIV] PEER_ENDPOINT_READ_VALIDATED round=1 cores=2 jettys_per_core=1
BATCH_GET_PERF_CORRECTNESS_PASS request_length=8192 miss_count=8192
PERF_RESULT operation=batch_get samples=5 mean_ms=1.250 p50_ms=1.100 p90_ms=1.500 p99_ms=1.800 useful_GiBps=4.004
PERF_RESULT operation=batch_get_kernel_total mean_ms=0.010 p50_ms=0.008 mean_cycles=10000 p50_cycles=8000 p90_cycles=12000 p99_cycles=16000
PERF_RESULT operation=batch_get_kernel_post mean_cycles=1000 p50_cycles=800 p90_cycles=1200 p99_cycles=1600
PERF_RESULT operation=batch_get_kernel_completion_wait mean_cycles=5000 p50_cycles=4000 p90_cycles=6000 p99_cycles=8000
BATCH_GET_PERF_PHASE_DONE phase=batch_get
"""

    def arguments(self, directory):
        return ["/tmp/build", "--peer-eids", "1" * 32 + "," + "2" * 32,
                "--npu-eid", "a" * 32, "--cores", "2", "--jettys", "1",
                "--log-dir", str(directory), "--kernel-profile-clock-mhz", "800"]

    def run_main(self, arguments, runner):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(output), redirect_stderr(output):
            with patch.object(sweep.subprocess, "run", side_effect=runner) as run:
                status = sweep.main(arguments)
        return status, output.getvalue(), run

    def successful_runner(self, command, *, stdout, **kwargs):
        stdout.write(self.LOG)
        return subprocess.CompletedProcess(command, 0)

    def read_csv(self, path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_summary_matches_cases_and_converts_cycles_without_rounding(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "logs, with spaces"
            status, output, run = self.run_main(self.arguments(directory), self.successful_runner)
            self.assertEqual(status, 0, output)
            self.assertEqual(run.call_count, 2)
            rows = self.read_csv(directory / "peer_eid_sweep_summary.csv")
            self.assertEqual([row["mode"] for row in rows], ["shared", "distinct"])
            self.assertEqual(rows[0]["peer_eids_by_core"], ",".join(["1" * 32] * 2))
            self.assertEqual(rows[1]["peer_eids_by_core"], "1" * 32 + "," + "2" * 32)
            for row in rows:
                self.assertNotIn(None, row)  # Commas in EIDs and paths must be CSV-escaped.
                self.assertEqual(row["status"], "PASS")
                self.assertEqual(row["error"], "")
                self.assertEqual(row["aiv_cores"], "2")
                self.assertEqual(row["jettys_per_core"], "1")
                self.assertEqual(row["mean_us"], "12.500")
                self.assertEqual(row["p50_us"], "10.000")
                self.assertEqual(row["p50_GBps"], "537.395")
                self.assertEqual(row["post_p50_us"], "1.000")
                self.assertEqual(row["completion_wait_p50_us"], "5.000")
                self.assertEqual(row["batch_get_mean_ms"], "1.250")
                self.assertEqual(row["batch_get_useful_GiBps"], "4.004")
                self.assertEqual(Path(row["log_file"]).read_text(), self.LOG)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--kernel-profile-clock-mhz") + 1], "800")
            self.assertIn("planned=2 passed=2 failed=0", output)
            self.assertIn("PEER_EID_SWEEP_CSV_WRITTEN", output)

    def test_environment_paths_and_command_line_override(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root).resolve()
            env = {"KVAF_PEER_EID_SWEEP_LOG_DIR": str(directory / "env_logs"),
                   "KVAF_PEER_EID_SWEEP_CSV": str(directory / "env_summary.csv"),
                   "KVAF_BATCH_GET_KERNEL_CLOCK_MHZ": "1200"}
            arguments = self.arguments(directory)[:-4]
            with patch.dict(os.environ, env, clear=True):
                args = sweep.parse_args(arguments)
                self.assertEqual(args.output_dir, directory / "env_logs")
                self.assertEqual(args.csv, directory / "env_summary.csv")
                self.assertEqual(args.kernel_profile_clock_mhz, 1200)
                args = sweep.parse_args(arguments + ["--output-dir", str(directory / "cli_logs"),
                                                      "--csv", str(directory / "csv" / "summary.csv"),
                                                      "--kernel-profile-clock-mhz", "800"])
                self.assertEqual(args.output_dir, directory / "cli_logs")
                self.assertEqual(args.csv, directory / "csv" / "summary.csv")
                self.assertEqual(args.kernel_profile_clock_mhz, 800)
            status, output, _ = self.run_main(self.arguments(args.output_dir) + ["--csv", str(args.csv)],
                                               self.successful_runner)
            self.assertEqual(status, 0, output)
            self.assertEqual(len(self.read_csv(args.csv)), 2)
            self.assertFalse((args.output_dir / "peer_eid_sweep_summary.csv").exists())

    def test_invalid_clock_and_duplicate_shapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for extra in (["--kernel-profile-clock-mhz", "0"], ["--cores", "2", "2"],
                          ["--jettys", "1", "1"]):
                with self.subTest(extra=extra), patch.dict(os.environ, {}, clear=True), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        sweep.parse_args(self.arguments(root) + extra)
                    self.assertEqual(error.exception.code, 2)

    def test_failure_keeps_successful_rows_and_records_failed_case(self):
        for failure in ("exit", "skip", "profile", "number"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as root:
                directory = Path(root)
                calls = []

                def runner(command, *, stdout, **kwargs):
                    calls.append(command)
                    if len(calls) == 1:
                        # Header is already saved before hardware starts.
                        self.assertEqual(self.read_csv(directory / "peer_eid_sweep_summary.csv"), [])
                        return self.successful_runner(command, stdout=stdout)
                    self.assertEqual(len(self.read_csv(directory / "peer_eid_sweep_summary.csv")), 1)
                    if failure == "exit":
                        stdout.write("hardware failure\n")
                        raise subprocess.CalledProcessError(2, command)
                    contents = self.LOG
                    if failure == "skip":
                        contents = "SKIP: hardware unavailable\n"
                    elif failure == "profile":
                        contents = contents.replace("operation=batch_get_kernel_post", "operation=missing")
                    else:
                        contents = contents.replace("mean_cycles=5000", "mean_cycles=nan")
                    stdout.write(contents)
                    return subprocess.CompletedProcess(command, 0)

                status, output, run = self.run_main(self.arguments(directory) + ["--jettys", "1", "2"], runner)
                self.assertEqual(status, 1)
                self.assertEqual(run.call_count, 2)  # Stop before the next shape.
                rows = self.read_csv(directory / "peer_eid_sweep_summary.csv")
                self.assertEqual([row["status"] for row in rows], ["PASS", "FAIL"])
                self.assertTrue(rows[1]["error"])
                self.assertEqual(rows[1]["mean_us"], "")
                self.assertEqual(rows[1]["batch_get_mean_ms"], "")
                self.assertTrue(Path(rows[1]["log_file"]).exists())
                self.assertIn("planned=4 passed=1 failed=1", output)

    def test_dry_run_creates_no_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / "not_created"
            status, output, run = self.run_main(self.arguments(directory) + ["--dry-run"], self.successful_runner)
            self.assertEqual(status, 0)
            run.assert_not_called()
            self.assertFalse(directory.exists())
            self.assertIn("PEER_EID_SWEEP_CSV", output)

    def test_existing_outputs_and_colliding_paths_are_rejected_before_running(self):
        for filename in ("peer_eid_sweep_summary.csv", "cores_2_jettys_1_distinct.log"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as root:
                directory = Path(root)
                existing = directory / filename
                existing.write_text("preserve me")
                status, output, run = self.run_main(self.arguments(directory), self.successful_runner)
                self.assertEqual(status, 1)
                run.assert_not_called()
                self.assertEqual(existing.read_text(), "preserve me")
                self.assertEqual(list(directory.iterdir()), [existing])
                self.assertIn("refusing to overwrite", output)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            status, _, run = self.run_main(self.arguments(directory) + [
                "--csv", str(directory / "cores_2_jettys_1_shared.log")], self.successful_runner)
            self.assertEqual(status, 1)
            run.assert_not_called()
            self.assertEqual(list(directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
