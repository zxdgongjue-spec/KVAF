#!/usr/bin/env python3
"""Host-only mixed-peer routing, baseline comparison and real Bash/gate tests."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import batch_get_jetty_profile as profile
import run_mixed_peer_batch_get as mixed
import test_npu_batch_get_perf as perf


def fake_perf(argv):
    args = perf.parse_args(argv)
    output = Path(args.jetty_csv).parent
    if args.peer_device == "cpu":
        time.sleep(0.25)  # The NPU worker must wait for this slower preparation.
    record = dict(vars(args), device=os.environ["DEVICE_ID"],
                  priorities=os.environ["KVAF_BATCH_GET_JETTY_PRIORITIES"],
                  ready_ns=time.monotonic_ns())
    perf.wait_for_concurrent_start()
    record["start_ns"] = time.monotonic_ns()
    time.sleep(0.1)
    record["finish_ns"] = time.monotonic_ns()
    (output / "invocation.json").write_text(json.dumps(record))
    cycles = 2000 if output.parent.name == "concurrent" else 1000
    stages = ("total", "post", "completion_wait", "wqe_fill", "mte_copy", "doorbell")
    for stage in stages:
        print(f"PERF_RESULT operation=batch_get_kernel_{stage} " + " ".join(
            f"{stat}_ms={cycles / 1e6} {stat}_cycles={cycles}"
            for stat in ("mean", "p50", "p90", "p99")))
    if args.peer_device == "cpu":
        print(f"[KVAF_AIV] UB_DEVICE_SELECTED ub_device={args.peer_ub_device}")
    row = dict.fromkeys(profile.FIELDS, "0")
    row.update(peer_device=args.peer_device, batch_get_mode=args.batch_get_mode,
               token_bytes=args.token_bytes, miss_count=args.miss_count,
               aiv_cores=args.aiv_cores, jettys_per_core=args.jettys_per_core,
               priority=record["priorities"])
    profile.write_csv(args.jetty_csv, [row])
    print(f"BATCH_GET_JETTY_CSV_WRITTEN path={args.jetty_csv} rows=1")
    return 0


class MixedPeerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "results"
        env = {key: value for key, value in os.environ.items() if not key.startswith("KVAF_")}
        env.update(PYTHONDONTWRITEBYTECODE="1", KVAF_REMOTE_CPU_PERF_TIMEOUT="10")
        patcher = patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.argv = ["build-npu", "--output-dir", str(self.output),
                     "--cpu-service-addr", "cpu-host:29000"]

    def test_fixed_mixed_routes_override_inherited_peer_on_same_device(self):
        with patch.dict(os.environ, KVAF_PEER_DEVICE="cpu", KVAF_PEER_SERVICE_ADDR="wrong:1"):
            runs = mixed.build_runs(mixed.parse_args(self.argv))
            self.assertEqual([phase for phase, _ in runs], ["concurrent"])
            for peer, (name, command, env) in zip(("npu", "cpu"), runs[0][1]):
                parsed = perf.parse_args(command[3:])
                self.assertEqual(parsed.peer_device, peer)
                self.assertEqual(parsed.peer_service_addr,
                                 "127.0.0.1:28999" if peer == "npu" else "cpu-host:29000")
                self.assertEqual(parsed.peer_ub_device, "" if peer == "npu" else "udmac0d1e3")
                self.assertEqual(env["DEVICE_ID"], "6")
                self.assertEqual(Path(env["KVAF_BATCH_GET_MISS_SWEEP_CSV"]).parent.name, name)
                if peer == "npu":
                    self.assertEqual(parsed.adaptive_peer_ub_devices, [])

    def test_per_flow_modes_eids_priorities_and_shape_override(self):
        args = mixed.parse_args(self.argv + [
            "--device", "7", "--npu-priorities", "3,5", "--cpu-priorities", "13",
            "--common-args=--miss-count 4096 --aiv-cores 1 --jettys-per-core 2 --iterations 30",
            "--npu-args=--batch-get-mode raw --npu-eids " + "1" * 32,
            "--cpu-args=--batch-get-mode no-pipeline --miss-count 8192 --npu-eids " + "2" * 32])
        for peer, (_, command, env) in zip(("npu", "cpu"), mixed.build_runs(args)[0][1]):
            parsed = perf.parse_args(command[3:])
            self.assertEqual(parsed.batch_get_mode, "raw" if peer == "npu" else "no-pipeline")
            self.assertEqual(parsed.npu_eids, [("1" if peer == "npu" else "2") * 32])
            self.assertEqual(parsed.iterations, 30)
            self.assertEqual(env["DEVICE_ID"], "7")
            self.assertEqual(env["KVAF_BATCH_GET_JETTY_PRIORITIES"], "3,5" if peer == "npu" else "13")
            self.assertEqual(env["KVAF_BATCH_GET_SWEEP_MISS_COUNTS"], "4096" if peer == "npu" else "8192")

    def test_invalid_configuration_launches_nothing(self):
        for option in ("--npu-args=--peer-device cpu", "--cpu-args=--peer-ub-device=wrong",
                       "--common-args=--adaptive-ub-device", "--npu-priorities=16",
                       "--cpu-args=--iterations 0"):
            with self.subTest(option=option), contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(mixed.concurrent, "run_flows") as run:
                self.assertEqual(mixed.main(self.argv + [option]), 1)
                run.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_existing_results_are_preserved(self):
        self.output.mkdir()
        sentinel = self.output / "keep.log"
        sentinel.write_text("keep")
        with contextlib.redirect_stdout(io.StringIO()), patch.object(mixed.concurrent, "run_flows") as run:
            self.assertEqual(mixed.main(self.argv), 1)
            run.assert_not_called()
        self.assertEqual(sentinel.read_text(), "keep")

    def test_baseline_failure_stops_experiment_and_preserves_exit_code(self):
        with contextlib.redirect_stdout(io.StringIO()), \
                patch.object(mixed.concurrent, "run_flows", return_value=9) as run:
            self.assertEqual(mixed.main(self.argv + ["--with-baselines"]), 9)
        self.assertEqual(run.call_count, 1)
        self.assertFalse((self.output / "concurrent").exists())
        self.assertFalse((self.output / "comparison.csv").exists())

    def test_zero_baseline_has_no_ratio(self):
        for peer in ("npu", "cpu"):
            for phase in (f"baseline_{peer}", "concurrent"):
                directory = self.output / phase / f"{peer}_peer"
                directory.mkdir(parents=True)
                (directory / "batch_get_miss_sweep_summary.csv").write_text(
                    "mean_us,p50_us,p99_us,p50_GBps\n0,0,0,0\n")
        with contextlib.redirect_stdout(io.StringIO()):
            mixed.write_comparison(self.output)
        with (self.output / "comparison.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["p50_GBps_ratio"] == "" for row in rows))

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "POSIX Bash integration")
    def test_real_sweeps_gate_baseline_order_and_comparison(self):
        runner = self.root / "fake_perf.sh"
        runner.write_text("#!/usr/bin/env bash\nshift\nexec " + shlex.quote(sys.executable)
                          + " -B " + shlex.quote(str(Path(__file__).resolve()))
                          + ' --fake-perf "$@"\n')
        runner.chmod(0o755)
        (self.root / "python3").symlink_to(sys.executable)
        env = dict(os.environ, KVAF_BATCH_GET_PERF_RUNNER=str(runner),
                   PATH=str(self.root) + os.pathsep + os.environ.get("PATH", ""))
        result = subprocess.run([sys.executable, "-B", mixed.__file__, *self.argv,
                                 "--with-baselines", "--npu-priorities=3", "--cpu-priorities=13",
                                 "--common-args=--miss-count 8192 --aiv-cores 1 --jettys-per-core 1 --iterations 7",
                                 "--npu-args=--batch-get-mode raw", "--cpu-args=--batch-get-mode no-pipeline"],
                                env=env, text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        records = {}
        for phase, peers in (("baseline_npu", ("npu",)), ("baseline_cpu", ("cpu",)),
                             ("concurrent", ("npu", "cpu"))):
            for peer in peers:
                directory = self.output / phase / f"{peer}_peer"
                record = json.loads((directory / "invocation.json").read_text())
                records[phase, peer] = record
                self.assertEqual(record["device"], "6")
                self.assertEqual(record["peer_device"], peer)
                self.assertEqual(record["peer_ub_device"], "" if peer == "npu" else "udmac0d1e3")
                self.assertEqual(record["priorities"], "3" if peer == "npu" else "13")
                self.assertEqual(record["iterations"], 7)
                self.assertTrue((directory / "batch_get_jetty_summary.csv").is_file())
                with (directory / "batch_get_miss_sweep_summary.csv").open() as stream:
                    summary = list(csv.DictReader(stream))
                self.assertEqual(len(summary), 1)
                self.assertEqual(summary[0]["batch_get_mode"], "raw" if peer == "npu" else "no-pipeline")
                self.assertEqual(summary[0]["selected_ub_device"], "" if peer == "npu" else "udmac0d1e3")
        self.assertLess(records["baseline_npu", "npu"]["finish_ns"], records["baseline_cpu", "cpu"]["ready_ns"])
        ready = max(records["concurrent", peer]["ready_ns"] for peer in ("npu", "cpu"))
        for peer in ("npu", "cpu"):
            self.assertLess(records["baseline_cpu", "cpu"]["finish_ns"], records["concurrent", peer]["ready_ns"])
            self.assertGreaterEqual(records["concurrent", peer]["start_ns"], ready)
        with (self.output / "comparison.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["flow"] for row in rows], ["npu_peer", "cpu_peer"])
        for row in rows:
            self.assertEqual(float(row["mean_us_ratio"]), 2.0)
            self.assertAlmostEqual(float(row["p50_GBps_ratio"]), 0.5, places=5)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fake-perf"]:
        raise SystemExit(fake_perf(sys.argv[2:]))
    unittest.main()
