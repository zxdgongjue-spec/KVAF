#!/usr/bin/env python3
"""Host-only rendezvous/process tests and original Bash output compatibility checks.

Run: python3 -B -m unittest discover -s tests/python -p test_concurrent_batch_get.py -v
Bash integration checks run on POSIX hosts; no NPU or KVAF build is required.
"""

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

import batch_get_jetty_profile as jetty_profile
import run_concurrent_batch_get as launcher
import test_npu_batch_get_perf as perf


ROOT = Path(__file__).resolve().parents[2]
FLOW_ARGS = ("--token-bytes 656 --miss-count 8192 --aiv-cores 1 --jettys-per-core 4 "
             "--table-capacity 16384 --warmup 3 --iterations 100")
STEM = "token_bytes_656_miss_count_8192_aiv_cores_1_jettys_per_core_4"
OUTPUTS = {STEM + ".log", STEM + ".jetty.csv",
           "batch_get_miss_sweep_summary.csv", "batch_get_jetty_summary.csv"}


def worker(mode, output):
    """Real subprocess using the benchmark's actual ready hook."""
    output = Path(output)
    (output / f"worker-{os.environ['DEVICE_ID']}.pid").write_text(str(os.getpid()))
    if mode == "slow":
        time.sleep(0.3)
    if mode == "hang":
        time.sleep(60)
    if mode in ("fail", "skip"):
        time.sleep(0.2)
        return 7 if mode == "fail" else 0
    child = None
    if mode == "descendant":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        (output / "child.pid").write_text(str(child.pid))
    ready_ns = time.monotonic_ns()
    perf.wait_for_concurrent_start()
    started_ns = time.monotonic_ns()
    (output / (os.environ["DEVICE_ID"] + ".json")).write_text(json.dumps({
        "ready_ns": ready_ns, "started_ns": started_ns,
        "device": os.environ["DEVICE_ID"],
        "priorities": os.environ["KVAF_BATCH_GET_JETTY_PRIORITIES"],
    }))
    print("JETTY_PROFILE_CONTRACT original", flush=True)
    print("JETTY_PROFILE_TABLE", flush=True)
    print("original table row", flush=True)
    print("BATCH_GET_MISS_SWEEP_SUMMARY original", flush=True)
    if child is not None:
        child.terminate()
        child.wait()
    return 9 if mode == "fail_after_start" else 0


def fake_perf(argv):
    """Deterministic PERF_RESULT/Jetty fixture consumed by the unchanged sweep."""
    args = perf.parse_args(argv)
    perf.wait_for_concurrent_start()
    print("BATCH_GET_PERF_CORRECTNESS_PASS fixture=1")
    for stage in ("kernel_total", "kernel_post", "kernel_completion_wait"):
        print(f"PERF_RESULT operation=batch_get_{stage} " + " ".join(
            f"{stat}_ms=0.001 {stat}_cycles=1000.000" for stat in ("mean", "p50", "p90", "p99")))
    print("[KVAF_AIV] UB_DEVICE_SELECTED ub_device=udmac0d1e2")
    rows = []
    for core in range(args.aiv_cores):
        for slot in range(args.jettys_per_core):
            row = dict.fromkeys(jetty_profile.FIELDS, "0")
            row.update(token_bytes=args.token_bytes, miss_count=args.miss_count,
                       aiv_cores=args.aiv_cores, jettys_per_core=args.jettys_per_core,
                       core=core, jetty_slot=slot,
                       priority=os.environ.get("KVAF_BATCH_GET_JETTY_PRIORITIES", "15"))
            rows.append(row)
    jetty_profile.write_csv(args.jetty_csv, rows)
    print("JETTY_PROFILE_CONTRACT fixture=1")
    print("JETTY_PROFILE_TABLE")
    print("original table row")
    print(f"BATCH_GET_JETTY_CSV_WRITTEN path={args.jetty_csv} rows={len(rows)}")
    return 0


class ConcurrentFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith("KVAF_")}
        self.env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                        KVAF_REMOTE_CPU_PERF_TIMEOUT="10")

    def args(self, device2=7, options=FLOW_ARGS):
        with patch.dict(os.environ, self.env, clear=True):
            return launcher.parse_args([
                "build-npu", "--output-dir", str(self.output / "pair"),
                "--flow1-device", "7", "--flow2-device", str(device2),
                "--flow1-priorities", "3", "--flow2-priorities", "13",
                "--flow1-args=" + options, "--flow2-args=" + options])

    def flows(self, mode1="normal", mode2="slow"):
        return [(name, [sys.executable, "-B", str(Path(__file__).resolve()),
                        "--worker", mode, str(self.output)],
                 dict(self.env, DEVICE_ID=device, KVAF_BATCH_GET_JETTY_PRIORITIES=priority))
                for name, device, priority, mode in
                (("flow1", "7", "3", mode1), ("flow2", "6", "13", mode2))]

    def run_pair(self, flows, timeout=5):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = launcher.run_flows(flows, timeout)
        return code, captured.getvalue()


class ConcurrentTests(ConcurrentFixture):
    def test_single_benchmark_hook_is_silent_noop(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(output), \
                patch("socket.create_connection", side_effect=AssertionError("unexpected connection")):
            perf.wait_for_concurrent_start()
        self.assertEqual(output.getvalue(), "")

    def test_independent_devices_priorities_and_paths(self):
        inherited = dict(self.env, KVAF_BATCH_GET_MISS_SWEEP_CSV="shared.csv",
                         KVAF_BATCH_GET_JETTY_SWEEP_CSV="shared_jetty.csv",
                         KVAF_BATCH_GET_SWEEP_MISS_COUNTS="1 2 3", DEVICE_ID="0")
        for device2 in (7, 6):
            args = self.args(device2)
            with patch.dict(os.environ, inherited, clear=True):
                first, second = [launcher.build_flow(args, name) for name in ("flow1", "flow2")]
                self.assertEqual(os.environ["DEVICE_ID"], "0")
            for flow, device, priority in ((first, 7, "3"), (second, device2, "13")):
                name, command, env = flow
                self.assertEqual(env["DEVICE_ID"], str(device))
                self.assertEqual(env["KVAF_BATCH_GET_JETTY_PRIORITIES"], priority)
                self.assertEqual(env["KVAF_BATCH_GET_SWEEP_MISS_COUNTS"], "8192")
                self.assertEqual(env["KVAF_BATCH_GET_SWEEP_AIV_CORES"], "1")
                self.assertEqual(env["KVAF_BATCH_GET_SWEEP_JETTYS_PER_CORE"], "4")
                self.assertEqual(Path(env["KVAF_BATCH_GET_MISS_SWEEP_CSV"]).parent.name, name)
                self.assertEqual(Path(env["KVAF_BATCH_GET_JETTY_SWEEP_CSV"]).parent.name, name)
                self.assertTrue(command[1].endswith("run_remote_cpu_batch_get_miss_count_perf.sh"))

    def test_invalid_configuration_fails_before_launch(self):
        args = self.args(options="--request-length 10 --miss-count 11")
        with patch.dict(os.environ, self.env, clear=True), self.assertRaisesRegex(ValueError, "miss-count"):
            launcher.build_flow(args, "flow1")
        args = self.args()
        args.flow1_priorities = "3,16"
        with patch.dict(os.environ, self.env, clear=True), self.assertRaisesRegex(ValueError, "priorities"):
            launcher.build_flow(args, "flow1")
        args = self.args(options="--adaptive-ub-device")
        with patch.dict(os.environ, self.env, clear=True), self.assertRaisesRegex(ValueError, "adaptive"):
            launcher.build_flow(args, "flow1")

    def test_npu_peer_and_operator_modes_are_forwarded(self):
        for mode in ("lookup", "raw", "no-pipeline"):
            options = (f"--peer-device npu --peer-service-addr npu-peer:28999 "
                       f"--batch-get-mode {mode} --npu-eids {'1' * 32}")
            args = self.args(options=options)
            with patch.dict(os.environ, self.env, clear=True):
                _, command, env = launcher.build_flow(args, "flow1")
                parsed = perf.parse_args(command[3:])
            self.assertEqual(parsed.peer_device, "npu")
            self.assertEqual(parsed.peer_service_addr, "npu-peer:28999")
            self.assertEqual(parsed.batch_get_mode, mode)
            self.assertEqual(parsed.adaptive_peer_ub_devices, [])
            self.assertEqual(env["KVAF_BATCH_GET_SWEEP_MISS_COUNTS"], str(parsed.miss_count))

    def test_waits_for_late_worker_and_preserves_summary_blocks(self):
        code, text = self.run_pair(self.flows())
        self.assertEqual(code, 0)
        results = [json.loads((self.output / f"{device}.json").read_text()) for device in (7, 6)]
        last_ready = max(result["ready_ns"] for result in results)
        self.assertTrue(all(result["started_ns"] >= last_ready for result in results))
        self.assertGreater(results[0]["started_ns"] - results[0]["ready_ns"], 200_000_000)
        self.assertEqual([r["priorities"] for r in results], ["3", "13"])
        block = "JETTY_PROFILE_CONTRACT original\nJETTY_PROFILE_TABLE\noriginal table row\nBATCH_GET_MISS_SWEEP_SUMMARY original\n"
        self.assertEqual(text.count(block), 2)
        self.assertLess(text.index("flow1 / DEVICE_ID=7"), text.index("flow2 / DEVICE_ID=6"))

    def assert_cleanup_on_error(self, flows, exception, timeout=5):
        created = []
        popen = subprocess.Popen

        def track(*args, **kwargs):
            process = popen(*args, **kwargs)
            created.append(process)
            return process

        with patch.object(launcher.subprocess, "Popen", side_effect=track), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(exception):
            launcher.run_flows(flows, timeout)
        self.assertEqual(len(created), 2)
        self.assertTrue(all(process.poll() is not None for process in created))
        self.assertFalse((self.output / "7.json").exists())

    def test_preparation_failure_and_skip_release_no_worker(self):
        for mode in ("fail", "skip"):
            with self.subTest(mode=mode):
                self.assert_cleanup_on_error(self.flows(mode2=mode), RuntimeError)

    def test_startup_timeout_reaps_workers(self):
        self.assert_cleanup_on_error(self.flows(mode2="hang"), TimeoutError, timeout=0.6)

    def test_interrupt_reaps_workers(self):
        with patch.object(launcher.socket.socket, "accept", side_effect=KeyboardInterrupt):
            self.assert_cleanup_on_error(self.flows(), KeyboardInterrupt)

    def test_failure_after_start_returns_original_exit_code(self):
        code, text = self.run_pair(self.flows(mode1="fail_after_start"))
        self.assertEqual(code, 9)
        self.assertIn("flow2 exit_status=0", text)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process group check")
    def test_failure_kills_descendant_in_worker_group(self):
        self.assert_cleanup_on_error(self.flows(mode1="descendant", mode2="fail"), RuntimeError)
        pid = int((self.output / "child.pid").read_text())
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists():
            # A killed orphan may remain a zombie until the host's init reaps it.
            self.assertEqual(stat.read_text().split(") ", 1)[1].split()[0], "Z")

    def test_existing_results_are_not_overwritten(self):
        pair = self.output / "pair"
        (pair / "flow1").mkdir(parents=True)
        existing = pair / "flow1" / "result.log"
        existing.write_text("keep me")
        with patch.dict(os.environ, self.env, clear=True), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(launcher, "run_flows", side_effect=AssertionError("must not launch")):
            code = launcher.main(["build-npu", "--output-dir", str(pair),
                                  "--flow1-device", "7", "--flow2-device", "6"])
        self.assertEqual(code, 1)
        self.assertEqual(existing.read_text(), "keep me")
        self.assertFalse((pair / "flow2").exists())


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "POSIX Bash integration")
class SweepCompatibilityTests(ConcurrentFixture):
    def setUp(self):
        super().setUp()
        runner = self.output / "fake_perf.sh"
        runner.write_text("#!/usr/bin/env bash\nshift\nexec " + shlex.quote(sys.executable)
                          + " -B " + shlex.quote(str(Path(__file__).resolve()))
                          + ' --fake-perf "$@"\n')
        runner.chmod(0o755)
        # The existing sweep invokes python3 for its unchanged CSV merger.
        python3 = self.output / "python3"
        python3.symlink_to(sys.executable)
        self.env.update(KVAF_BATCH_GET_PERF_RUNNER=str(runner),
                        PATH=str(self.output) + os.pathsep + self.env.get("PATH", ""))

    def test_default_single_sweep_still_runs_original_three_shapes(self):
        output = self.output / "single"
        env = dict(self.env, KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR=str(output))
        result = subprocess.run(["bash", str(ROOT / "tests/run_remote_cpu_batch_get_miss_count_perf.sh"),
                                 "build-npu"], env=env, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with (output / "batch_get_miss_sweep_summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["miss_count"] for r in rows], ["2048", "4096", "8192"])
        self.assertTrue(all(r["aiv_cores"] == "2" and r["jettys_per_core"] == "2" for r in rows))
        self.assertEqual(len(list(output.iterdir())), 8)
        self.assertNotIn("[launcher]", result.stdout)

    def test_pair_outputs_match_single_sweep_verbatim(self):
        for device2 in (7, 6):
            with self.subTest(device2=device2):
                args = self.args(device2)
                args.output_dir = self.output / f"pair-{device2}"
                with patch.dict(os.environ, self.env, clear=True):
                    flows = [launcher.build_flow(args, name) for name in ("flow1", "flow2")]
                code, text = self.run_pair(flows, timeout=10)
                self.assertEqual(code, 0, text)
                for name, command, env in flows:
                    pair_output = Path(env["KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR"])
                    solo_output = self.output / f"solo-{device2}-{name}"
                    solo_env = dict(env, KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR=str(solo_output),
                                    KVAF_BATCH_GET_MISS_SWEEP_CSV=str(solo_output / "batch_get_miss_sweep_summary.csv"),
                                    KVAF_BATCH_GET_JETTY_SWEEP_CSV=str(solo_output / "batch_get_jetty_summary.csv"))
                    single = subprocess.run(command, env=solo_env, text=True, capture_output=True, timeout=10)
                    self.assertEqual(single.returncode, 0, single.stdout + single.stderr)
                    self.assertEqual({p.name for p in pair_output.iterdir()}, OUTPUTS)
                    for filename in OUTPUTS:
                        self.assertEqual(
                            (pair_output / filename).read_text().replace(str(pair_output), "OUTPUT"),
                            (solo_output / filename).read_text().replace(str(solo_output), "OUTPUT"))
                    with (pair_output / "batch_get_miss_sweep_summary.csv").open() as stream:
                        header = next(csv.reader(stream))
                    self.assertEqual(header, [
                        "token_bytes", "miss_count", "aiv_cores", "jettys_per_core",
                        "mean_us", "p50_us", "p90_us", "p99_us", "p50_GBps",
                        "post_mean_us", "post_p50_us", "post_p90_us", "post_p99_us",
                        "completion_wait_mean_us", "completion_wait_p50_us",
                        "completion_wait_p90_us", "completion_wait_p99_us", "selected_ub_device",
                        "address_mode", "batch_get_mode",
                        *[f"{stage}_{stat}_us" for stage in ("wqe_fill", "mte_copy", "doorbell")
                          for stat in ("mean", "p50", "p90", "p99")]])

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux timeout/process group check")
    def test_real_perf_timeout_stays_in_launcher_group(self):
        build = self.output / "build"
        (build / "bin").mkdir(parents=True)
        (build / "python/kvaf").mkdir(parents=True)
        for operation in ("get", "put"):
            (build / f"bin/kvaf_execute_batch_{operation}_kernels.o").write_text("fixture")
        shim = self.output / "python3"
        shim.unlink()
        shim.write_text("#!/usr/bin/env bash\nexec " + shlex.quote(sys.executable)
                        + " -B " + shlex.quote(str(Path(__file__).resolve()))
                        + " --worker hang " + shlex.quote(str(self.output)) + "\n")
        shim.chmod(0o755)
        flows = [(f"flow{index}", ["bash", str(ROOT / "tests/run_remote_cpu_batch_get_perf.sh"), str(build)],
                  dict(self.env, DEVICE_ID=str(device), KVAF_BATCH_GET_JETTY_PRIORITIES="3"))
                 for index, device in ((1, 7), (2, 6))]
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(TimeoutError):
            launcher.run_flows(flows, timeout=1)
        survivors = []
        for device in (7, 6):
            pid = int((self.output / f"worker-{device}.pid").read_text())
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists() and stat.read_text().split(") ", 1)[1].split()[0] != "Z":
                survivors.append(pid)
                os.kill(pid, 9)
        self.assertEqual(survivors, [], "timeout must not isolate workers from launcher cleanup")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        raise SystemExit(worker(*sys.argv[2:]))
    if sys.argv[1:2] == ["--fake-perf"]:
        raise SystemExit(fake_perf(sys.argv[2:]))
    unittest.main()
