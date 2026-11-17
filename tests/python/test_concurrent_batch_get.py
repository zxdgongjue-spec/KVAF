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
import batch_get_concurrency as concurrency
import run_concurrent_batch_get as launcher
import test_npu_batch_get_perf as perf
from npu_eids import peer_eid_path


ROOT = Path(__file__).resolve().parents[2]
FLOW_ARGS = ("--token-bytes 656 --miss-count 8192 --aiv-cores 1 --jettys-per-core 4 "
             "--table-capacity 16384 --warmup 3 --iterations 100")
STEM = "token_bytes_656_miss_count_8192_aiv_cores_1_jettys_per_core_4"
OUTPUTS = {STEM + ".log", STEM + ".jetty.csv",
           "batch_get_miss_sweep_summary.csv", "batch_get_jetty_summary.csv"}
PEERS = ["1" * 32, "2" * 32]
NPU_EIDS = ["3" * 32, "4" * 32]
PEER_OPTIONS = [option for index in range(2) for option in (
    f"--flow{index + 1}-peer-eids", ",".join(PEERS),
    f"--flow{index + 1}-npu-eid", NPU_EIDS[index])]


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
    rounds = []
    iterations = int(os.environ.get("KVAF_BATCH_GET_CONCURRENT_ITERATIONS", "3"))
    execution_mode = os.environ.get("KVAF_TEST_EXECUTION_MODE", "eager")
    calls = int(os.environ.get("KVAF_TEST_GRAPH_RING", "8")) if execution_mode == "graph" else 1
    with perf.ConcurrentMeasurement(iterations, 1024 * calls,
                                    execution_mode=execution_mode, calls_per_round=calls,
                                    torch_npu_profiler=os.environ.get("KVAF_TEST_PROFILER") == "1") as measurement:
        for iteration in range(iterations):
            if mode == "slow_round" and iteration == 1:
                time.sleep(0.2)
            round_ready = time.monotonic_ns()
            measurement.before(iteration)
            begin = time.monotonic_ns()
            if mode == "fail_in_round" and iteration == 1:
                return 9
            if mode == "hang_in_round" and iteration == 1:
                time.sleep(60)
            time.sleep(0.02 if mode == "slow_round" else 0.002)
            end = time.monotonic_ns()
            measurement.after(iteration, begin, end,
                              device_elapsed_ms=float(calls) if execution_mode == "graph" else None)
            rounds.append(dict(ready_ns=round_ready, begin_ns=begin, end_ns=end))
    started_ns = rounds[0]["begin_ns"]
    (output / (os.environ["DEVICE_ID"] + ".json")).write_text(json.dumps({
        "ready_ns": ready_ns, "started_ns": started_ns,
        "rounds": rounds,
        "device": os.environ["DEVICE_ID"],
        "cpu_service_addr": os.environ.get("KVAF_CPU_SERVICE_ADDR"),
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
    """Emit deterministic configuration, PERF_RESULT and Jetty records for the sweep."""
    args = perf.parse_args(argv)
    error = perf.validate_args(args)
    if error:
        raise ValueError(error)
    calls = args.graph_ring if args.execution_mode == "graph" else 1
    print(f"BATCH_GET_PERF_CONFIG table_capacity={args.table_capacity} "
          f"tokens_per_block={args.tokens_per_block} address_space_budget_bytes={args.address_space_bytes or 0} "
          f"address_space_bytes={args.table_capacity * args.tokens_per_block * args.token_bytes} "
          f"execution_mode={args.execution_mode} calls_per_round={calls}")
    with perf.ConcurrentMeasurement(args.iterations, args.miss_count * args.token_bytes * calls,
                                    execution_mode=args.execution_mode, calls_per_round=calls) as measurement:
        for iteration in range(args.iterations):
            measurement.before(iteration)
            begin = time.monotonic_ns()
            end = time.monotonic_ns()
            measurement.after(iteration, begin, end,
                              device_elapsed_ms=float(calls) if args.execution_mode == "graph" else None)
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
                       priority=os.environ.get("KVAF_BATCH_GET_JETTY_PRIORITIES", "15"),
                       execution_mode=args.execution_mode, calls_per_replay=calls,
                       profile_scope="last_get_in_replay" if args.execution_mode == "graph" else "single_get")
            rows.append(row)
    jetty_profile.write_csv(args.jetty_csv, rows)
    print("JETTY_PROFILE_CONTRACT fixture=1")
    print("JETTY_PROFILE_TABLE")
    print("original table row")
    print(f"BATCH_GET_JETTY_CSV_WRITTEN path={args.jetty_csv} rows={len(rows)}")
    if args.peer_eids:
        profiles = [{"jettys": [dict(
            eid_path=peer_eid_path(core, slot, args.jettys_per_core, len(args.peer_eids)),
            requests=128, completion_count=1,
            first_doorbell_cycle=10 + slot, last_doorbell_cycle=20 + slot,
            first_completion_cycle=100 + slot, last_completion_cycle=200 + slot)
            for slot in range(args.jettys_per_core)]} for core in range(args.aiv_cores)]
        perf.peer_eid_profile.report([profiles], args.peer_eids, args.aiv_cores,
                                    args.jettys_per_core, args.token_bytes, args.peer_csv,
                                    execution_mode=args.execution_mode, calls_per_replay=calls)
    return 0


class ConcurrentFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name).resolve()
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith("KVAF_")}
        self.env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                        KVAF_REMOTE_CPU_PERF_TIMEOUT="10")

    def args(self, device2=7, options=FLOW_ARGS, extra=()):
        with patch.dict(os.environ, self.env, clear=True):
            return launcher.parse_args([
                "build-npu", "--output-dir", str(self.output / "pair"),
                "--flow1-device", "7", "--flow2-device", str(device2),
                "--flow1-priorities", "3", "--flow2-priorities", "13",
                "--flow1-args=" + options, "--flow2-args=" + options, *extra])

    def flows(self, mode1="normal", mode2="slow", *extra_modes):
        flows = []
        for index, mode in enumerate((mode1, mode2, *extra_modes), start=1):
            command = [sys.executable, "-B", str(Path(__file__).resolve()),
                       "--worker", mode, str(self.output)]
            env = dict(self.env, DEVICE_ID=str(8 - index),
                       KVAF_BATCH_GET_JETTY_PRIORITIES="3" if index == 1 else "13")
            flows.append((f"flow{index}", command, env))
        return flows

    def run_pair(self, flows, timeout=5):
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = launcher.run_flows(flows, timeout)
        return code, captured.getvalue()


class ConcurrentTests(ConcurrentFixture):
    def test_profiler_options_and_paths_are_isolated_per_flow(self):
        args = self.args(extra=["--torch-npu-profiler", "--profiler-wait", "2",
                                "--profiler-warmup", "1", "--profiler-active", "4",
                                "--profiler-level", "Level2"])
        args.flow1_args += " --profiler-dir ignored-shared-path --profiler-active 5"
        with patch.dict(os.environ, self.env, clear=True):
            for name, active in (("flow1", 5), ("flow2", 4)):
                _, command, env = launcher.build_flow(args, name)
                parsed = perf.parse_args(command[3:])
                self.assertTrue(parsed.torch_npu_profiler)
                self.assertEqual(parsed.profiler_dir, args.output_dir.resolve() / name / "profiler")
                self.assertEqual((parsed.profiler_wait, parsed.profiler_warmup, parsed.profiler_active,
                                  parsed.profiler_level), (2, 1, active, "Level2"))
                self.assertEqual(env["KVAF_BATCH_GET_FLOW_NAME"], name)
            args.flow2_args += " --no-torch-npu-profiler"
            _, command, _ = launcher.build_flow(args, "flow2")
            self.assertFalse(perf.parse_args(command[3:]).torch_npu_profiler)

    def test_profiler_needs_enough_rounds_before_any_launch(self):
        args = self.args(options="--iterations 2", extra=["--torch-npu-profiler"])
        with patch.dict(os.environ, self.env, clear=True), self.assertRaisesRegex(ValueError, "iterations"):
            launcher.build_flow(args, "flow1")
        self.assertFalse(args.output_dir.exists())

    def test_graph_defaults_and_explicit_execution_overrides(self):
        for extra, flow_options, expected_mode, expected_ring in (
                ([], "", "graph", 8),
                (["--graph-ring", "4"], "", "graph", 4),
                (["--execution-mode", "eager"], "", "eager", 8),
                (["--graph-ring", "4"], "--execution-mode graph --graph-ring 2", "graph", 2),
                ([], "--execution-mode eager", "eager", 8)):
            with self.subTest(extra=extra, flow_options=flow_options), patch.dict(os.environ, self.env, clear=True):
                args = self.args(options=flow_options, extra=extra)
                _, command, _ = launcher.build_flow(args, "flow1")
                benchmark = perf.parse_args(command[3:])
                self.assertEqual(benchmark.execution_mode, expected_mode)
                self.assertEqual(benchmark.graph_ring, expected_ring)
        self.assertEqual(perf.parse_args([]).execution_mode, "eager")

    def test_graph_summary_uses_ring_bytes_and_separate_device_event_durations(self):
        flows = self.flows("normal", "normal")
        for (_, _, env), ring in zip(flows, (8, 4)):
            env.update(KVAF_TEST_EXECUTION_MODE="graph", KVAF_TEST_GRAPH_RING=str(ring), KVAF_TEST_PROFILER="1")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.run_flows(flows, 5, self.output), 0)
        summary = json.loads((self.output / concurrency.REPORT_FILES[2]).read_text())
        self.assertEqual(summary["total_payload_bytes"], 3 * (8 + 4) * 1024)
        self.assertEqual(summary["execution_mode"], "graph")
        self.assertTrue(summary["torch_npu_profiler"])
        self.assertEqual(summary["timing"], "host_graph_replay_to_event_completion")
        for name, ring in (("flow1", 8), ("flow2", 4)):
            row = summary["flows"][name]
            self.assertEqual(row["calls_per_round"], ring)
            self.assertEqual(row["payload_bytes_per_round"], ring * 1024)
            self.assertEqual(row["device_event_mean_us_per_replay"], ring * 1000)
            self.assertEqual(row["device_event_mean_us_per_get"], 1000)
            self.assertEqual(row["profile_scope"], "last_get_in_replay")
            self.assertTrue(row["torch_npu_profiler"])
            self.assertAlmostEqual(row["device_event_GBps"], 0.001024)
        with (self.output / concurrency.REPORT_FILES[1]).open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(row["device_elapsed_ms"]) for row in rows], [8.0, 4.0] * 3)

    def test_three_four_and_eight_flows_wait_for_every_round(self):
        popen = subprocess.Popen
        for count in (3, 4, 8):
            output = self.output / f"run-{count}"
            shared_addr = "192.0.2.10:29000"
            argv = ["build-npu", "--output-dir", str(output), "--address-space-bytes", "1048576"]
            for i in range(1, count + 1):
                options = f"--iterations 3 --token-bytes {128 * i}"
                if i == count:
                    options += " --address-space-bytes 2097152"
                argv.extend([f"--flow{i}-device={i - 1}", f"--flow{i}-priorities", str(i),
                             f"--flow{i}-npu-eid", NPU_EIDS[0],
                             f"--flow{i}-peer-eids", ",".join(PEERS),
                             f"--flow{i}-args={options}"])
                if count == 4:
                    argv.extend([f"--flow{i}-cpu-service-addr", f"192.0.2.{i}:29000"])

            def start_worker(command, **kwargs):
                folder = Path(kwargs["env"]["KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR"])
                options = perf.parse_args(command[3:])
                self.assertEqual(options.execution_mode, "graph")
                kwargs["env"].update(KVAF_TEST_EXECUTION_MODE=options.execution_mode,
                                     KVAF_TEST_GRAPH_RING=str(options.graph_ring))
                self.assertEqual(options.peer_eids, PEERS)
                self.assertEqual(options.npu_eids, NPU_EIDS[:1])
                self.assertEqual(Path(options.peer_csv), folder / "peer_samples.csv")
                flow_number = options.token_bytes // 128
                self.assertEqual(folder, output / f"flow{flow_number}")
                expected_bytes = 2097152 if flow_number == count else 1048576
                self.assertEqual(options.table_capacity, expected_bytes // (options.token_bytes * 128))
                mode = "slow" if folder.name == f"flow{count}" else "normal"
                return popen([sys.executable, "-B", str(Path(__file__).resolve()),
                              "--worker", mode, str(folder)], **kwargs)

            env = dict(self.env, KVAF_CPU_SERVICE_ADDR=shared_addr)
            with self.subTest(count=count), patch.dict(os.environ, env, clear=True), \
                    patch.object(launcher.subprocess, "Popen", side_effect=start_worker), \
                    contextlib.redirect_stdout(io.StringIO()) as console:
                self.assertEqual(launcher.main(argv), 0, console.getvalue())
                results = [json.loads((output / f"flow{i}" / f"{i - 1}.json").read_text())
                           for i in range(1, count + 1)]
                for i, row in enumerate(results, start=1):
                    expected_addr = f"192.0.2.{i}:29000" if count == 4 else shared_addr
                    self.assertEqual(row["cpu_service_addr"], expected_addr)
                    self.assertIn(f"DEVICE_ID={i - 1} KVAF_CPU_SERVICE_ADDR={expected_addr}",
                                  console.getvalue())
                self.assertEqual([row["priorities"] for row in results],
                                 [str(i) for i in range(1, count + 1)])
                self.assertGreaterEqual(min(row["started_ns"] for row in results),
                                        max(row["ready_ns"] for row in results))
                self.assertIn(f"all {count} flows ready", console.getvalue())
                for iteration in range(3):
                    self.assertGreaterEqual(min(row["rounds"][iteration]["begin_ns"] for row in results),
                                            max(row["rounds"][iteration]["ready_ns"] for row in results))
                    if iteration:
                        self.assertGreaterEqual(min(row["rounds"][iteration]["begin_ns"] for row in results),
                                                max(row["rounds"][iteration - 1]["end_ns"] for row in results))
                summary = json.loads((output / concurrency.REPORT_FILES[2]).read_text())
                self.assertEqual(summary["samples"], 3)
                self.assertEqual(summary["flow_count"], count)
                self.assertEqual(summary["total_payload_bytes"], count * 3 * 1024 * 8)
                self.assertEqual(summary["execution_mode"], "graph")

    def test_service_override_only_changes_its_own_flow(self):
        shared_addr = "192.0.2.1:29000"
        override = "192.0.2.1:29001"
        args = self.args(device2=6, extra=["--flow1-cpu-service-addr", override])
        for inherited in (shared_addr, None):
            env = self.env.copy()
            if inherited is not None:
                env["KVAF_CPU_SERVICE_ADDR"] = inherited
            with self.subTest(inherited=inherited), patch.dict(os.environ, env, clear=True):
                _, _, first = launcher.build_flow(args, "flow1")
                _, _, second = launcher.build_flow(args, "flow2")
                self.assertEqual(first["KVAF_CPU_SERVICE_ADDR"], override)
                self.assertEqual(second.get("KVAF_CPU_SERVICE_ADDR"), inherited)
                self.assertEqual(os.environ.get("KVAF_CPU_SERVICE_ADDR"), inherited)

    def test_extra_flow_requires_consecutive_numbers_and_device(self):
        for options in (["--flow4-device", "5"], ["--flow3-priorities", "15"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                self.args(extra=options)

    def test_three_peer_four_core_four_jetty_defaults_match_peer_sweep(self):
        import run_peer_eid_sweep as sweep

        peers = ["00000000000402000010000050100504",
                 "00000000000403000010000050100504",
                 "00000000000404000010000050100504"]
        options = ("--token-bytes 656 --miss-count 8192 --aiv-cores 4 --jettys-per-core 4 "
                   "--table-capacity 1277952 --warmup 20 --iterations 200")
        args = self.args(device2=4, options=options, extra=PEER_OPTIONS)
        args.flow1_device = 6
        args.flow1_npu_eid = ["00000000003f030000100000df00cb00"]
        args.flow2_npu_eid = ["00000000003f030000100000df008b00"]
        args.flow1_peer_eids = args.flow2_peer_eids = peers
        with patch.dict(os.environ, self.env, clear=True):
            for name in ("flow1", "flow2"):
                _, command, _ = launcher.build_flow(args, name)
                parsed = perf.parse_args(command[3:])
                _, (reference,) = sweep.parse_args([
                    "build-npu", "--peer-eids", ",".join(peers),
                    "--npu-eid", getattr(args, f"{name}_npu_eid")[0], "--cores", "4",
                    "--jettys", "4", "--miss-count", "8192", "--output-dir", "unused"])
                self.assertEqual(parsed.max_inflight_total, 2048)
                self.assertEqual(parsed.max_inflight_total, reference.max_inflight_total)
                self.assertEqual(parsed.peer_eids, peers)
                self.assertIsNone(perf.validate_args(parsed))

    def test_inflight_defaults_stay_fixed_across_flow_shapes_and_peer_counts(self):
        args = self.args(extra=PEER_OPTIONS)
        args.flow1_args = FLOW_ARGS + " --aiv-cores 4"
        with patch.dict(os.environ, self.env, clear=True):
            for name in ("flow1", "flow2"):
                _, command, _ = launcher.build_flow(args, name)
                self.assertEqual(perf.parse_args(command[3:]).max_inflight_total, 2048)
            for peers in (None, PEERS[:1]):
                args.flow1_peer_eids = peers
                _, command, _ = launcher.build_flow(args, "flow1")
                self.assertEqual(perf.parse_args(command[3:]).max_inflight_total, 2048)

    def test_explicit_inflight_cli_and_environment_are_never_raised(self):
        for extra, env_budget, expected in (
                ("--max-inflight-total 0", None, 0),
                ("--max-inflight-total=8192", None, 8192),
                ("--max-inflight 8192", None, 8192),
                ("", "8192", 8192), ("", "0", 0),
                ("--max-inflight-total 0", "8192", 0),
                ("--max-inflight-total 2048", None, 2048),
                ("--max-inflight-total=2048", None, 2048),
                ("", "2048", 2048)):
            with self.subTest(extra=extra, env_budget=env_budget):
                args = self.args(options=FLOW_ARGS + " --aiv-cores 4 " + extra, extra=PEER_OPTIONS)
                env = dict(self.env)
                if env_budget is not None:
                    env["KVAF_BATCH_GET_MAX_INFLIGHT_TOTAL"] = env_budget
                with patch.dict(os.environ, env, clear=True):
                    _, command, _ = launcher.build_flow(args, "flow1")
                    self.assertEqual(perf.parse_args(command[3:]).max_inflight_total, expected)

    def test_each_flow_passes_both_cpu_eids_and_its_own_npu_eid(self):
        for dedicated_options in (False, True):
            with self.subTest(dedicated_options=dedicated_options):
                args = self.args(device2=6, extra=PEER_OPTIONS if dedicated_options else ())
                # Explicit per-flow flags win over both nested options and the environment.
                for index, name in enumerate(("flow1", "flow2")):
                    source = "5" * 32 if dedicated_options else NPU_EIDS[index]
                    peers = ["6" * 32] if dedicated_options else PEERS
                    setattr(args, f"{name}_args", FLOW_ARGS +
                            f" --peer-eids {','.join(peers)} --npu-eids {source} --peer-csv shared.csv")
                with patch.dict(os.environ, dict(self.env, KVAF_NPU_EIDS="7" * 32), clear=True):
                    flows = [launcher.build_flow(args, name) for name in ("flow1", "flow2")]
                for index, (name, command, env) in enumerate(flows):
                    parsed = perf.parse_args(command[3:])
                    self.assertIsNone(perf.validate_args(parsed))
                    self.assertEqual(parsed.peer_eids, PEERS)
                    self.assertEqual(parsed.npu_eids, [NPU_EIDS[index]])
                    self.assertEqual(Path(parsed.peer_csv), args.output_dir / name / "peer_samples.csv")
                    self.assertEqual(env["DEVICE_ID"], str((7, 6)[index]))
                    self.assertEqual(env["KVAF_BATCH_GET_JETTY_PRIORITIES"], ("3", "13")[index])

    def test_invalid_per_flow_eids_fail_before_launch(self):
        for extra in (["--flow1-peer-eids", ""], ["--flow1-peer-eids", "bad"],
                      ["--flow1-peer-eids", ",".join([PEERS[0]] * 2)],
                      ["--flow1-npu-eid", ""], ["--flow1-npu-eid", ",".join(NPU_EIDS)]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                self.args(extra=extra)
        args = self.args(extra=["--flow1-peer-eids", ",".join(PEERS)])
        with patch.dict(os.environ, self.env, clear=True), self.assertRaisesRegex(ValueError, "one npu-eid"):
            launcher.build_flow(args, "flow1")

    def test_dual_peer_workers_synchronize_and_write_separate_reports(self):
        args = self.args(device2=6, extra=PEER_OPTIONS)
        with patch.dict(os.environ, self.env, clear=True):
            flows = [launcher.build_flow(args, name) for name in ("flow1", "flow2")]
        workers = []
        for name, command, env in flows:
            output = args.output_dir / name
            output.mkdir(parents=True)
            workers.append((name, [sys.executable, "-B", str(Path(__file__).resolve()),
                                  "--fake-perf", *command[3:], "--jetty-csv", str(output / "jettys.csv")], env))
        code, text = self.run_pair(workers)
        self.assertEqual(code, 0, text)
        self.assertIn("both flows ready; synchronized start scheduled", text)
        self.assertEqual(text.count("PEER_EID_CONCURRENCY"), 2)
        for name, _, _ in flows:
            with (args.output_dir / name / "peer_samples.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["peer_eid"] for row in rows], PEERS)
            self.assertTrue(all(int(row["bytes"]) > 0 for row in rows))

    def test_single_benchmark_hook_is_silent_noop(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(output), \
                patch("socket.create_connection", side_effect=AssertionError("unexpected connection")):
            with perf.ConcurrentMeasurement(2, 1024) as measurement:
                for iteration in range(2):
                    measurement.before(iteration)
                    measurement.after(iteration, 10, 20)
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

    def test_later_rounds_wait_for_slow_preparation_and_completion(self):
        with contextlib.redirect_stdout(io.StringIO()) as console:
            code = launcher.run_flows(self.flows("normal", "slow_round"), 5, self.output)
        self.assertEqual(code, 0, console.getvalue())
        results = [json.loads((self.output / f"{device}.json").read_text()) for device in (7, 6)]
        fast, slow = [row["rounds"] for row in results]
        for iteration in range(3):
            self.assertGreaterEqual(fast[iteration]["begin_ns"], slow[iteration]["ready_ns"])
            if iteration:
                self.assertGreaterEqual(fast[iteration]["begin_ns"], slow[iteration - 1]["end_ns"])
        self.assertGreater(fast[1]["begin_ns"] - fast[1]["ready_ns"], 150_000_000)
        with (self.output / concurrency.REPORT_FILES[0]).open() as stream:
            samples = list(csv.DictReader(stream))
        self.assertEqual(len(samples), 3)
        for iteration, sample in enumerate(samples):
            expected_ns = max(row["rounds"][iteration]["end_ns"] for row in results) - min(
                row["rounds"][iteration]["begin_ns"] for row in results)
            self.assertEqual(int(sample["window_ns"]), expected_ns)
            self.assertAlmostEqual(float(sample["aggregate_GBps"]), 2048 / expected_ns)
        with (self.output / concurrency.REPORT_FILES[1]).open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["device_id"] for row in rows}, {"6", "7"})

    def test_mismatched_iterations_fail_before_creating_outputs(self):
        argv = ["build-npu", "--output-dir", str(self.output / "mismatch"),
                "--flow1-device", "7", "--flow2-device", "6",
                "--flow1-args=--iterations 3", "--flow2-args=--iterations 4"]
        with patch.dict(os.environ, self.env, clear=True), contextlib.redirect_stdout(io.StringIO()) as console, \
                patch.object(launcher.subprocess, "Popen", side_effect=AssertionError("must not launch")):
            self.assertEqual(launcher.main(argv), 1)
        self.assertIn("same --iterations", console.getvalue())
        self.assertFalse((self.output / "mismatch").exists())

    def test_failed_or_stalled_round_reaps_waiting_workers(self):
        self.assert_cleanup_on_error(self.flows("normal", "fail_in_round"), RuntimeError)
        self.assert_cleanup_on_error(self.flows("normal", "hang_in_round"), TimeoutError, timeout=0.7)

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
        self.assertEqual(len(created), len(flows))
        self.assertTrue(all(process.poll() is not None for process in created))
        for _, _, env in flows:
            self.assertFalse((self.output / (env["DEVICE_ID"] + ".json")).exists())

    def test_preparation_failure_and_skip_release_no_worker(self):
        for mode in ("fail", "skip"):
            with self.subTest(mode=mode):
                self.assert_cleanup_on_error(self.flows(mode2=mode), RuntimeError)
        self.assert_cleanup_on_error(self.flows("normal", "normal", "normal", "fail"), RuntimeError)

    def test_startup_timeout_reaps_workers(self):
        self.assert_cleanup_on_error(self.flows(mode2="hang"), TimeoutError, timeout=0.6)

    def test_interrupt_reaps_workers(self):
        with patch.object(launcher.socket.socket, "accept", side_effect=KeyboardInterrupt):
            self.assert_cleanup_on_error(self.flows(), KeyboardInterrupt)

    def test_failure_after_start_returns_original_exit_code(self):
        code, text = self.run_pair(self.flows(mode1="fail_after_start"))
        self.assertEqual(code, 9)
        self.assertNotIn("BATCH_GET_CONCURRENT_SUMMARY", text)

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

    def test_existing_concurrent_summary_is_not_overwritten(self):
        report = self.output / concurrency.REPORT_FILES[2]
        report.write_text("keep me")
        with patch.dict(os.environ, self.env, clear=True), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(launcher, "run_flows", side_effect=AssertionError("must not launch")):
            code = launcher.main(["build-npu", "--output-dir", str(self.output),
                                  "--flow1-device", "7", "--flow2-device", "6"])
        self.assertEqual(code, 1)
        self.assertEqual(report.read_text(), "keep me")
        self.assertFalse((self.output / "flow1").exists())


class AggregateTests(unittest.TestCase):
    def test_uses_common_window_and_byte_weighted_total_not_sum_of_rates(self):
        first = concurrency.round_sample(0, 100, [
            dict(begin_ns=100, end_ns=1100, payload_bytes=1000),
            dict(begin_ns=200, end_ns=2200, payload_bytes=1000)])
        second = concurrency.round_sample(1, 3000, [
            dict(begin_ns=3000, end_ns=4000, payload_bytes=1000),
            dict(begin_ns=5000, end_ns=6000, payload_bytes=1000)])
        self.assertEqual(first["window_ns"], 2100)
        self.assertEqual(first["start_skew_ns"], 100)
        self.assertEqual(first["host_overlap_ns"], 900)
        self.assertAlmostEqual(first["aggregate_GBps"], 2000 / 2100)
        self.assertEqual(second["host_overlap_ns"], 0)
        summary = concurrency.report_results([first, second], [], None, lambda _: None)
        self.assertAlmostEqual(summary["aggregate_GBps"], 4000 / 5100)
        self.assertEqual(summary["all_flow_host_overlap_samples"], 1)

    def test_rejects_invalid_intervals_and_accepts_zero_payload(self):
        for begin, end in ((99, 200), (100, 100), (200, 100), (None, 200), (100, "200")):
            with self.subTest(begin=begin, end=end), self.assertRaises(RuntimeError):
                concurrency.round_sample(0, 100, [dict(begin_ns=begin, end_ns=end, payload_bytes=1000)])
        sample = concurrency.round_sample(0, 100, [dict(begin_ns=100, end_ns=200, payload_bytes=0)])
        self.assertEqual(sample["aggregate_GBps"], 0)


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

    def test_dual_peer_pair_keeps_sweep_outputs_and_adds_peer_reports(self):
        args = self.args(device2=6, extra=[*PEER_OPTIONS, "--address-space-bytes", "1048576"])
        args.flow2_args += " --address-space-bytes 2097152"
        with patch.dict(os.environ, self.env, clear=True):
            flows = [launcher.build_flow(args, name) for name in ("flow1", "flow2")]
        code, text = self.run_pair(flows, timeout=10)
        self.assertEqual(code, 0, text)
        self.assertEqual(text.count("PEER_EID_CONCURRENCY"), 2)
        for name, _, _ in flows:
            output = args.output_dir / name
            self.assertEqual({p.name for p in output.iterdir()}, OUTPUTS | {"peer_samples.csv"})
            with (output / "peer_samples.csv").open() as stream:
                self.assertEqual([row["peer_eid"] for row in csv.DictReader(stream)], PEERS)
            with (output / "batch_get_miss_sweep_summary.csv").open() as stream:
                row, = csv.DictReader(stream)
            self.assertEqual((row["address_space_budget_bytes"], row["table_capacity"]),
                             ("1048576", "12") if name == "flow1" else ("2097152", "24"))

    def test_budget_recomputes_capacity_for_each_sweep_shape(self):
        env = dict(self.env, KVAF_BATCH_GET_SWEEP_TOKEN_BYTES="128 656",
                   KVAF_BATCH_GET_SWEEP_MISS_COUNTS="2048",
                   KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR=str(self.output))
        command = ["bash", str(ROOT / "tests/run_remote_cpu_batch_get_miss_count_perf.sh"),
                   "build-npu", "--address-space-bytes", "1048576", "--table-capacity", "99"]
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with (self.output / "batch_get_miss_sweep_summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([(row["token_bytes"], row["table_capacity"], row["address_space_bytes"])
                          for row in rows], [("128", "64", "1048576"), ("656", "12", "1007616")])
        self.assertTrue(all(row["address_space_budget_bytes"] == "1048576" for row in rows))

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
                        "table_capacity", "tokens_per_block", "address_space_budget_bytes", "address_space_bytes",
                        "execution_mode", "calls_per_replay", "profile_scope"])
                    with (pair_output / "batch_get_miss_sweep_summary.csv").open() as stream:
                        row, = csv.DictReader(stream)
                    self.assertEqual((row["execution_mode"], row["calls_per_replay"], row["profile_scope"]),
                                     ("graph", "8", "last_get_in_replay"))

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
