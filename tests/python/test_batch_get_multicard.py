"""Exercise multi-card service routing and saved reports without NPU hardware."""

import csv
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tests/run_remote_cpu_batch_get_multicard_perf.sh"

# Use the real single-card parser and controller barrier, replacing device work.
FAKE_WORKER = """import json
import os
import sys
import time

from test_npu_batch_get_perf import parse_args, runtime_options, validate_args
from utils.get_options import parallelism_field

args = parse_args(["--build-dir", sys.argv[1], *sys.argv[2:]])
assert validate_args(args) is None
assert "--peer" in sys.argv
assert args.peer == os.environ["KVAF_CPU_SERVICE_ADDR"]
control = args.contention_control_dir
(control / f"ready_{args.device_id}").touch()
deadline = time.monotonic() + 10
while not (control / "start.json").exists():
    if (control / "abort").exists() or time.monotonic() > deadline:
        raise RuntimeError("controller did not start the worker")
    time.sleep(0.01)
window = json.loads((control / "start.json").read_text())
if args.kernel_profile:
    from test_batch_get_multicard import run_profiled_worker
    run_profiled_worker(args, window)
    sys.exit(0)
result = dict(device=args.device_id, peer=args.peer, backend=args.backend,
              start=window["start"], end=window["stop"], payload_bytes=1000000,
              bandwidth_GBps=1000000 / (window["stop"] - window["start"]) / 1e9,
              verified=True, jettys_per_core=args.jettys_per_core)
result[parallelism_field(args.backend)] = args.parallelism
result["get_core_fe_map"] = runtime_options(args).get("get_core_fe_map")
args.contention_result.write_text(json.dumps(result))
"""


def run_profiled_worker(args, window):
    from test_npu_batch_get_perf import MAX_JETTYS_PER_CORE, summarize_multicard_profile
    from utils.multicard import run_worker

    # Two timed replays with independently timestamped cores and node leases.
    graph, runtime = Mock(), Mock()
    leases = list(range(1, args.contention_graph_calls + 1))
    def snapshot(*, graph_lease):
        assert graph_lease == leases[-1]
        assert graph.synchronize.call_count > 1 + args.warmup
        workers = []
        for core in range(args.parallelism):
            start = 1000 + 100 * core
            profile = dict.fromkeys(("active_requests", "total_posts", "submitted_group_count",
                                     "completed_group_count", "completion_count", "post_active_cycles",
                                     "drain_wait_cycles", "peak_submitted_not_drained"), 1)
            profile.update(core=core, kernel_start_cycle=start, kernel_finish_cycle=start + 500,
                           setup_done_cycle=start + 10, compact_done_cycle=start + 20,
                           first_doorbell_cycle=start + 30, completion_done_cycle=start + 400,
                           jetty_completion_wait_cycles=[0] * MAX_JETTYS_PER_CORE,
                           jetty_completion_polls=[0] * MAX_JETTYS_PER_CORE)
            workers.append(profile)
        return workers
    runtime.batch_get_kernel_profile.side_effect = snapshot
    start, stop = window["start"], window["stop"]
    clock = [start, start, start, (start + stop) / 2, stop, stop]
    with patch("utils.multicard.captured_get_nodes", return_value=nullcontext((graph, leases))), \
         patch("utils.multicard.time.monotonic", side_effect=clock):
        run_worker(None, Mock(), args, runtime, Mock(), None, Mock(), Mock(), Mock(), summarize_multicard_profile)
    assert runtime.batch_get_kernel_profile.call_args_list == [call(graph_lease=leases[-1])] * 2


def read_csv(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


class MultiCardServices(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        worker = self.directory / "worker.py"
        worker.write_text(f"#!{sys.executable}\n" + FAKE_WORKER)
        worker.chmod(0o755)
        self.env = dict(os.environ, KVAF_BATCH_GET_PERF_RUNNER=str(worker),
                        KVAF_CPU_SERVICE_ADDR="ambient:29999", PYTHON=sys.executable,
                        PYTHONPATH=str(ROOT / "tests/python"), KVAF_NPU_EIDS="")

    def invoke(self, arguments, expected=0):
        result = subprocess.run(["bash", str(RUNNER), "build-npu", *arguments],
                                env=self.env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def run_case(self, name, backend="gather_read", peers=(), sweep=False):
        output = self.directory / name
        field = "--aicpu-workers" if backend in ("gather_read", "cpu_push_scatter") else "--aiv-cores"
        self.invoke(["--backend", backend, "--devices", "3", "1", field,
                     *(["1", "2"] if sweep else ["1"]),
                     "--request-length", "4", "--miss-count", "4", "--table-capacity", "4",
                     "--tokens-per-block", "1", "--token-bytes", "8",
                     "--warmup", "0", "--seconds", "0.1", "--ready-timeout", "10",
                     "--output", str(output), *peers])
        return output

    def check_case(self, directory, peers):
        config = json.loads((directory / "config.json").read_text())
        self.assertEqual(config["device_peers"], peers)
        self.assertEqual(config["devices"], [3, 1])
        rows = read_csv(directory / "summary.csv")
        self.assertEqual([(row["device"], row["peer"]) for row in rows], list(peers.items()))
        aggregate = read_csv(directory / "aggregate.csv")[0]
        self.assertEqual(json.loads(aggregate["device_peers"]), peers)
        self.assertEqual(float(aggregate["overlap_percent"]), 100)
        for device, peer in peers.items():
            for phase in (f"isolated_{device}", "concurrent"):
                result = json.loads((directory / phase / f"device_{device}.json").read_text())
                self.assertEqual(result["peer"], peer)
            self.assertIn(f"| {device} | {peer} |", (directory / "report.md").read_text())

    def test_independent_services_all_backends_and_sweeps(self):
        peers = {"3": "cpu-b:21112", "1": "cpu-a:21111"}
        for backend in ("aiv", "ubmem", "gather_read", "cpu_push_scatter"):
            with self.subTest(backend=backend):
                directory = self.run_case(backend, backend, sweep=True, peers=(
                    "--device-peer", "1=cpu-a:21111", "--device-peer", "3=cpu-b:21112"))
                self.assertEqual(json.loads((directory / "config.json").read_text())["device_peers"], peers)
                prefix = "workers" if backend in ("gather_read", "cpu_push_scatter") else "cores"
                for count in (1, 2):
                    self.check_case(directory / f"{prefix}_{count}", peers)
                self.assertEqual(len(read_csv(directory / "summary.csv")), 4)
                self.assertEqual(len(read_csv(directory / "aggregate.csv")), 2)
                self.assertIn("| 3 | cpu-b:21112 |", (directory / "report.md").read_text())
                saved_report = (directory / "report.md").read_text()
                self.env["KVAF_CPU_SERVICE_ADDR"] = "changed:30000"
                self.invoke(["--summarize", str(directory)])
                self.assertEqual((directory / "report.md").read_text(), saved_report)

    def test_default_service_with_one_override(self):
        directory = self.run_case("mixed", peers=(
            "--peer", "default:21111", "--device-peer", "1=other:21112"))
        self.check_case(directory, {"3": "default:21111", "1": "other:21112"})

    def test_core_fe_maps_follow_device_through_both_phases(self):
        for name, maps in (("both", {"1": [0, 0, 1], "3": [2, 2, 1]}),
                           ("partial", {"1": [0, 0, 1]})):
            with self.subTest(case=name):
                options = ["--aiv-cores", "3", "--jettys-per-core", "2", "--peer-ub-device", "E2,E3,E4"]
                for device, mapping in maps.items():
                    options.extend(["--device-core-fe-map", f"{device}=" + ",".join(map(str, mapping))])
                directory = self.run_case(name, "aiv", peers=options)
                self.assertEqual(json.loads((directory / "config.json").read_text())["device_core_fe_maps"], maps)
                for device in ("3", "1"):
                    for phase in (f"isolated_{device}", "concurrent"):
                        result = json.loads((directory / phase / f"device_{device}.json").read_text())
                        self.assertEqual(result["get_core_fe_map"], maps.get(device))

    def test_invalid_core_fe_maps_fail_before_workers_start(self):
        output = self.directory / "invalid"
        for options in (["--device-core-fe-map", "3=2,2,1"], ["--backend", "gather_read"],
                        ["--aiv-cores", "2"], ["--peer-ub-device", "E2,E3"]):
            self.invoke(["--backend", "aiv", "--devices", "3", "1", "--device-core-fe-map", "3=2,2,1",
                         "--output", str(output), *options], expected=2)
            self.assertFalse(output.exists())
        for value in ("2=0,0,1", "3=-1,0,1", "3=0,,1", "3="):
            self.invoke(["--backend", "aiv", "--devices", "3", "1", "--device-core-fe-map", value,
                         "--output", str(output)], expected=2)
            self.assertFalse(output.exists())

    def test_kernel_profile_sample_scope_timing_and_saved_summary(self):
        directory = self.run_case("profile", "aiv", peers=("--kernel-profile", "--graph-calls", "3",
                                  "--kernel-profile-clock-mhz", "500", "--aiv-cores", "2", "--jettys-per-core", "1"))
        rows = read_csv(directory / "summary.csv")
        for row in rows:
            # Each 32-byte sample spans 600 cycles wall or 500 cycles on the longest core.
            self.assertEqual(int(row["concurrent_kernel_samples"]), 2)
            self.assertAlmostEqual(float(row["concurrent_kernel_mean_us"]), 1.2)
            self.assertAlmostEqual(float(row["concurrent_kernel_mean_GBps"]), 32 / 1200)
            self.assertAlmostEqual(float(row["isolated_kernel_max_core_p50_GBps"]), 32 / 1000)
            path = directory / "concurrent" / f"device_{row['device']}"
            worker = json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(worker["payload_bytes"], 2 * 3 * 32)
            self.assertEqual(worker["kernel_profile"]["payload_bytes_per_sample"], 32)
            self.assertIn("KERNEL_PROFILE_CORE", path.with_suffix(".log").read_text())
        self.invoke(["--summarize", str(directory)])
        self.assertEqual(read_csv(directory / "summary.csv"), rows)

    def test_shared_environment_service_and_legacy_summary(self):
        directory = self.run_case("shared")
        peers = {"3": "ambient:29999", "1": "ambient:29999"}
        self.check_case(directory, peers)
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        config.pop("device_peers")
        config_path.write_text(json.dumps(config))
        for path in directory.glob("*/device_*.json"):
            result = json.loads(path.read_text())
            result.pop("peer")
            path.write_text(json.dumps(result))
        self.env["KVAF_CPU_SERVICE_ADDR"] = "changed:30000"
        self.invoke(["--summarize", str(directory)])
        rows = read_csv(directory / "summary.csv")
        self.assertEqual({row["peer"] for row in rows}, {"ambient:29999"})

    def test_summary_rejects_worker_service_mismatch(self):
        directory = self.run_case("mismatch")
        path = directory / "concurrent/device_1.json"
        result = json.loads(path.read_text())
        result["peer"] = "wrong:21111"
        path.write_text(json.dumps(result))
        error = self.invoke(["--summarize", str(directory)], expected=1)
        self.assertIn("CPU service mismatch", error.stderr)

    def test_invalid_mapping_fails_before_workers_start(self):
        output = self.directory / "invalid"
        invalid = [
            ["--device-peer", value] for value in (
                "1", "x=host:21111", "-1=host:21111", "1=", "1=host", "1=:21111",
                "1=host:0", "1=host:65536", "1=host:abc", "1=host:21111 ", "2=host:21111")
        ]
        invalid += [["--device-peer", "1=host:21111", "--device-peer", "1=other:21112"],
                    ["--peer", ""]]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                self.invoke(["--devices", "3", "1", "--output", str(output), *arguments], expected=2)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
