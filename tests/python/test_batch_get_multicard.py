"""Exercise multi-card service routing and saved reports without NPU hardware."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tests/run_remote_cpu_batch_get_multicard_perf.sh"

# Use the real single-card parser and controller barrier, replacing device work.
FAKE_WORKER = """import json
import os
import sys
import time

from test_npu_batch_get_perf import parse_args
from utils.get_options import parallelism_field

args = parse_args(["--build-dir", sys.argv[1], *sys.argv[2:]])
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
result = dict(device=args.device_id, peer=args.peer, backend=args.backend,
              start=window["start"], end=window["stop"], payload_bytes=1000000,
              bandwidth_GBps=1000000 / (window["stop"] - window["start"]) / 1e9,
              verified=True, jettys_per_core=args.jettys_per_core)
result[parallelism_field(args.backend)] = args.parallelism
args.contention_result.write_text(json.dumps(result))
"""


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
