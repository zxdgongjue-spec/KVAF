"""Host-only checks for profile aggregation and the owned service lifecycle."""
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "access_operator/demo/aicpu_batch_get/python"))
import run_aicpu_gather_read_numa as diagnostic
import benchmark_gather_read as benchmark


def cpu_line(sequence, lane, copy, valid=1):
    return (f"CPU_GATHER_PROFILE lane={lane} sequence={sequence} region=0x1000 descriptor=0x2000 "
            f"bytes=128 threads=1 prepare_us=1 copy_us={copy} first_ready_us=2 total_us={copy + 3} "
            f"cpu_begin={lane} cpu_end={lane} valid={valid}\n")


class ProfileTests(unittest.TestCase):
    def test_excludes_warmup_and_preserves_rounds_and_lanes(self):
        text = "".join(cpu_line(seq, lane, value) for lane in range(2)
                       for seq, value in ((2, 999), (3, 10 + lane), (4, 20 + lane)))
        summary, samples = diagnostic.parse_cpu_profiles(text, 2, 2)
        self.assertEqual([sample["sequence"] for sample in samples], [3, 4])
        self.assertEqual(summary["cpu_copy_us"], 16)
        self.assertEqual(summary["cpu_copy_p50_us"], 16)
        self.assertEqual(len(samples[0]["lanes"]), 2)

    def test_incomplete_failed_and_duplicate_records_fail(self):
        complete = cpu_line(3, 0, 10) + cpu_line(3, 1, 11)
        for text in ("", cpu_line(3, 0, 10), complete + cpu_line(3, 0, 10),
                     cpu_line(3, 0, 10) + cpu_line(3, 1, 11, valid=0)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                diagnostic.parse_cpu_profiles(text, 2, 1)

    def test_benchmark_summary_exposes_ack_and_queries(self):
        def worker(value):
            return {key: value for key in ("request_us", "source_write_us", "notify_us", "read_us",
                                          "wr_build_us", "ready_wait_us", "post_us", "cq_wait_us",
                                          "ack_us", "ready_queries")}
        summary = benchmark.summarize_measurements([
            {"device_span_us": 10, "workers": [worker(2), worker(4)]},
            {"device_span_us": 12, "workers": [worker(8), worker(3)]}], 256)
        self.assertEqual(summary["ack_us"], 6)
        self.assertEqual(summary["ready_queries"], 6)

    def test_busy_port_is_rejected(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaisesRegex(RuntimeError, "existing service"):
                diagnostic.require_free_port(listener.getsockname()[1])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = self.root / "build-cpu/runtime/kvaf_cpu_service"
        self.service.parent.mkdir(parents=True)
        self.service.write_text(f"#!{sys.executable}\n" + '''
import os, sys, time
if '--help' in sys.argv:
    print('--gather-profile')
    sys.exit(0)
with open(os.environ['TEST_SERVICE_PIDS'], 'a') as log:
    log.write(str(os.getpid()) + '\\n')
port = sys.argv[sys.argv.index('--port') + 1]
print('[KVAF_CPU] LISTEN address=127.0.0.1 port=' + port, flush=True)
while True:
    time.sleep(0.1)
''')
        self.service.chmod(0o755)
        self.numactl = self.root / "numactl"
        self.numactl.write_text(f"#!{sys.executable}\n" + '''
import os, sys
if '--hardware' in sys.argv:
    print('fake NUMA topology')
else:
    os.execv(sys.argv[3], sys.argv[3:])
''')
        self.numactl.chmod(0o755)
        self.benchmark = self.root / "benchmark.py"
        self.benchmark.write_text('''
import json, os, sys
from pathlib import Path
if os.environ.get('TEST_FAIL_NPU'):
    sys.exit(3)
def arg(name):
    return sys.argv[sys.argv.index(name) + 1]
output = Path(arg('--profile-output'))
n = int(arg('--iterations'))
log = output.parent / 'cpu.log'
with log.open('a') as stream:
    for lane in range(2):
        for sequence in range(2, n + 4):
            stream.write(f'CPU_GATHER_PROFILE lane={lane} sequence={sequence} region=0x1000 descriptor=0x2000 '
                         f'bytes=128 threads=1 prepare_us=1 copy_us={sequence + lane} first_ready_us=2 '
                         f'total_us={sequence + lane + 3} cpu_begin={lane} cpu_end={lane} valid=1\\n')
summary = dict(device_p50_us=25, p50_GBps=1, cpu_gather_wait_us=5, cq_wait_us=3)
output.write_text(json.dumps(dict(summary=summary,
    profiles=[dict(workers=[dict(bytes=128), dict(bytes=128)]) for _ in range(n)])))
print('FAKE_NPU_PASS')
''')
        self.pids = self.root / "pids"

    def run_matrix(self, fail=False):
        real_run = subprocess.run

        def run(command, **kwargs):
            if command[1:2] == ["-c"]:
                return subprocess.CompletedProcess(command, 0)
            return real_run(command, **kwargs)

        environment = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ["PATH"],
                           TEST_SERVICE_PIDS=str(self.pids))
        if fail:
            environment["TEST_FAIL_NPU"] = "1"
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        with patch.dict(os.environ, environment, clear=True), \
             patch.object(diagnostic.sys, "platform", "linux"), \
             patch.object(diagnostic, "BENCHMARK", self.benchmark), \
             patch.object(diagnostic.shutil, "which", side_effect=lambda name: str(self.numactl) if name == "numactl" else None), \
             patch.object(diagnostic.subprocess, "run", side_effect=run), \
             contextlib.redirect_stdout(io.StringIO()):
            diagnostic.main(["--cpu-build", str(self.root / "build-cpu"), "--npu-build", str(self.root / "npu"),
                             "--workers", "2", "--gather-threads", "2", "--token-bytes", "64", "--miss-count", "4",
                             "--iterations", "2", "--warmup", "1", "--port", str(port),
                             "--output", str(self.root / "report")])

    def assert_services_stopped(self):
        for pid in self.pids.read_text().splitlines():
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid), 0)

    def test_complete_matrix_keeps_logs_and_cleans_its_services(self):
        self.run_matrix()
        self.assert_services_stopped()
        self.assertEqual(len(self.pids.read_text().splitlines()), 4)
        report = self.root / "report"
        self.assertEqual(len((report / "summary.csv").read_text().splitlines()), 5)
        self.assertEqual(len(list(report.glob("run_*/cpu_profiles.json"))), 4)
        self.assertTrue((report / "config.json").exists())

    def test_failed_client_stops_owned_service_and_keeps_failure_log(self):
        with self.assertRaisesRegex(RuntimeError, "NPU benchmark failed"):
            self.run_matrix(fail=True)
        self.assert_services_stopped()
        self.assertTrue((self.root / "report/run_1_cpu_1_mem_1/npu.log").exists())


if __name__ == "__main__":
    unittest.main()
