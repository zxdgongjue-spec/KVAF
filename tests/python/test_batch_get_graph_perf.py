#!/usr/bin/env python3
"""Host checks for graph timing boundaries, cleanup and sweep units."""

import contextlib
import csv
import io
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_npu_batch_get_perf as perf


class GraphTimingFixture:
    def __init__(self, fail_capture=False, fail_replay=False):
        self.now_ms = 0.0
        self.input = -1
        self.output = None
        self.capturing = False
        self.ready = False
        self.reset_done = False
        self.validated_inputs = []
        self.executed_inputs = []
        self.launch_calls = 0
        self.replays = 0
        self.fail_capture = fail_capture
        self.fail_replay = fail_replay
        self.stream = SimpleNamespace(synchronize=lambda: None)
        self.torch = SimpleNamespace(npu=SimpleNamespace(
            stream=lambda stream: contextlib.nullcontext(),
            Event=self.event,
        ))

    def event(self, enable_timing):
        assert enable_timing
        fixture = self

        class Event:
            timestamp = None
            complete = False

            def record(self, stream):
                assert stream is fixture.stream and not fixture.capturing
                self.timestamp = fixture.now_ms
                self.complete = False

            def synchronize(self):
                self.complete = True

            def elapsed_time(self, other):
                assert other.complete
                return other.timestamp - self.timestamp

        return Event()

    def prepare(self):
        assert not self.capturing
        self.input += 1
        self.output = None
        # Large preparation/validation delays expose incorrect timing boundaries.
        self.now_ms += 100.0

    def launch(self):
        self.launch_calls += 1
        if self.capturing:
            if self.fail_capture:
                raise RuntimeError("capture failed")
        else:
            self.execute()

    def execute(self):
        self.now_ms += 0.2375
        self.output = self.input
        self.executed_inputs.append(self.input)

    def validate(self):
        assert self.output == self.input
        self.validated_inputs.append(self.input)
        self.now_ms += 200.0

    def capture_begin(self, stream):
        assert stream is self.stream
        self.capturing = True

    def capture_end(self):
        self.capturing = False
        self.ready = True

    def replay(self, stream):
        assert self.ready and stream is self.stream
        if self.fail_replay:
            raise RuntimeError("replay failed")
        self.replays += 1
        self.execute()

    def reset(self):
        assert not self.capturing
        self.ready = False
        self.reset_done = True

    def measure(self, warmup=2, iterations=3):
        return perf.measure_graph_event_us(
            self.torch, self, self.stream, self.launch, self.prepare,
            self.validate, warmup, iterations,
        )


class GraphPerfTests(unittest.TestCase):
    def test_replay_samples_exclude_capture_inputs_and_validation(self):
        for warmup in (0, 2):
            with self.subTest(warmup=warmup):
                fixture = GraphTimingFixture()
                samples = fixture.measure(warmup=warmup)
                self.assertEqual(len(samples), 3)
                for sample in samples:
                    self.assertAlmostEqual(sample, 237.5)
                self.assertEqual(fixture.launch_calls, 2)
                self.assertEqual(fixture.replays, 1 + warmup + 3)
                self.assertEqual(fixture.validated_inputs, [1, 1 + warmup + 3])
                self.assertEqual(fixture.executed_inputs, list(range(2 + warmup + 3)))
                self.assertTrue(fixture.reset_done)

    def test_graph_is_reset_after_capture_or_replay_error(self):
        for failure in ("capture", "replay"):
            with self.subTest(failure=failure):
                fixture = GraphTimingFixture(
                    fail_capture=failure == "capture", fail_replay=failure == "replay",
                )
                with self.assertRaisesRegex(RuntimeError, failure):
                    fixture.measure()
                self.assertFalse(fixture.capturing)
                self.assertTrue(fixture.reset_done)

    def test_mode_validation_and_environment_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(perf.parse_args([]).execution_mode, "eager")
            args = perf.parse_args(["--execution-mode", "graph", "--kernel-profile"])
            self.assertIn("--no-kernel-profile", perf.validate_args(args))
            args = perf.parse_args([
                "--execution-mode", "graph", "--no-kernel-profile",
                "--kernel-profile-clock-mhz", "0",
            ])
            self.assertIsNone(perf.validate_args(args))
        with patch.dict(os.environ, {"KVAF_BATCH_GET_PERF_EXECUTION_MODE": "graph"}, clear=True):
            self.assertEqual(perf.parse_args([]).execution_mode, "graph")
            self.assertEqual(perf.parse_args(["--execution-mode", "eager"]).execution_mode, "eager")

    def test_report_uses_us_for_percentiles_and_bandwidth(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            perf.print_samples("batch_get_graph_event", [125.0, 375.0], 2, 1, 1, 1024)
        fields = dict(field.split("=", 1) for field in output.getvalue().split()[1:])
        self.assertEqual(fields["mean_us"], "250.000")
        self.assertEqual(fields["p50_us"], "250.000")
        self.assertEqual(fields["unit"], "us")
        self.assertEqual(fields["useful_GiBps"], "0.004")
        self.assertNotIn("mean_ms", fields)


class SweepModeTests(unittest.TestCase):
    def run_sweep(self, mode, adaptive=False):
        root = Path(__file__).resolve().parents[2]
        source = (root / "tests/run_remote_cpu_batch_get_miss_count_perf.sh").read_text()
        # Run one real sweep case, with a fake runner replacing NPU hardware.
        for name, values in (
            ("TOKEN_BYTES", "656"), ("MISS_COUNTS", "32"),
            ("AIV_CORES", "2"), ("JETTYS_PER_CORE", "2"),
        ):
            prefix = f"KVAF_BATCH_GET_{name}=("
            start = source.index(prefix)
            end = source.index(")", start)
            source = source[:start] + prefix + values + source[end:]

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            script = directory / "sweep.sh"
            script.write_text(source)
            runner = directory / "runner.py"
            runner.write_text('''#!/usr/bin/env python3
import sys
args = sys.argv[2:]
def option(name, default):
    return args[args.index(name) + 1] if name in args else default
mode = option('--execution-mode', 'eager')
profile_flags = [arg for arg in args if arg in ('--kernel-profile', '--no-kernel-profile')]
if '--adaptive-calibration-only' in args:
    print('BATCH_GET_PERF_ADAPTIVE_CALIBRATION_DONE')
    sys.exit(0)
is_calibration = option('--iterations', '') == '5'
assert profile_flags[-1] == ('--no-kernel-profile' if mode == 'graph' or is_calibration else '--kernel-profile')
operation = 'batch_get_graph_event' if mode == 'graph' else ('batch_get' if is_calibration else 'batch_get_kernel_total')
if '--peer-ub-device' in args:
    print('[KVAF_AIV] UB_DEVICE_REUSED ub_device=' + option('--peer-ub-device', ''))
print('BATCH_GET_PERF_CONFIG execution_mode=' + mode + ' token_bytes=656 miss_count=32 aiv_cores=2 jettys_per_core=2')
print('PERF_RESULT operation=' + operation + ' mean_us=250.125 p50_us=200.000 p90_us=300.000 p99_us=400.000 unit=us useful_GiBps=0.100')
if mode == 'eager' and not is_calibration:
    for stage in ('post', 'completion_wait'):
        print('PERF_RESULT operation=batch_get_kernel_' + stage + ' mean_us=100.000 p50_us=80.000 p90_us=120.000 p99_us=160.000 unit=us')
''')
            runner.chmod(0o755)
            env = {key: value for key, value in os.environ.items() if not key.startswith("KVAF_")}
            env.update(
                KVAF_BATCH_GET_PERF_RUNNER=str(runner),
                KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR=str(directory / "logs"),
                # Event timing must not depend on an AIV clock.
                KVAF_BATCH_GET_KERNEL_CLOCK_MHZ="invalid" if mode == "graph" else "1000",
            )
            command = ["bash", str(script), "/unused/build", "--execution-mode", mode]
            if adaptive:
                command.append("--adaptive-ub-device")
            result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            with (directory / "logs/batch_get_miss_sweep_summary.csv").open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 1)
            return rows[0], result.stdout

    def test_sweep_graph_us_and_empty_diagnostic_columns(self):
        row, output = self.run_sweep("graph", adaptive=True)
        self.assertEqual(row["mean_us"], "250.125")
        self.assertEqual(row["p50_GBps"], "0.105")
        self.assertEqual(row["metric"], "batch_get_graph_event")
        self.assertEqual(row["post_mean_us"], "")
        self.assertEqual(row["selected_ub_device"], "udmac0d1e2")
        self.assertNotIn("stage=kernel_post", output)
        self.assertNotIn("clock_mhz=", output)

    def test_sweep_eager_keeps_kernel_stages_in_us(self):
        row, output = self.run_sweep("eager")
        self.assertEqual(row["mean_us"], "250.125")
        self.assertEqual(row["post_mean_us"], "100.000")
        self.assertEqual(row["completion_wait_p99_us"], "160.000")
        self.assertEqual(row["metric"], "batch_get_kernel_total")
        self.assertIn("stage=kernel_post", output)


if __name__ == "__main__":
    unittest.main()
