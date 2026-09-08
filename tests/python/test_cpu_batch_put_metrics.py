#!/usr/bin/env python3
"""Hardware-independent regression checks for CPU BatchPut measurements."""

import contextlib
import csv
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import test_cpu_batch_put_perf as perf


def records(output):
    return {
        fields["operation"]: fields
        for line in output.splitlines()
        if line.startswith("PERF_RESULT ")
        for fields in [dict(item.split("=", 1) for item in line.split()[1:])]
    }


class MetricsTest(unittest.TestCase):
    def test_percentiles_and_unrounded_bandwidth(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            perf.print_samples("batch_put_transport_total", [9000, 1000, 4000, 1001],
                               5, 8192, 1024, 1024)
            perf.print_samples("batch_put", [10000, 10010, 40000, 90000],
                               5, 8192, 1024, 1024)
        result = records(out.getvalue())
        total = result["batch_put_transport_total"]
        self.assertEqual(total["mean_us"], "3.750")
        self.assertEqual(total["p90_us"], "7.500")
        self.assertEqual(total["p99_us"], "8.850")
        self.assertEqual(total["p50_GBps"], f"{1048576 / 2500.5:.3f}")
        self.assertEqual(result["batch_put"]["p50_GBps"], f"{1048576 / 25005:.3f}")
        self.assertEqual(total["payload_bytes_per_call"], "1048576")
        with contextlib.redirect_stdout(out):
            perf.print_samples("batch_put_transport_total", [1000, 1001], 0, 8192, 1024, 1024)
        total = records(out.getvalue())["batch_put_transport_total"]
        self.assertEqual(total["p50_GBps"], "1048.052")
        rounded_bandwidth = 1048576 / float(total["p50_us"]) / 1000
        self.assertNotEqual(total["p50_GBps"], f"{rounded_bandwidth:.3f}")

    def test_invalid_timings_fail_instead_of_becoming_zero_bandwidth(self):
        valid = dict(total_scope="transport_call", prepare_ns=100,
                     post_ns=200, poll_ns=300, total_ns=1000)
        self.assertEqual(perf.validate_timing(valid, 2000)["total_ns"], 1000)
        for changes in ({"total_scope": None}, {"total_ns": 0}, {"total_ns": 500},
                        {"total_ns": 2001}, {"poll_ns": -1}, {"post_ns": None},
                        {"prepare_ns": 1.5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                perf.validate_timing(dict(valid, **changes), 2000)
        for samples in ([], [0], [-1, 2]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                perf.print_samples("batch_put_transport_total", samples, 0, 32, 32, 128)
        self.assertIsNotNone(perf.validate_args(perf.parse_args(["--miss-count", "0"])))

    def test_preflight_and_warmup_excluded_and_timing_read_before_free(self):
        class FakeRuntime:
            initialized = False

            def __init__(self):
                self.totals = iter([99999999, 88888888, 6001, 10000, 20000, 8000])

            def initialize(self, *args, **kwargs):
                self.initialized = True

            def batch_put(self, keys, sources, offsets, lengths, miss_count):
                self.payload = sources[0][:sum(lengths[0][:miss_count])]
                self.last_batch_timing = dict(total_scope="transport_call", total_ns=next(self.totals),
                                             prepare_ns=100, post_ns=200, poll_ns=300)

            def batch_free(self, keys):
                self.last_batch_timing = None

            def prepare_access_table(self, keys):
                return keys

            def batch_get(self, table, rows, offsets, output_rows, output, token_bytes):
                output[:] = self.payload

            def release_access_table(self, table):
                pass

            def shutdown(self):
                self.initialized = False

        out = io.StringIO()
        with patch.dict(sys.modules, {"kvaf": types.SimpleNamespace(CpuRuntime=FakeRuntime)}), \
             patch.object(perf.time, "perf_counter_ns", side_effect=[
                 0, 60010, 100000, 200000, 300000, 500000, 600000, 680000]), \
             contextlib.redirect_stdout(out):
            self.assertEqual(perf.main(["--request-length", "64", "--miss-count", "32",
                                        "--token-bytes", "128", "--warmup", "1",
                                        "--iterations", "4"]), 0)
        result = records(out.getvalue())
        total = result["batch_put_transport_total"]
        self.assertEqual(total["samples"], "4")
        self.assertEqual(total["mean_us"], "11.000")
        self.assertEqual(total["p50_us"], "9.000")
        self.assertEqual(total["p90_us"], "17.000")
        self.assertEqual(total["p99_us"], "19.700")
        self.assertEqual(total["p50_GBps"], "0.455")
        self.assertEqual(result["batch_put"]["p50_GBps"], "0.046")

    def test_preflight_payload_distinguishes_rows(self):
        for token in (128, 256, 512, 656, 1024, 2048, 4096):
            data = perf.make_source(32 * token, token)
            self.assertEqual(len(data), 32 * token)
            self.assertEqual(len({data[i * token:(i + 1) * token] for i in range(32)}), 32)

    def test_sweep_selects_transport_record_for_all_63_combinations(self):
        # Run the real aggregator with an isolated runner emitting both scopes.
        with tempfile.TemporaryDirectory(prefix="kvaf_metrics_") as tmp:
            root = Path(tmp)
            scripts = root / "tests"
            scripts.mkdir()
            sweep = scripts / "run_cpu_batch_put_miss_count_perf.sh"
            shutil.copy(Path(__file__).resolve().parents[1] / sweep.name, sweep)
            runner = scripts / "run_cpu_batch_put_perf.sh"
            runner.write_text('''#!/usr/bin/env bash
exec "$METRICS_PYTHON" "$METRICS_FIXTURE" "$@"
''')
            runner.chmod(0o755)
            fixture = root / "fixture.py"
            fixture.write_text('''import os, sys
sys.path.insert(0, os.environ["METRICS_MODULE_DIR"])
import test_cpu_batch_put_perf as perf
token = int(sys.argv[sys.argv.index("--token-bytes") + 1])
miss = int(sys.argv[sys.argv.index("--miss-count") + 1])
perf.print_samples("batch_put_transport_total", [1000, 1001], 0, 8192, miss, token)
perf.print_samples("batch_put", [10000, 10010], 0, 8192, miss, token)
''')
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("KVAF_CPU_BATCH_PUT_")}
            env.update(METRICS_PYTHON=sys.executable, METRICS_FIXTURE=str(fixture),
                       METRICS_MODULE_DIR=str(Path(__file__).resolve().parent),
                       KVAF_CPU_BATCH_PUT_SWEEP_LOG_DIR=str(root / "results"))
            run = subprocess.run(["bash", str(sweep), str(root / "build")],
                                 env=env, text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            files = list((root / "results").glob("*.csv"))
            self.assertEqual(len(files), 1)
            with files[0].open() as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], ["token_bytes", "miss_count", "mean_us", "p50_us",
                                       "p90_us", "p99_us", "p50_GBps"])
            pairs = [(token, miss) for token in (128, 256, 512, 656, 1024, 2048, 4096)
                     for miss in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)]
            self.assertEqual(len(rows), 64)
            for row, (token, miss) in zip(rows[1:], pairs):
                self.assertEqual(row, [str(token), str(miss), f"{1.0005:.3f}",
                                       f"{1.0005:.3f}", "1.001", "1.001",
                                       f"{token * miss / 1000.5:.3f}"])
            # Missing records/failed runners must never produce a data row.
            runner.write_text("#!/usr/bin/env bash\nexit 0\n")
            run = subprocess.run(["bash", str(sweep), str(root / "build")],
                                 env=env, text=True, capture_output=True)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn("incomplete CPU BatchPut results", run.stderr)
            self.assertEqual(len(files[0].read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
