"""Timing boundaries, payload accounting and report validation without NPU."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scatter_perf_common import CPU_MAGIC, PROFILE_MAGIC, profile_metrics, summarize
from test_aiv_scatter_perf import arguments, main, measure_pass, write_results


def profiles(cores=3, requests=19, token=656):
    result = []
    for core in range(cores):
        count = requests * (core + 1) // cores - requests * core // cores
        # AIV absolute cycles are deliberately unrelated to CPU nanoseconds.
        base = 9000000000000 + core * 1000
        result.append(dict(core=core, magic=PROFILE_MAGIC, sequence=99,
                           start_cycle=base, inputs_ready_cycle=base + 10,
                           notify_cycle=base + 20 if not core else 0,
                           done_cycle=base + 100, finish_cycle=base + 140 + core,
                           requests=count, bytes=count * token,
                           cpu_magic=CPU_MAGIC ^ (99 << 32) if not core else 0,
                           cpu_header_ns=500, cpu_descriptor_ns=600,
                           cpu_write_ns=5000, cpu_comm_ns=3000,
                           cpu_post_ns=1000, cpu_cq_wait_ns=1500))
    return result


class ScatterPerfTests(unittest.TestCase):
    def test_decimal_bandwidth_uses_sum_of_times(self):
        stats = summarize([100, 200, 300], 1000)
        self.assertEqual(stats["effective_gbps"], 5)
        self.assertAlmostEqual(stats["p95_ms"], 290 / 1e6)
        for values in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                summarize(values, 1000)

    def test_profile_clock_domains_and_slowest_worker(self):
        result = profile_metrics(profiles(), 3, 19, 656, 1000)
        self.assertEqual(result["cpu_comm_ns"], 3000)
        self.assertEqual(result["aiv_body_ns"], 142)
        self.assertEqual(result["aiv_scatter_ns"], 42)
        self.assertEqual(result["aiv_build_notify_ns"], 10)
        self.assertEqual(result["aiv_notify_to_done_ns"], 80)
        self.assertEqual(profile_metrics(profiles(), 3, 19, 656, 50)["aiv_scatter_ns"], 840)

    def test_idle_workers_do_not_change_payload_numerator(self):
        result = profile_metrics(profiles(8, 1, 32), 8, 1, 32, 1000)
        self.assertGreater(result["aiv_scatter_ns"], 0)

    def test_stale_missing_or_invalid_profiles_fail(self):
        for field, value in (("sequence", 98), ("magic", 0), ("bytes", 1),
                             ("done_cycle", 1), ("notify_cycle", 0),
                             ("cpu_magic", 0), ("cpu_comm_ns", 0), ("cpu_write_ns", 1)):
            records = profiles()
            records[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                profile_metrics(records, 3, 19, 656, 1000)
        with self.assertRaises(ValueError):
            profile_metrics(profiles()[:-1], 3, 19, 656, 1000)
        stale_cpu = profiles()
        for record in stale_cpu:
            record["sequence"] += 1
        with self.assertRaises(ValueError):
            profile_metrics(stale_cpu, 3, 19, 656, 1000)

    def test_snapshot_is_outside_per_call_time(self):
        now = [0]
        called = []
        def call(index):
            called.append(index)
            now[0] += 10
        def snapshot():
            now[0] += 1000
            return {"cpu_comm_ns": 4}
        samples, wall = measure_pass(call, 3, 32, snapshot, lambda: now[0])
        self.assertEqual(called, [0, 1, 2])
        self.assertEqual([s["end_to_end_ns"] for s in samples], [10, 10, 10])
        self.assertEqual([s["snapshot_ns"] for s in samples], [1000, 1000, 1000])
        self.assertEqual(wall, 3030)
        self.assertTrue(all(s["payload_bytes"] == 32 for s in samples))

    def test_no_snapshot_work_in_baseline(self):
        now = [0]
        def call(_):
            now[0] += 10
        samples, wall = measure_pass(call, 2, 64, clock=lambda: now[0])
        self.assertEqual(wall, 20)
        self.assertTrue(all("cpu_comm_ns" not in s for s in samples))

    def test_cli_validation_and_skip_before_hardware_import(self):
        for args in (["--requests", "0"], ["--cores", "9"], ["--iterations", "0"],
                     ["--aiv-clock-mhz", "nan"], ["--token-bytes", "65536"],
                     ["--json", "same", "--csv", "same"],
                     ["--token-bytes", "65535", "--tokens-per-block", "65536"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(args)
        with patch.dict("os.environ", {"KVAF_RUN_NPU_TESTS": "0"}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([]), 77)

    def test_json_and_csv_keep_units_and_raw_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = arguments(["--json", str(root / "result.json"), "--csv", str(root / "result.csv")])
            report = dict(config={"units": "decimal GB/s"}, summary={}, samples={
                "baseline": [dict(iteration=0, payload_bytes=656, end_to_end_ns=1000)],
                "profiled": [dict(iteration=0, payload_bytes=656, end_to_end_ns=2000,
                                  cpu_comm_ns=500, workers=profiles())]})
            write_results(args, report)
            self.assertEqual(json.loads(args.json.read_text()), report)
            with args.csv.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["phase"] for r in rows], ["baseline", "profiled"])
            self.assertEqual(rows[1]["cpu_comm_ns"], "500")
            self.assertNotIn("workers", rows[1])


if __name__ == "__main__":
    unittest.main()
