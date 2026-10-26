"""Host-only checks of measured Jetty accounting, timings and CSV export."""

import contextlib
import copy
import csv
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import batch_get_jetty_profile as report
import test_npu_batch_get_perf as perf


def sample(end):
    jettys = []
    for slot, (requests, priority) in enumerate(((128, 3), (1, 13), (0, 15))):
        jettys.append(dict(requests=requests, posts=int(requests > 0),
                          first_doorbell_cycle=1000 if requests else 0,
                          last_doorbell_cycle=1000 if requests else 0,
                          first_completion_cycle=end if requests else 0,
                          last_completion_cycle=end if requests else 0,
                          completion_count=int(requests > 0), priority=priority,
                          jetty_id=100 + slot, eid_path=slot % 2))
    return [dict(core=0, active_requests=129, completion_count=2,
                 kernel_start_cycle=100, kernel_finish_cycle=10000,
                 jettys=jettys, jetty_completion_wait_cycles=[100, 20, 0],
                 jetty_completion_polls=[5, 2, 0])]


class JettyProfileTests(unittest.TestCase):
    def rows(self, samples=None, **options):
        arguments = dict(clock_mhz=1000, npu_core_count=1, jettys_per_core=3,
                         token_bytes=1000, miss_count=129, request_length=256,
                         max_inflight_total=2048, warmup=3)
        arguments.update(options)
        return report.summarize(samples if samples is not None else [sample(2000), sample(5000)], **arguments)

    def test_actual_bytes_and_ratio_of_sums(self):
        row = self.rows()[0]
        self.assertEqual(row["requests_total"], 256)
        self.assertEqual(row["bytes_total"], 256000)
        self.assertEqual(row["requests_mean"], 128)
        self.assertEqual(row["observed_window_us_mean"], 2.5)
        self.assertEqual(row["observed_window_us_p50"], 2.5)
        self.assertAlmostEqual(row["observed_window_us_p99"], 3.97)
        self.assertAlmostEqual(row["effective_GBps"], 51.2)  # not mean(128, 32) = 80
        self.assertEqual(row["first_post_us_mean"], .9)
        self.assertEqual(row["samples"], 2)
        self.assertEqual(row["warmup_excluded"], 3)
        self.assertEqual(self.rows()[1]["bytes_total"], 2000)

    def test_unused_jetty_is_not_zero_latency_or_infinite_throughput(self):
        row = self.rows()[2]
        self.assertEqual(row["priority"], 15)
        self.assertEqual(row["active_samples"], 0)
        self.assertEqual(row["requests_total"], 0)
        self.assertEqual(row["effective_GBps"], "")
        self.assertEqual(row["observed_window_us_p50"], "")

    def test_priority_is_observed_metadata_not_current_environment(self):
        with patch.dict(os.environ, {"KVAF_BATCH_GET_JETTY_PRIORITIES": "0"}):
            self.assertEqual([row["priority"] for row in self.rows()], [3, 13, 15])

    def test_zero_miss_launch_keeps_resource_rows(self):
        profiles = sample(2000)
        profiles[0].update(active_requests=0, completion_count=0,
                           jetty_completion_wait_cycles=[0, 0, 0],
                           jetty_completion_polls=[0, 0, 0])
        for jetty in profiles[0]["jettys"]:
            for key in ("requests", "posts", "completion_count", "first_doorbell_cycle",
                        "last_doorbell_cycle", "first_completion_cycle", "last_completion_cycle"):
                jetty[key] = 0
        rows = self.rows([profiles], miss_count=0)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["bytes_total"] == 0 and row["effective_GBps"] == "" for row in rows))

    def test_multiple_cores_and_unsorted_snapshot(self):
        profiles = sample(2000)
        second = copy.deepcopy(profiles[0])
        second["core"] = 1
        profiles.insert(0, second)
        rows = self.rows([profiles], npu_core_count=2, miss_count=258)
        self.assertEqual([r["entity_index"] for r in rows], list(range(6)))
        self.assertEqual(sum(r["bytes_total"] for r in rows), 258000)

    def test_timing_is_local_to_core_and_allows_pipelined_completions(self):
        profiles = sample(5000)
        profiles[0]["jettys"][0].update(last_doorbell_cycle=4000, first_completion_cycle=2000)
        self.assertEqual(self.rows([profiles])[0]["observed_window_us_mean"], 4)

    def test_accounting_and_timeline_failures(self):
        for key, value in (("requests", 129), ("completion_count", 2),
                           ("last_completion_cycle", 500), ("last_doorbell_cycle", 9000),
                           ("priority", 16)):
            profiles = sample(2000)
            profiles[0]["jettys"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.rows([profiles])
        profiles = sample(2000)
        profiles[0]["jettys"][2]["last_completion_cycle"] = 10
        with self.assertRaisesRegex(ValueError, "Unused"):
            self.rows([profiles])
        with self.assertRaisesRegex(ValueError, "miss_count"):
            self.rows(miss_count=130)

    def test_stale_binding_and_changed_transport_fail(self):
        profiles = sample(2000)
        del profiles[0]["jettys"]
        with self.assertRaisesRegex(ValueError, "rebuild"):
            self.rows([profiles])
        second = sample(5000)
        second[0]["jettys"][0]["priority"] = 10
        with self.assertRaisesRegex(ValueError, "changed"):
            self.rows([sample(2000), second])

    def test_csv_roundtrip_and_merge(self):
        with tempfile.TemporaryDirectory() as folder:
            one, two, merged = (Path(folder) / name for name in ("one.csv", "two.csv", "merged.csv"))
            report.write_csv(one, self.rows())
            report.write_csv(two, self.rows(token_bytes=656))
            with contextlib.redirect_stdout(io.StringIO()):
                report.merge_csv(merged, [one, two])
            with merged.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
                self.assertEqual(reader.fieldnames, list(report.FIELDS))
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[2]["effective_GBps"], "")
            self.assertAlmostEqual(float(rows[0]["effective_GBps"]), 51.2)
            self.assertEqual(rows[3]["token_bytes"], "656")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                report.merge_csv(one, [one])
            two.write_text("wrong,header\n1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema"):
                report.merge_csv(merged, [two])

    def test_csv_option_requires_profile_and_supports_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIn("requires", perf.validate_args(perf.parse_args(["--jetty-csv", "result.csv"])))
            self.assertIsNone(perf.validate_args(perf.parse_args(["--kernel-profile", "--jetty-csv", "result.csv"])))
        with patch.dict(os.environ, {"KVAF_BATCH_GET_JETTY_CSV": "from_env.csv"}):
            self.assertEqual(perf.parse_args(["--kernel-profile"]).jetty_csv, "from_env.csv")


if __name__ == "__main__":
    unittest.main()
