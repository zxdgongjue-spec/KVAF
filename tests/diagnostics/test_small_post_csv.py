import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

from summarize_small_post import compare_runs, main, parse_run


class SmallPostCsvTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def log(self, name, p50_cycles=3485.5, seed=20260903, clock=500,
            first_delay_cycles=None):
        config = dict(
            token_bytes=656, miss_count=32, aiv_cores=2, jettys_per_core=2,
            request_length=32768, table_capacity=16384, tokens_per_block=128,
            max_inflight_total=2048, source_access_pattern="random",
            output_access_pattern="contiguous", sort_input_indices=0,
            seed=seed, warmup=20, iterations=1000, npu_eids="auto", eid_paths=1,
            peer_ub_device="auto", adaptive_ub_device=0,
            adaptive_ub_miss_count=2048, kernel_profile_clock_mhz=clock,
        )
        lines = ["BATCH_GET_PERF_CONFIG " + " ".join(f"{key}={value}" for key, value in config.items())]
        lines.append(f"KERNEL_PROFILE_CONTRACT clock_mhz={clock} aiv_cores=2 "
                     "jettys_per_core=2 request_length=32768 miss_count=32")
        stages = {"total": p50_cycles, "wall": p50_cycles + 100,
                  "post": 600, "completion_wait": 2000}
        if first_delay_cycles is not None:
            stages.update(first_doorbell=first_delay_cycles,
                          transfer_window=p50_cycles - first_delay_cycles)
        for stage, cycles in stages.items():
            lines.append(
                f"PERF_RESULT operation=batch_get_kernel_{stage} samples=1000 "
                f"warmup_excluded=20 mean_ms=0.007 p50_ms=0.007 "
                f"mean_cycles={cycles * 1.1} p50_cycles={cycles} "
                f"p90_cycles={cycles * 1.2} p99_cycles={cycles * 1.5}"
            )
        posts = 2 if "small-post" in name else 1
        for core in range(2):
            lines.append(f"KERNEL_PROFILE_CORE core={core} samples=1000 "
                         f"active_requests_mean=16 total_posts_mean={posts} "
                         f"completion_count_mean={posts}")
        lines += ["BATCH_GET_PERF_PHASE_DONE phase=batch_get",
                  "BATCH_GET_PERF_PHASE_DONE phase=shutdown"]
        path = self.root / name
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_uses_raw_cycles_and_logged_clock(self):
        run = parse_run(self.log("r1_n32_baseline.log"))
        self.assertAlmostEqual(run["p50_us"], 6.971)
        self.assertAlmostEqual(run["p50_GBps"], 20992 / 6971)
        self.assertEqual(run["posts_per_call"], 2)
        self.assertEqual(run["completions_per_call"], 2)
        self.assertEqual(run["first_doorbell_p50_us"], "")

    def test_timeline_metrics_and_paired_summary(self):
        baseline = parse_run(self.log("r1_n32_baseline.log", first_delay_cycles=500))
        small = parse_run(self.log("r1_n32_small-post.log", first_delay_cycles=250))
        self.assertEqual(baseline["first_doorbell_p50_us"], 1)
        self.assertEqual(small["first_doorbell_p50_us"], 0.5)
        pairs, summary = compare_runs([baseline, small])
        self.assertEqual(pairs[0]["first_doorbell_p50_change_us"], -0.5)
        self.assertEqual(summary[0]["first_doorbell_rounds"], 1)
        self.assertEqual(summary[0]["transfer_window_rounds"], 1)
        self.assertEqual(summary[0]["median_first_doorbell_p50_change_us"], -0.5)

    def test_mixed_old_new_timeline_is_not_treated_as_zero(self):
        baseline = parse_run(self.log("r1_n32_baseline.log"))
        small = parse_run(self.log("r1_n32_small-post.log", first_delay_cycles=250))
        pairs, summary = compare_runs([baseline, small])
        self.assertEqual(small["first_doorbell_p50_us"], 0.5)
        self.assertEqual(pairs[0]["baseline_first_doorbell_p50_us"], "")
        self.assertEqual(pairs[0]["small_post_first_doorbell_p50_us"], "")
        self.assertEqual(pairs[0]["first_doorbell_p50_change_us"], "")
        self.assertEqual(summary[0]["first_doorbell_rounds"], 0)
        self.assertEqual(summary[0]["median_first_doorbell_p50_change_us"], "")

    def test_exports_three_csv_files(self):
        self.log("r1_n32_baseline.log", p50_cycles=5000)
        self.log("r1_n32_small-post.log", p50_cycles=2500)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root)]), 0)
        for filename, expected_rows in (("small_post_runs.csv", 2),
                                        ("small_post_pairs.csv", 1),
                                        ("small_post_summary.csv", 1)):
            with (self.root / filename).open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), expected_rows)
            if filename == "small_post_pairs.csv":
                self.assertEqual(rows[0]["p50_speedup"], "2.000000")
                self.assertEqual(rows[0]["p50_latency_reduction_pct"], "50.000000")
                self.assertEqual(rows[0]["p50_bandwidth_gain_pct"], "100.000000")

    def test_summary_is_median_of_round_statistics_and_paired_speedups(self):
        runs = [
            parse_run(self.log("r1_n32_baseline.log", p50_cycles=5000)),
            parse_run(self.log("r1_n32_small-post.log", p50_cycles=2500)),
            parse_run(self.log("r2_n32_baseline.log", p50_cycles=15000)),
            parse_run(self.log("r2_n32_small-post.log", p50_cycles=15000)),
        ]
        pairs, summary = compare_runs(runs)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(summary[0]["rounds"], 2)
        self.assertEqual(summary[0]["median_baseline_p50_us"], 20)
        self.assertEqual(summary[0]["median_small_post_p50_us"], 17.5)
        self.assertEqual(summary[0]["median_p50_speedup"], 1.5)

    def test_rejects_mismatched_configuration_before_csv_output(self):
        self.log("r1_n32_baseline.log", seed=1)
        self.log("r1_n32_small-post.log", seed=2)
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main([str(self.root)]), 2)
        self.assertIn("seed", errors.getvalue())
        self.assertEqual(list(self.root.glob("*.csv")), [])

    def test_rejects_missing_pair_and_duplicate_variant(self):
        run = parse_run(self.log("r1_n32_baseline.log"))
        with self.assertRaisesRegex(ValueError, "missing baseline/small-post pair"):
            compare_runs([run])
        with self.assertRaisesRegex(ValueError, "duplicate baseline run"):
            compare_runs([run, run])

    def test_rejects_truncated_failed_and_unprofiled_logs(self):
        path = self.log("r1_n32_baseline.log")
        original = path.read_text(encoding="utf-8")
        variants = (
            original.replace("BATCH_GET_PERF_PHASE_DONE phase=shutdown", ""),
            original + "BATCH_GET_PERF_UNHANDLED_ERROR type=RuntimeError\n",
            "\n".join(line for line in original.splitlines()
                      if not line.startswith("KERNEL_PROFILE_CONTRACT ")),
            original.replace("operation=batch_get_kernel_post", "operation=unknown"),
            original.replace("active_requests_mean=16", "active_requests_mean=15"),
            original.replace("samples=1000", "samples=999"),
        )
        for content in variants:
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    parse_run(path)

    def test_rejects_filename_miss_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "filename miss_count"):
            parse_run(self.log("r1_n64_baseline.log"))


if __name__ == "__main__":
    unittest.main()
