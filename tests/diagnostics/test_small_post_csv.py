import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

from summarize_small_post import JETTY_METRICS, compare_runs, main, parse_run


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

    def test_exports_four_csv_files(self):
        self.log("r1_n32_baseline.log", p50_cycles=5000)
        self.log("r1_n32_small-post.log", p50_cycles=2500)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root)]), 0)
        for filename, expected_rows in (("small_post_runs.csv", 2),
                                        ("small_post_pairs.csv", 1),
                                        ("small_post_summary.csv", 1),
                                        ("small_post_jettys.csv", 0)):
            with (self.root / filename).open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), expected_rows)
            if filename == "small_post_pairs.csv":
                self.assertEqual(rows[0]["p50_speedup"], "2.000000")
                self.assertEqual(rows[0]["p50_latency_reduction_pct"], "50.000000")
                self.assertEqual(rows[0]["p50_bandwidth_gain_pct"], "100.000000")

    def timeline_log(self, name, mode="round_robin"):
        path = self.log(name)
        text = path.read_text(encoding="utf-8")
        text = text.replace("KERNEL_PROFILE_CONTRACT ",
                            f"KERNEL_PROFILE_CONTRACT profile_version=3 completion_poll_mode={mode} ")
        small = "small-post" in name
        extra = []
        for core in range(2):
            extra.append(f"KERNEL_PROFILE_SCHEDULE core={core} samples=1000 "
                         f"small_post_max_requests={256 if small else 0} "
                         f"post_width_min={8 if small else 128} post_width_max={8 if small else 128} "
                         f"small_post_active_samples={1000 if small else 0} "
                         f"small_post_groups_per_call={int(small)} fused_groups_per_call={int(small)} "
                         "wqe_build_calls_per_call=1")
            for jetty in range(2):
                active = small or jetty == 0
                reads = (8 if small else 16) if active else 0
                record = {key: "n/a" for key in JETTY_METRICS}
                record.update(core=core, jetty=jetty, samples=1000, completion_poll_mode=mode,
                              active_samples=1000 if active else 0, reads_per_call=reads,
                              posts_per_call=int(active), completions_per_call=int(active),
                              payload_bytes_per_call=reads * 656, polls_per_call=3 if active else 0)
                if active:
                    record.update({key: 1 for key in JETTY_METRICS if key.endswith(("_mean", "_p50", "_p99"))})
                    if mode == "serial": record["probe_active_us_mean"] = "n/a"
                extra.append("KERNEL_PROFILE_JETTY " + " ".join(f"{k}={v}" for k, v in record.items()))
        path.write_text(text + "\n".join(extra) + "\n", encoding="utf-8")
        return path

    def test_actual_schedule_and_per_jetty_csv(self):
        baseline = parse_run(self.timeline_log("r1_n32_baseline.log"))
        small = parse_run(self.timeline_log("r1_n32_small-post.log"))
        self.assertEqual(small["fused_groups_per_call"], 2)
        self.assertEqual(small["wqe_build_calls_per_call"], 2)
        self.assertEqual(baseline["_jettys"][1]["last_db_to_last_cqe_us_mean"], "")
        self.assertEqual(small["_jettys"][1]["reads_per_call"], 8)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root)]), 0)
        with (self.root / "small_post_jettys.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 8)

    def test_rejects_polling_mode_mismatch_and_incomplete_jetty_evidence(self):
        baseline = parse_run(self.timeline_log("r1_n32_baseline.log", "serial"))
        small_path = self.timeline_log("r1_n32_small-post.log")
        small = parse_run(small_path)
        with self.assertRaisesRegex(ValueError, "completion_poll_mode"):
            compare_runs([baseline, small])
        text = small_path.read_text(encoding="utf-8")
        small_path.write_text(text.replace("reads_per_call=8", "reads_per_call=9"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reads_per_call"):
            parse_run(small_path)

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
