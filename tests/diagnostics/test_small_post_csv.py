import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

from summarize_small_post import JETTY_METRICS, compare_jetty_scaling, compare_runs, main, parse_run
from summarize_batch_get import main as lean_main


class SmallPostCsvTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def log(self, name, p50_cycles=3485.5, seed=20260903, clock=500,
            first_delay_cycles=None, miss_count=32, aiv_cores=2, jettys_per_core=2,
            small_post_max_requests=256, p99_cycles=None):
        config = dict(
            token_bytes=656, miss_count=miss_count, aiv_cores=aiv_cores, jettys_per_core=jettys_per_core,
            request_length=32768, table_capacity=16384, tokens_per_block=128,
            max_inflight_total=2048, source_access_pattern="random",
            output_access_pattern="contiguous", sort_input_indices=0,
            seed=seed, warmup=20, iterations=1000, npu_eids="auto", eid_paths=1,
            peer_ub_device="auto", adaptive_ub_device=0,
            adaptive_ub_miss_count=2048, kernel_profile_clock_mhz=clock,
        )
        lines = ["BATCH_GET_PERF_CONFIG " + " ".join(f"{key}={value}" for key, value in config.items())]
        lines.append(f"KERNEL_PROFILE_CONTRACT clock_mhz={clock} aiv_cores={aiv_cores} "
                     f"jettys_per_core={jettys_per_core} request_length=32768 miss_count={miss_count}")
        stages = {"total": p50_cycles, "wall": p50_cycles + 100,
                  "post": 600, "completion_wait": 2000}
        if first_delay_cycles is not None:
            stages.update(first_doorbell=first_delay_cycles,
                          transfer_window=p50_cycles - first_delay_cycles)
        for stage, cycles in stages.items():
            tail = p99_cycles if stage == "total" and p99_cycles is not None else cycles * 1.5
            lines.append(
                f"PERF_RESULT operation=batch_get_kernel_{stage} samples=1000 "
                f"warmup_excluded=20 mean_ms=0.007 p50_ms=0.007 "
                f"mean_cycles={cycles * 1.1} p50_cycles={cycles} "
                f"p90_cycles={cycles * 1.2} p99_cycles={tail}"
            )
        small = "small-post" in name and 0 < miss_count <= small_post_max_requests and jettys_per_core > 1
        for core in range(aiv_cores):
            reads = miss_count * (core + 1) // aiv_cores - miss_count * core // aiv_cores
            width = min(128, (reads + jettys_per_core - 1) // jettys_per_core) if small and reads else 128
            posts = (reads + width - 1) // width
            lines.append(f"KERNEL_PROFILE_CORE core={core} samples=1000 "
                         f"active_requests_mean={reads} total_posts_mean={posts} "
                         f"completion_count_mean={min(posts, jettys_per_core)}")
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
        self.assertEqual(run["reads_per_core_min"], 16)
        self.assertEqual(run["reads_per_core_max"], 16)
        self.assertEqual(run["reads_per_core_mean"], 16)
        self.assertEqual(run["configured_jettys_total"], 4)
        self.assertEqual(run["active_jettys_per_call"], "")

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

    def test_exports_six_csv_files(self):
        self.log("r1_n32_baseline.log", p50_cycles=5000)
        self.log("r1_n32_small-post.log", p50_cycles=2500)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root)]), 0)
        for filename, expected_rows in (("small_post_runs.csv", 2),
                                        ("small_post_pairs.csv", 1),
                                        ("small_post_summary.csv", 1),
                                        ("small_post_jettys.csv", 0),
                                        ("small_post_jetty_pairs.csv", 0),
                                        ("small_post_jetty_summary.csv", 0)):
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
        self.assertEqual(baseline["active_jettys_per_call"], 2)
        self.assertEqual(small["active_jettys_per_call"], 4)
        self.assertEqual(small["small_post_max_requests"], 256)
        self.assertEqual(small["post_width_min"], 8)
        self.assertEqual(small["post_width_max"], 8)
        self.assertEqual(baseline["_jettys"][1]["last_db_to_last_cqe_us_mean"], "")
        self.assertEqual(small["_jettys"][1]["reads_per_call"], 8)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root)]), 0)
        with (self.root / "small_post_jettys.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 8)

    def test_recovers_native_log_attached_to_na_value(self):
        path = self.timeline_log("r1_n32_baseline.log")
        text = path.read_text(encoding="utf-8")
        text = text.replace("last_db_to_last_cqe_us_p99=n/a",
                            "last_db_to_last_cqe_us_p99=n/a[KVAF_AIV] "
                            "DIRECT_HCCP_TEARDOWN phase=delete_transport\n", 1)
        path.write_text(text, encoding="utf-8")
        run = parse_run(path)
        self.assertEqual(run["_jettys"][1]["last_db_to_last_cqe_us_p99"], "")
        self.assertEqual(run["_native_interleavings_recovered"], 1)
        self.assertEqual(run["active_jettys_per_call"], 2)

    def management_log(self, name, enabled):
        path = self.timeline_log(name)
        text = path.read_text(encoding="utf-8")
        active = 2 if "small-post" in name else 1
        managed = active if enabled else 2
        extension = (
            f"schedule_metadata_version=1 lean_schedule_enabled={int(enabled)} "
            f"configured_jettys=2 managed_jettys_min={managed} managed_jettys_max={managed} "
            f"managed_jettys_per_call={managed} single_group_fast_path_samples={1000 if enabled else 0} "
        )
        path.write_text(text.replace("KERNEL_PROFILE_SCHEDULE ",
                                     "KERNEL_PROFILE_SCHEDULE " + extension), encoding="utf-8")
        return path

    def test_management_proof_and_backward_compatible_export(self):
        baseline = parse_run(self.management_log("r1_n32_baseline.log", False))
        small = parse_run(self.management_log("r1_n32_small-post.log", True))
        self.assertEqual(baseline["managed_jettys_per_call"], 4)
        self.assertEqual(small["managed_jettys_per_call"], 4)
        self.assertEqual(small["single_group_fast_path_calls"], 2)
        self.assertEqual(small["_jettys"][0]["managed_jettys_min"], 2)
        pairs, summary = compare_runs([baseline, small])
        self.assertEqual(pairs[0]["baseline_lean_schedule_enabled"], 0)
        self.assertEqual(summary[0]["median_small_post_single_group_fast_path_calls"], 2)
        # An older V3 log has no extension: leave it missing, never infer OFF.
        old_base = parse_run(self.timeline_log("r2_n32_baseline.log"))
        old_small = parse_run(self.timeline_log("r2_n32_small-post.log"))
        self.assertEqual(old_base["lean_schedule_enabled"], "")
        self.assertEqual(len(compare_runs([baseline, small, old_base, old_small])[1]), 2)
        mixed_base = parse_run(self.timeline_log("r3_n32_baseline.log"))
        mixed_small = parse_run(self.management_log("r3_n32_small-post.log", True))
        mixed_pair = compare_runs([mixed_base, mixed_small])[0][0]
        self.assertEqual(mixed_pair["baseline_lean_schedule_enabled"], "")
        self.assertEqual(mixed_pair["small_post_lean_schedule_enabled"], 1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root)]), 0)

    def test_trims_management_without_changing_active_reads(self):
        run = parse_run(self.management_log("r1_n32_baseline.log", True))
        self.assertEqual(run["managed_jettys_per_call"], 2)
        self.assertEqual(run["active_jettys_per_call"], 2)
        self.assertEqual(run["single_group_fast_path_calls"], 2)
        path = self.management_log("r1_n32_small-post.log", True)
        original = path.read_text(encoding="utf-8")
        for old, new in (("managed_jettys_per_call=2", "managed_jettys_per_call=0"),
                         ("single_group_fast_path_samples=1000", "single_group_fast_path_samples=1001"),
                         ("configured_jettys=2", "configured_jettys=8"),
                         ("schedule_metadata_version=1 ", "")):
            with self.subTest(old=old):
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                with self.assertRaises(ValueError):
                    parse_run(path)

    def test_fixed_post_lean_export_has_unambiguous_names(self):
        self.management_log("r1_n32_baseline.log", False)
        self.management_log("r1_n32_lean.log", True)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(lean_main([str(self.root)]), 0)
        files = list(self.root.glob("batch_get_*.csv"))
        self.assertEqual(len(files), 6)
        with (self.root / "batch_get_pairs.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            row = next(reader)
            self.assertIn("lean_p50_us", reader.fieldnames)
            self.assertFalse(any("small_post" in name or "fused" in name for name in reader.fieldnames))
        self.assertEqual(row["baseline_lean_schedule_enabled"], "0")
        self.assertEqual(row["lean_lean_schedule_enabled"], "1")
        self.assertEqual(row["baseline_active_jettys_per_call"], row["lean_active_jettys_per_call"])
        self.assertEqual(row["lean_managed_jettys_per_call"], "2.000000")
        self.assertEqual(row["lean_single_group_fast_path_calls"], "2.000000")
        self.assertEqual(list(self.root.glob("small_post_*.csv")), [])

    def test_fixed_post_export_rejects_wrong_build_or_old_experiment(self):
        self.management_log("r1_n32_baseline.log", False)
        lean = self.management_log("r1_n32_lean.log", True)
        text = lean.read_text(encoding="utf-8")
        lean.write_text(text.replace("lean_schedule_enabled=1", "lean_schedule_enabled=0"), encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(lean_main([str(self.root)]), 2)
        self.assertEqual(list(self.root.glob("*.csv")), [])
        lean.write_text(text, encoding="utf-8")
        self.management_log("r1_n32_small-post.log", True)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(lean_main([str(self.root)]), 2)

    def test_rejoins_numeric_fields_keys_and_completion_markers(self):
        path = self.timeline_log("r1_n32_baseline.log")
        original = path.read_text(encoding="utf-8")
        # Native fields deliberately overlap benchmark field names. They must
        # remain separate rather than overwrite the actual samples/core values.
        native = "[KVAF_AIV] DIRECT_HCCP_TEARDOWN core=999 samples=7\n"
        cases = (
            ("p50_cycles=3485.5", "p50_cycles=34" + native + "85.5"),
            ("active_samples=0", "active_sam" + native + "ples=0"),
            ("phase=shutdown", "phase=shut" + native + "down"),
            ("p50_cycles=3485.5", "p50_cycles=34" + native + native + "85.5"),
        )
        for old, new in cases:
            with self.subTest(old=old, new=new):
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                run = parse_run(path)
                self.assertAlmostEqual(run["p50_us"], 6.971)
                self.assertEqual(run["_jettys"][1]["active_samples"], 0)
                self.assertEqual(run["posts_per_call"], 2)

    def test_retains_native_device_selection_order_after_recovery(self):
        path = self.timeline_log("r1_n32_baseline.log")
        text = path.read_text(encoding="utf-8")
        text = "[KVAF_AIV] UB_DEVICE_SELECTED ub_device=first\n" + text
        text = text.replace("last_db_to_last_cqe_us_p99=n/a",
                            "last_db_to_last_cqe_us_p99=n/a[KVAF_AIV] "
                            "UB_DEVICE_REUSED ub_device=second\n", 1)
        path.write_text(text, encoding="utf-8")
        self.assertEqual(parse_run(path)["selected_ub_device"], "second")

    def test_does_not_hide_corrupt_values_or_ambiguous_interleaving(self):
        path = self.timeline_log("r1_n32_baseline.log")
        original = path.read_text(encoding="utf-8")
        cases = (
            original.replace("last_db_to_last_cqe_us_p99=n/a",
                             "last_db_to_last_cqe_us_p99=broken[KVAF_AIV] teardown=1\n", 1),
            original.replace("last_db_to_last_cqe_us_p99=n/a",
                             "last_db_to_last_cqe_us_p99=n/a[UNKNOWN] teardown=1\n", 1),
            original.replace("KERNEL_PROFILE_JETTY ", "[KVAF_AIV] partial KERNEL_PROFILE_JETTY ", 1),
            original + "[KVAF_AIV] partial PERF_RESULT operation=batch_get_kernel_setup\n",
            original.rstrip("\n") + "[KVAF_AIV] truncated",
            original.replace("BATCH_GET_PERF_PHASE_DONE phase=shutdown", ""),
        )
        for text in cases:
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    parse_run(path)

    def test_exports_existing_interleaved_logs_without_modifying_them(self):
        baseline = self.timeline_log("r1_n32_baseline.log")
        self.timeline_log("r1_n32_small-post.log")
        text = baseline.read_text(encoding="utf-8").replace(
            "last_db_to_last_cqe_us_p99=n/a", "last_db_to_last_cqe_us_p99=n/a"
            "[KVAF_AIV] DIRECT_HCCP_TEARDOWN phase=delete_transport\n", 1)
        baseline.write_text(text, encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([str(self.root)]), 0)
        self.assertIn("SMALL_POST_LOG_RECOVERED", output.getvalue())
        self.assertEqual(baseline.read_text(encoding="utf-8"), text)
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

    def test_accepts_jetty_and_core_jetty_filename_formats(self):
        for name in ("r1_n32_2_baseline.log", "r1_n32_c2_j2_small-post.log"):
            with self.subTest(name=name):
                run = parse_run(self.log(name))
                self.assertEqual(run["round"], 1)
                self.assertEqual(run["jettys_per_core"], "2")
                self.assertEqual(run["variant"], "small-post" if "small-post" in name else "baseline")

    def test_rejects_filename_core_and_jetty_mismatch(self):
        for name, field in (("r1_n32_1_baseline.log", "jettys_per_core"),
                            ("r1_n32_c2_j4_baseline.log", "jettys_per_core"),
                            ("r1_n32_c1_j2_baseline.log", "aiv_cores")):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "filename " + field):
                    parse_run(self.log(name))

    def test_sweep_preserves_core_jetty_shapes_and_uses_matching_reference(self):
        runs = []
        for cores in (1, 2):
            for jettys in (1, 2, 4, 8):
                for variant in ("baseline", "small-post"):
                    path = self.log(f"r1_n128_c{cores}_j{jettys}_{variant}.log",
                                    miss_count=128, aiv_cores=cores, jettys_per_core=jettys,
                                    p50_cycles=5000 * cores if variant == "baseline" else 4000 * cores)
                    runs.append(parse_run(path))
        ab_pairs, ab_summary = compare_runs(runs)
        self.assertEqual(len(ab_pairs), 8)
        self.assertEqual(len(ab_summary), 8)
        pairs, summary = compare_jetty_scaling(runs)
        self.assertEqual(len(pairs), 16)
        self.assertEqual(len(summary), 16)
        for pair in pairs:
            cores = int(pair["aiv_cores"])
            self.assertEqual(pair["reads_per_core_mean"], 128 / cores)
            self.assertEqual(pair["reference_p50_us"], 10 * cores)
            self.assertIn(f"c{cores}_j1_baseline.log", pair["reference_log"])
            expected_gain = 20 if pair["variant"] == "small-post" else 0
            self.assertAlmostEqual(pair["p50_latency_reduction_pct"], expected_gain)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root)]), 0)
        with (self.root / "small_post_jetty_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 16)

    def test_uneven_per_core_read_counts(self):
        run = parse_run(self.log("r1_n257_c2_j2_baseline.log", miss_count=257))
        self.assertEqual(run["reads_per_core_min"], 128)
        self.assertEqual(run["reads_per_core_max"], 129)
        self.assertEqual(run["reads_per_core_mean"], 128.5)

    def test_jetty_summary_reports_paired_wins_and_tail_regressions(self):
        runs = []
        for round_id, candidate_p50 in ((1, 4000), (2, 6000)):
            for variant in ("baseline", "small-post"):
                runs.append(parse_run(self.log(f"r{round_id}_n32_c2_j1_{variant}.log",
                                               jettys_per_core=1, p50_cycles=5000)))
                runs.append(parse_run(self.log(f"r{round_id}_n32_c2_j2_{variant}.log",
                                               p50_cycles=candidate_p50, p99_cycles=9000)))
        _, summary = compare_jetty_scaling(runs)
        row = next(row for row in summary if row["variant"] == "small-post" and row["jettys_per_core"] == "2")
        self.assertEqual(row["paired_rounds"], 2)
        self.assertEqual(row["round_ids"], "1;2")
        self.assertEqual(row["p50_win_rounds"], 1)
        self.assertEqual(row["p99_regression_rounds"], 2)
        self.assertAlmostEqual(row["median_p50_latency_reduction_pct"], 0)
        self.assertAlmostEqual(row["worst_p50_latency_reduction_pct"], -20)
        self.assertAlmostEqual(row["worst_p99_latency_reduction_pct"], -20)

    def test_missing_jetty_reference_never_uses_another_core_or_round(self):
        runs = [parse_run(self.log("r1_n32_c1_j1_baseline.log", aiv_cores=1, jettys_per_core=1)),
                parse_run(self.log("r1_n32_c2_j2_baseline.log")),
                parse_run(self.log("r2_n32_c1_j2_baseline.log", aiv_cores=1))]
        pairs, summary = compare_jetty_scaling(runs)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(summary), 1)
        self.assertEqual(pairs[0]["candidate_log"], pairs[0]["reference_log"])

    def test_jetty_reference_configuration_mismatch_prevents_csv_output(self):
        for jettys in (1, 2):
            for variant in ("baseline", "small-post"):
                self.log(f"r1_n32_c2_j{jettys}_{variant}.log", jettys_per_core=jettys, seed=jettys)
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(main([str(self.root)]), 2)
        self.assertIn("Jetty reference configuration mismatch", errors.getvalue())
        self.assertEqual(list(self.root.glob("*.csv")), [])

    def test_duplicate_names_do_not_create_independent_samples(self):
        runs = [parse_run(self.log("r1_n32_baseline.log")),
                parse_run(self.log("r1_n32_2_baseline.log"))]
        with self.assertRaisesRegex(ValueError, "duplicate baseline run"):
            compare_runs(runs)

    def test_different_compiled_limits_are_separate_summary_groups(self):
        runs = []
        for round_id, limit in ((1, 256), (2, 1024)):
            for variant in ("baseline", "small-post"):
                path = self.timeline_log(f"r{round_id}_n32_{variant}.log")
                if variant == "small-post":
                    text = path.read_text(encoding="utf-8").replace(
                        "small_post_max_requests=256", f"small_post_max_requests={limit}")
                    path.write_text(text, encoding="utf-8")
                runs.append(parse_run(path))
        _, summary = compare_runs(runs)
        self.assertEqual(len(summary), 2)
        self.assertEqual({row["rounds"] for row in summary}, {1})


if __name__ == "__main__":
    unittest.main()
