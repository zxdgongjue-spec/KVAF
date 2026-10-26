"""Hardware-free checks for profile attribution and diagnostic interpretation."""

from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("analyze_ddr_init", Path(__file__).with_name("analyze_ddr_init.py"))
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def profile(event, phase=None, *, role="cpu", pid=10, attempt=1, **fields):
    result = {"event": event, "role": role, "pid": pid, "attempt": attempt,
              "bytes": 64 * 1024**3, "candidate": 0, "page_mode": "thp"}
    if phase:
        result["phase"] = phase
    result.update(fields)
    return "KVAF_INIT_PROFILE " + json.dumps(result)


def probe(event, **fields):
    return "DDR_INIT_PROBE " + json.dumps({"event": event, "pid": 20, "device": 7, **fields})


def region(*, pid=10, attempt=1, role="cpu", session=9, base=0x200000):
    return profile("region", role=role, pid=pid, attempt=attempt,
                   session=session, base=base)


class AnalyzeDdrInitTests(unittest.TestCase):
    def parse(self, *logs):
        with tempfile.TemporaryDirectory() as tmp:
            inputs = []
            for index, (role, lines) in enumerate(logs):
                path = Path(tmp) / f"{role}-{index}.log"
                path.write_text("\n".join(lines) + "\n")
                inputs.append((role, path))
            records, warnings, errors = ANALYZER.read_logs(inputs)
            return ANALYZER.analyze(records, warnings, errors)

    def test_mixed_log_completed_probe_not_shutdown_or_eid_count(self):
        report = self.parse(("npu", [
            probe("config", allocated_bytes=64 * 1024**3),
            probe("done", operation="set_device", elapsed_s=20.0),
            probe("begin", operation="initialize"),
            "[KVAF_AIV] PEER_LINK_NPU_EID_DISCOVERED eid=first",
            "[KVAF_AIV] PEER_LINK_NPU_EID_DISCOVERED eid=second",
            probe("done", operation="initialize", elapsed_s=7.0),
            probe("ready"),
            probe("done", operation="shutdown", elapsed_s=0.12),
        ]))
        run = report["runs"][0]
        self.assertEqual(run["initialize_elapsed_s"], 7.0)
        self.assertEqual(run["status"], "success")
        self.assertIsNone(run["retry_count"])
        self.assertIsNone(run["candidate_attempt_count"])
        self.assertIsNone(run["additional_candidate_attempts"])
        self.assertEqual(report["error_signals"], [])
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            ANALYZER.print_report(report)
        self.assertIn("candidates=? retries=?", stdout.getvalue())

    def test_done_without_ready_is_not_success(self):
        report = self.parse(("npu", [probe("done", operation="initialize", elapsed_s=1.0)]))
        self.assertEqual(report["runs"][0]["status"], "done_without_ready")

    def test_explicit_failed_candidate_then_success_is_one_retry(self):
        lines = [probe("config"), probe("begin", operation="initialize")]
        for attempt, outcome in ((1, "failed"), (2, "success")):
            lines.extend([
                profile("begin", "candidate_initialize", role="npu", pid=20, attempt=attempt),
                profile("end", "candidate_initialize", role="npu", pid=20, attempt=attempt,
                        elapsed_ms=300.0, outcome=outcome),
            ])
        lines.extend([probe("done", operation="initialize", elapsed_s=.6), probe("ready")])
        report = self.parse(("npu", lines))
        self.assertEqual(report["runs"][0]["retry_count"], 1)
        self.assertEqual(report["runs"][0]["status"], "success")
        summary = report["phase_summaries"][0]
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["failed_count"], 1)

    def test_multiple_successful_candidate_probes_are_not_retries(self):
        lines = [probe("config")]
        for attempt in (1, 2):
            lines.append(profile("end", "candidate_initialize", role="npu", pid=20,
                                 attempt=attempt, elapsed_ms=300, outcome="success"))
        report = self.parse(("npu", lines))
        self.assertEqual(report["runs"][0]["candidate_attempt_count"], 2)
        self.assertEqual(report["runs"][0]["additional_candidate_attempts"], 1)
        self.assertEqual(report["runs"][0]["retry_count"], 0)

    def test_pending_candidate_begin_counts_as_an_observed_attempt(self):
        report = self.parse(("npu", [probe("config"), probe("begin", operation="initialize"),
                                    profile("begin", "candidate_initialize", role="npu", pid=20)]))
        run = report["runs"][0]
        self.assertEqual(run["candidate_attempt_count"], 1)
        self.assertEqual(run["additional_candidate_attempts"], 0)
        self.assertEqual(run["retry_count"], 0)
        self.assertEqual(run["status"], "incomplete")

    def test_unique_region_correlation_assigns_cpu_mode(self):
        report = self.parse(
            ("cpu", [profile("end", "memset", elapsed_ms=1500, outcome="success", page_mode="no_thp"),
                     region().replace('"page_mode": "thp"', '"page_mode": "no_thp"')]),
            ("npu", [region(role="npu", pid=20, attempt=4).replace('"base": 2097152', '"base": "0x200000"'),
                     profile("end", "transport_initialize", role="runtime", pid=20,
                             attempt=4, elapsed_ms=20, page_mode="na")]))
        self.assertEqual(report["correlations"][0]["confidence"], "inferred_unique")
        self.assertEqual(report["attempts"][1]["page_mode"], "no_thp")
        self.assertEqual(report["attempts"][2]["page_mode"], "no_thp")
        self.assertEqual(report["phase_rows"][1]["page_mode"], "no_thp")
        self.assertEqual(report["correlations"][0]["cpu_attempts"], ["attempt-1"])

    def test_duplicate_server_identity_after_restart_is_ambiguous(self):
        report = self.parse(("cpu", [region(pid=10)]), ("cpu", [region(pid=11)]),
                            ("npu", [region(role="npu", pid=20)]))
        correlation = report["correlations"][0]
        self.assertEqual(correlation["status"], "ambiguous")
        self.assertEqual(len(correlation["cpu_attempts"]), 2)
        self.assertEqual(correlation["confidence"], "none")

    def test_reused_attempt_in_concatenated_log_stays_separate(self):
        lines = []
        for _ in range(2):
            lines.extend([profile("snapshot", "before_allocate"),
                          profile("begin", "total_allocate"), region(),
                          profile("end", "total_allocate", elapsed_ms=2000)])
        report = self.parse(("cpu", lines), ("npu", [region(role="npu", pid=20)]))
        self.assertEqual(len([a for a in report["attempts"] if a["role"] == "cpu"]), 2)
        self.assertEqual(report["correlations"][0]["status"], "ambiguous")

    def test_snapshot_deltas_keep_gauges_and_omit_counter_resets(self):
        report = self.parse(("cpu", [
            profile("snapshot", "before_allocate", global_vmstat={"compact_stall": 5, "pswpout": 8},
                    numa_memfree_kib={"node0": 200000}, process_status={"VmRSS_kib": 100},
                    process_anon_huge_pages_kib=0),
            profile("snapshot", "after_memset", global_vmstat={"compact_stall": 9, "pswpout": 8},
                    numa_memfree_kib={"node0": 100000}, process_status={"VmRSS_kib": 100100},
                    process_anon_huge_pages_kib=98304),
            profile("snapshot", "after_register", global_vmstat={"compact_stall": 10, "pswpout": 1},
                    numa_memfree_kib={"node0": 90000}, process_status={"VmRSS_kib": 100200},
                    process_anon_huge_pages_kib=98304),
        ]))
        delta = next(d for d in report["attempts"][0]["snapshot_deltas"]
                     if d["from"] == "before_allocate" and d["to"] == "after_register")
        self.assertEqual(delta["global_vmstat_delta"], {"compact_stall": 5})
        self.assertEqual(delta["counter_decreases"], ["pswpout"])
        self.assertEqual(delta["numa_memfree_delta_kib"]["node0"], -110000)
        self.assertEqual(delta["process_anon_huge_pages_delta_kib"], 98304)
        self.assertTrue(any("not proof" in note for note in report["notes"]))

    def test_malformed_profile_retains_good_records_and_error_signals(self):
        report = self.parse(("npu", [
            "arbitrary log text",
            "KVAF_INIT_PROFILE {not json}",
            "[KVAF_AIV] BATCH_PUT_EXECUTION_ERROR result=-5",
            "timestamp " + profile("end", "memset", elapsed_ms=7, outcome="success"),
        ]))
        self.assertEqual(len(report["warnings"]), 1)
        self.assertEqual(len(report["raw_records"]), 1)
        self.assertEqual(len(report["error_signals"]), 1)
        # An error signal may be a recovered candidate. It must not invent a
        # failed runtime initialization or discard the measured phase.
        self.assertEqual(report["runs"], [])
        self.assertEqual(report["phase_rows"][0]["outcome"], "success")

    def test_separate_configs_with_reused_pid_remain_separate_runs(self):
        report = self.parse(("npu", [probe("config"), probe("done", operation="initialize", elapsed_s=7),
                                    probe("ready"), probe("config"),
                                    probe("failed", operation="initialize", elapsed_s=38)]))
        self.assertEqual([r["status"] for r in report["runs"]], ["success", "failed"])
        self.assertEqual([r["initialize_elapsed_s"] for r in report["runs"]], [7, 38])

    def test_cli_writes_raw_json_csv_and_readable_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "cpu.log"
            output, csv_path = Path(tmp) / "report.json", Path(tmp) / "phases.csv"
            log.write_text(profile("end", "memset", elapsed_ms=1000, outcome="success",
                                   user_cpu_ms=100, system_cpu_ms=900, minor_faults=32768,
                                   usage_scope="thread") + "\n")
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = ANALYZER.main(["--cpu-log", str(log), "--output", str(output), "--csv", str(csv_path)])
            self.assertEqual(result, 0)
            parsed = json.loads(output.read_text())
            self.assertEqual(parsed["phase_rows"][0]["minor_faults"], 32768)
            self.assertEqual(len(parsed["raw_records"]), 1)
            with csv_path.open() as stream:
                self.assertEqual(list(csv.DictReader(stream))[0]["phase"], "memset")
            self.assertIn("memset", stdout.getvalue())
            self.assertIn("do not sum", stdout.getvalue())

    def test_snapshot_overhead_and_small_sample_tail_are_separate(self):
        report = self.parse(("cpu", [
            profile("snapshot", "before_allocate", collection_elapsed_ms=3),
            profile("end", "memset", elapsed_ms=7000, outcome="success"),
            profile("snapshot", "after_memset", collection_elapsed_ms=5),
            profile("snapshot", "after_register", collection_elapsed_ms=7),
            profile("snapshot", "after_release", collection_elapsed_ms=11),
        ]))
        self.assertEqual(report["attempts"][0]["snapshot_collection_ms"],
                         {"allocation": 15, "release": 11})
        self.assertEqual(report["phase_summaries"][0]["median_ms"], 7000)
        self.assertIsNone(report["phase_summaries"][0]["p95_ms"])

    def test_snapshot_overhead_missing_category_is_unknown(self):
        allocation = self.parse(("cpu", [profile("snapshot", "before_allocate", collection_elapsed_ms=3)]))
        self.assertEqual(allocation["attempts"][0]["snapshot_collection_ms"],
                         {"allocation": 3, "release": None})
        release = self.parse(("cpu", [profile("snapshot", "after_release", collection_elapsed_ms=2)]))
        self.assertEqual(release["attempts"][0]["snapshot_collection_ms"],
                         {"allocation": None, "release": 2})

    def test_repeated_native_discovery_keeps_each_capacity_without_probe_config(self):
        lines = []
        for gib in (1, 8):
            for phase in ("candidate_query", "eid_discovery"):
                lines.extend([
                    profile("begin", phase, role="npu", pid=20, attempt=0,
                            bytes=gib * 1024**3, candidate=None, page_mode="na"),
                    profile("end", phase, role="npu", pid=20, attempt=0,
                            bytes=gib * 1024**3, candidate=None, page_mode="na",
                            elapsed_ms=gib, outcome="success"),
                ])
        report = self.parse(("npu", lines))
        self.assertEqual(len(report["attempts"]), 2)
        self.assertEqual([p["bytes"] for p in report["phase_rows"]],
                         [1024**3, 1024**3, 8 * 1024**3, 8 * 1024**3])

    def test_phase_metadata_overrides_aggregate_and_na_mode_uses_inferred_mode(self):
        report = self.parse(("cpu", [
            profile("end", "first", bytes=1024**3, candidate=None, page_mode="thp", elapsed_ms=1),
            profile("end", "second", bytes=8 * 1024**3, candidate=7, page_mode="no-thp", elapsed_ms=8),
        ]))
        first, second = report["phase_rows"]
        self.assertEqual(first["bytes"], 1024**3)
        self.assertIsNone(first["candidate"])
        self.assertEqual(first["page_mode"], "thp")
        self.assertEqual(second["bytes"], 8 * 1024**3)
        self.assertEqual(second["candidate"], 7)
        self.assertEqual(second["page_mode"], "no-thp")

    def test_trace_ending_inside_memset_and_probe_remains_pending(self):
        report = self.parse(("cpu", [
            profile("begin", "total_allocate", timestamp_unix_ns=123),
            profile("begin", "memset", timestamp_unix_ns=124),
        ]), ("npu", [
            probe("config"), probe("begin", operation="initialize"),
            profile("begin", "startup_probe", role="npu", pid=20),
        ]))
        cpu, npu = report["attempts"]
        self.assertEqual([p["phase"] for p in cpu["incomplete_phases"]],
                         ["total_allocate", "memset"])
        self.assertEqual(cpu["incomplete_phases"][1]["line"], 2)
        self.assertEqual(cpu["incomplete_phases"][1]["timestamp_unix_ns"], 124)
        self.assertEqual(npu["incomplete_phases"][0]["phase"], "startup_probe")
        self.assertEqual(report["runs"][0]["status"], "incomplete")
        self.assertEqual(report["phase_rows"], [])
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            ANALYZER.print_report(report)
        self.assertIn("Pending in this log excerpt: total_allocate", stdout.getvalue())
        self.assertIn("not evidence of failure", stdout.getvalue())

    def test_duplicate_phase_begins_match_ends_as_a_stack(self):
        report = self.parse(("cpu", [profile("begin", "memset"), profile("begin", "memset"),
                                    profile("end", "memset", elapsed_ms=3)]))
        pending = report["attempts"][0]["incomplete_phases"]
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["line"], 1)

    def test_meminfo_deltas_are_gauges_and_may_decrease(self):
        report = self.parse(("cpu", [
            profile("snapshot", "before_allocate", global_meminfo_kib={"MemAvailable": 1000}),
            profile("snapshot", "after_register", global_meminfo_kib={"MemAvailable": 200}),
        ]))
        delta = report["attempts"][0]["snapshot_deltas"][0]
        self.assertEqual(delta["global_meminfo_delta_kib"], {"MemAvailable": -800})
        self.assertEqual(delta["counter_decreases"], [])

    def test_exact_leaf_counter_windows_do_not_absorb_outside_activity(self):
        snapshots = [
            ("before_allocate", 5, 0),
            ("before_memset", 7, 20),  # Earlier unrelated/outer work.
            ("after_memset", 12, 20),
            ("before_register", 12, 20),
            ("after_register", 12, 20),
        ]
        report = self.parse(("cpu", [profile("snapshot", phase, global_vmstat={
            "compact_stall": compact, "pgscan_direct": scan})
            for phase, compact, scan in snapshots]))
        windows = {(d["from"], d["to"]): d for d in report["attempts"][0]["snapshot_deltas"]}
        memset = windows[("before_memset", "after_memset")]["global_vmstat_delta"]
        register = windows[("before_register", "after_register")]["global_vmstat_delta"]
        total = windows[("before_allocate", "after_register")]["global_vmstat_delta"]
        self.assertEqual(memset, {"compact_stall": 5, "pgscan_direct": 0})
        self.assertEqual(register, {"compact_stall": 0, "pgscan_direct": 0})
        self.assertEqual(total, {"compact_stall": 7, "pgscan_direct": 20})
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            ANALYZER.print_report(report)
        self.assertIn('before_memset->after_memset counters={"compact_stall": 5}', stdout.getvalue())
        self.assertIn('before_register->after_register counters={}', stdout.getvalue())
        self.assertIn('before_allocate->after_register counters={"compact_stall": 7, "pgscan_direct": 20}',
                      stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
