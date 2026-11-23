import contextlib
import csv
import io
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from summarize_aicpu_gap import (Trace, bounded_report, csv_report, main, report_execution,
                                graph_scope_comparison, load_graph_samples, entry_boundary_report)


def event(name, start, duration, pid=1, tid=1, **args):
    return dict(ph="X", name=name, ts=start, dur=duration, pid=pid, tid=tid, args=args)


def trace_round(start=100, token=10, flows=False):
    identity = {} if flows else {"Stream Id": 3, "Task Id": token}
    copy_identity = {} if flows else {"Stream Id": 3, "Task Id": token + 1}
    rows = [
        event("aclrtLaunchKernelWithHostArgs", start, 10, **identity),
        event("rtCpuKernelLaunch", start + 1, 8, **identity),
        event("KvafAicpuGatherReadExecutionV3", start + 20, 100, 2, 3, **identity),
        event("aclrtMemcpyAsync", start + 15, 10, **copy_identity),
        event("MemcpyAsync", start + 125, 2, 2, 3, **dict(copy_identity, **{"Task Type": "MEMCPY_ASYNC", "Direction": "D2H"})),
        event("aclrtSynchronizeStream", start + 30, 110),
    ]
    if flows:
        for flow_id, source_ts, dest_ts in ((token, start + 10, start + 20), (token + 1, start + 25, start + 125)):
            rows.extend([dict(ph="s", name="HostToDevice", cat="flow", id=flow_id, pid=1, tid=1, ts=source_ts),
                         dict(ph="f", name="HostToDevice", cat="flow", id=flow_id, pid=2, tid=3, ts=dest_ts)])
    return rows


def write_trace(root, rows, filename="msprof_1.json", wrapped=True):
    metadata = [dict(ph="M", name="process_name", pid=1, args=dict(name="CANN")),
                dict(ph="M", name="process_name", pid=2, args=dict(name="Ascend Hardware")),
                dict(ph="M", name="thread_name", pid=2, tid=3, args=dict(name="Stream 3"))]
    data = dict(traceEvents=metadata + rows, displayTimeUnit="ns") if wrapped else metadata + rows
    (root / filename).write_text(json.dumps(data), encoding="utf-8-sig")


def report(root, last=0):
    return Trace(root, re.compile("KvafAicpuGatherRead", re.I)).report(last, 5, False)


def graph_round(start=100, count=1, token=100):
    end = start + 30 + count * 120
    rows = [event("aclrtRecordEvent", start - 20, 2, **{"Connection Id": token}),
            event("aclmdlRIExecuteAsync", start, 8, **{"Connection Id": token + 1}),
            event("aclrtRecordEvent", start + 10, 2, **{"Connection Id": token + 2}),
            event("aclrtSynchronizeEvent", start + 14, end + 20 - start - 14),
            event("EVENT_RECORD", start - 10, 2, 2, 3,
                  **{"Task Type": "EVENT_RECORD", "Connection Id": token}),
            event("EVENT_RECORD", end, 2, 2, 3,
                  **{"Task Type": "EVENT_RECORD", "Connection Id": token + 2})]
    rows.extend(event("KvafAicpuGatherReadExecutionV4", start + 20 + i * 120, 100, 2, 3,
                      **{"Connection Id": 7, "Task Id": i + 1}) for i in range(count))
    return rows


def sample_document(starts=(600, 1100), gets=1):
    origin = 9_000_000_000_000
    rounds = []
    for ordinal, start in enumerate(starts, 1):
        nodes = []
        for node in range(gets):
            base = origin + (start + 30 + node * 120) * 1000
            nodes.append(dict(node=node, workers=[dict(worker=worker,
                device_entry_ns=base + worker * 5000, device_start_ns=base + 5000 + worker * 5000,
                device_finish_ns=base + 65000 - worker * 5000, device_return_ns=base + 70000 - worker * 5000)
                for worker in range(2)]))
        rounds.append(dict(round=ordinal, host_begin_ns=1_000_000_000 + (start - 25) * 1000,
            host_end_ns=1_000_000_000 + (start + gets * 120 + 60) * 1000,
            event_elapsed_ns=(gets * 120 + 40) * 1000, nodes=nodes))
    return dict(format="kvaf-aicpu-samples-v1", execution="graph", backend="gather_read",
        graph_api="acl", graph_submit="native", device_clock="aicpu_steady_ns", host_clock="python_perf_counter_ns",
        device_id=1, warmup=0, iterations=len(starts), graph_replays=8, aicpu_workers=2,
        gets_per_replay=gets, timed_prefix_replays=1, rounds=rounds)


def sample_trace(root, gets=1, last=0):
    rows = []
    for token, start in enumerate((100, 600, 1100), 1):
        rows.extend(graph_round(start, gets, token * 100))
    write_trace(root, rows)
    data = sample_document(gets=gets)
    (root / "aicpu_samples.json").write_text(json.dumps(data))
    data = load_graph_samples(root)
    trace = Trace(root, re.compile("KvafAicpuGatherRead", re.I))
    timeline = trace.report(last, 5, False, "graph", gets, data)
    return trace, timeline, data


class GapExtractorTest(unittest.TestCase):
    def test_formal_rounds_exclude_native_event_warmup(self):
        with TemporaryDirectory() as folder:
            trace, timeline, data = sample_trace(Path(folder))
            self.assertEqual(timeline["timed_replay_candidates"], 3)
            self.assertEqual(timeline["selected_replays"], 2)
            self.assertEqual(timeline["formal_timed_replays"], 2)
            graph_scope_comparison({}, timeline)
            self.assertTrue(timeline["scope_comparison_selection_complete"])
            result = entry_boundary_report(trace, timeline, data)
            self.assertEqual(result["stats_us"]["task_outside_entry"]["mean"], 30)
            self.assertEqual(result["clock"]["status"], "bounds_only_no_independent_calibration")
            self.assertNotIn("task_begin_to_first_entry", result["stats_us"])
            low, high = result["clock"]["task_begin_to_first_entry_mean_bounds_us"]
            self.assertLessEqual(low, 10)
            self.assertGreaterEqual(high, 10)

    def test_calibrated_boundaries_last_and_multiple_nodes(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            trace, timeline, data = sample_trace(root, gets=2, last=1)
            path = root / "clock.json"
            path.write_text(json.dumps(dict(format="kvaf-aicpu-clock-map-v1", device_id=1,
                source_clock="aicpu_steady_ns", target_clock="msprof_trace_us",
                provenance="independent_timestamp_calibration", anchors=[
                    dict(device_ns=9_000_000_000_000, trace_us=0, uncertainty_us=.1),
                    dict(device_ns=9_000_002_000_000, trace_us=2000, uncertainty_us=.1)])))
            result = entry_boundary_report(trace, timeline, data, path)
            self.assertEqual(result["selected_rounds"], 1)
            self.assertEqual(result["selected_nodes"], 2)
            self.assertEqual(result["stats_us"]["task_begin_to_first_entry"]["mean"], 10)
            self.assertEqual(result["stats_us"]["last_return_to_task_end"]["mean"], 20)
            self.assertEqual(result["stats_us"]["calibrated_task_outside_entry"]["mean"], 30)
            self.assertEqual(result["stats_us"]["boundary_reconciliation_error"]["mean"], 0)
            self.assertEqual(result["representative_nodes"][0]["round"], 2)
            mapping = json.loads(path.read_text())
            for anchor in mapping["anchors"]:
                anchor["trace_us"] += 100
            path.write_text(json.dumps(mapping))
            failed = entry_boundary_report(trace, timeline, data, path)
            self.assertEqual(failed["clock"]["status"], "calibration_inconsistent_with_task_containment")
            self.assertNotIn("task_begin_to_first_entry", failed["stats_us"])

    def test_changed_event_sample_and_host_cadence_are_rejected(self):
        with TemporaryDirectory() as folder:
            trace, timeline, data = sample_trace(Path(folder))
            data["rounds"][0]["event_elapsed_ns"] += 10000
            self.assertEqual(entry_boundary_report(trace, timeline, data)["status"], "sample_trace_event_window_mismatch")
            data["rounds"][0]["event_elapsed_ns"] -= 10000
            for field in ("host_begin_ns", "host_end_ns"):
                data["rounds"][1][field] += 500000
            self.assertEqual(entry_boundary_report(trace, timeline, data)["status"], "sample_trace_host_cadence_mismatch")

    def test_missing_replay_and_clock_drift_do_not_create_a_split(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            trace, timeline, data = sample_trace(root)
            for worker in data["rounds"][1]["nodes"][0]["workers"]:
                for key in ("device_entry_ns", "device_start_ns", "device_finish_ns", "device_return_ns"):
                    worker[key] += 100000
            result = entry_boundary_report(trace, timeline, data)
            self.assertEqual(result["clock"]["status"], "no_consistent_unit_rate_offset")
            write_trace(root, graph_round(100) + graph_round(1100, token=200))
            trace = Trace(root, re.compile("KvafAicpuGatherRead", re.I))
            timeline = trace.report(1, 5, False, "graph", 1, data)
            self.assertEqual(entry_boundary_report(trace, timeline, data)["status"], "sample_trace_replay_count_mismatch")

    def test_reused_csv_task_identity_uses_exported_timestamp(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            trace, timeline, data = sample_trace(root, last=1)
            (root / "aicpu_1.csv").write_text(
                "Task ID,Stream ID,Timestamp(us),Dispatch Time(us)\n"
                "1,3,130,99\n1,3,630,88\n1,3,1130,5\n")
            result = csv_report(root, trace.selected_csv_tasks, trace.selected_task_events)["selected_aicpu_rows"]
            self.assertEqual(result["matched_tasks"], 1)
            self.assertEqual(result["timestamp_matched_rows"], 1)
            self.assertEqual(result["row_duration_us"]["dispatchtime"]["mean"], 5)

    def test_boundary_cli_keeps_absolute_times_and_payload_private(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            trace, timeline, data = sample_trace(root)
            data["payload"] = "CONTROLLED_PAYLOAD"
            (root / "aicpu_samples.json").write_text(json.dumps(data))
            (root / "summary.csv").write_text(
                "backend,execution,gets_per_replay,kernel_mean_us,e2e_mean_us,graph_api,graph_submit\n"
                "gather_read,graph,1,60,160,acl,native\n")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["--msprof-dir", str(root), "--benchmark-dir", str(root), "--last", "1"])
            self.assertEqual(code, 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["entry_boundary"]["selected_rounds"], 1)
            self.assertEqual(report["entry_boundary"]["stats_us"]["task_outside_entry"]["mean"], 30)
            self.assertNotIn("CONTROLLED", output.getvalue())
            self.assertNotIn("9000000000000", output.getvalue())
            self.assertNotIn(str(root), output.getvalue())

    def test_unique_event_identity_survives_host_device_clock_inversion(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = graph_round()
            for row in rows:
                if row["pid"] == 2:
                    row["ts"] -= 13
            write_trace(root, rows)
            result = Trace(root, re.compile("KvafAicpuGatherRead", re.I)).report(1, 1, False, "graph", 1)
            self.assertEqual(result["selected_replays"], 1)
            self.assertEqual(result["cross_clock_order_warnings"], 1)
            self.assertEqual(result["chain_stats_us"]["device_event_start_to_task"]["mean"], 28)
            self.assertEqual(result["chain_stats_us"]["aligned_launch_to_task_begin"], {"n": 0})

    def test_partial_graph_selection_does_not_subtract_full_benchmark_means(self):
        timeline = dict(execution="graph", selected_replays=2, timed_replay_candidates=50, requested_replays=50)
        graph_scope_comparison({}, timeline)
        self.assertFalse(timeline["scope_comparison_selection_complete"])
        self.assertNotIn("scope_mean_comparison_us", timeline)

    def test_graph_event_brackets_recover_replay_without_model_task_id_links(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            capture = [event("aclrtLaunchKernelWithHostArgs", 10, 5, **{"Connection Id": 7})]
            write_trace(root, capture + graph_round(count=2) + graph_round(600, 2, 200))
            trace = Trace(root, re.compile("KvafAicpuGatherRead", re.I))
            result = trace.report(1, 5, False, "graph", 2)
            self.assertEqual(result["model_execute_calls"], 2)
            self.assertEqual(result["timed_replay_brackets"], 2)
            self.assertEqual(result["selected_replays"], 1)
            self.assertEqual(result["linked_target_events"], 4)
            chain = result["representative_chains"][0]
            self.assertEqual(chain["launch_api"], "aclmdlRIExecuteAsync")
            self.assertEqual(chain["link_methods"], ["event_bracket_same_stream"])
            self.assertEqual(chain["metrics_us"]["task_union_per_get"], 100)
            self.assertEqual(chain["metrics_us"]["inter_task_gaps"], 20)
            self.assertEqual(chain["metrics_us"]["device_event_start_to_task"], 28)
            self.assertEqual(chain["metrics_us"]["device_task_to_end_event"], 30)
            self.assertEqual(chain["metrics_us"]["aligned_end_event_end_to_sync_return"], 18)
            self.assertEqual(trace.selected_csv_tasks, {})  # Replay Task IDs repeat.

    def test_graph_does_not_fall_back_to_eager_when_brackets_are_missing(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = trace_round() + [row for row in graph_round(500) if row["name"] != "EVENT_RECORD"]
            write_trace(root, rows)
            result = Trace(root, re.compile("KvafAicpuGatherRead", re.I)).report(1, 1, False, "graph", 1)
            self.assertEqual(result["complete_chains"], 0)
            self.assertEqual(result["target_task_duration_us"], {"n": 0})
            self.assertEqual(result["replay_rejections"]["missing_or_ambiguous_device_event_links"], 1)

    def test_graph_requires_expected_task_count_and_same_stream(self):
        for change in ("count", "stream", "unrelated"):
            with self.subTest(change=change), TemporaryDirectory() as folder:
                root = Path(folder)
                rows = graph_round(count=2 if change == "count" else 1)
                if change == "stream":
                    next(row for row in rows if row["name"].startswith("Kvaf"))["tid"] = 4
                if change == "unrelated":
                    rows.append(event("MemcpyAsync", 225, 3, 2, 3, **{"Task Type": "MEMCPY_ASYNC"}))
                write_trace(root, rows)
                result = Trace(root, re.compile("KvafAicpuGatherRead", re.I)).report(1, 1, False, "graph", 1)
                self.assertEqual(result["selected_replays"], 0)

    def test_graph_physical_stream_identity_can_cross_logical_tracks(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = graph_round()
            for row in rows:
                if row["pid"] == 2:
                    row["args"]["Physic Stream Id"] = 3
                    row["args"]["Stream Id"] = 3
                if row["name"].startswith("Kvaf"):
                    row["tid"] = 4
                    row["args"]["Stream Id"] = 99
            write_trace(root, rows)
            result = Trace(root, re.compile("KvafAicpuGatherRead", re.I)).report(1, 1, False, "graph", 1)
            self.assertEqual(result["selected_replays"], 1)
            self.assertEqual(result["internal_stream_replays"], 0)
            self.assertEqual(result["representative_chains"][0]["stream_identity"], "physical_stream")

    def test_graph_internal_stream_requires_captured_origin_in_same_host_process(self):
        for owner in (1, 99):
            with self.subTest(owner=owner), TemporaryDirectory() as folder:
                root = Path(folder)
                rows = [event("aclrtLaunchKernelWithHostArgs", 10, 5, owner, 1, **{"Connection Id": 7})] + graph_round()
                for row in rows:
                    if row["pid"] == 2:
                        row["args"]["Physic Stream Id"] = 3
                    if row["name"].startswith("Kvaf"):
                        row["tid"] = 4
                        row["args"]["Physic Stream Id"] = 9
                write_trace(root, rows)
                result = Trace(root, re.compile("KvafAicpuGatherRead", re.I)).report(1, 1, False, "graph", 1)
                self.assertEqual(result["selected_replays"], 1 if owner == 1 else 0)
                self.assertEqual(result["graph_match_diagnostics"]["same_stream"], {"0": 1})
                self.assertEqual(result["graph_match_diagnostics"]["same_device_inner"], {"1": 1})
                if owner == 1:
                    self.assertEqual(result["internal_stream_replays"], 1)
                    self.assertEqual(result["representative_chains"][0]["link_methods"],
                                     ["event_bracket_captured_get_internal_stream"])
                else:
                    example = result["graph_match_examples"][0]["candidates"][0]
                    self.assertFalse(example["physical_stream_equal"])
                    self.assertFalse(example["captured_origin_same_host_process"])

    def test_graph_execution_is_selected_from_benchmark_metadata(self):
        graph = {"gather_read/graph": {"summary": {"gets_per_replay": 8}}}
        self.assertEqual(report_execution(graph, "auto"), ("graph", 8))
        with self.assertRaises(ValueError):
            report_execution(graph, "eager")
        with self.assertRaises(ValueError):
            report_execution(dict(graph, **{"gather_read/eager": {}}), "auto")

    def test_graph_ignores_nested_runtime_sync_and_allows_serial_host_queue_thread(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = graph_round()
            next(row for row in rows if row["name"] == "aclmdlRIExecuteAsync")["tid"] = 2
            rows.append(event("rtStreamSynchronize", 80.5, .5))  # Inside the start RecordEvent API.
            write_trace(root, rows)
            result = Trace(root, re.compile("KvafAicpuGatherRead", re.I)).report(1, 1, False, "graph", 1)
            self.assertEqual(result["selected_replays"], 1)
            self.assertEqual(result["representative_chains"][0]["host_order"], "process_serial")

    def test_graph_cli_auto_selection_and_controlled_fields_remain_private(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = graph_round()
            for row in rows:
                row["args"]["payload"] = "CONTROLLED_PAYLOAD"
                row["args"]["address"] = "CONTROLLED_ADDRESS"
            write_trace(root, rows)
            (root / "summary.csv").write_text(
                "backend,execution,gets_per_replay,kernel_mean_us,e2e_mean_us,graph_submit,host_replay_mean_us\n"
                "gather_read,graph,1,90,160,native,180\n", encoding="utf-8")
            (root / "profile.csv").write_text(
                "backend,execution,kind,name,metric,value\n"
                "gather_read,graph,host_graph,stream_resolve,mean_us,2\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["--msprof-dir", str(root), "--benchmark-dir", str(root), "--last", "1"])
            self.assertEqual(code, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["timeline"]["execution"], "graph")
            self.assertEqual(result["timeline"]["selected_replays"], 1)
            self.assertEqual(result["timeline"]["scope_mean_comparison_us"]["gather_read/graph"][
                "trace_task_minus_benchmark_body"], 10)
            benchmark = result["benchmark"]["gather_read/graph"]
            self.assertEqual(benchmark["configuration"]["graph_submit"], "native")
            self.assertEqual(benchmark["summary"]["host_replay_mean_us"], 180)
            self.assertEqual(benchmark["profile_us"]["host_graph.stream_resolve"]["mean_us"], 2)
            self.assertNotIn("CONTROLLED", output.getvalue())

    def test_id_links_nested_apis_and_completion_chain(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            write_trace(root, trace_round())
            result = report(root)
            self.assertEqual(result["complete_chains"], 1)
            chain = result["representative_chains"][0]
            self.assertEqual(chain["host_launch_us"], [0, 10])
            self.assertEqual(chain["device_task_us"], [20, 120])
            self.assertEqual(chain["device_copy_us"], [125, 127])
            self.assertEqual(chain["metrics_us"]["launch_to_sync_return"], 140)
            self.assertEqual(chain["metrics_us"]["device_task_end_to_copy_begin"], 5)
            self.assertEqual(chain["metrics_us"]["aligned_copy_end_to_sync_return"], 13)
            self.assertTrue(chain["cross_clock_order_ok"])
            self.assertEqual(chain["copy_direction"], "d2h")

    def test_flow_links_without_task_ids(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            write_trace(root, trace_round(flows=True), wrapped=False)
            result = report(root)
            self.assertEqual(result["chains_with_device_copy"], 1)
            self.assertEqual(result["representative_chains"][0]["link_methods"], ["flow"])

    def test_graph_model_event_sync_and_linked_record_boundaries(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [
                event("aclrtRecordEvent", 80, 2, **{"Connection Id": 9}),
                event("aclmdlRIExecuteAsync", 100, 8, **{"Connection Id": 10}),
                event("aclrtRecordEvent", 110, 2, **{"Connection Id": 11}),
                event("aclrtSynchronizeEvent", 114, 251),
                event("EVENT_RECORD", 90, 2, 2, 3, **{"Task Type": "EVENT_RECORD", "Connection Id": 9}),
                event("KvafAicpuGatherReadExecutionV4", 120, 100, 2, 3, **{"Connection Id": 10}),
                event("KvafAicpuGatherReadExecutionV4", 240, 100, 2, 3, **{"Connection Id": 10}),
                event("EVENT_RECORD", 343, 2, 2, 3, **{"Task Type": "EVENT_RECORD", "Connection Id": 11}),
            ]
            write_trace(root, rows)
            result = report(root, 1)
            self.assertEqual(result["selected_launches"], 1)
            chain = result["representative_chains"][0]
            self.assertEqual(chain["launch_api"], "aclmdlRIExecuteAsync")
            self.assertEqual(chain["sync_api"], "aclrtSynchronizeEvent")
            self.assertEqual(chain["metrics_us"]["task_union"], 200)
            self.assertEqual(chain["metrics_us"]["inter_task_gaps"], 20)
            self.assertEqual(chain["metrics_us"]["aligned_task_end_to_sync_return"], 25)
            self.assertEqual(chain["metrics_us"]["device_event_start_to_task"], 28)
            self.assertEqual(chain["metrics_us"]["device_task_to_end_event"], 3)
            self.assertTrue(chain["event_bracket_order_ok"])

    def test_last_counts_launches_not_worker_events_and_shards_are_deduped(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = trace_round() + trace_round(300, 20)
            rows.append(event("KvafAicpuGatherReadExecutionV3", 325, 90, 2, 4, **{"Stream Id": 3, "Task Id": 20}))
            write_trace(root, rows)
            write_trace(root, rows, "msprof_2.json")
            result = report(root, 1)
            self.assertEqual(result["target_events"], 3)
            self.assertEqual(result["linked_launches"], 2)
            self.assertEqual(result["selected_launches"], 1)
            self.assertEqual(result["representative_chains"][0]["workers_or_task_events"], 2)
            self.assertGreater(result["warnings"]["DUPLICATE_SHARD_EVENT_REMOVED"], 0)

    def test_no_guessed_links_when_ids_and_flows_are_absent(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [row for row in trace_round(flows=True) if row["ph"] == "X"]
            write_trace(root, rows)
            result = report(root, 50)
            self.assertEqual(result["linked_target_events"], 0)
            self.assertEqual(result["complete_chains"], 0)
            self.assertEqual(result["target_task_duration_us"]["mean"], 100)
            self.assertEqual(result["warnings"]["LAST_NOT_APPLIED_WITHOUT_LAUNCH_LINKS"], 1)

    def test_reused_task_ids_choose_the_current_launch(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            write_trace(root, trace_round() + trace_round(300))
            result = report(root)
            self.assertEqual(result["linked_launches"], 2)
            self.assertEqual(result["chain_stats_us"]["aligned_launch_to_task_begin"]["mean"], 20)

    def test_capture_kernel_id_is_not_a_replay_link(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            write_trace(root, [
                event("aclrtLaunchKernelWithHostArgs", 100, 10, **{"Connection Id": 7}),
                event("aclmdlRIExecuteAsync", 300, 10, **{"Connection Id": 9}),
                event("KvafAicpuGatherReadExecutionV4", 320, 100, 2, 3, **{"Connection Id": 7}),
                event("aclrtSynchronizeEvent", 315, 120),
            ])
            result = report(root, 1)
            self.assertEqual(result["linked_target_events"], 0)
            self.assertGreater(result["warnings"]["CAPTURE_LAUNCH_NOT_REPLAY_LINK"], 0)

    def test_begin_end_api_events_and_nan_are_handled(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = trace_round()
            launch = rows.pop(0)
            rows.extend([dict(launch, ph="B"), dict(ph="E", ts=110, pid=1, tid=1),
                         event("aclrtMemcpyAsync", 0, float("nan"))])
            rows.sort(key=lambda row: row["ts"])
            write_trace(root, rows)
            self.assertEqual(report(root)["complete_chains"], 1)

    def test_csv_units_and_weighted_api_mean(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "task_time_1.csv").write_text("Task Type,Task Duration(ns)\nAI_CPU,100000\nAI_CPU,120000\n", encoding="utf-8-sig")
            (root / "api_statistic_1.csv").write_text("Name,Count,Total Time(us)\naclrtMemcpyAsync,2,20\naclrtMemcpyAsync,8,160\n", encoding="utf-8")
            result = csv_report(root)
            self.assertEqual(result["all_rows_duration_us"]["task_time.aicpu"]["mean"], 110)
            self.assertEqual(result["aggregate_api_us"]["aclrtMemcpyAsync"], dict(calls=10, total=180, mean=18))

    def test_selected_csv_does_not_include_other_task_rows(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            write_trace(root, trace_round() + trace_round(300, 20))
            trace = Trace(root, re.compile("KvafAicpuGatherRead", re.I))
            trace.report(1, 1, False)
            (root / "aicpu_1.csv").write_text(
                "Task ID,Stream ID,Dispatch Time(us)\n10,3,50\n20,3,5\n20,4,99\n", encoding="utf-8")
            selected = csv_report(root, trace.selected_csv_tasks)["selected_aicpu_rows"]
            self.assertEqual(selected["matched_tasks"], 1)
            self.assertEqual(selected["row_duration_us"]["dispatchtime"]["mean"], 5)
            write_trace(root, trace_round() + trace_round(300))
            trace = Trace(root, re.compile("KvafAicpuGatherRead", re.I))
            result = trace.report(1, 1, False)
            self.assertEqual(trace.selected_csv_tasks, {})
            self.assertEqual(result["warnings"]["SELECTED_CSV_IDS_MISSING_OR_REUSED"], 1)

    def test_cli_report_is_bounded_and_does_not_export_controlled_fields(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            rows = trace_round()
            for row in rows:
                row["args"].update(dict(src="CONTROLLED_ADDRESS_0xabcdef", eid="CONTROLLED_EID",
                                        filename="CONTROLLED_PATH", payload="CONTROLLED_DATA"))
            rows.append(event("CONTROLLED_OPERATOR_NAME", 300, 2))
            write_trace(root, rows)
            output = root / "bounded.txt"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = main(["--msprof-dir", str(root), "--max-chars", "4096", "--output", str(output)])
            self.assertEqual(code, 0)
            text = stream.getvalue()
            self.assertLessEqual(len(text), 4096)
            self.assertLessEqual(len(text.encode("utf-8")), 4096)
            self.assertNotIn("CONTROLLED", text)
            self.assertNotIn(str(root), text)
            self.assertNotIn("0xabcdef", text)
            self.assertEqual(output.read_bytes(), text.encode("ascii"))
            self.assertIn("timeline", json.loads(text))

    def test_multiple_captures_fail_without_mixing_runs(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "PROF_1").mkdir()
            (root / "PROF_2").mkdir()
            stream, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(errors):
                code = main(["--msprof-dir", str(root)])
            self.assertEqual(code, 2)
            self.assertEqual(stream.getvalue(), "")
            self.assertIn("MULTIPLE_CAPTURES", errors.getvalue())
            self.assertNotIn(str(root), errors.getvalue())


if __name__ == "__main__":
    unittest.main()
