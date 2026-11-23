import contextlib
import csv
import io
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest

from summarize_aicpu_gap import Trace, bounded_report, csv_report, main


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
                dict(ph="M", name="process_name", pid=2, args=dict(name="Ascend Hardware"))]
    data = dict(traceEvents=metadata + rows, displayTimeUnit="ns") if wrapped else metadata + rows
    (root / filename).write_text(json.dumps(data), encoding="utf-8-sig")


def report(root, last=0):
    return Trace(root, re.compile("KvafAicpuGatherRead", re.I)).report(last, 5, False)


class GapExtractorTest(unittest.TestCase):
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
