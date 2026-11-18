"""Deterministic device traces for overlap selection, gap and bandwidth math."""

import csv
from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

import batch_get_concurrency as concurrency
import batch_get_profiler_overlap as overlap
import run_concurrent_batch_get as launcher
import test_npu_batch_get_perf as perf


SHAPE = "token_bytes_10_miss_count_100_aiv_cores_1_jettys_per_core_1"


def make_trace(folder, device, intervals, *, first=0, kernel_name=overlap.KERNEL_NAME, ring=None):
    """All fixture event timestamps are original, unshifted trace microseconds."""
    shape = folder / "profiler" / SHAPE
    output = shape / "worker_ascend_pt" / "ASCEND_PROFILER_OUTPUT"
    output.mkdir(parents=True)
    worker = f"flow{device + 1}_device{device}_pid{100 + device}"
    metadata = dict(worker=worker, device_id=str(device), execution_mode="graph",
                    calls_per_round=ring or len(intervals[0]), active=len(intervals),
                    first_active_round=first, payload_bytes_per_get=1000, kernel_name=overlap.KERNEL_NAME)
    (shape / "collection.json").write_text(json.dumps(metadata))
    events = []
    for iteration, rows in enumerate(intervals, start=first):
        start, end = rows[0][0], rows[-1][1]
        events.extend([
            dict(ph="X", name=f"KVAF/{worker}/round{iteration}/graph_replay", cat="cpu_op",
                 pid=100 + device, tid=1, ts=start - 20, dur=1),
            dict(ph="X", name=f"KVAF/{worker}/round{iteration}/wait_completion", cat="cpu_op",
                 pid=100 + device, tid=1, ts=start - 19, dur=end - start + 29),
            # A CPU duplicate with the same symbol must never count as a Get.
            dict(ph="X", name=kernel_name, cat="cpu_op", pid=100 + device, tid=1, ts=start, dur=1),
            dict(ph="X", name="some_other_kernel", cat="kernel", pid=200 + device, tid=9,
                 ts=start, dur=1),
        ])
        for begin, finish in rows:
            events.append(dict(ph="X", name=kernel_name, cat="kernel", pid=200 + device, tid=8,
                               ts=begin, dur=finish - begin, args={"Task Type": "AI_VECTOR_CORE"}))
    path = output / "trace_view.json"
    path.write_text(json.dumps(events))
    with (folder / "batch_get_miss_sweep_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["p50_GBps", "profile_scope"])
        writer.writeheader()
        writer.writerow(dict(p50_GBps="777.000", profile_scope="last_get_in_replay"))
    return path, events


class OverlapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def four_cards(self):
        groups = [((100, 110), (112, 122), (124, 134)),
                  ((103, 108), (110, 115), (117, 122)),
                  ((105, 113), (114, 122), (123, 131)),
                  ((102, 109), (111, 118), (120, 127))]
        timelines = []
        for device, group in enumerate(groups):
            folder = self.root / f"flow{device + 1}"
            make_trace(folder, device, [group])
            timelines.append(overlap.load_flow(folder / "profiler", folder.name, device))
        return timelines

    def test_four_card_selection_is_latest_first_start_to_earliest_last_end(self):
        result = overlap.analyze_timelines(self.four_cards())
        self.assertEqual([(row["common_begin_ns"], row["common_end_ns"]) for row in result["rounds"]],
                         [(105000, 122000)] * 4)
        self.assertEqual([row["selected_gets"] for row in result["rounds"]], [1, 2, 2, 1])
        self.assertEqual([row["selected_kernel_ns"] for row in result["rounds"]], [10000, 10000, 16000, 7000])
        for device, expected in enumerate((0.1, 0.2, 0.125, 1 / 7)):
            self.assertAlmostEqual(result["summary"]["devices"][str(device)]["mean_GBps"], expected)
        self.assertEqual([row["get_index"] for row in result["gets"]
                          if row["flow"] == "flow1" and row["selected"]], [1])
        # Exact start/end equality is included; partial boundary Gets are excluded.
        self.assertEqual([row["get_index"] for row in result["gets"]
                          if row["flow"] == "flow3" and row["selected"]], [0, 1])
        self.assertAlmostEqual(result["summary"]["flows"]["flow1"]["gap_fraction"], 4 / 34)
        self.assertEqual(result["summary"]["flows"]["flow1"]["gap_max_us"], 2)

    def test_bandwidth_is_ratio_of_total_bytes_and_selected_durations_not_average_rates(self):
        folder = self.root / "flow1"
        make_trace(folder, 0, [[(100, 110), (112, 122)], [(200, 220), (222, 242)]])
        flow = overlap.load_flow(folder / "profiler", "flow1", 0)
        result = overlap.analyze_timelines([flow])
        self.assertAlmostEqual(result["summary"]["flows"]["flow1"]["mean_GBps"], 4000 / 60000)
        self.assertNotAlmostEqual(result["summary"]["flows"]["flow1"]["mean_GBps"], (0.1 + 0.05) / 2)

    def test_no_intersection_and_no_complete_get_use_null_bandwidth(self):
        flows = self.four_cards()[:2]
        for row in flows[1]["rounds"][0]:
            row["begin_ns"] += 1000000
            row["end_ns"] += 1000000
        result = overlap.analyze_timelines(flows)
        self.assertTrue(all(row["status"] == "no_common_interval" for row in result["rounds"]))
        self.assertTrue(all(row["mean_GBps"] is None for row in result["rounds"]))
        # An intersection can exist while a long Get spans both its boundaries.
        first = self.root / "long"
        second = self.root / "short"
        make_trace(first, 0, [[(100, 200)]])
        make_trace(second, 1, [[(120, 130)]])
        result = overlap.analyze_timelines([overlap.load_flow(first / "profiler", "long", 0),
                                           overlap.load_flow(second / "profiler", "short", 1)])
        self.assertIsNone(result["summary"]["flows"]["long"]["mean_GBps"])
        self.assertEqual(result["summary"]["flows"]["long"]["selected_gets"], 0)

    def test_only_shared_recorded_rounds_are_used(self):
        one, two = self.root / "one", self.root / "two"
        make_trace(one, 0, [[(100, 110)], [(200, 210)]], first=0)
        make_trace(two, 1, [[(200, 210)], [(300, 310)]], first=1)
        result = overlap.analyze_timelines([overlap.load_flow(one / "profiler", "one", 0),
                                           overlap.load_flow(two / "profiler", "two", 1)])
        self.assertEqual(result["summary"]["round_indices"], [1])
        self.assertEqual(result["summary"]["flows"]["one"]["profiler_round_indices"], [0, 1])

    def test_hundred_gets_per_graph_and_real_start_skew_are_preserved(self):
        groups = []
        for device in range(4):
            start = 1000 + 2 * device
            rows = [(start + i * 10, start + i * 10 + 8) for i in range(100)]
            folder = self.root / str(device)
            make_trace(folder, device, [rows])
            groups.append(overlap.load_flow(folder / "profiler", str(device), device))
        result = overlap.analyze_timelines(groups)
        self.assertEqual(len(result["gets"]), 400)
        self.assertEqual(result["rounds"][0]["common_begin_ns"], 1006000)
        self.assertEqual(result["rounds"][0]["common_end_ns"], 1998000)
        self.assertTrue(all(row["selected_gets"] < 100 for row in result["rounds"]))

    def test_trace_units_preserve_fractional_ns_at_large_absolute_timestamps(self):
        self.assertEqual(overlap.us_to_ns(Decimal("1700000000000000.125")), 1700000000000000125)
        for invalid in ("NaN", "Infinity", "garbage", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                overlap.us_to_ns(invalid)

    def test_incomplete_wrong_symbol_duplicate_and_cpu_only_traces_fail(self):
        for mode in ("missing", "wrong_symbol", "duplicate", "missing_marker", "cpu_only"):
            with self.subTest(mode=mode):
                folder = self.root / mode
                path, events = make_trace(folder, 0, [[(100, 110), (120, 130)]])
                if mode == "missing":
                    events.pop()
                elif mode == "wrong_symbol":
                    for event in events:
                        if event.get("cat") == "kernel":
                            event["name"] = "unknown_kernel"
                elif mode == "duplicate":
                    events.append(events[-1])
                elif mode == "missing_marker":
                    events.pop(1)
                elif mode == "cpu_only":
                    events = [event for event in events if event.get("cat") != "kernel"]
                path.write_text(json.dumps(events))
                with self.assertRaises(ValueError):
                    overlap.load_flow(folder / "profiler", mode, 0)

    def test_reports_enrich_json_and_each_flow_csv_and_keep_last_get_p50(self):
        self.four_cards()
        flows = [(f"flow{i + 1}", [], dict(DEVICE_ID=str(i),
                  KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR=str(self.root / f"flow{i + 1}"),
                  KVAF_BATCH_GET_MISS_SWEEP_CSV=str(self.root / f"flow{i + 1}" / "batch_get_miss_sweep_summary.csv")))
                 for i in range(4)]
        result = overlap.analyze_flows(flows)
        overlap.write_reports(result, self.root, flows)
        for filename in overlap.REPORT_FILES:
            self.assertTrue((self.root / filename).is_file())
        for name, _, env in flows:
            with Path(env["KVAF_BATCH_GET_MISS_SWEEP_CSV"]).open() as stream:
                row, = csv.DictReader(stream)
            self.assertEqual(row["p50_GBps"], "777.000")
            self.assertAlmostEqual(float(row["overlap_mean_GBps"]), result["summary"]["flows"][name]["mean_GBps"])
        measurements = [dict(flow=name, device_id=env["DEVICE_ID"], execution_mode="graph",
                             torch_npu_profiler=True, calls_per_round=3, payload_bytes=3000,
                             device_elapsed_ms=1) for name, _, env in flows]
        sample = concurrency.round_sample(0, 0, [dict(begin_ns=1, end_ns=10000, payload_bytes=3000)] * 4)
        summary = concurrency.report_results([sample], measurements, None, lambda unused: None,
                                             profiler_overlap=result["summary"])
        self.assertAlmostEqual(summary["flows"]["flow1"]["overlap_mean_GBps"], 0.1)
        self.assertEqual(summary["profiler_overlap"]["rounds"], 1)

    def test_offline_reanalysis_accepts_older_collection_metadata_and_writes_fresh_reports(self):
        self.four_cards()
        for folder in self.root.glob("flow*"):
            metadata, = folder.glob("profiler/*/collection.json")
            document = json.loads(metadata.read_text())
            del document["payload_bytes_per_get"]
            metadata.write_text(json.dumps(document))
        report = self.root / "report"
        self.assertEqual(overlap.main([str(self.root), "--report-dir", str(report)]), 0)
        summary = json.loads((report / "profiler_overlap_summary.json").read_text())
        self.assertEqual(summary["flows"]["flow1"]["selected_payload_bytes"], 1000)

    def test_default_graph_contains_hundred_gets(self):
        self.assertEqual(perf.parse_args([]).graph_ring, 100)
        args = launcher.parse_args(["build", "--output-dir", "out", "--flow1-device", "0", "--flow2-device", "1"])
        self.assertEqual(args.graph_ring, 100)


if __name__ == "__main__":
    unittest.main()
