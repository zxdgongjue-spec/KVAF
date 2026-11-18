"""Select complete BatchGets in the common interval of parsed NPU timelines.

Trace timestamps are the original CANN-aligned timestamps, in microseconds.
Never align workers by their first event: that would erase real start skew.
"""

import argparse
from bisect import bisect_left
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import statistics


REPORT_FILES = ("concurrent_batch_get_overlap_rounds.csv", "concurrent_batch_get_gets.csv",
                "concurrent_batch_get_devices.csv")
KERNEL_NAME = "KvafExecuteBatchGetAivKernelV3"


def us_to_ns(value):
    try:
        number = Decimal(str(value))
        if not number.is_finite():
            raise ValueError("non-finite profiler timestamp/duration")
        return int(number * 1000)
    except (InvalidOperation, TypeError) as error:
        raise ValueError("invalid profiler timestamp/duration") from error


def interval(event):
    begin, duration = us_to_ns(event.get("ts")), us_to_ns(event.get("dur"))
    if begin < 0 or duration <= 0:
        raise ValueError("profiler events require non-negative timestamps and positive durations")
    return begin, begin + duration


def load_flow(profiler_root, name, device_id):
    root = Path(profiler_root)
    configurations = list(root.glob("*/collection.json"))
    if len(configurations) != 1:
        raise ValueError(f"{name}: expected one profiler collection under {root}")
    metadata_path, = configurations
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["execution_mode"] != "graph" or str(metadata["device_id"]) != str(device_id):
        raise ValueError(f"{name}: profiler collection does not match the graph flow/device")
    ring = metadata["calls_per_round"]
    payload = metadata.get("payload_bytes_per_get")
    if payload is None:
        # Older collection.json files predate this field; the worker's shape
        # directory has always contained these two resolved sizes.
        shape = re.fullmatch(r"token_bytes_([0-9]+)_miss_count_([0-9]+)_aiv_cores_[0-9]+_jettys_per_core_[0-9]+",
                             metadata_path.parent.name)
        if shape:
            payload = int(shape[1]) * int(shape[2])
    if type(ring) is not int or ring < 1 or type(payload) is not int or payload < 0:
        raise ValueError(f"{name}: invalid ring or Get payload in profiler metadata")
    traces = list(metadata_path.parent.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json"))
    if len(traces) != 1:
        raise ValueError(f"{name}: expected one parsed trace_view.json under {metadata_path.parent}")
    with traces[0].open(encoding="utf-8") as stream:
        document = json.load(stream, parse_float=Decimal)
    if isinstance(document, dict) and document.get("baseTimeNanoseconds"):
        raise ValueError(f"{name}: relative trace time base is unsupported; use the original Ascend trace")
    events = document if isinstance(document, list) else document.get("traceEvents", [])
    marker = re.compile(r"^KVAF/" + re.escape(metadata["worker"])
                        + r"/round([0-9]+)/(graph_replay|wait_completion)$")
    bounds, kernels, host_pids = {}, [], set()
    for event in events:
        if event.get("ph") != "X":
            continue
        match = marker.fullmatch(str(event.get("name", "")))
        if match:
            iteration, phase = int(match[1]), match[2]
            phases = bounds.setdefault(iteration, {})
            if phase in phases:
                raise ValueError(f"{name}: duplicate {phase} marker for round {iteration}")
            phases[phase] = (*interval(event), event.get("pid"), event.get("tid"))
            host_pids.add(event.get("pid"))
    kernel_name = metadata.get("kernel_name", KERNEL_NAME)
    for event in events:
        if event.get("ph") != "X" or kernel_name.lower() not in str(event.get("name", "")).lower():
            continue
        # Match the actual device task, never CPU API/annotation duplicates.
        if event.get("pid") in host_pids:
            continue
        category = str(event.get("cat", "")).lower()
        task = str(event.get("args", {}).get("Task Type", "")).upper()
        if category in {"cpu_op", "user_annotation", "python_function"}:
            continue
        if category not in {"kernel", "npu", "aiv"} and task not in {"AI_CORE", "AI_VECTOR_CORE", "AIV"}:
            continue
        begin, end = interval(event)
        kernels.append(dict(begin_ns=begin, end_ns=end, duration_ns=end - begin,
                            stream_pid=str(event.get("pid")), stream_tid=str(event.get("tid"))))
    kernels.sort(key=lambda row: row["begin_ns"])
    starts = [row["begin_ns"] for row in kernels]
    rounds = {}
    first = metadata["first_active_round"]
    for iteration in range(first, first + metadata["active"]):
        phases = bounds.get(iteration, {})
        if set(phases) != {"graph_replay", "wait_completion"}:
            raise ValueError(f"{name}: incomplete profiler markers for active round {iteration}")
        submit, complete = phases["graph_replay"], phases["wait_completion"]
        if submit[2:] != complete[2:] or submit[1] > complete[0]:
            raise ValueError(f"{name}: invalid submission/completion markers for round {iteration}")
        begin, end = submit[0], complete[1]
        candidates = kernels[bisect_left(starts, begin):bisect_left(starts, end)]
        if len(candidates) != ring or any(row["end_ns"] > end for row in candidates):
            raise ValueError(f"{name}: round {iteration} needs {ring} device Get tasks, found "
                             f"{len(candidates)}; inspect kernel name {kernel_name}, task timestamps "
                             "and CANN profiler output (CPU ranges are not device timings)")
        if len({(row["stream_pid"], row["stream_tid"]) for row in candidates}) != 1:
            raise ValueError(f"{name}: round {iteration} Get tasks span multiple device streams")
        if any(left["end_ns"] > right["begin_ns"] for left, right in zip(candidates, candidates[1:])):
            raise ValueError(f"{name}: overlapping/duplicate Get tasks on one stream in round {iteration}")
        rounds[iteration] = [dict(row, iteration=iteration, flow=name, device_id=str(device_id),
                                 get_index=index, payload_bytes=payload)
                             for index, row in enumerate(candidates)]
    return dict(name=name, device_id=str(device_id), rounds=rounds, trace=str(traces[0]))


def gap_statistics(rows_by_round):
    gaps, span = [], 0
    for rows in rows_by_round:
        gaps.extend(right["begin_ns"] - left["end_ns"] for left, right in zip(rows, rows[1:]))
        span += rows[-1]["end_ns"] - rows[0]["begin_ns"]
    return dict(gap_count=len(gaps), gap_total_ns=sum(gaps), graph_span_ns=span,
                gap_mean_us=statistics.mean(gaps) / 1000 if gaps else 0,
                gap_p50_us=statistics.median(gaps) / 1000 if gaps else 0,
                gap_max_us=max(gaps, default=0) / 1000,
                gap_fraction=sum(gaps) / span if span else None)


def selected_statistics(rows):
    selected = [row for row in rows if row["selected"]]
    payload = sum(row["payload_bytes"] for row in selected)
    duration = sum(row["duration_ns"] for row in selected)
    return dict(selected_gets=len(selected), selected_payload_bytes=payload, selected_kernel_ns=duration,
                mean_GBps=payload / duration if duration else None)


def analyze_timelines(timelines):
    if not timelines:
        raise ValueError("no profiler timelines")
    common_rounds = sorted(set.intersection(*(set(flow["rounds"]) for flow in timelines)))
    if not common_rounds:
        raise ValueError("flows have no common profiler-active round; use matching profiler schedules")
    rounds, gets = [], []
    for iteration in common_rounds:
        groups = [flow["rounds"][iteration] for flow in timelines]
        begin = max(group[0]["begin_ns"] for group in groups)
        end = min(group[-1]["end_ns"] for group in groups)
        for group in groups:
            selected_rows = []
            for row in group:
                reason = ("no_common_interval" if end <= begin else
                          "starts_before_common_interval" if row["begin_ns"] < begin else
                          "ends_after_common_interval" if row["end_ns"] > end else "")
                selected_rows.append(dict(row, selected=not reason, rejection_reason=reason))
            gets.extend(selected_rows)
            stats = selected_statistics(selected_rows)
            rounds.append(dict(iteration=iteration, flow=group[0]["flow"],
                               device_id=group[0]["device_id"],
                               common_begin_ns=begin, common_end_ns=end,
                               common_window_ns=max(0, end - begin),
                               status=("no_common_interval" if end <= begin else
                                       "no_complete_get" if not stats["selected_gets"] else "ok"),
                               total_gets=len(group), excluded_gets=len(group) - stats["selected_gets"], **stats,
                               **gap_statistics([group])))
    summary = dict(timing="profiler_device_kernel", timestamp_basis="original_cann_aligned_trace",
                   selection="get_begin>=common_begin_and_get_end<=common_end",
                   bandwidth="selected_payload_bytes/sum_selected_kernel_ns",
                   boundary_policy="exclude_partial_gets", rounds=len(common_rounds),
                   round_indices=common_rounds, flows={}, devices={})
    for flow in timelines:
        rows = [row for row in gets if row["flow"] == flow["name"]]
        summary["flows"][flow["name"]] = dict(device_id=flow["device_id"], trace=flow["trace"],
            profiler_round_indices=sorted(flow["rounds"]), **selected_statistics(rows),
            rounds_with_selected_gets=sum(row["selected_gets"] > 0 for row in rounds
                                          if row["flow"] == flow["name"]),
            **gap_statistics([flow["rounds"][iteration] for iteration in common_rounds]))
    for device in dict.fromkeys(flow["device_id"] for flow in timelines):
        rows = [row for row in gets if row["device_id"] == device]
        summary["devices"][device] = dict(device_id=device,
            flows=[flow["name"] for flow in timelines if flow["device_id"] == device],
            **selected_statistics(rows))
    return dict(summary=summary, rounds=rounds, gets=gets)


def analyze_flows(flows):
    return analyze_timelines([load_flow(Path(env["KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR"]) / "profiler",
                                       name, env["DEVICE_ID"]) for name, _, env in flows])


def write_csv(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_reports(result, output_dir, flows=()):
    if output_dir is not None:
        output = Path(output_dir)
        devices = [dict(row, flows=",".join(row["flows"]))
                   for row in result["summary"]["devices"].values()]
        for filename, rows in zip(REPORT_FILES, (result["rounds"], result["gets"], devices)):
            write_csv(output / filename, rows)
    # Keep the original last-Get p50 columns and add clearly named mean fields.
    for name, _, env in flows:
        path = Path(env["KVAF_BATCH_GET_MISS_SWEEP_CSV"])
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        if len(rows) != 1:
            raise ValueError(f"{name}: concurrent profiler summary requires one sweep shape")
        stats = result["summary"]["flows"][name]
        extra = dict(overlap_mean_GBps=stats["mean_GBps"], overlap_selected_gets=stats["selected_gets"],
                     overlap_selected_payload_bytes=stats["selected_payload_bytes"],
                     overlap_selected_kernel_ns=stats["selected_kernel_ns"],
                     overlap_rounds=result["summary"]["rounds"],
                     gap_mean_us=stats["gap_mean_us"], gap_p50_us=stats["gap_p50_us"],
                     gap_max_us=stats["gap_max_us"], gap_fraction=stats["gap_fraction"])
        rows[0].update(extra)
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*fields, *extra])
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path,
                        help="existing concurrent output containing flowN/profiler")
    parser.add_argument("--report-dir", type=Path, required=True,
                        help="new directory for an offline reanalysis")
    args = parser.parse_args(argv)
    folders = sorted(args.output_dir.glob("flow[0-9]*"))
    timelines = []
    for folder in folders:
        metadata, = (folder / "profiler").glob("*/collection.json")
        device = json.loads(metadata.read_text(encoding="utf-8"))["device_id"]
        timelines.append(load_flow(folder / "profiler", folder.name, device))
    result = analyze_timelines(timelines)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    write_reports(result, args.report_dir)
    with (args.report_dir / "profiler_overlap_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result["summary"], stream, indent=2)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
