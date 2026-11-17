#!/usr/bin/env python3
"""Extract the Host and device events needed to time a Scatter Graph replay."""

import argparse
import csv
import json
import sys
from contextlib import nullcontext
from decimal import Decimal, InvalidOperation
from pathlib import Path


COLUMNS = (
    "kind", "name", "start_us", "end_us", "duration_us", "pid", "tid", "category",
    "device_id", "stream_id", "physical_stream_id", "task_id", "model_id", "connection_id", "event_id",
)
DEVICE_TASKS = {"MODEL_EXECUTE", "NOTIFY_WAIT", "NOTIFY_RECORD"}
SCATTER_KERNEL = "kvafscatterbatchgetaivkernel"


def normalized_args(event):
    args = event.get("args")
    if not isinstance(args, dict):
        return {}
    return {"".join(ch for ch in key.lower() if ch.isalnum()): value
            for key, value in args.items()}


def event_kind(name, args):
    label = name.rsplit("@", 1)[-1].strip()
    task_type = str(args.get("tasktype", "")).upper()
    if label in DEVICE_TASKS or task_type in DEVICE_TASKS:
        return label if label in DEVICE_TASKS else task_type
    api = label.lower()
    if api.startswith(("aclmdlriexecute", "aclmdlexecute", "rtmodelexecute", "modelexecute")):
        return "host_replay"
    if "synchronizestream" in api or "streamsynchronize" in api:
        return "host_sync"
    for candidate in (api, args.get("kernelname", ""), args.get("opname", "")):
        if SCATTER_KERNEL in str(candidate).lower():
            return "aiv_scatter"
    return None


def collect_events(path):
    with path.open(encoding="utf-8-sig") as source:
        trace = json.load(source, parse_float=Decimal)
    events = trace.get("traceEvents") if isinstance(trace, dict) else trace
    if not isinstance(events, list):
        raise ValueError("expected a Chrome trace event array or an object with traceEvents")

    rows = []
    for event in events:
        if not isinstance(event, dict) or "ts" not in event or "dur" not in event:
            continue
        name = str(event.get("name", ""))
        args = normalized_args(event)
        kind = event_kind(name, args)
        if kind is None:
            continue
        start = Decimal(str(event["ts"]))
        duration = Decimal(str(event["dur"]))
        rows.append({
            "kind": kind, "name": name, "start_us": start, "end_us": start + duration,
            "duration_us": duration, "pid": event.get("pid"), "tid": event.get("tid"),
            "category": event.get("cat"), "device_id": args.get("deviceid"),
            "stream_id": args.get("streamid", args.get("logicstreamid", args.get("logicalstreamid"))),
            "physical_stream_id": args.get("physicalstreamid", args.get("physicstreamid")),
            "task_id": args.get("taskid"), "model_id": args.get("modelid"),
            "connection_id": args.get("connectionid", event.get("connection_id")),
            "event_id": event.get("id"),
        })
    return sorted(rows, key=lambda row: row["start_us"])


def select_window(rows, last, context_us):
    scatter = [row for row in rows if row["kind"] == "aiv_scatter"]
    if not scatter:
        print("No Scatter kernel events found; exporting the matching API/task events without a time filter.",
              file=sys.stderr)
        return rows
    if last > len(scatter):
        print(f"Requested {last} Scatter replays but found {len(scatter)}.", file=sys.stderr)
    if last and last < len(scatter):
        start = scatter[-last - 1]["end_us"]
    else:
        start = scatter[0]["start_us"] - context_us
    end = scatter[-1]["end_us"] + context_us
    return [row for row in rows if row["end_us"] >= start and row["start_us"] <= end
            and (row["kind"] != "aiv_scatter" or row["start_us"] >= start)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="one msprof_*.json timeline")
    parser.add_argument("-o", "--output", type=Path, help="CSV path; defaults to standard output")
    parser.add_argument("--last", type=int, default=0,
                        help="keep the last N Scatter replays; 0 keeps all")
    parser.add_argument("--context-us", type=int, default=500,
                        help="time around the first and last Scatter task, in microseconds")
    args = parser.parse_args()
    if args.last < 0 or args.context_us < 0:
        parser.error("--last and --context-us must be nonnegative")
    if args.output and args.output.resolve() == args.trace.resolve():
        parser.error("the output CSV must differ from the input timeline")

    try:
        rows = select_window(collect_events(args.trace), args.last, args.context_us)
        if not rows:
            parser.error("no matching events were found in the timeline")
        destination = (args.output.open("w", encoding="utf-8", newline="") if args.output
                       else nullcontext(sys.stdout))
        with destination as output:
            writer = csv.DictWriter(output, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    except (OSError, ValueError, InvalidOperation) as error:
        parser.error(str(error))

    print(f"Extracted {len(rows)} events.", file=sys.stderr)
    present = {row["kind"] for row in rows}
    for kind in ("host_replay", "host_sync"):
        if kind not in present:
            print(f"No {kind} events found in the selected window.", file=sys.stderr)


if __name__ == "__main__":
    main()
