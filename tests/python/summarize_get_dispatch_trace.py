#!/usr/bin/env python3
"""Summarize opt-in per-call Get dispatcher timing CSV from the NPU runtime."""

import argparse
import csv
from pathlib import Path


SUBMIT_FIELDS = (
    "total_ns", "lock_ns", "resolve_ns", "output_check_ns", "previous_drain_ns", "context_ns",
    "describe_ns", "acquire_ns", "bind_ns", "requirements_ns", "warmup_ns",
    "provider_ns", "provider_prepare_ns", "provider_submit_ns", "provider_completion_ns",
    "provider_cleanup_ns", "provider_other_ns", "context_release_ns", "outside_provider_ns",
)
AIV_FIELDS = (
    "aiv_capture_prepare_ns", "aiv_capture_submit_ns", "aiv_status_reset_ns",
    "aiv_kernel_launch_ns", "aiv_submit_other_ns",
)
DRAIN_FIELDS = ("total_ns", "drain_provider_ns", "drain_release_ns")


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(rows, algorithm, path, event, fields):
    if not rows:
        print(f"GET_DISPATCH_TIMING algorithm={algorithm} path={path} event={event} samples=0")
        return
    print(f"GET_DISPATCH_TIMING algorithm={algorithm} path={path} event={event} "
          f"samples={len(rows)} unit=us")
    for field in fields:
        values = [row[field] / 1000 for row in rows]
        print(f"  {field.removesuffix('_ns')}: p50={percentile(values, 0.5):.2f} "
              f"p90={percentile(values, 0.9):.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--last", type=int, default=0,
                        help="take the last N samples of each event, excluding preflight and warmup")
    args = parser.parse_args()
    if args.last < 0:
        parser.error("--last must be nonnegative")
    with args.trace.open(newline="") as source:
        rows = list(csv.DictReader(source))
    groups = sorted({(row.get("algorithm") or "unknown", row.get("path") or "ordinary")
                     for row in rows})
    for algorithm, path in groups:
        for event, fields in (("submit", SUBMIT_FIELDS), ("drain", DRAIN_FIELDS)):
            selected = [row for row in rows if row["event"] == event and
                        (row.get("algorithm") or "unknown") == algorithm and
                        (row.get("path") or "ordinary") == path]
            if args.last:
                selected = selected[-args.last:]
            parsed = [{key: int(value) for key, value in row.items()
                       if key not in ("event", "index", "algorithm", "path")}
                      for row in selected]
            for row in parsed:
                row["outside_provider_ns"] = row["total_ns"] - row["provider_ns"]
                row["provider_other_ns"] = max(0, row["provider_ns"] - sum(
                    row.get(field, 0) for field in ("provider_prepare_ns", "provider_submit_ns",
                                                    "provider_completion_ns", "provider_cleanup_ns")))
                row["aiv_submit_other_ns"] = max(0, row.get("provider_submit_ns", 0) - sum(
                    row.get(field, 0) for field in ("aiv_capture_submit_ns", "aiv_status_reset_ns",
                                                    "aiv_kernel_launch_ns")))
                for field in (*fields, *AIV_FIELDS):
                    row.setdefault(field, 0)
            if selected:
                display_fields = fields
                if event == "submit" and any(
                    any(row[field] for field in AIV_FIELDS[:-1]) for row in parsed
                ):
                    prepare_at = fields.index("provider_prepare_ns") + 1
                    submit_at = fields.index("provider_submit_ns") + 1
                    display_fields = (fields[:prepare_at] + AIV_FIELDS[:1] +
                                      fields[prepare_at:submit_at] + AIV_FIELDS[1:] +
                                      fields[submit_at:])
                summarize(parsed, algorithm, path, event, display_fields)


if __name__ == "__main__":
    main()
