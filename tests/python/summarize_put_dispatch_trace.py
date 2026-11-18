#!/usr/bin/env python3
"""Summarize KVAF_PUT_DISPATCH_TRACE without mixing backends, paths or recovery."""

import argparse
import csv
from pathlib import Path

from utils.statistics import percentile


FIELDS = (
    "total_ns", "lock_ns", "validation_ns", "resolve_ns", "prepare_ns",
    "metadata_prepare_ns", "execute_ns", "metadata_finalize_ns", "acknowledge_ns",
    "dispatch_lock_ns", "dispatch_resolve_ns", "context_ns", "describe_ns",
    "acquire_ns", "bind_ns", "provider_ns", "provider_prepare_ns",
    "provider_registration_ns", "provider_metadata_ns", "provider_submit_ns",
    "provider_completion_ns", "provider_cleanup_ns", "provider_other_ns",
    "outside_provider_ns",
)
PROVIDER_PHASES = (
    "provider_prepare_ns", "provider_metadata_ns", "provider_submit_ns",
    "provider_completion_ns", "provider_cleanup_ns",
)
GROUP_FIELDS = ("transport", "executor", "path", "event", "result")


def summarize(rows, last=0):
    groups = {}
    for source in rows:
        group = tuple(source.get(field) or "unknown" for field in GROUP_FIELDS)
        row = {field: int(source.get(field) or 0) for field in FIELDS}
        if not source.get("provider_other_ns"):
            # registration is nested within provider_prepare: never subtract it twice.
            row["provider_other_ns"] = max(0, row["provider_ns"] - sum(row[f] for f in PROVIDER_PHASES))
        row["outside_provider_ns"] = max(0, row["total_ns"] - row["provider_ns"])
        groups.setdefault(group, []).append(row)
    for group, samples in sorted(groups.items()):
        if last:
            samples = samples[-last:]
        labels = " ".join(f"{key}={value}" for key, value in zip(GROUP_FIELDS, group))
        print(f"PUT_DISPATCH_TIMING {labels} samples={len(samples)} unit=us")
        for field in FIELDS:
            values = [row[field] / 1000 for row in samples]
            print(f"  {field.removesuffix('_ns')}: p50={percentile(values, 0.5):.2f} "
                  f"p90={percentile(values, 0.9):.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--last", type=int, default=0,
                        help="last N samples per backend/path/event/result; 0 keeps all")
    args = parser.parse_args()
    if args.last < 0:
        parser.error("--last must be nonnegative")
    with args.trace.open(newline="") as source:
        reader = csv.DictReader(source)
        required = {*GROUP_FIELDS, "total_ns", "provider_ns"}
        if not required.issubset(reader.fieldnames or ()):
            parser.error("not a Put trace: missing " + ", ".join(sorted(required - set(reader.fieldnames or ()))))
        rows = list(reader)
    if not rows:
        parser.error("Put trace contains no samples")
    summarize(rows, args.last)


if __name__ == "__main__":
    main()
