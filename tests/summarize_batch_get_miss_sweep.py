#!/usr/bin/env python3
"""Recover the BatchGet miss-sweep CSV from existing per-case logs, offline."""

import argparse
import csv
import math
import re
import sys
from pathlib import Path


SHAPE_FIELDS = ("token_bytes", "miss_count", "aiv_cores", "jettys_per_core")
STATS = ("mean", "p50", "p90", "p99")
STAGES = {
    "batch_get_kernel_total": "",
    "batch_get_kernel_post": "post_",
    "batch_get_kernel_completion_wait": "completion_wait_",
}
FIELDNAMES = [
    *SHAPE_FIELDS,
    *(f"{stat}_us" for stat in STATS),
    "p50_GBps",
    *(f"post_{stat}_us" for stat in STATS),
    *(f"completion_wait_{stat}_us" for stat in STATS),
    "selected_ub_device",
]
LOG_PATTERN = "token_bytes_*_miss_count_*_aiv_cores_*_jettys_per_core_*.log"
LOG_NAME = re.compile(
    r"token_bytes_(\d+)_miss_count_(\d+)_aiv_cores_(\d+)_jettys_per_core_(\d+)\.log"
)


def fields(line):
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


def number(value, label, positive=False):
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"invalid {label}: {value!r}") from None
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"invalid {label}: {value!r}")
    return result


def parse_log(path, clock_override=None):
    config = {}
    profiles = {}
    contract = {}
    device = ""
    config_count = 0
    with path.open(encoding="utf-8", errors="replace") as log:
        for line in log:
            parts = line.split()
            if not parts:
                continue
            record = fields(line)
            if parts[0] == "BATCH_GET_PERF_CONFIG":
                config_count += 1
                if config_count > 1:
                    raise ValueError("multiple runs in one log; use the per-case .log files")
                config = record
            elif parts[0] == "KERNEL_PROFILE_CONTRACT":
                contract = record
            elif parts[0] == "PERF_RESULT" and record.get("operation") in STAGES:
                profiles[record["operation"]] = record
            elif parts[0] == "[KVAF_AIV]" and len(parts) > 1 and parts[1] in (
                "UB_DEVICE_SELECTED", "UB_DEVICE_REUSED"
            ):
                device = record.get("ub_device", "")
                if not device:
                    raise ValueError("UB device record has no ub_device")
            elif parts[0] in (
                "BATCH_GET_PERF_UNHANDLED_ERROR", "BATCH_GET_KERNEL_PROFILE_DISABLE_ERROR"
            ) or line.startswith("Traceback (most recent call last):"):
                raise ValueError("benchmark error reported in log")

    missing = STAGES.keys() - profiles.keys()
    if missing:
        raise ValueError(f"incomplete profile: missing {', '.join(sorted(missing))}")

    match = LOG_NAME.fullmatch(path.name)
    filename_shape = dict(zip(SHAPE_FIELDS, match.groups())) if match else {}
    row = {}
    for name in SHAPE_FIELDS:
        value = config.get(name, filename_shape.get(name, ""))
        if not value.isdigit() or (int(value) == 0 and name != "miss_count"):
            raise ValueError(f"missing or invalid {name}: {value!r}")
        row[name] = int(value)
        if name in filename_shape and row[name] != int(filename_shape[name]):
            raise ValueError(f"{name} differs between filename and log")

    clock = number(
        clock_override if clock_override is not None else contract.get(
            "clock_mhz", config.get("kernel_profile_clock_mhz", 1000)
        ),
        "clock_mhz", positive=True,
    )
    p50_cycles = None
    for operation, prefix in STAGES.items():
        profile = profiles[operation]
        for stat in STATS:
            name = f"{stat}_cycles"
            cycles = number(profile.get(name), f"{operation}.{name}")
            row[f"{prefix}{stat}_us"] = f"{cycles / clock:.3f}"
            if operation == "batch_get_kernel_total" and stat == "p50":
                p50_cycles = cycles
    payload_bytes = row["miss_count"] * row["token_bytes"]
    gbps = payload_bytes * clock / p50_cycles / 1000 if p50_cycles > 0 else 0
    row["p50_GBps"] = f"{gbps:.3f}"
    row["selected_ub_device"] = device
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs", nargs="+", type=Path,
        help="sweep log directory (searched recursively), or individual per-case logs",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="CSV path; defaults to LOG_DIR/batch_get_miss_sweep_summary.csv for one directory",
    )
    parser.add_argument(
        "--clock-mhz", type=float,
        help="override clock frequency; otherwise read each log, falling back to 1000 MHz",
    )
    args = parser.parse_args(argv)
    if args.clock_mhz is not None:
        try:
            number(args.clock_mhz, "--clock-mhz", positive=True)
        except ValueError as error:
            parser.error(str(error))
    if args.output is None:
        if len(args.logs) != 1 or not args.logs[0].is_dir():
            parser.error("--output is required unless the input is one log directory")
        args.output = args.logs[0] / "batch_get_miss_sweep_summary.csv"

    paths = set()
    for source in args.logs:
        if source.is_dir():
            paths.update(path.resolve() for path in source.rglob(LOG_PATTERN) if path.is_file())
        elif source.is_file():
            paths.add(source.resolve())
        else:
            parser.error(f"log path does not exist: {source}")
    if not paths:
        parser.error("no per-case logs found")
    if args.output.resolve() in paths:
        parser.error("output must not overwrite an input log")

    rows = []
    skipped = 0
    for path in sorted(paths):
        try:
            rows.append(parse_log(path, args.clock_mhz))
        except (OSError, ValueError) as error:
            print(f"SKIP path={path}: {error}", file=sys.stderr)
            skipped += 1
    if not rows:
        print("ERROR: no complete profiles; CSV was not written", file=sys.stderr)
        return 1
    rows.sort(key=lambda row: tuple(row[name] for name in SHAPE_FIELDS))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"BATCH_GET_MISS_SWEEP_CSV_WRITTEN path={args.output} "
        f"rows={len(rows)} skipped={skipped}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
