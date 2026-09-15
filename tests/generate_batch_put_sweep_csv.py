#!/usr/bin/env python3
"""Rebuild BatchPut end-to-end and optional kernel timing CSVs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import sys
from typing import Sequence


RUNTIME_RESULT_PREFIX = "PERF_RESULT operation=batch_put "
KERNEL_RESULT_PREFIX = "PERF_RESULT operation=batch_put_kernel "
HOST_RESULT_PREFIX = "PERF_RESULT operation=batch_put_host "
PERF_CONFIG_PREFIX = "BATCH_PUT_PERF_CONFIG "
RUNTIME_CONFIG_MARKER = "RUNTIME_BATCH_PUT_TRANSPORT_READY "
RUNTIME_REQUIRED_FIELDS = ("entry_bytes", "row_count", "mean_ms", "useful_GBps")
KERNEL_REQUIRED_FIELDS = (
    "entry_bytes",
    "row_count",
    "mean_execute_cycles",
    "mean_non_execute_cycles",
)
CONFIG_REQUIRED_FIELDS = ("aiv_cores", "jettys_per_core")
HOST_FIELDS = (
    "mean_caller_e2e_us",
    "mean_runtime_e2e_us",
    "mean_caller_overhead_us",
    "mean_runtime_non_execute_us",
    "mean_request_validation_us",
    "mean_prepare_put_us",
    "mean_metadata_reserve_us",
    "mean_metadata_pack_us",
    "mean_metadata_h2d_us",
    "mean_local_buffer_acquire_us",
    "mean_execute_sync_us",
    "mean_local_buffer_release_us",
    "mean_commit_put_us",
    "mean_metadata_bytes",
)
HOST_REQUIRED_FIELDS = HOST_FIELDS
BASE_CSV_FIELDS = (
    "entry_bytes",
    "row_count",
    "aiv_cores",
    "jettys_per_core",
    "mean_us",
    "mean_GBps",
    "host_profile",
    *HOST_FIELDS,
)
KERNEL_CSV_FIELDS = (
    "kernel_profile",
    "mean_kernel_execute_cycles",
    "kernel_execute_mean_us",
    "mean_kernel_non_execute_cycles",
    "kernel_non_execute_mean_us",
)
# Compatibility alias for callers that imported the original fixed schema.
CSV_FIELDS = BASE_CSV_FIELDS + KERNEL_CSV_FIELDS


@dataclass(frozen=True)
class LogResult:
    path: Path
    modified_ns: int
    runtime_fields: dict[str, str] | None
    kernel_fields: dict[str, str] | None
    host_fields: dict[str, str] | None
    aiv_cores: int
    jettys_per_core: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read completed BatchPut benchmark logs and rebuild an end-to-end "
            "CSV with optional AIV kernel timing columns."
        )
    )
    parser.add_argument(
        "log_dir",
        nargs="?",
        type=Path,
        default=Path("results/batch_put_sweep"),
        help="log directory to scan recursively (default: results/batch_put_sweep)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "output CSV path (default: "
            "<log_dir>/batch_put_sweep_summary.csv)"
        ),
    )
    parser.add_argument(
        "--clock-mhz",
        type=Decimal,
        default=Decimal(os.environ.get("KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ", "1000")),
        help=(
            "GetSystemCycle frequency used only for optional kernel timing "
            "(default: KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ or 1000)"
        ),
    )
    kernel_breakdown = parser.add_mutually_exclusive_group()
    kernel_breakdown.add_argument(
        "--kernel-breakdown",
        dest="kernel_breakdown",
        action="store_true",
        help="include optional AIV kernel timing columns",
    )
    kernel_breakdown.add_argument(
        "--no-kernel-breakdown",
        dest="kernel_breakdown",
        action="store_false",
        help="emit only end-to-end and host-side columns",
    )
    parser.set_defaults(
        kernel_breakdown=os.environ.get(
            "KVAF_BATCH_PUT_SWEEP_KERNEL_BREAKDOWN", "1"
        )
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    return parser.parse_args(argv)


def parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split():
        name, separator, value = token.partition("=")
        if separator:
            fields[name] = value
    return fields


def parse_config_fields(line: str) -> dict[str, str] | None:
    fields = parse_fields(line)
    if line.startswith(PERF_CONFIG_PREFIX):
        return {name: fields[name] for name in CONFIG_REQUIRED_FIELDS if name in fields}
    if RUNTIME_CONFIG_MARKER in line:
        config: dict[str, str] = {}
        if "cores" in fields:
            config["aiv_cores"] = fields["cores"]
        if "jettys_per_core" in fields:
            config["jettys_per_core"] = fields["jettys_per_core"]
        return config
    return None


def read_last_result(path: Path) -> LogResult | None:
    current_config: dict[str, str] | None = None
    latest_runtime: dict[str, str] | None = None
    latest_runtime_config: dict[str, str] | None = None
    latest_kernel: dict[str, str] | None = None
    latest_kernel_config: dict[str, str] | None = None
    latest_host: dict[str, str] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            config = parse_config_fields(line)
            if config is not None and all(
                name in config for name in CONFIG_REQUIRED_FIELDS
            ):
                current_config = config
            if line.startswith(RUNTIME_RESULT_PREFIX):
                latest_runtime = parse_fields(line)
                latest_runtime_config = current_config
                latest_host = None
            elif line.startswith(KERNEL_RESULT_PREFIX):
                latest_kernel = parse_fields(line)
                latest_kernel_config = current_config
            elif line.startswith(HOST_RESULT_PREFIX):
                latest_host = parse_fields(line)

    runtime_valid = latest_runtime is not None and all(
        name in latest_runtime for name in RUNTIME_REQUIRED_FIELDS
    )
    kernel_valid = latest_kernel is not None and all(
        name in latest_kernel for name in KERNEL_REQUIRED_FIELDS
    )
    if not runtime_valid and not kernel_valid:
        return None
    primary = latest_runtime if runtime_valid else latest_kernel
    config = latest_runtime_config if runtime_valid else latest_kernel_config
    if config is None:
        print(f"SKIP path={path} reason=missing_perf_config", file=sys.stderr)
        return None
    try:
        entry_bytes = int(primary["entry_bytes"])
        row_count = int(primary["row_count"])
        aiv_cores = int(config["aiv_cores"])
        jettys_per_core = int(config["jettys_per_core"])
    except (KeyError, ValueError):
        print(f"SKIP path={path} reason=invalid_required_numeric_field", file=sys.stderr)
        return None

    try:
        if runtime_valid:
            Decimal(latest_runtime["mean_ms"])
            Decimal(latest_runtime["useful_GBps"])
        if kernel_valid:
            Decimal(latest_kernel["mean_execute_cycles"])
            Decimal(latest_kernel["mean_non_execute_cycles"])
        if latest_host is not None:
            for name in HOST_REQUIRED_FIELDS:
                Decimal(latest_host[name])
    except (InvalidOperation, KeyError, ValueError):
        if not runtime_valid:
            print(f"SKIP path={path} reason=invalid_profile_numeric_field", file=sys.stderr)
            return None
        # A malformed optional kernel or host profile must not discard a
        # valid end-to-end result.
        latest_host = None
    if (
        entry_bytes <= 0
        or row_count <= 0
        or aiv_cores <= 0
        or jettys_per_core <= 0
    ):
        print(f"SKIP path={path} reason=non_positive_value", file=sys.stderr)
        return None
    return LogResult(
        path=path,
        modified_ns=path.stat().st_mtime_ns,
        runtime_fields=latest_runtime if runtime_valid else None,
        kernel_fields=latest_kernel if kernel_valid else None,
        host_fields=latest_host,
        aiv_cores=aiv_cores,
        jettys_per_core=jettys_per_core,
    )


def format_decimal(value: Decimal) -> str:
    return f"{value:.3f}"


def csv_row(
    result: LogResult, clock_mhz: Decimal, kernel_breakdown: bool
) -> dict[str, str]:
    if result.runtime_fields is not None:
        fields = result.runtime_fields
        entry_bytes = int(fields["entry_bytes"])
        row_count = int(fields["row_count"])
        mean_us = Decimal(fields["mean_ms"]) * Decimal(1000)
        mean_gbps = Decimal(fields["useful_GBps"])
    else:
        assert result.kernel_fields is not None
        fields = result.kernel_fields
        entry_bytes = int(fields["entry_bytes"])
        row_count = int(fields["row_count"])
        mean_cycles = Decimal(fields["mean_execute_cycles"]) + Decimal(
            fields["mean_non_execute_cycles"]
        )
        payload_bytes = Decimal(entry_bytes * row_count)
        mean_us = mean_cycles / clock_mhz
        mean_gbps = payload_bytes * clock_mhz / mean_cycles / Decimal(1000)
    row = {
        "entry_bytes": str(entry_bytes),
        "row_count": str(row_count),
        "aiv_cores": str(result.aiv_cores),
        "jettys_per_core": str(result.jettys_per_core),
        "mean_us": format_decimal(mean_us),
        "mean_GBps": format_decimal(mean_gbps),
        "host_profile": "1" if result.host_fields is not None else "0",
    }
    row.update(
        {
            name: (
                result.host_fields.get(name, "")
                if result.host_fields is not None
                else ""
            )
            for name in HOST_FIELDS
        }
    )
    if not kernel_breakdown:
        return row
    row["kernel_profile"] = "1" if result.kernel_fields is not None else "0"
    if result.kernel_fields is None:
        row.update({name: "" for name in KERNEL_CSV_FIELDS[1:]})
        return row
    execute_cycles = Decimal(result.kernel_fields["mean_execute_cycles"])
    non_execute_cycles = Decimal(result.kernel_fields["mean_non_execute_cycles"])
    row["mean_kernel_execute_cycles"] = format_decimal(execute_cycles)
    row["kernel_execute_mean_us"] = format_decimal(execute_cycles / clock_mhz)
    row["mean_kernel_non_execute_cycles"] = format_decimal(non_execute_cycles)
    row["kernel_non_execute_mean_us"] = format_decimal(
        non_execute_cycles / clock_mhz
    )
    return row


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    log_dir = args.log_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else log_dir / "batch_put_sweep_summary.csv"
    )
    if args.clock_mhz <= 0:
        print("ERROR: --clock-mhz must be positive", file=sys.stderr)
        return 2
    if not log_dir.is_dir():
        print(f"ERROR: log directory does not exist: {log_dir}", file=sys.stderr)
        return 2

    selected: dict[tuple[int, int, int, int], LogResult] = {}
    scanned = 0
    skipped = 0
    duplicates = 0
    for path in sorted(log_dir.rglob("*.log")):
        scanned += 1
        result = read_last_result(path)
        if result is None:
            skipped += 1
            continue
        fields = result.runtime_fields or result.kernel_fields
        assert fields is not None
        key = (
            int(fields["entry_bytes"]),
            int(fields["row_count"]),
            result.aiv_cores,
            result.jettys_per_core,
        )
        previous = selected.get(key)
        if previous is not None:
            duplicates += 1
            if (result.modified_ns, str(result.path)) <= (
                previous.modified_ns,
                str(previous.path),
            ):
                continue
        selected[key] = result

    if not selected:
        print(
            f"ERROR: no completed BatchPut performance results under {log_dir}",
            file=sys.stderr,
        )
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    fields = BASE_CSV_FIELDS + (
        KERNEL_CSV_FIELDS if args.kernel_breakdown else ()
    )
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key in sorted(selected):
            writer.writerow(
                csv_row(selected[key], args.clock_mhz, args.kernel_breakdown)
            )
    temporary.replace(output)

    print(
        "BATCH_PUT_SWEEP_CSV_REBUILT "
        f"path={output} rows={len(selected)} logs_scanned={scanned} "
        f"incomplete_skipped={skipped} duplicates_replaced={duplicates} "
        f"clock_mhz={format_decimal(args.clock_mhz)} "
        f"kernel_breakdown={int(args.kernel_breakdown)} format=e2e_host"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
