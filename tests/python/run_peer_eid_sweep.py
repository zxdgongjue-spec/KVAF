#!/usr/bin/env python3
"""Compare shared and distinct CPU EIDs at 2/4/8 AIV cores on real hardware."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from npu_eids import parse_npu_eids


PERCENTILES = ("mean", "p50", "p90", "p99")
SUMMARY_FIELDS = [
    "case", "mode", "status", "error", "token_bytes", "tokens_per_block",
    "request_length", "miss_count", "table_capacity", "aiv_cores", "jettys_per_core",
    "max_inflight_total", "npu_eid", "peer_eids_by_core", "warmup", "iterations",
    "clock_mhz",
    *[f"{stat}_us" for stat in PERCENTILES], "p50_GBps",
    *[f"post_{stat}_us" for stat in PERCENTILES],
    *[f"completion_wait_{stat}_us" for stat in PERCENTILES],
    *[f"batch_get_{stat}_ms" for stat in PERCENTILES], "batch_get_useful_GiBps",
    "log_file",
]


def build_cases(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    if not args.cores or not args.jettys or any(c < 1 or c > 48 for c in args.cores):
        raise ValueError("cores must be in [1, 48]")
    if any(j < 1 or j > 64 for j in args.jettys):
        raise ValueError("jettys must be in [1, 64]")
    if len(set(args.cores)) != len(args.cores) or len(set(args.jettys)) != len(args.jettys):
        raise ValueError("cores and jettys must not contain duplicates")
    if len(args.peer_eids) < max(args.cores):
        raise ValueError("provide at least one distinct CPU EID per core in the largest case")
    if not 0 <= args.max_inflight_total <= 0xFFFFFFFF or (
        args.max_inflight_total and args.max_inflight_total < 2 * max(args.cores)
    ):
        raise ValueError("max-inflight-total must be zero or at least twice the largest core count")
    if args.requests < max(args.cores) * ((max(args.jettys) - 1) * 128 + 1):
        raise ValueError("requests must cover every core and Jetty (at least cores * ((jettys-1)*128+1))")
    runner = Path(__file__).resolve().parents[1] / "run_remote_cpu_batch_get_perf.sh"
    cases = []
    for cores in args.cores:
        for jettys in args.jettys:
            for mode in ("shared", "distinct"):
                eids = [args.peer_eids[0]] * cores if mode == "shared" else args.peer_eids[:cores]
                command = [
                    "bash", str(runner), str(args.build_dir.resolve()),
                    "--aiv-cores", str(cores), "--jettys-per-core", str(jettys),
                    "--peer-eids-by-core", ",".join(eids),
                    "--max-inflight-total", str(args.max_inflight_total),
                    "--request-length", str(args.requests), "--miss-count", str(args.requests),
                    "--table-capacity", str(args.table_capacity),
                    "--token-bytes", str(args.token_bytes),
                    "--tokens-per-block", str(args.tokens_per_block),
                    "--warmup", str(args.warmup), "--iterations", str(args.iterations),
                    "--seed", "20260903", "--kernel-profile",
                    "--kernel-profile-clock-mhz", str(args.kernel_profile_clock_mhz),
                ]
                if args.npu_eid:
                    command += ["--npu-eids", args.npu_eid]
                cases.append((f"cores_{cores}_jettys_{jettys}_{mode}", command))
    return cases


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--peer-eids", type=parse_npu_eids,
                        default=parse_npu_eids(os.environ.get("KVAF_PEER_EID_CANDIDATES")))
    parser.add_argument("--npu-eid", default="", help="one fixed NPU EID for comparable runs")
    parser.add_argument("--cores", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--jettys", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-inflight-total", type=int, default=2048)
    parser.add_argument("--requests", type=int, default=8192)
    parser.add_argument("--table-capacity", type=int, default=64)
    parser.add_argument("--token-bytes", type=int, default=656)
    parser.add_argument("--tokens-per-block", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--kernel-profile-clock-mhz", type=int,
                        default=os.environ.get("KVAF_BATCH_GET_KERNEL_CLOCK_MHZ", "1000"),
                        help="GetSystemCycle frequency used for AIV time conversion (default: 1000)")
    parser.add_argument("--output-dir", "--log-dir", dest="output_dir", type=Path,
                        default=os.environ.get("KVAF_PEER_EID_SWEEP_LOG_DIR") or "peer_eid_sweep_logs",
                        help="log directory (environment: KVAF_PEER_EID_SWEEP_LOG_DIR)")
    parser.add_argument("--csv", type=Path,
                        default=os.environ.get("KVAF_PEER_EID_SWEEP_CSV") or None,
                        help="summary path; defaults to LOG_DIR/peer_eid_sweep_summary.csv "
                             "(environment: KVAF_PEER_EID_SWEEP_CSV)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        local_eids = parse_npu_eids(args.npu_eid or os.environ.get("KVAF_NPU_EIDS"))
        if len(local_eids) != 1:
            raise ValueError("set --npu-eid (or KVAF_NPU_EIDS) to exactly one NPU EID for the sweep")
        args.npu_eid = local_eids[0]
        if min(args.table_capacity, args.token_bytes, args.tokens_per_block, args.iterations) <= 0 or args.warmup < 0:
            raise ValueError("shape and iteration counts must be positive; warmup must be non-negative")
        if args.kernel_profile_clock_mhz <= 0:
            raise ValueError("kernel-profile-clock-mhz must be positive")
        build_cases(args)
    except ValueError as error:
        parser.error(str(error))
    args.output_dir = args.output_dir.expanduser().resolve()
    args.csv = (args.csv or args.output_dir / "peer_eid_sweep_summary.csv").expanduser().resolve()
    return args


def parse_results(contents: str, clock_mhz: int, payload_bytes: int) -> dict[str, str]:
    for marker in ("BATCH_GET_PERF_CORRECTNESS_PASS", "BATCH_GET_PERF_PHASE_DONE phase=batch_get",
                   "PEER_ENDPOINT_READ_VALIDATED round=0", "PEER_ENDPOINT_READ_VALIDATED round=1"):
        if marker not in contents:
            raise ValueError(f"case skipped or validation incomplete: missing {marker}")
    results = {}
    for line in contents.splitlines():
        if line.startswith("PERF_RESULT "):
            fields = dict(field.split("=", 1) for field in line.split()[1:] if "=" in field)
            results[fields.get("operation", "")] = fields

    def number(operation: str, field: str) -> float:
        try:
            value = float(results[operation][field])
        except (KeyError, ValueError) as error:
            raise ValueError(f"missing or invalid PERF_RESULT {operation} {field}") from error
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid PERF_RESULT {operation} {field}: {value}")
        return value

    metrics = {}
    # Convert cycles directly; rounded milliseconds lose precision for short kernels.
    for operation, prefix in (("batch_get_kernel_total", ""), ("batch_get_kernel_post", "post_"),
                              ("batch_get_kernel_completion_wait", "completion_wait_")):
        for stat in PERCENTILES:
            metrics[f"{prefix}{stat}_us"] = f"{number(operation, stat + '_cycles') / clock_mhz:.3f}"
    p50_cycles = number("batch_get_kernel_total", "p50_cycles")
    if p50_cycles == 0:
        raise ValueError("batch_get_kernel_total p50_cycles must be positive")
    metrics["p50_GBps"] = f"{payload_bytes * clock_mhz / p50_cycles / 1000:.3f}"
    for stat in PERCENTILES:
        metrics[f"batch_get_{stat}_ms"] = f"{number('batch_get', stat + '_ms'):.3f}"
    metrics["batch_get_useful_GiBps"] = f"{number('batch_get', 'useful_GiBps'):.3f}"
    return metrics


def case_summary(args: argparse.Namespace, name: str, command: list[str], log: Path) -> dict:
    def option(flag: str) -> str:
        return command[command.index(flag) + 1]

    return {
        "case": name, "mode": name.rsplit("_", 1)[1], "status": "PASS", "error": "",
        "token_bytes": args.token_bytes, "tokens_per_block": args.tokens_per_block,
        "request_length": args.requests, "miss_count": args.requests,
        "table_capacity": args.table_capacity, "aiv_cores": option("--aiv-cores"),
        "jettys_per_core": option("--jettys-per-core"),
        "max_inflight_total": args.max_inflight_total, "npu_eid": args.npu_eid,
        "peer_eids_by_core": option("--peer-eids-by-core"), "warmup": args.warmup,
        "iterations": args.iterations, "clock_mhz": args.kernel_profile_clock_mhz,
        "log_file": str(log),
    }


def run_sweep(args: argparse.Namespace) -> int:
    cases = build_cases(args)
    print(f"PEER_EID_SWEEP_LOG_DIR path={args.output_dir}", flush=True)
    print(f"PEER_EID_SWEEP_CSV path={args.csv}", flush=True)
    if args.dry_run:
        for name, command in cases:
            print(f"{name}: {shlex.join(command)}", flush=True)
        return 0

    logs = [args.output_dir / f"{name}.log" for name, _ in cases]
    if args.csv in logs:
        raise ValueError("CSV path must differ from every case log path")
    # Check all outputs before starting hardware work; exclusive creation also prevents overwrites.
    for path in [args.csv, *logs]:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    passed = failed = 0
    with args.csv.open("x", newline="", encoding="utf-8") as summary:
        writer = csv.DictWriter(summary, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        summary.flush()
        try:
            for (name, command), log in zip(cases, logs):
                print(f"{name}: {shlex.join(command)}", flush=True)
                row = case_summary(args, name, command, log)
                try:
                    with log.open("x", encoding="utf-8") as output:
                        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=True)
                    row.update(parse_results(log.read_text(encoding="utf-8"), args.kernel_profile_clock_mhz,
                                             args.requests * args.token_bytes))
                except (OSError, ValueError, subprocess.CalledProcessError) as error:
                    row.update(status="FAIL", error=str(error))
                    failed += 1
                    writer.writerow(row)
                    summary.flush()
                    print(f"FAIL {name}: {error}; log={log}", file=sys.stderr, flush=True)
                    return 1
                writer.writerow(row)
                summary.flush()
                passed += 1
                print(f"PASS {name}: {log}", flush=True)
        finally:
            print(f"PEER_EID_SWEEP_SUMMARY planned={len(cases)} passed={passed} failed={failed}", flush=True)
            print(f"PEER_EID_SWEEP_CSV_WRITTEN path={args.csv} rows={passed + failed}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_sweep(args)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
