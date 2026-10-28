#!/usr/bin/env python3
"""Export existing baseline/small-post A/B logs to run, pair and summary CSVs.

This script uses only the Python standard library and never starts a benchmark.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re
import statistics
import sys


LOG_NAME = re.compile(r"r([1-9][0-9]*)_n(0|[1-9][0-9]*)_(baseline|small-post)\.log")
SHAPE_FIELDS = ("token_bytes", "miss_count", "aiv_cores", "jettys_per_core")
CONFIG_FIELDS = SHAPE_FIELDS + (
    "request_length", "table_capacity", "tokens_per_block", "max_inflight_total",
    "source_access_pattern", "output_access_pattern", "sort_input_indices",
    "seed", "warmup", "iterations", "npu_eids", "eid_paths", "peer_ub_device",
    "adaptive_ub_device", "adaptive_ub_miss_count", "selected_ub_device", "clock_mhz",
)
STAGES = {
    "batch_get_kernel_total": "",
    "batch_get_kernel_wall": "wall_",
    "batch_get_kernel_post": "post_",
    "batch_get_kernel_completion_wait": "completion_wait_",
}
STATS = ("mean", "p50", "p90", "p99")
RUN_METRICS = tuple(prefix + stat + "_us" for prefix in STAGES.values() for stat in STATS)
PAIR_METRICS = (
    "mean_us", "p50_us", "p90_us", "p99_us", "p50_GBps",
    "wall_p50_us", "wall_p99_us", "post_p50_us", "completion_wait_p50_us",
    "posts_per_call", "completions_per_call",
)
DELTA_FIELDS = (
    "p50_speedup", "p50_latency_reduction_pct", "p99_latency_reduction_pct",
    "p50_bandwidth_gain_pct", "post_p50_change_us", "completion_wait_p50_change_us",
)
PAIR_VALUES = tuple(
    variant + "_" + metric
    for metric in PAIR_METRICS for variant in ("baseline", "small_post")
) + DELTA_FIELDS


def fields(line: str) -> dict[str, str]:
    return dict(word.split("=", 1) for word in line.split() if "=" in word)


def number(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"expected a finite non-negative number, got {value!r}")
    return result


def parse_run(path: Path) -> dict:
    try:
        return _parse_run(path)
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def _parse_run(path: Path) -> dict:
    match = LOG_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError("expected filename r<round>_n<count>_<baseline|small-post>.log")
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for marker in ("BATCH_GET_PERF_PHASE_DONE phase=batch_get",
                   "BATCH_GET_PERF_PHASE_DONE phase=shutdown"):
        if marker not in lines:
            raise ValueError(f"incomplete benchmark: missing {marker}")
    if any(line.startswith(("FAIL:", "BATCH_GET_PERF_UNHANDLED_ERROR ",
                            "BATCH_GET_KERNEL_PROFILE_DISABLE_ERROR ",
                            "Traceback (most recent call last):")) for line in lines):
        raise ValueError("benchmark reported a failure")

    configs = [fields(line) for line in lines if line.startswith("BATCH_GET_PERF_CONFIG ")]
    contracts = [fields(line) for line in lines if line.startswith("KERNEL_PROFILE_CONTRACT ")]
    if len(configs) != 1 or len(contracts) != 1:
        raise ValueError("expected one PERF_CONFIG and one KERNEL_PROFILE_CONTRACT; use --kernel-profile")
    config, contract = configs[0], contracts[0]
    clock = int(contract["clock_mhz"])
    if clock <= 0 or clock != int(config["kernel_profile_clock_mhz"]):
        raise ValueError("invalid or inconsistent profile clock MHz")
    if int(config["miss_count"]) != int(match[2]):
        raise ValueError("filename miss_count differs from PERF_CONFIG")
    if any(config[name] != contract[name] for name in
           ("aiv_cores", "jettys_per_core", "request_length", "miss_count")):
        raise ValueError("profile contract differs from PERF_CONFIG")

    selected_device = ""
    for line in lines:
        if line.startswith(("[KVAF_AIV] UB_DEVICE_SELECTED ", "[KVAF_AIV] UB_DEVICE_REUSED ")):
            selected_device = fields(line)["ub_device"]
    row = {name: config[name] for name in CONFIG_FIELDS
           if name not in ("selected_ub_device", "clock_mhz")}
    row.update(round=int(match[1]), variant=match[3], clock_mhz=clock,
               selected_ub_device=selected_device, log_file=str(path.resolve()))

    results = {}
    for line in lines:
        if line.startswith("PERF_RESULT "):
            record = fields(line)
            operation = record["operation"]
            if operation in results:
                raise ValueError(f"duplicate PERF_RESULT operation={operation}")
            results[operation] = record
    for operation, prefix in STAGES.items():
        if operation not in results:
            raise ValueError(f"missing PERF_RESULT operation={operation}")
        record = results[operation]
        if (int(record["samples"]) != int(config["iterations"]) or
                int(record["warmup_excluded"]) != int(config["warmup"])):
            raise ValueError(f"sample/warmup count mismatch for {operation}")
        for stat in STATS:
            # Do not use *_ms: the upstream formatter rounds it to 1 us.
            row[prefix + stat + "_us"] = number(record[stat + "_cycles"]) / clock
    if row["p50_us"] <= 0 or row["p99_us"] <= 0:
        raise ValueError("kernel total P50/P99 must be positive")
    row["payload_bytes"] = int(config["miss_count"]) * int(config["token_bytes"])
    row["p50_GBps"] = row["payload_bytes"] / row["p50_us"] / 1000.0

    cores = [fields(line) for line in lines if line.startswith("KERNEL_PROFILE_CORE ")]
    if sorted(int(core["core"]) for core in cores) != list(range(int(config["aiv_cores"]))):
        raise ValueError("incomplete or duplicate per-core profile")
    if any(int(core["samples"]) != int(config["iterations"]) for core in cores):
        raise ValueError("per-core profile sample count mismatch")
    active = sum(number(core["active_requests_mean"]) for core in cores)
    if not math.isclose(active, int(config["miss_count"]), rel_tol=0, abs_tol=1e-4):
        raise ValueError("per-core active request sum differs from miss_count")
    row["posts_per_call"] = sum(number(core["total_posts_mean"]) for core in cores)
    row["completions_per_call"] = sum(number(core["completion_count_mean"]) for core in cores)
    return row


def compare_runs(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = {}
    for run in runs:
        key = (run["round"], *(run[name] for name in SHAPE_FIELDS))
        pair = grouped.setdefault(key, {})
        if run["variant"] in pair:
            raise ValueError(f"duplicate {run['variant']} run for round/shape={key}")
        pair[run["variant"]] = run

    pairs = []
    for key in sorted(grouped, key=lambda key: tuple(int(value) for value in key)):
        variants = grouped[key]
        if set(variants) != {"baseline", "small-post"}:
            raise ValueError(f"missing baseline/small-post pair for round/shape={key}")
        base, small = variants["baseline"], variants["small-post"]
        mismatches = [name for name in CONFIG_FIELDS if base[name] != small[name]]
        if mismatches:
            raise ValueError(f"configuration mismatch for round/shape={key}: {', '.join(mismatches)}")
        pair = {name: base[name] for name in CONFIG_FIELDS}
        pair["round"] = base["round"]
        for metric in PAIR_METRICS:
            pair["baseline_" + metric] = base[metric]
            pair["small_post_" + metric] = small[metric]
        pair.update(
            p50_speedup=base["p50_us"] / small["p50_us"],
            p50_latency_reduction_pct=100 * (1 - small["p50_us"] / base["p50_us"]),
            p99_latency_reduction_pct=100 * (1 - small["p99_us"] / base["p99_us"]),
            p50_bandwidth_gain_pct=(100 * (small["p50_GBps"] / base["p50_GBps"] - 1)
                                     if base["p50_GBps"] else ""),
            post_p50_change_us=small["post_p50_us"] - base["post_p50_us"],
            completion_wait_p50_change_us=(small["completion_wait_p50_us"] -
                                           base["completion_wait_p50_us"]),
            baseline_log=base["log_file"], small_post_log=small["log_file"],
        )
        pairs.append(pair)

    configurations = {}
    for pair in pairs:
        configurations.setdefault(tuple(pair[name] for name in CONFIG_FIELDS), []).append(pair)
    summary = []
    for group in configurations.values():
        row = {name: group[0][name] for name in CONFIG_FIELDS}
        row.update(rounds=len(group), round_ids=";".join(str(pair["round"]) for pair in group))
        for metric in PAIR_VALUES:
            values = [pair[metric] for pair in group if pair[metric] != ""]
            row["median_" + metric] = statistics.median(values) if values else ""
        summary.append(row)
    return pairs, summary


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.6f}" if isinstance(value, float) else value
                             for key, value in row.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="directory containing r*_n*_<variant>.log; searched recursively")
    parser.add_argument("--output-dir", type=Path, help="CSV destination; defaults to log_dir")
    args = parser.parse_args(argv)
    try:
        if not args.log_dir.is_dir():
            raise ValueError(f"log directory does not exist: {args.log_dir}")
        paths = sorted(args.log_dir.rglob("r*_n*_*.log"))
        if not paths:
            raise ValueError(f"no A/B logs found under {args.log_dir}")
        runs = [parse_run(path) for path in paths]
        runs.sort(key=lambda row: (row["round"], *(int(row[name]) for name in SHAPE_FIELDS), row["variant"]))
        pairs, summary = compare_runs(runs)
        output = args.output_dir or args.log_dir
        output.mkdir(parents=True, exist_ok=True)
        outputs = (
            ("small_post_runs.csv", runs,
             ("round", "variant") + CONFIG_FIELDS + ("payload_bytes",) + RUN_METRICS +
             ("p50_GBps", "posts_per_call", "completions_per_call", "log_file")),
            ("small_post_pairs.csv", pairs,
             ("round",) + CONFIG_FIELDS + PAIR_VALUES + ("baseline_log", "small_post_log")),
            ("small_post_summary.csv", summary,
             CONFIG_FIELDS + ("rounds", "round_ids") + tuple("median_" + name for name in PAIR_VALUES)),
        )
        # Validate every log and pair before creating any result CSV.
        for filename, rows, columns in outputs:
            path = output / filename
            write_csv(path, rows, columns)
            print(f"SMALL_POST_CSV_WRITTEN path={path} rows={len(rows)}")
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
