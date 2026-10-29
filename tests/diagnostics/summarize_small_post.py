#!/usr/bin/env python3
"""Export baseline/small-post logs and compare core/Jetty sweep configurations.

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


LOG_NAME = re.compile(
    r"r(?P<round>[1-9][0-9]*)_n(?P<count>0|[1-9][0-9]*)_"
    r"(?:(?:c(?P<cores>[1-9][0-9]*)_j(?P<jettys>[1-9][0-9]*)|"
    r"(?P<legacy_jettys>[1-9][0-9]*))_)?"
    r"(?P<variant>baseline|small-post)\.log"
)
SHAPE_FIELDS = ("token_bytes", "miss_count", "aiv_cores", "jettys_per_core")
SHAPE_METRICS = ("reads_per_core_min", "reads_per_core_max", "reads_per_core_mean",
                 "configured_jettys_total")
CONFIG_FIELDS = SHAPE_FIELDS + (
    "request_length", "table_capacity", "tokens_per_block", "max_inflight_total",
    "source_access_pattern", "output_access_pattern", "sort_input_indices",
    "seed", "warmup", "iterations", "npu_eids", "eid_paths", "peer_ub_device",
    "adaptive_ub_device", "adaptive_ub_miss_count", "selected_ub_device", "clock_mhz",
    "completion_poll_mode", "profile_version",
)
STAGES = {
    "batch_get_kernel_total": "",
    "batch_get_kernel_wall": "wall_",
    "batch_get_kernel_post": "post_",
    "batch_get_kernel_completion_wait": "completion_wait_",
}
OPTIONAL_STAGES = {
    "batch_get_kernel_first_doorbell": "first_doorbell_",
    "batch_get_kernel_transfer_window": "transfer_window_",
    "batch_get_kernel_setup": "setup_",
    "batch_get_kernel_finalize": "finalize_",
}
SCHEDULE_METRICS = ("small_post_groups_per_call", "fused_groups_per_call", "wqe_build_calls_per_call")
PROOF_METRICS = ("small_post_max_requests", "post_width_min", "post_width_max",
                 "active_jettys_per_call")
JETTY_METRICS = (
    "active_samples", "reads_per_call", "posts_per_call", "completions_per_call",
    "payload_bytes_per_call", "polls_per_call", "probe_active_us_mean",
    "first_doorbell_offset_us_mean", "last_doorbell_offset_us_mean",
    "first_cqe_seen_offset_us_mean", "last_cqe_seen_offset_us_mean",
    "first_db_to_last_cqe_us_mean", "first_db_to_last_cqe_us_p50", "first_db_to_last_cqe_us_p99",
    "last_db_to_last_cqe_us_mean", "last_db_to_last_cqe_us_p50", "last_db_to_last_cqe_us_p99",
)
SCHEDULE_FIELDS = ("small_post_max_requests", "post_width_min", "post_width_max",
                   "small_post_active_samples") + SCHEDULE_METRICS
STATS = ("mean", "p50", "p90", "p99")
RUN_METRICS = tuple(prefix + stat + "_us"
                    for prefix in (*STAGES.values(), *OPTIONAL_STAGES.values())
                    for stat in STATS)
PAIR_METRICS = (
    "mean_us", "p50_us", "p90_us", "p99_us", "p50_GBps",
    "wall_p50_us", "wall_p99_us", "post_p50_us", "completion_wait_p50_us",
    "posts_per_call", "completions_per_call",
    "first_doorbell_p50_us", "first_doorbell_p99_us",
    "transfer_window_p50_us", "transfer_window_p99_us",
    "setup_p50_us", "finalize_p50_us",
) + SCHEDULE_METRICS + PROOF_METRICS
DELTA_FIELDS = (
    "p50_speedup", "p50_latency_reduction_pct", "p99_latency_reduction_pct",
    "p50_bandwidth_gain_pct", "post_p50_change_us", "completion_wait_p50_change_us",
    "first_doorbell_p50_change_us", "transfer_window_p50_change_us",
    "setup_p50_change_us", "finalize_p50_change_us",
)
PAIR_VALUES = tuple(
    variant + "_" + metric
    for metric in PAIR_METRICS for variant in ("baseline", "small_post")
) + DELTA_FIELDS
JETTY_PAIR_VALUES = tuple(
    prefix + metric
    for metric in PAIR_METRICS for prefix in ("reference_", "candidate_")
) + DELTA_FIELDS
JETTY_SUMMARY_COUNTS = ("p50_win_rounds", "p99_regression_rounds",
                        "worst_p50_latency_reduction_pct", "worst_p99_latency_reduction_pct")


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
        raise ValueError("expected r<round>_n<count>_[c<cores>_j<jettys>_|<jettys>_]"
                         "<baseline|small-post>.log")
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
    if int(config["miss_count"]) != int(match["count"]):
        raise ValueError("filename miss_count differs from PERF_CONFIG")
    for name, value in (("aiv_cores", match["cores"]),
                        ("jettys_per_core", match["jettys"] or match["legacy_jettys"])):
        if value is not None and int(config[name]) != int(value):
            raise ValueError(f"filename {name} differs from PERF_CONFIG")
    if any(int(config[name]) <= 0 for name in ("aiv_cores", "jettys_per_core", "iterations")):
        raise ValueError("core, Jetty and iteration counts must be positive")
    if any(config[name] != contract[name] for name in
           ("aiv_cores", "jettys_per_core", "request_length", "miss_count")):
        raise ValueError("profile contract differs from PERF_CONFIG")

    selected_device = ""
    for line in lines:
        if line.startswith(("[KVAF_AIV] UB_DEVICE_SELECTED ", "[KVAF_AIV] UB_DEVICE_REUSED ")):
            selected_device = fields(line)["ub_device"]
    row = {name: config[name] for name in CONFIG_FIELDS
           if name not in ("selected_ub_device", "clock_mhz", "completion_poll_mode", "profile_version")}
    row.update(round=int(match["round"]), variant=match["variant"], clock_mhz=clock,
               selected_ub_device=selected_device, log_file=str(path.resolve()),
               completion_poll_mode=contract.get("completion_poll_mode", "serial"),
               profile_version=contract.get("profile_version", "0"))
    if row["completion_poll_mode"] not in ("serial", "round_robin"):
        raise ValueError("unknown completion polling mode")

    results = {}
    for line in lines:
        if line.startswith("PERF_RESULT "):
            record = fields(line)
            operation = record["operation"]
            if operation in results:
                raise ValueError(f"duplicate PERF_RESULT operation={operation}")
            results[operation] = record
    for operation, prefix in (*STAGES.items(), *OPTIONAL_STAGES.items()):
        if operation not in results:
            if operation in OPTIONAL_STAGES:
                # Old benchmark logs have no first-doorbell stage. An absent
                # measurement is distinct from the zero for an idle core.
                row.update({prefix + stat + "_us": "" for stat in STATS})
                continue
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
    core_reads = [number(core["active_requests_mean"]) for core in cores]
    row.update(reads_per_core_min=min(core_reads), reads_per_core_max=max(core_reads),
               reads_per_core_mean=active / len(cores),
               configured_jettys_total=len(cores) * int(config["jettys_per_core"]))
    schedules = [fields(line) for line in lines if line.startswith("KERNEL_PROFILE_SCHEDULE ")]
    jettys = [fields(line) for line in lines if line.startswith("KERNEL_PROFILE_JETTY ")]
    row["_jettys"] = []
    row.update({name: "" for name in SCHEDULE_METRICS + PROOF_METRICS})
    if schedules or jettys or int(contract.get("profile_version", 0)) >= 3:
        core_count, jetty_count = int(config["aiv_cores"]), int(config["jettys_per_core"])
        if sorted(int(s["core"]) for s in schedules) != list(range(core_count)):
            raise ValueError("incomplete or duplicate scheduling proof")
        if sorted((int(j["core"]), int(j["jetty"])) for j in jettys) != [
                (c, j) for c in range(core_count) for j in range(jetty_count)]:
            raise ValueError("incomplete or duplicate per-Jetty profile")
        for name in SCHEDULE_METRICS:
            row[name] = sum(number(s[name]) for s in schedules)
        for core in cores:
            core_id = int(core["core"])
            schedule = next(s for s in schedules if int(s["core"]) == core_id)
            if int(schedule["samples"]) != int(config["iterations"]):
                raise ValueError("scheduling proof sample count mismatch")
            for name in SCHEDULE_FIELDS:
                number(schedule[name])
            if not 1 <= int(schedule["post_width_min"]) <= int(schedule["post_width_max"]) <= 128:
                raise ValueError("invalid selected post width")
            traces = [j for j in jettys if int(j["core"]) == core_id]
            for metric, core_metric in (("reads_per_call", "active_requests_mean"),
                                        ("posts_per_call", "total_posts_mean"),
                                        ("completions_per_call", "completion_count_mean")):
                if not math.isclose(sum(number(j[metric]) for j in traces), number(core[core_metric]),
                                    rel_tol=0, abs_tol=1e-4):
                    raise ValueError(f"per-Jetty {metric} differs from core counters")
            for trace in traces:
                if (int(trace["samples"]) != int(config["iterations"]) or
                        trace["completion_poll_mode"] != row["completion_poll_mode"]):
                    raise ValueError("per-Jetty sample count or polling mode mismatch")
                record = dict(core=core_id, jetty=int(trace["jetty"]))
                record.update({name: number(schedule[name]) for name in SCHEDULE_FIELDS})
                record.update({name: "" if trace[name] == "n/a" and "_us_" in name
                               else number(trace[name])
                               for name in JETTY_METRICS})
                active = record["active_samples"]
                if active != int(active) or not 0 <= active <= int(config["iterations"]):
                    raise ValueError("invalid per-Jetty active sample count")
                if (active == 0) != (record["reads_per_call"] == 0):
                    raise ValueError("per-Jetty active samples differ from READ count")
                timing = [name for name in JETTY_METRICS if "_us_" in name
                          and name != "probe_active_us_mean"]
                if any((record[name] == "") != (active == 0) for name in timing):
                    raise ValueError("missing active or populated idle per-Jetty timing")
                probe_available = active > 0 and row["completion_poll_mode"] == "round_robin"
                if (record["probe_active_us_mean"] != "") != probe_available:
                    raise ValueError("per-Jetty probe timing differs from polling mode")
                if not math.isclose(record["payload_bytes_per_call"],
                                    record["reads_per_call"] * int(config["token_bytes"]),
                                    rel_tol=0, abs_tol=1e-4):
                    raise ValueError("per-Jetty payload differs from READ count")
                row["_jettys"].append(record)
        limits = {int(s["small_post_max_requests"]) for s in schedules}
        if len(limits) != 1:
            raise ValueError("per-core small-post limits differ")
        row.update(small_post_max_requests=limits.pop(),
                   post_width_min=min(int(s["post_width_min"]) for s in schedules),
                   post_width_max=max(int(s["post_width_max"]) for s in schedules),
                   active_jettys_per_call=sum(j["active_samples"] for j in row["_jettys"]) /
                                         int(config["iterations"]))
    return row


def comparison_deltas(reference: dict, candidate: dict) -> dict:
    result = dict(
        p50_speedup=reference["p50_us"] / candidate["p50_us"],
        p50_latency_reduction_pct=100 * (1 - candidate["p50_us"] / reference["p50_us"]),
        p99_latency_reduction_pct=100 * (1 - candidate["p99_us"] / reference["p99_us"]),
        p50_bandwidth_gain_pct=(100 * (candidate["p50_GBps"] / reference["p50_GBps"] - 1)
                                if reference["p50_GBps"] else ""),
        post_p50_change_us=candidate["post_p50_us"] - reference["post_p50_us"],
        completion_wait_p50_change_us=(candidate["completion_wait_p50_us"] -
                                       reference["completion_wait_p50_us"]),
    )
    for prefix in OPTIONAL_STAGES.values():
        metric = prefix + "p50_us"
        result[prefix + "p50_change_us"] = (
            candidate[metric] - reference[metric]
            if reference[metric] != "" and candidate[metric] != "" else ""
        )
    return result


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
        pair.update({name: base[name] for name in SHAPE_METRICS})
        pair["round"] = base["round"]
        for metric in PAIR_METRICS:
            available = base[metric] != "" and small[metric] != ""
            pair["baseline_" + metric] = base[metric] if available else ""
            pair["small_post_" + metric] = small[metric] if available else ""
        pair.update(comparison_deltas(base, small))
        pair.update(baseline_log=base["log_file"], small_post_log=small["log_file"])
        pairs.append(pair)

    configurations = {}
    for pair in pairs:
        key = tuple(pair[name] for name in CONFIG_FIELDS) + (
            pair["baseline_small_post_max_requests"], pair["small_post_small_post_max_requests"])
        configurations.setdefault(key, []).append(pair)
    summary = []
    for group in configurations.values():
        row = {name: group[0][name] for name in CONFIG_FIELDS}
        row.update({name: group[0][name] for name in SHAPE_METRICS})
        row.update(rounds=len(group), round_ids=";".join(str(pair["round"]) for pair in group))
        for prefix in OPTIONAL_STAGES.values():
            row[prefix + "rounds"] = sum(
                pair["baseline_" + prefix + "p50_us"] != "" for pair in group
            )
        for metric in PAIR_VALUES:
            values = [pair[metric] for pair in group if pair[metric] != ""]
            row["median_" + metric] = statistics.median(values) if values else ""
        summary.append(row)
    return pairs, summary


def compare_jetty_scaling(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pair every configuration with the same-round/core baseline using one Jetty.

    A missing reference leaves the original A/B export usable, but produces no
    Jetty comparison for that round/core. Never substitute a different round or
    core count, or silently compare different routing/profile configurations.
    """
    grouped = {}
    shape = tuple(name for name in SHAPE_FIELDS if name != "jettys_per_core")
    for run in runs:
        key = (run["round"], *(run[name] for name in shape))
        candidates = grouped.setdefault(key, {})
        identity = (run["variant"], int(run["jettys_per_core"]))
        if identity in candidates:
            raise ValueError(f"duplicate Jetty candidate for round/shape={key}: {identity}")
        candidates[identity] = run

    pairs = []
    for key in sorted(grouped, key=lambda key: tuple(int(value) for value in key)):
        candidates = grouped[key]
        reference = candidates.get(("baseline", 1))
        if reference is None:
            continue
        for identity in sorted(candidates, key=lambda item: (item[1], item[0])):
            candidate = candidates[identity]
            mismatches = [name for name in CONFIG_FIELDS
                          if name != "jettys_per_core" and reference[name] != candidate[name]]
            if mismatches:
                raise ValueError(f"Jetty reference configuration mismatch for round/shape={key}: "
                                 + ", ".join(mismatches))
            pair = {name: candidate[name] for name in CONFIG_FIELDS + SHAPE_METRICS}
            pair.update(round=candidate["round"], variant=candidate["variant"],
                        reference_variant="baseline", reference_jettys_per_core=1,
                        reference_log=reference["log_file"], candidate_log=candidate["log_file"])
            for metric in PAIR_METRICS:
                available = reference[metric] != "" and candidate[metric] != ""
                pair["reference_" + metric] = reference[metric] if available else ""
                pair["candidate_" + metric] = candidate[metric] if available else ""
            pair.update(comparison_deltas(reference, candidate))
            pairs.append(pair)

    configurations = {}
    for pair in pairs:
        key = (pair["variant"], *(pair[name] for name in CONFIG_FIELDS),
               pair["reference_small_post_max_requests"], pair["candidate_small_post_max_requests"])
        configurations.setdefault(key, []).append(pair)
    summary = []
    for group in configurations.values():
        row = {name: group[0][name] for name in ("variant",) + CONFIG_FIELDS + SHAPE_METRICS}
        row.update(reference_variant="baseline", reference_jettys_per_core=1,
                   paired_rounds=len(group), round_ids=";".join(str(pair["round"]) for pair in group),
                   p50_win_rounds=sum(pair["p50_latency_reduction_pct"] > 0 for pair in group),
                   p99_regression_rounds=sum(pair["p99_latency_reduction_pct"] < 0 for pair in group),
                   worst_p50_latency_reduction_pct=min(pair["p50_latency_reduction_pct"] for pair in group),
                   worst_p99_latency_reduction_pct=min(pair["p99_latency_reduction_pct"] for pair in group))
        for metric in JETTY_PAIR_VALUES:
            values = [pair[metric] for pair in group if pair[metric] != ""]
            row["median_" + metric] = statistics.median(values) if values else ""
        summary.append(row)
    return pairs, summary


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{row[key]:.6f}" if isinstance(row[key], float) else row[key]
                             for key in columns})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path,
                        help="recursively read r<round>_n<count>_[c<cores>_j<jettys>_|<jettys>_]<variant>.log")
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
        jetty_pairs, jetty_summary = compare_jetty_scaling(runs)
        jettys = [{**{key: run[key] for key in ("round", "variant") + CONFIG_FIELDS}, **trace}
                  for run in runs for trace in run["_jettys"]]
        output = args.output_dir or args.log_dir
        output.mkdir(parents=True, exist_ok=True)
        outputs = (
            ("small_post_runs.csv", runs,
             ("round", "variant") + CONFIG_FIELDS + SHAPE_METRICS + ("payload_bytes",) + RUN_METRICS +
             ("p50_GBps", "posts_per_call", "completions_per_call") + SCHEDULE_METRICS + PROOF_METRICS + ("log_file",)),
            ("small_post_pairs.csv", pairs,
             ("round",) + CONFIG_FIELDS + SHAPE_METRICS + PAIR_VALUES + ("baseline_log", "small_post_log")),
            ("small_post_summary.csv", summary,
             CONFIG_FIELDS + SHAPE_METRICS + ("rounds", "round_ids") + tuple(prefix + "rounds" for prefix in OPTIONAL_STAGES.values()) +
             tuple("median_" + name for name in PAIR_VALUES)),
            ("small_post_jettys.csv", jettys,
             ("round", "variant") + CONFIG_FIELDS + ("core", "jetty") + SCHEDULE_FIELDS + JETTY_METRICS),
            ("small_post_jetty_pairs.csv", jetty_pairs,
             ("round", "variant") + CONFIG_FIELDS + SHAPE_METRICS +
             ("reference_variant", "reference_jettys_per_core") + JETTY_PAIR_VALUES +
             ("reference_log", "candidate_log")),
            ("small_post_jetty_summary.csv", jetty_summary,
             ("variant",) + CONFIG_FIELDS + SHAPE_METRICS +
             ("reference_variant", "reference_jettys_per_core", "paired_rounds", "round_ids") +
             JETTY_SUMMARY_COUNTS + tuple("median_" + name for name in JETTY_PAIR_VALUES)),
        )
        # Validate every log and pair before creating any result CSV.
        print("SMALL_POST_JETTY_REFERENCE variant=baseline jettys_per_core=1 "
              f"matched_runs={len(jetty_pairs)} missing_reference_runs={len(runs) - len(jetty_pairs)}")
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
