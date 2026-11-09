#!/usr/bin/env python3
"""Gather performance comparison: edit the configuration below, then run on NPU.

Device/core selection, service/kernel paths, output layout and optional SGE
coalescing use environment variables. Request tensors are prepared once and
reused; CPU source requests remain fixed and random across all output layouts.
"""
from __future__ import annotations

# ---- Edit benchmark settings here. ------------------------------------------
TOKEN_BYTES = 656
REQUEST_COUNT = 8192             # Try 1024, 8192, 8193, 65536; all requests are active.
WARMUP = 20                     # Excluded from performance results.
ITERATIONS = 100                # Profile OFF samples; 0 skips E2E for profile debugging.
PROFILE_SAMPLES = 10            # Profile ON: diagnostic stage samples, separately.
TABLE_CAPACITY = 16_384         # Number of remote source blocks.
TOKENS_PER_BLOCK = 128
GATHER_SGL_SIZE = 6             # 1..13; compare 5 / 9 / 13 on the confirmed NPU.
SEED = 20261030
PROFILE_CLOCK_MHZ = 1000        # Confirmed Ascend 950DT; A2/A3 use 50 (GetSystemCycle).
POPULATE_CHUNK_ENTRIES = 16     # Setup only; does not change BatchGet request size.
CSV_OUTPUT_DIR = "gather-perf-results"  # Relative to the working directory.
RUN_LABEL = "gather_progress_control"  # KVAF_AIV_GATHER_CHUNK_BYTES=0 selects the AIV-only baseline.
# ---------------------------------------------------------------------------

import csv
import math
import os
import statistics
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aiv_batch_get_test_common import kernel_paths, shutdown_runtime


STAGE_BOUNDARIES = (
    ("input", "kernel_start_cycle", "input_done_cycle"),
    ("descriptor", "input_done_cycle", "descriptor_done_cycle"),
    ("publish", "descriptor_done_cycle", "notify_done_cycle"),
    ("wait_cpu", "notify_done_cycle", "ack_done_cycle"),
    ("read", "ack_done_cycle", "read_done_cycle"),
    ("finalize", "read_done_cycle", "kernel_finish_cycle"),
    ("total", "kernel_start_cycle", "kernel_finish_cycle"),
)
CALL_HOST_FIELDS = tuple("call_" + phase + "_ns" for phase in (
    "binding", "tensor_metadata", "stream_resolve", "native", "runtime_validation",
    "operator", "mutex_wait", "capture_query", "prior_drain", "input", "launch",
    "status_copy_enqueue", "stream_sync", "status_validate", "output_acquire",
))
HOST_FIELDS = (
    "prepare_ns", "launch_ns", "status_enqueue_ns", "wait_ns", "finish_ns", "total_ns", *CALL_HOST_FIELDS,
)
COUNTERS = (
    "data_wrs", "ack_queries", "data_wqebbs", "data_doorbells",
    "data_cq_wait_cycles", "read_build_cycles", "ready_queries", "ready_wait_cycles",
    "pipeline_chunk_bytes", "pipeline_shards", "first_read_ready_bytes",
    "read_vf_calls", "ready_no_change_queries", "ready_query_skips", "max_ready_backlog_wrs",
    "ready_idle_waits", "multi_run_posts", "packed_posts", "read_plan_segments",
    "progress_mode", "ready_local_polls", "ready_local_updates", "ready_ack_checks", "ready_control_wrs",
)
PIPELINE_EVENTS = ("first_ready_cycle", "first_read_submit_cycle")


def configured_chunk_bytes() -> int:
    value = os.environ.get("KVAF_AIV_GATHER_CHUNK_BYTES", "") or "131072"
    if not value.isascii() or not value.isdecimal():
        raise ValueError("KVAF_AIV_GATHER_CHUNK_BYTES must be 0 or a uint32 multiple of 64")
    chunk = int(value)
    if chunk > (1 << 32) - 1 or (chunk and chunk % 64):
        raise ValueError("KVAF_AIV_GATHER_CHUNK_BYTES must be 0 or a uint32 multiple of 64")
    return chunk


def profile_stage_cycles(lane: Mapping[str, object]) -> dict[str, int]:
    """Serial stages partition time; chunk-mode wait_cpu and read overlap."""
    pipeline = int(lane.get("pipeline_chunk_bytes", 0)) != 0
    durations = {}
    for name, begin, end in STAGE_BOUNDARIES:
        first, last = int(lane[begin]), int(lane[end])
        if pipeline and name == "read":
            first = int(lane["first_read_submit_cycle"])
        elif pipeline and name == "finalize":
            first = max(first, int(lane["ack_done_cycle"]))
        if first < 0 or last < first:
            raise ValueError(f"invalid {name} profile timestamp order")
        durations[name] = last - first
    if pipeline:
        notify, ready, submit, read, ack, finish = (
            int(lane[key]) for key in ("notify_done_cycle", "first_ready_cycle",
                "first_read_submit_cycle", "read_done_cycle", "ack_done_cycle", "kernel_finish_cycle"))
        if not notify <= ready <= submit <= read <= finish or not notify <= ack <= finish:
            raise ValueError("invalid overlapping gather profile timestamp order")
        if not 0 < int(lane.get("first_read_ready_bytes", 0)) <= int(lane["requests"]) * TOKEN_BYTES:
            raise ValueError("invalid first READ ready prefix")
    return durations


def validate_config(cores: int) -> None:
    """Reject invalid shapes before allocating memory or connecting to the CPU."""
    configured_chunk_bytes()
    if (os.environ.get("KVAF_AIV_GATHER_PROGRESS_MODE", "pull") or "pull") not in ("legacy", "pull", "push"):
        raise ValueError("KVAF_AIV_GATHER_PROGRESS_MODE must be legacy/pull/push")
    for name in ("KVAF_AIV_GATHER_SHARD_PIPELINE", "KVAF_GATHER_TRUSTED_INPUTS", "KVAF_AIV_GATHER_PACKED_POSTS"):
        if (os.environ.get(name, "1") or "1") not in ("0", "1"):
            raise ValueError(f"{name} must be 0 or 1")
    positive = {
        "TOKEN_BYTES": TOKEN_BYTES, "REQUEST_COUNT": REQUEST_COUNT,
        "TABLE_CAPACITY": TABLE_CAPACITY,
        "TOKENS_PER_BLOCK": TOKENS_PER_BLOCK,
        "POPULATE_CHUNK_ENTRIES": POPULATE_CHUNK_ENTRIES,
    }
    for name, value in positive.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in {"WARMUP": WARMUP, "ITERATIONS": ITERATIONS,
                        "PROFILE_SAMPLES": PROFILE_SAMPLES}.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if ITERATIONS == 0 and PROFILE_SAMPLES == 0:
        raise ValueError("ITERATIONS and PROFILE_SAMPLES cannot both be zero")
    if TOKEN_BYTES > 65535 or REQUEST_COUNT > 65536:
        raise ValueError("TOKEN_BYTES must be <= 65535 and REQUEST_COUNT <= 65536")
    if TABLE_CAPACITY > (1 << 31) - 1:
        raise ValueError("TABLE_CAPACITY exceeds the DSA int32 directory range")
    if (TOKENS_PER_BLOCK - 1) * TOKEN_BYTES > (1 << 31) - 1:
        raise ValueError("token offsets must fit int32")
    if not 1 <= cores <= 8 or not 1 <= GATHER_SGL_SIZE <= 13:
        raise ValueError("KVAF_NPU_CORE_COUNT must be 1..8 and GATHER_SGL_SIZE 1..13")
    layout = os.environ.get("KVAF_GATHER_OUTPUT_LAYOUT", "random")
    if layout not in ("random", "segmented", "contiguous"):
        raise ValueError("KVAF_GATHER_OUTPUT_LAYOUT must be random, segmented or contiguous")
    if layout == "segmented" and TOKEN_BYTES * GATHER_SGL_SIZE > 65536:
        raise ValueError("segmented output requires TOKEN_BYTES * GATHER_SGL_SIZE <= 65536")
    coalesce = os.environ.get("KVAF_AIV_GATHER_COALESCE_SGES", "0") or "0"
    if coalesce not in ("0", "1"):
        raise ValueError("KVAF_AIV_GATHER_COALESCE_SGES must be 0 or 1")
    if coalesce == "1" and (layout == "random" or TOKEN_BYTES * GATHER_SGL_SIZE > 65536):
        raise ValueError("coalesced gather requires segmented/contiguous output and TOKEN_BYTES * SGL <= 65536")
    if not isinstance(SEED, int) or isinstance(SEED, bool) or not 0 <= SEED < 1 << 63:
        raise ValueError("SEED must be an integer in [0, 2**63)")
    if PROFILE_CLOCK_MHZ is not None and (
        isinstance(PROFILE_CLOCK_MHZ, bool)
        or not isinstance(PROFILE_CLOCK_MHZ, (int, float))
        or not math.isfinite(PROFILE_CLOCK_MHZ)
        or PROFILE_CLOCK_MHZ <= 0
    ):
        raise ValueError("PROFILE_CLOCK_MHZ must be None or a verified positive frequency")
    if not isinstance(CSV_OUTPUT_DIR, str) or not CSV_OUTPUT_DIR.strip():
        raise ValueError("CSV_OUTPUT_DIR must be a non-empty directory path")
    if not isinstance(RUN_LABEL, str):
        raise ValueError("RUN_LABEL must be a string")


def make_output_rows(requests: int, cores: int, sgl: int, layout: str, randperm) -> list[int]:
    """Keep guard rows untouched; shuffle whole WQE destinations, retaining tails."""
    if layout == "random":
        return [row + 1 for row in randperm(requests)]
    rows = list(range(1, requests + 1))
    if layout == "segmented":
        starts = []
        for core in range(cores):
            first, last = requests * core // cores, requests * (core + 1) // cores
            starts.extend(range(first, last - sgl + 1, sgl))
        # Full segments can move across lanes; partial tails keep their rows.
        for target, index in zip(starts, randperm(len(starts))):
            source = starts[index]
            rows[target:target + sgl] = range(source + 1, source + sgl + 1)
    return rows


def check_coalesced_profile(profile) -> None:
    for lane in profile["cores"]:
        wrs = (int(lane["requests"]) + GATHER_SGL_SIZE - 1) // GATHER_SGL_SIZE
        if int(lane["data_wrs"]) != wrs or int(lane["data_wqebbs"]) != wrs:
            raise RuntimeError("SGE coalescing did not preserve READ count / emit one WQEBB per READ; "
                               "rebuild and deploy both gather Host and kernel")


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("percentile requires samples and a fraction in [0, 1]")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_profiles(
    samples: Sequence[Mapping[str, object]], cores: int, requests: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Validate a batch timeline and keep parallel lanes separate in the summary."""
    if not samples or not 1 <= cores <= 8 or requests < 0:
        raise ValueError("profile summary requires samples, 1..8 cores and a request count")
    host_values = {field: [] for field in HOST_FIELDS}
    core_values = [
        {field: [] for field in ("requests", *COUNTERS, *(s[0] for s in STAGE_BOUNDARIES))}
        for _ in range(cores)
    ]
    last_sequence = -1
    for snapshot in samples:
        sequence = int(snapshot["sequence"])
        if sequence <= last_sequence:
            raise ValueError("profile sequence must increase for each measured call")
        last_sequence = sequence
        host = snapshot["host"]
        lanes = snapshot["cores"]
        if not isinstance(host, Mapping) or not isinstance(lanes, Sequence) or len(lanes) != cores:
            raise ValueError("profile host or core count does not match this benchmark")
        for field in HOST_FIELDS:
            value = int(host[field])
            if value < 0:
                raise ValueError("negative host profile duration")
            host_values[field].append(value)
        seen = set()
        for lane in lanes:
            core = int(lane["core"])
            if core not in range(cores) or core in seen:
                raise ValueError("duplicate or invalid core in profile")
            seen.add(core)
            if int(lane["sequence"]) != sequence or int(lane.get("result", 0)) != 0:
                raise ValueError("profile has a stale sequence or failed lane")
            expected_requests = requests * (core + 1) // cores - requests * core // cores
            if int(lane["requests"]) != expected_requests:
                raise ValueError("profile request count does not match the lane partition")
            values = core_values[core]
            values["requests"].append(expected_requests)
            for name, duration in profile_stage_cycles(lane).items():
                values[name].append(duration)
            for field in COUNTERS:
                value = int(lane.get(field, 0))
                if value < 0:
                    raise ValueError("negative core profile counter")
                values[field].append(value)
    return (
        {field: statistics.mean(values) for field, values in host_values.items()},
        [{field: statistics.mean(values) for field, values in lane.items()} for lane in core_values],
    )


def print_profiles(samples: list[dict[str, object]], cores: int) -> None:
    host, lanes = summarize_profiles(samples, cores, REQUEST_COUNT)
    print(
        f"GATHER_PROFILE_CONFIG profile=on cpu_profile=external samples={len(samples)} "
        f"first_sequence={samples[0]['sequence']} last_sequence={samples[-1]['sequence']} "
        f"clock_mhz={PROFILE_CLOCK_MHZ if PROFILE_CLOCK_MHZ is not None else 'unverified'} "
        "overhead_included=1 snapshot_copy_outside_call=1",
        flush=True,
    )
    print("GATHER_PROFILE_HOST " + " ".join(
        f"{field.removesuffix('_ns')}_us={value / 1000:.3f}" for field, value in host.items()
    ))
    for core, lane in enumerate(lanes):
        fields = [f"core={core}", f"requests={lane['requests']:.0f}"]
        fields += [f"{name}_cycles={lane[name]:.1f}" for name, _, _ in STAGE_BOUNDARIES]
        fields += [f"{field}={lane[field]:.1f}" for field in COUNTERS]
        if PROFILE_CLOCK_MHZ is not None:
            fields += [f"{name}_us={lane[name] / PROFILE_CLOCK_MHZ:.3f}"
                       for name, _, _ in STAGE_BOUNDARIES]
        print("GATHER_PROFILE_CORE " + " ".join(fields))


def latency_summary(samples_us: Sequence[float], payload_bytes: int) -> dict[str, float]:
    mean_us = statistics.mean(samples_us)
    p50_us = percentile(samples_us, 0.50)
    return {
        "mean_us": mean_us, "p50_us": p50_us,
        "p95_us": percentile(samples_us, 0.95), "p99_us": percentile(samples_us, 0.99),
        "useful_GiBps": payload_bytes / (p50_us / 1e6) / 1024**3,
    }


def csv_metadata(cores: int, device_id: int, kernel: str, put_kernel: str) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    local = now.astimezone(timezone(timedelta(hours=8)))
    revision, dirty = "", "unknown"
    try:
        repo = Path(__file__).resolve().parents[2]
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                                  capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=repo, check=True, capture_output=True, text=True).stdout
        dirty = int(bool(status.strip()))
    except (OSError, subprocess.CalledProcessError):
        pass  # A deployed source bundle need not contain .git.
    return {
        "schema_version": 7,
        "run_id": now.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8],
        "started_at_utc": now.isoformat(), "run_label": RUN_LABEL,
        "started_at_local": local.isoformat(), "timezone": "Asia/Shanghai",
        "run_directory": (local.strftime("%Y%m%d_%H%M%S") +
                          f"-cores{cores}-token-bytes{TOKEN_BYTES}-request-count{REQUEST_COUNT}"),
        "git_revision": revision, "git_dirty": dirty,
        "mode": "gather_read", "device_id": device_id, "cores": cores,
        "token_bytes": TOKEN_BYTES, "request_count": REQUEST_COUNT,
        "payload_bytes": REQUEST_COUNT * TOKEN_BYTES, "table_capacity": TABLE_CAPACITY,
        "tokens_per_block": TOKENS_PER_BLOCK, "sgl": GATHER_SGL_SIZE, "seed": SEED,
        "output_layout": os.environ.get("KVAF_GATHER_OUTPUT_LAYOUT", "random"),
        "coalesce_sges": int(os.environ.get("KVAF_AIV_GATHER_COALESCE_SGES", "0") or "0"),
        "configured_chunk_bytes": configured_chunk_bytes(),
        "configured_progress_mode": os.environ.get("KVAF_AIV_GATHER_PROGRESS_MODE", "pull") or "pull",
        "configured_packed_posts": int(os.environ.get("KVAF_AIV_GATHER_PACKED_POSTS", "1") or "1"),
        "configured_shard_pipeline": int(os.environ.get("KVAF_AIV_GATHER_SHARD_PIPELINE", "1") or "1"),
        "trusted_gather_inputs": int(os.environ.get("KVAF_GATHER_TRUSTED_INPUTS", "1") or "1"),
        "warmup": WARMUP, "iterations": ITERATIONS, "profile_samples": PROFILE_SAMPLES,
        "clock_mhz": PROFILE_CLOCK_MHZ if PROFILE_CLOCK_MHZ is not None else "",
        "clock_source": "configured" if PROFILE_CLOCK_MHZ is not None else "unverified",
        "cpu_profile": "external", "kernel_path": kernel, "put_kernel_path": put_kernel,
        "cpu_service_addr": os.environ.get("KVAF_CPU_SERVICE_ADDR", ""),
        "request_plan": "fixed_random", "timing_contract": "async_call_to_stream_completion",
    }


def export_csv_results(
    directory: str | Path, metadata: Mapping[str, object], samples_us: Sequence[float],
    snapshots: Sequence[Mapping[str, object]], profile_calls_us: Sequence[float],
    error: BaseException | None = None,
) -> list[Path]:
    """Write only outside measured loops; preserve completed samples on failure.

    All four files share run_id/config columns. Host and AIV rows are separate
    so multi-core rows cannot accidentally multiply the Host time in a sum.
    """
    if len(snapshots) != len(profile_calls_us):
        raise ValueError("profile snapshots and synchronous call samples must match")
    cores, requests = int(metadata["cores"]), int(metadata["request_count"])
    host_mean, core_means = summarize_profiles(snapshots, cores, requests) if snapshots else ({}, [])
    clock = metadata["clock_mhz"]
    stages = [name for name, _, _ in STAGE_BOUNDARIES]
    boundaries = [STAGE_BOUNDARIES[0][1]] + [end for _, _, end in STAGE_BOUNDARIES[:-1]]
    host_columns = [field.removesuffix("_ns") + "_us" for field in HOST_FIELDS]
    timed_counters = [field for field in COUNTERS if field.endswith("_cycles")]
    core_columns = ["requests", *boundaries, *PIPELINE_EVENTS, *(name + "_cycles" for name in stages),
                    *(name + "_us" for name in stages), *COUNTERS,
                    *(field.removesuffix("_cycles") + "_us" for field in timed_counters)]

    def core_fields(lane: Mapping[str, object], mean: bool = False) -> dict[str, object]:
        row = {"requests": lane["requests"], **{key: lane.get(key, 0) for key in COUNTERS}}
        if not mean:
            row.update({key: lane[key] for key in boundaries})
            row.update({key: lane.get(key, 0) for key in PIPELINE_EVENTS})
        durations = None if mean else profile_stage_cycles(lane)
        for name, begin, end in STAGE_BOUNDARIES:
            cycles = lane[name] if mean else durations[name]
            row[name + "_cycles"] = cycles
            row[name + "_us"] = cycles / float(clock) if clock != "" else ""
        for field in timed_counters:
            row[field.removesuffix("_cycles") + "_us"] = float(row[field]) / float(clock) if clock != "" else ""
        return row

    def host_fields(host: Mapping[str, object]) -> dict[str, object]:
        return {field.removesuffix("_ns") + "_us": float(host[field]) / 1000 for field in HOST_FIELDS}

    summary = [{"row_kind": "sampling_status", "sampling_status": "failed" if error else "complete",
                "error_type": type(error).__name__ if error else "", "error_message": str(error) if error else ""}]
    e2e_rows = []
    if samples_us:
        summary.append({"row_kind": "e2e_summary", "profile": "off", "samples": len(samples_us),
                        **latency_summary(samples_us, int(metadata["payload_bytes"]))})
        e2e_rows = [{"sample_index": index, "profile": "off", "e2e_us": elapsed,
                     "useful_GiBps": int(metadata["payload_bytes"]) / (elapsed / 1e6) / 1024**3}
                    for index, elapsed in enumerate(samples_us, 1)]
    host_rows, core_rows = [], []
    if snapshots:
        sequence_range = {"first_sequence": snapshots[0]["sequence"],
                          "last_sequence": snapshots[-1]["sequence"]}
        summary.append({"row_kind": "profile_e2e_summary", "profile": "on", "samples": len(snapshots),
                        **sequence_range, **latency_summary(profile_calls_us, int(metadata["payload_bytes"]))})
        summary.append({"row_kind": "profile_host_mean", "profile": "on", "samples": len(snapshots),
                        **sequence_range, **host_fields(host_mean)})
        for core, lane in enumerate(core_means):
            summary.append({"row_kind": "profile_core_mean", "profile": "on", "core": core,
                            "samples": len(snapshots), **sequence_range, **core_fields(lane, mean=True)})
        for index, (snapshot, elapsed) in enumerate(zip(snapshots, profile_calls_us), 1):
            key = {"sample_index": index, "profile": "on", "sequence": snapshot["sequence"]}
            host_rows.append({**key, "call_e2e_us": elapsed, **host_fields(snapshot["host"])})
            for lane in snapshot["cores"]:
                core_rows.append({**key, "core": lane["core"], **core_fields(lane)})
    output_root = Path(directory).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    # Atomically reserve a directory for this test, including same-second runs.
    # The readable name carries the test configuration; CSV run_id remains the
    # unique join key. Never mix or overwrite files from an earlier test.
    suffix = 1
    while True:
        folder = str(metadata["run_directory"])
        target = output_root / (folder if suffix == 1 else f"{folder}_{suffix}")
        try:
            target.mkdir()
            break
        except FileExistsError:
            suffix += 1
    metadata = {**metadata, "run_directory": target.name}
    # Keep less frequently compared transport/timing metadata before mode,
    # leaving the shape and measurements together on the right. Refer to field
    # names here: spreadsheet letters change whenever the schema is reordered.
    before_mode = ("clock_mhz", "clock_source", "cpu_profile", "kernel_path", "put_kernel_path",
                   "request_plan", "timing_contract")
    metadata_columns = [key for key in metadata if key not in before_mode]
    mode_index = metadata_columns.index("mode")
    metadata_columns[mode_index:mode_index] = before_mode
    stage_cycle_columns = [name + "_cycles" for name in stages]
    stage_time_columns = [name + "_us" for name in stages if name != "total"]
    summary_core_columns = [field for field in core_columns
                            if field not in (*stage_cycle_columns, *stage_time_columns, "total_us")]
    base_host_columns = [field for field in host_columns if not field.startswith("call_")]
    call_host_columns = [field for field in host_columns if field.startswith("call_")]
    outputs = (
        ("summary", ["row_kind", "profile",
                     "samples", "first_sequence", "last_sequence", "mean_us", "p50_us", "p95_us", "p99_us", "useful_GiBps",
                     *base_host_columns, *stage_time_columns, *stage_cycle_columns, "sampling_status", "core",
                     "error_type", "error_message", *call_host_columns, *summary_core_columns], summary),
        ("e2e", ["sample_index", "profile", "e2e_us", "useful_GiBps"], e2e_rows),
        ("profile_host", ["sample_index", "profile", "sequence", "call_e2e_us", *host_columns], host_rows),
        ("profile_core", ["sample_index", "profile", "sequence", "core", *core_columns], core_rows),
    )
    paths = []
    for name, columns, rows in outputs:
        path = target / f"{name}.csv"
        # A new per-test directory plus exclusive creation protect measurements.
        with path.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*metadata_columns, *dict.fromkeys(columns)])
            writer.writeheader()
            writer.writerows({**metadata, **row,
                              "timing_contract": "synchronous_call_with_status_check"
                              if row.get("profile") == "on" else metadata["timing_contract"]}
                             for row in rows)
        paths.append(path)
    return paths


def token_pattern(torch, token_ids, token_bytes: int):
    """Deterministic bytes depend on the source token as well as its byte offset."""
    x = token_ids.to(torch.int64).unsqueeze(1)
    y = torch.arange(token_bytes, dtype=torch.int64).unsqueeze(0)
    return ((x * 37 + y * 13 + (y >> 3) + (x >> 8) * (y % 7)) % 251).to(torch.uint8)


def main() -> int:
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the A5/CANN target")
        return 77
    if os.environ.get("KVAF_AIV_GATHER_PROBE"):
        print("FAIL: gather prefix probes cannot collect performance; use test_aiv_gather_pipeline_debug.py",
              file=sys.stderr)
        return 2
    if os.environ.get("KVAF_AIV_GATHER_CHECK_STATUS") == "1":
        print("FAIL: unset KVAF_AIV_GATHER_CHECK_STATUS for the asynchronous performance baseline",
              file=sys.stderr)
        return 2
    try:
        cores = int(os.environ.get("KVAF_NPU_CORE_COUNT", "1"))
        validate_config(cores)
        kernel, put_kernel = kernel_paths("gather_read")
        endpoint = os.environ.get("KVAF_CPU_SERVICE_ADDR")
        if not endpoint:
            raise ValueError("KVAF_CPU_SERVICE_ADDR is not set")
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    from backend import detect_torch_backend
    from dsa_keys import build_dsa_keys
    from npu_eids import parse_npu_eids
    import kvaf

    backend = detect_torch_backend()
    if backend.name != "npu" or not hasattr(kvaf, "NpuRuntime"):
        raise RuntimeError("hardware benchmark was requested but NPU Runtime is unavailable")
    coalesce = os.environ.get("KVAF_AIV_GATHER_COALESCE_SGES") == "1"
    if (PROFILE_SAMPLES or coalesce) and not all(hasattr(kvaf.NpuRuntime, method) for method in (
        "start_gather_batch_get_profile", "gather_batch_get_profile", "stop_gather_batch_get_profile",
    )):
        raise RuntimeError("rebuild KVAF Host/Python and gather kernel for gather profiling")
    torch = backend.torch
    device_id = int(os.environ.get("DEVICE_ID", "0"))
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    block_bytes = TOKEN_BYTES * TOKENS_PER_BLOCK
    payload_bytes = REQUEST_COUNT * TOKEN_BYTES
    runtime = kvaf.NpuRuntime(
        kernel, TOKEN_BYTES, TOKENS_PER_BLOCK,
        device_id=device_id, npu_core_count=cores,
        batch_put_kernel_object_path=put_kernel,
        npu_eids=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")),
        aiv_batch_get_mode="gather_read", aiv_gather_sgl_size=GATHER_SGL_SIZE,
        batch_get_max_inflight_total=0,
        trusted_gather_inputs=(os.environ.get("KVAF_GATHER_TRUSTED_INPUTS", "1") or "1") == "1",
    )
    metadata = csv_metadata(cores, device_id, kernel, put_kernel)
    layout = metadata["output_layout"]
    samples_us, snapshots, profile_calls_us = [], [], []
    try:
        runtime.link_to(endpoint)
        print(
            f"GATHER_PERF_CONFIG mode=gather_read cores={cores} token_bytes={TOKEN_BYTES} "
            f"requests={REQUEST_COUNT} payload_bytes={payload_bytes} table_capacity={TABLE_CAPACITY} "
            f"tokens_per_block={TOKENS_PER_BLOCK} sgl={GATHER_SGL_SIZE} seed={SEED} output_layout={layout} "
            f"coalesce_sges={metadata['coalesce_sges']} "
            f"configured_chunk_bytes={metadata['configured_chunk_bytes']} "
            f"warmup={WARMUP} iterations={ITERATIONS} profile_samples={PROFILE_SAMPLES} "
            "plan=fixed_random timing=async_call_to_stream_completion",
            flush=True,
        )
        runtime.initialize(TABLE_CAPACITY * block_bytes, block_bytes)
        for start in range(0, TABLE_CAPACITY, POPULATE_CHUNK_ENTRIES):
            count = min(POPULATE_CHUNK_ENTRIES, TABLE_CAPACITY - start)
            ids = torch.arange(start * TOKENS_PER_BLOCK, (start + count) * TOKENS_PER_BLOCK)
            source = token_pattern(torch, ids, TOKEN_BYTES).reshape(count, block_bytes).to(device)
            runtime.batch_put(
                build_dsa_keys(count, start=start, q_capacity=TABLE_CAPACITY), source,
                [i * block_bytes for i in range(count)], [block_bytes] * count,
            )
        table = runtime.prepare_access_table(build_dsa_keys(TABLE_CAPACITY, q_capacity=TABLE_CAPACITY))
        view = runtime.access_table_view(table, device="npu", capacity=TABLE_CAPACITY)
        generator = torch.Generator().manual_seed(SEED)
        indices_cpu = torch.randint(TABLE_CAPACITY, (REQUEST_COUNT,), generator=generator, dtype=torch.int32)
        slots_cpu = torch.randint(TOKENS_PER_BLOCK, (REQUEST_COUNT,), generator=generator, dtype=torch.int32)
        rows_cpu = torch.tensor(make_output_rows(
            REQUEST_COUNT, cores, GATHER_SGL_SIZE, layout,
            lambda count: torch.randperm(count, generator=generator).tolist()), dtype=torch.int32)
        indices = indices_cpu.to(device)
        offsets = (slots_cpu * TOKEN_BYTES).to(device)
        rows = rows_cpu.to(device)
        miss_count = torch.tensor([REQUEST_COUNT], dtype=torch.int32, device=device)
        output = torch.full((REQUEST_COUNT + 2, TOKEN_BYTES), 0xAD, dtype=torch.uint8, device=device)
        torch.npu.synchronize()

        def batch_get(*, checked: bool = False) -> None:
            runtime.batch_get(view, indices, offsets, miss_count, rows, output, synchronize=checked)
            if not checked:
                # Same timing contract as the direct benchmark: asynchronous
                # submission followed by caller-owned completion, without D2H.
                torch.npu.synchronize()

        # Check all payload bytes and the two untouched guard rows outside timing.
        if coalesce:
            runtime.start_gather_batch_get_profile()
        batch_get(checked=True)
        if coalesce:
            preflight_profile = runtime.gather_batch_get_profile()
            runtime.stop_gather_batch_get_profile()
            check_coalesced_profile(preflight_profile)
        expected = torch.full((REQUEST_COUNT + 2, TOKEN_BYTES), 0xAD, dtype=torch.uint8)
        source_ids = indices_cpu.to(torch.int64) * TOKENS_PER_BLOCK + slots_cpu
        for start in range(0, REQUEST_COUNT, 128):
            end = min(start + 128, REQUEST_COUNT)
            expected[rows_cpu[start:end].long()] = token_pattern(torch, source_ids[start:end], TOKEN_BYTES)
        if not torch.equal(output.cpu(), expected):
            raise AssertionError("gather correctness preflight payload/guard mismatch")
        print("GATHER_PERF_CORRECTNESS_PASS", flush=True)

        for _ in range(WARMUP):
            batch_get()
        for _ in range(ITERATIONS):
            begin = time.perf_counter_ns()
            batch_get()
            samples_us.append((time.perf_counter_ns() - begin) / 1000)
        if samples_us:
            result = latency_summary(samples_us, payload_bytes)
            print(
                f"GATHER_PERF_RESULT profile=off cpu_profile=external samples={ITERATIONS} warmup_excluded={WARMUP} "
                f"mean_us={result['mean_us']:.3f} p50_us={result['p50_us']:.3f} "
                f"p95_us={result['p95_us']:.3f} p99_us={result['p99_us']:.3f} "
                f"useful_GiBps={result['useful_GiBps']:.3f} bandwidth_basis=p50_us",
                flush=True,
            )
        else:
            print("GATHER_PERF_SKIPPED profile=off reason=iterations_zero", flush=True)
        if PROFILE_SAMPLES:
            print(f"GATHER_PROFILE_PHASE_BEGIN warmup=1 samples={PROFILE_SAMPLES}", flush=True)
            runtime.start_gather_batch_get_profile()
            batch_get(checked=True)  # Warm the profiling path; exclude this sample.
            for _ in range(PROFILE_SAMPLES):
                begin = time.perf_counter_ns()
                batch_get(checked=True)
                elapsed = (time.perf_counter_ns() - begin) / 1000
                snapshots.append(dict(runtime.gather_batch_get_profile()))
                profile_calls_us.append(elapsed)
            runtime.stop_gather_batch_get_profile()
            print_profiles(snapshots, cores)
        return 0
    finally:
        # CSV I/O is outside both timed groups. Preserve successful baseline or
        # profile samples even when a later sample fails; retain the original
        # device exception if export or shutdown fails as well.
        original = sys.exc_info()[1]
        try:
            paths = export_csv_results(CSV_OUTPUT_DIR, metadata, samples_us, snapshots,
                                       profile_calls_us, original)
            for path in paths:
                print(f"GATHER_CSV_FILE path={path}", flush=True)
        except Exception as export_error:
            if original is None:
                raise
            print(f"GATHER_CSV_ERROR preserving original exception: {export_error}", file=sys.stderr)
        finally:
            # Keep all request/output tensors alive through transport shutdown.
            shutdown_runtime(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
