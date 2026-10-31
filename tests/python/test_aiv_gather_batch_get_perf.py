#!/usr/bin/env python3
"""Gather serial baseline: edit the configuration block below, then run on NPU.

Only device/core selection and the existing service/kernel paths use environment
variables. This script has no command-line options. Request tensors are prepared
once and reused, so the measurements describe a warmed, fixed random workload.
"""
from __future__ import annotations

# ---- Edit benchmark settings here. ------------------------------------------
TOKEN_BYTES = 656
REQUEST_COUNT = 8193             # Try 1024, 8193, 65536; all requests are active.
WARMUP = 20                     # Excluded from performance results.
ITERATIONS = 100                # Profile OFF samples; 0 skips E2E for profile debugging.
PROFILE_SAMPLES = 10            # Profile ON: diagnostic stage samples, separately.
TABLE_CAPACITY = 256            # Number of remote source blocks.
TOKENS_PER_BLOCK = 128
GATHER_SGL_SIZE = 6
SEED = 20261030
PROFILE_CLOCK_MHZ = None        # Set only after verifying GetSystemCycle frequency.
POPULATE_CHUNK_ENTRIES = 16     # Setup only; does not change BatchGet request size.
CSV_OUTPUT_DIR = "gather-perf-results"  # Relative to the working directory.
RUN_LABEL = "serial"            # Change to identify later pipeline versions.
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
HOST_FIELDS = (
    "prepare_ns", "launch_ns", "status_enqueue_ns", "wait_ns", "finish_ns", "total_ns",
)
COUNTERS = (
    "data_wrs", "ack_queries", "data_wqebbs", "data_doorbells",
    "data_cq_wait_cycles", "read_build_cycles",
)


def validate_config(cores: int) -> None:
    """Reject invalid shapes before allocating memory or connecting to the CPU."""
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
    if not 1 <= cores <= 8 or not 1 <= GATHER_SGL_SIZE <= 6:
        raise ValueError("KVAF_NPU_CORE_COUNT must be 1..8 and GATHER_SGL_SIZE 1..6")
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
            for name, begin, end in STAGE_BOUNDARIES:
                first, last = int(lane[begin]), int(lane[end])
                if first < 0 or last < first:
                    raise ValueError(f"invalid {name} profile timestamp order")
                values[name].append(last - first)
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
    return {
        "mean_us": mean_us, "p50_us": percentile(samples_us, 0.50),
        "p95_us": percentile(samples_us, 0.95), "p99_us": percentile(samples_us, 0.99),
        "useful_GiBps": payload_bytes / (mean_us / 1e6) / 1024**3,
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
        "schema_version": 2,
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
        "warmup": WARMUP, "iterations": ITERATIONS, "profile_samples": PROFILE_SAMPLES,
        "clock_mhz": PROFILE_CLOCK_MHZ if PROFILE_CLOCK_MHZ is not None else "",
        "clock_source": "configured" if PROFILE_CLOCK_MHZ is not None else "unverified",
        "cpu_profile": "external", "kernel_path": kernel, "put_kernel_path": put_kernel,
        "cpu_service_addr": os.environ.get("KVAF_CPU_SERVICE_ADDR", ""),
        "request_plan": "fixed_random", "timing_contract": "synchronous_call_with_status_check",
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
    core_columns = ["requests", *boundaries, *(name + "_cycles" for name in stages),
                    *(name + "_us" for name in stages), *COUNTERS,
                    *(field.removesuffix("_cycles") + "_us" for field in timed_counters)]

    def core_fields(lane: Mapping[str, object], mean: bool = False) -> dict[str, object]:
        row = {"requests": lane["requests"], **{key: lane.get(key, 0) for key in COUNTERS}}
        if not mean:
            row.update({key: lane[key] for key in boundaries})
        for name, begin, end in STAGE_BOUNDARIES:
            cycles = lane[name] if mean else int(lane[end]) - int(lane[begin])
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
    outputs = (
        ("summary", ["row_kind", "sampling_status", "profile", "core",
                     "samples", "first_sequence", "last_sequence", "mean_us", "p50_us", "p95_us", "p99_us", "useful_GiBps",
                     *host_columns, "error_type", "error_message", *core_columns], summary),
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
            writer.writerows({**metadata, **row} for row in rows)
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
    if PROFILE_SAMPLES and not all(hasattr(kvaf.NpuRuntime, method) for method in (
        "start_gather_batch_get_profile", "gather_batch_get_profile", "stop_gather_batch_get_profile",
    )):
        raise RuntimeError("rebuild KVAF Host/Python and gather kernel for serial profiling")
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
    )
    metadata = csv_metadata(cores, device_id, kernel, put_kernel)
    samples_us, snapshots, profile_calls_us = [], [], []
    try:
        runtime.link_to(endpoint)
        print(
            f"GATHER_PERF_CONFIG mode=gather_read cores={cores} token_bytes={TOKEN_BYTES} "
            f"requests={REQUEST_COUNT} payload_bytes={payload_bytes} table_capacity={TABLE_CAPACITY} "
            f"tokens_per_block={TOKENS_PER_BLOCK} sgl={GATHER_SGL_SIZE} seed={SEED} "
            f"warmup={WARMUP} iterations={ITERATIONS} profile_samples={PROFILE_SAMPLES} "
            "plan=fixed_random timing=synchronous_call_with_status_check",
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
        rows_cpu = torch.randperm(REQUEST_COUNT, generator=generator).to(torch.int32) + 1
        indices = indices_cpu.to(device)
        offsets = (slots_cpu * TOKEN_BYTES).to(device)
        rows = rows_cpu.to(device)
        miss_count = torch.tensor([REQUEST_COUNT], dtype=torch.int32, device=device)
        output = torch.full((REQUEST_COUNT + 2, TOKEN_BYTES), 0xAD, dtype=torch.uint8, device=device)
        torch.npu.synchronize()

        def batch_get() -> None:
            runtime.batch_get(view, indices, offsets, miss_count, rows, output, synchronize=True)

        # Check all payload bytes and the two untouched guard rows outside timing.
        batch_get()
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
            mean_us = statistics.mean(samples_us)
            print(
                f"GATHER_PERF_RESULT profile=off cpu_profile=external samples={ITERATIONS} warmup_excluded={WARMUP} "
                f"mean_us={mean_us:.3f} p50_us={percentile(samples_us, 0.50):.3f} "
                f"p95_us={percentile(samples_us, 0.95):.3f} p99_us={percentile(samples_us, 0.99):.3f} "
                f"useful_GiBps={payload_bytes / (mean_us / 1e6) / 1024**3:.3f}",
                flush=True,
            )
        else:
            print("GATHER_PERF_SKIPPED profile=off reason=iterations_zero", flush=True)
        if PROFILE_SAMPLES:
            print(f"GATHER_PROFILE_PHASE_BEGIN warmup=1 samples={PROFILE_SAMPLES}", flush=True)
            runtime.start_gather_batch_get_profile()
            batch_get()  # Warm the profiling path; exclude this sample.
            for _ in range(PROFILE_SAMPLES):
                begin = time.perf_counter_ns()
                batch_get()
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
