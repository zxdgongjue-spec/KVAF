#!/usr/bin/env python3
"""Configurable NPU Runtime BatchGet steady-state performance benchmark.

Unlike ``test_npu_runtime_perf.py``, this benchmark has no derived ``batch``
dimension. The request-array length, AccessTable capacity, block layout and
exact ``miss_count`` are independent inputs. BatchPut is used only to populate
the remote table before measurement and is never included in the timing.

The five primary settings can be supplied as command-line options or through
environment variables:

    --request-length       KVAF_BATCH_GET_PERF_REQUEST_LENGTH
    --table-capacity       KVAF_BATCH_GET_PERF_TABLE_CAPACITY
    --tokens-per-block     KVAF_BATCH_GET_PERF_TOKENS_PER_BLOCK
    --token-bytes          KVAF_BATCH_GET_PERF_TOKEN_BYTES
    --miss-count           KVAF_BATCH_GET_PERF_MISS_COUNT
    --max-inflight-total   KVAF_BATCH_GET_MAX_INFLIGHT_TOTAL
    --batch-get-mode       KVAF_BATCH_GET_MODE (lookup, no-pipeline, raw)
    --raw-addresses        KVAF_BATCH_GET_RAW_ADDRESSES=1
    --source-access-pattern
                           KVAF_BATCH_GET_PERF_SOURCE_ACCESS_PATTERN
    --output-access-pattern
                           KVAF_BATCH_GET_PERF_OUTPUT_ACCESS_PATTERN
    --sort-input-indices   KVAF_BATCH_GET_PERF_SORT_INPUT_INDICES=1
    --kernel-profile       KVAF_BATCH_GET_PERF_KERNEL_PROFILE=1
    --kernel-profile-clock-mhz
                           KVAF_BATCH_GET_PERF_KERNEL_PROFILE_CLOCK_MHZ
    adaptive UB devices    KVAF_ADAPTIVE_UB_DEVICES
    adaptive miss count    KVAF_ADAPTIVE_UB_MISS_COUNT
    --peer-device          KVAF_PEER_DEVICE (cpu or npu; default cpu)
    --peer-service-addr    KVAF_PEER_SERVICE_ADDR (falls back to KVAF_CPU_SERVICE_ADDR)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import detect_torch_backend
from npu_eids import parse_npu_eids


DEFAULT_REQUEST_LENGTH = 32_768
DEFAULT_TABLE_CAPACITY = 16_384 * 78
DEFAULT_TOKENS_PER_BLOCK = 128
DEFAULT_TOKEN_BYTES = 656
DEFAULT_AIV_CORES = 2
DEFAULT_JETTYS_PER_CORE = 1
MAX_JETTYS_PER_CORE = 64
DEFAULT_WARMUP = 3
DEFAULT_ITERATIONS = 100
DEFAULT_POPULATE_CHUNK_ENTRIES = 128
DEFAULT_KERNEL_PROFILE_CLOCK_MHZ = 1000
DEFAULT_ADAPTIVE_UB_MISS_COUNT = 2048
DEFAULT_ADAPTIVE_PEER_UB_DEVICES = (
    "udmac0d1e2",
    "udmac0d1e3",
    "udmac1d1e2",
    "udmac1d1e3",
)
MAX_INT32 = (1 << 31) - 1
MAX_UINT64 = (1 << 64) - 1


def environment_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def environment_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off")


def configured_adaptive_ub_devices() -> list[str]:
    configured = os.environ.get(
        "KVAF_ADAPTIVE_UB_DEVICES",
        ",".join(DEFAULT_ADAPTIVE_PEER_UB_DEVICES),
    )
    return [device.strip() for device in configured.split(",")]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=(
            "Measure NPU BatchGet performance with an independently configurable "
            "request length, table capacity, block layout and exact miss count."
        )
    )
    mode_options = parser.add_mutually_exclusive_group()
    mode_options.add_argument(
        "--batch-get-mode", choices=("lookup", "no-pipeline", "raw"),
        help="lookup: pipelined table lookup; no-pipeline: serial table lookup; raw: pipelined final addresses",
    )
    mode_options.add_argument(
        "--raw-addresses", action="store_const", const="raw", dest="batch_get_mode",
        help="alias for --batch-get-mode raw; address preparation is outside timed BatchGet",
    )
    parser.add_argument(
        "--peer-device", choices=("cpu", "npu"),
        default=os.environ.get("KVAF_PEER_DEVICE", "cpu"),
        help="remote memory kind: CPU DDR or NPU HBM",
    )
    parser.add_argument(
        "--peer-service-addr",
        default=(os.environ.get("KVAF_PEER_SERVICE_ADDR")
                 or os.environ.get("KVAF_CPU_SERVICE_ADDR")),
        help="peer host:port; CLI overrides the peer/legacy CPU address environment",
    )
    parser.add_argument(
        "--request-length",
        type=int,
        default=environment_int(
            "KVAF_BATCH_GET_PERF_REQUEST_LENGTH", DEFAULT_REQUEST_LENGTH
        ),
        help="length/capacity of indices, offsets and output_indices",
    )
    parser.add_argument(
        "--table-capacity",
        type=int,
        default=environment_int(
            "KVAF_BATCH_GET_PERF_TABLE_CAPACITY", DEFAULT_TABLE_CAPACITY
        ),
        help="number of valid AccessTable entries",
    )
    parser.add_argument(
        "--tokens-per-block",
        type=int,
        default=environment_int(
            "KVAF_BATCH_GET_PERF_TOKENS_PER_BLOCK", DEFAULT_TOKENS_PER_BLOCK
        ),
        help="number of tokens stored in each AccessTable entry",
    )
    parser.add_argument(
        "--token-bytes",
        type=int,
        default=environment_int("KVAF_BATCH_GET_PERF_TOKEN_BYTES", DEFAULT_TOKEN_BYTES),
        help="bytes transferred for each valid request",
    )
    parser.add_argument(
        "--miss-count",
        type=int,
        default=(
            int(os.environ["KVAF_BATCH_GET_PERF_MISS_COUNT"])
            if "KVAF_BATCH_GET_PERF_MISS_COUNT" in os.environ
            else None
        ),
        help="exact valid request-prefix length; defaults to request-length",
    )
    parser.add_argument(
        "--source-access-pattern",
        choices=("random", "contiguous", "shuffled-contiguous"),
        default=os.environ.get(
            "KVAF_BATCH_GET_PERF_SOURCE_ACCESS_PATTERN", "random"
        ),
        help=(
            "remote memory read order: random samples the full table, "
            "contiguous reads an ascending token range, and shuffled-contiguous "
            "sends the same contiguous range in random order"
        ),
    )
    parser.add_argument(
        "--output-access-pattern",
        choices=("random", "contiguous"),
        default=os.environ.get(
            "KVAF_BATCH_GET_PERF_OUTPUT_ACCESS_PATTERN", "contiguous"
        ),
        help=(
            "local NPU HBM output order; use contiguous while comparing source "
            "patterns to isolate remote memory access behavior"
        ),
    )
    parser.add_argument(
        "--aiv-cores",
        type=int,
        default=environment_int("KVAF_BATCH_GET_PERF_AIV_CORES", DEFAULT_AIV_CORES),
    )
    parser.add_argument(
        "--jettys-per-core",
        type=int,
        choices=range(1, MAX_JETTYS_PER_CORE + 1),
        default=environment_int(
            "KVAF_BATCH_GET_PERF_JETTYS_PER_CORE", DEFAULT_JETTYS_PER_CORE
        ),
    )
    parser.add_argument(
        "--npu-eids",
        type=parse_npu_eids,
        default=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")),
        help=(
            "comma-separated local NPU EIDs; workers are striped over all "
            "listed EIDs (environment: KVAF_NPU_EIDS)"
        ),
    )
    parser.add_argument(
        "--adaptive-ub-device", action="store_true",
        help=(
            "startup tuning over the N-side peer-device allowlist when K has "
            "no explicit device filter"
        ),
    )
    parser.add_argument(
        "--peer-ub-device",
        default="",
        help="strict K-side UB device selected by an earlier process",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=environment_int("KVAF_BATCH_GET_PERF_WARMUP", DEFAULT_WARMUP),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=environment_int("KVAF_BATCH_GET_PERF_ITERATIONS", DEFAULT_ITERATIONS),
    )
    parser.add_argument(
        "--populate-chunk-entries",
        type=int,
        default=environment_int(
            "KVAF_BATCH_GET_PERF_POPULATE_CHUNK_ENTRIES",
            DEFAULT_POPULATE_CHUNK_ENTRIES,
        ),
        help="setup-only BatchPut chunk size; it does not affect BatchGet shape",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=environment_int("KVAF_BATCH_GET_PERF_SEED", 20260903),
    )
    input_order = parser.add_mutually_exclusive_group()
    input_order.add_argument(
        "--sort-input-indices",
        dest="sort_input_indices",
        action="store_true",
        help=(
            "sort the active request prefix by the table-resolved entry and then offsets; "
            "offsets and output_indices follow the same permutation"
        ),
    )
    input_order.add_argument(
        "--no-sort-input-indices",
        dest="sort_input_indices",
        action="store_false",
        help="keep the generated request prefix in its configured order",
    )
    kernel_profile = parser.add_mutually_exclusive_group()
    kernel_profile.add_argument(
        "--kernel-profile",
        dest="kernel_profile",
        action="store_true",
        help="collect the per-core AIV kernel timeline for every measured sample",
    )
    kernel_profile.add_argument(
        "--no-kernel-profile",
        dest="kernel_profile",
        action="store_false",
        help="disable AIV kernel cycle profiling",
    )
    parser.add_argument(
        "--kernel-profile-clock-mhz",
        type=int,
        default=environment_int(
            "KVAF_BATCH_GET_PERF_KERNEL_PROFILE_CLOCK_MHZ",
            environment_int(
                "KVAF_RUNTIME_PERF_KERNEL_PROFILE_CLOCK_MHZ",
                DEFAULT_KERNEL_PROFILE_CLOCK_MHZ,
            ),
        ),
        help="AIV clock used to convert profile cycles to milliseconds",
    )
    parser.set_defaults(
        kernel_profile=environment_bool("KVAF_BATCH_GET_PERF_KERNEL_PROFILE"),
        sort_input_indices=environment_bool(
            "KVAF_BATCH_GET_PERF_SORT_INPUT_INDICES"
        ),
    )
    parser.add_argument(
        "--max-inflight-total", type=int,
        default=environment_int("KVAF_BATCH_GET_MAX_INFLIGHT_TOTAL", 2048),
        help="total unconfirmed READ limit across AIV cores; 0 uses the baseline",
    )
    args = parser.parse_args(argv)
    if args.batch_get_mode is None:
        args.batch_get_mode = (
            os.environ.get("KVAF_BATCH_GET_MODE") or
            ("raw" if os.environ.get("KVAF_BATCH_GET_RAW_ADDRESSES") == "1" else "lookup")
        )
    if args.batch_get_mode not in ("lookup", "no-pipeline", "raw"):
        parser.error("KVAF_BATCH_GET_MODE must be lookup, no-pipeline, or raw")
    args.raw_addresses = args.batch_get_mode == "raw"
    # CPU device names are not NPU service candidates. Do not forward the
    # CPU default allowlist to an NPU peer, even when tuning is disabled.
    args.adaptive_peer_ub_devices = (
        configured_adaptive_ub_devices() if args.peer_device == "cpu" else []
    )
    if args.miss_count is None:
        args.miss_count = args.request_length
    args.adaptive_ub_miss_count = environment_int(
        "KVAF_ADAPTIVE_UB_MISS_COUNT", DEFAULT_ADAPTIVE_UB_MISS_COUNT
    )
    return args


def validate_args(args: argparse.Namespace) -> str | None:
    if args.peer_device not in ("cpu", "npu"):
        return "peer-device must be cpu or npu"
    if args.peer_device == "npu" and (args.adaptive_ub_device or args.peer_ub_device):
        return "NPU peers do not support --adaptive-ub-device or --peer-ub-device"
    block_bytes = args.token_bytes * args.tokens_per_block
    ddr_bytes = args.table_capacity * block_bytes
    if args.request_length <= 0 or args.request_length > MAX_INT32:
        return "request-length must be in [1, INT32_MAX]"
    if args.table_capacity <= 0 or args.table_capacity > MAX_INT32:
        return "table-capacity must be in [1, INT32_MAX]"
    if args.tokens_per_block <= 0 or args.tokens_per_block > MAX_INT32:
        return "tokens-per-block must be in [1, INT32_MAX]"
    if args.token_bytes <= 0 or args.token_bytes > MAX_INT32:
        return "token-bytes must be in [1, INT32_MAX]"
    if (args.tokens_per_block - 1) * args.token_bytes > MAX_INT32:
        return "the largest per-block token offset must fit int32"
    if block_bytes <= 0 or ddr_bytes > MAX_UINT64:
        return "configured table size must fit uint64"
    if args.miss_count < 0 or args.miss_count > args.request_length:
        return "miss-count must be between zero and request-length"
    if args.adaptive_ub_device and args.miss_count == 0:
        return "adaptive UB device tuning requires a positive miss-count"
    if args.adaptive_ub_device and (
        args.adaptive_ub_miss_count <= 0 or args.adaptive_ub_miss_count > MAX_INT32
    ):
        return "KVAF_ADAPTIVE_UB_MISS_COUNT must be in [1, INT32_MAX]"
    if args.adaptive_ub_device and args.peer_ub_device:
        return "adaptive-ub-device and peer-ub-device are mutually exclusive"
    if args.adaptive_ub_device and (
        any(not device for device in args.adaptive_peer_ub_devices)
        or len(set(args.adaptive_peer_ub_devices))
        != len(args.adaptive_peer_ub_devices)
    ):
        return "KVAF_ADAPTIVE_UB_DEVICES must contain unique non-empty names"
    if args.aiv_cores <= 0 or args.aiv_cores > 48:
        return "aiv-cores must be in [1, 48]"
    if len(args.npu_eids) > args.aiv_cores * args.jettys_per_core:
        return "npu-eids count must not exceed aiv-cores * jettys-per-core"
    if not 0 <= args.max_inflight_total <= 0xFFFFFFFF:
        return "max-inflight-total must fit uint32"
    if args.max_inflight_total and args.max_inflight_total < 2 * args.aiv_cores:
        return "max-inflight-total must be zero or at least twice aiv-cores"
    if args.warmup < 0:
        return "warmup must be non-negative"
    if args.iterations <= 0:
        return "iterations must be positive"
    if args.populate_chunk_entries <= 0:
        return "populate-chunk-entries must be positive"
    if args.kernel_profile_clock_mhz <= 0:
        return "kernel-profile-clock-mhz must be positive"
    return None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def sort_request_plan_prefix(
    torch: object,
    address_table: object,
    indices: object,
    offsets: object,
    output_indices: object,
    active_count: int,
) -> tuple[object, object, object]:
    """Sort active requests by resolved table entry and offset."""
    if active_count <= 1:
        return indices, offsets, output_indices

    active_indices = indices[:active_count]
    active_offsets = offsets[:active_count]
    active_output_indices = output_indices[:active_count]
    entry_addresses = address_table.index_select(
        0, active_indices.to(torch.int64)
    )
    # Every valid table entry owns one non-overlapping, fixed-size block, so
    # sorting the resolved remote byte address is equivalent to sorting first
    # by the actual slot ID and then by the offset inside that slot.
    sort_keys = entry_addresses + active_offsets.to(torch.int64)
    _, permutation = torch.sort(sort_keys)
    sorted_indices = active_indices.index_select(0, permutation)
    sorted_offsets = active_offsets.index_select(0, permutation)
    sorted_output_indices = active_output_indices.index_select(0, permutation)
    indices[:active_count].copy_(sorted_indices)
    offsets[:active_count].copy_(sorted_offsets)
    output_indices[:active_count].copy_(sorted_output_indices)
    return indices, offsets, output_indices


def print_samples(
    operation: str,
    samples: list[float],
    warmup: int,
    request_length: int,
    miss_count: int,
    token_bytes: int,
) -> None:
    mean_ms = statistics.mean(samples)
    result = (
        "PERF_RESULT "
        f"operation={operation} samples={len(samples)} warmup_excluded={warmup} "
        f"request_length={request_length} miss_count={miss_count} "
        f"mean_ms={mean_ms:.3f} "
        f"p50_ms={percentile(samples, 0.50):.3f} "
        f"p90_ms={percentile(samples, 0.90):.3f} "
        f"p99_ms={percentile(samples, 0.99):.3f}"
    )
    if operation == "batch_get":
        payload_bytes = miss_count * token_bytes
        useful_gibps = (
            payload_bytes / (mean_ms / 1000.0) / (1024.0**3)
            if payload_bytes > 0
            else 0.0
        )
        result += (
            f" payload_bytes_per_call={payload_bytes} "
            f"useful_GiBps={useful_gibps:.3f}"
        )
    print(result)


def print_kernel_profile(
    samples: list[list[dict[str, object]]],
    warmup: int,
    clock_mhz: int,
    npu_core_count: int,
    jettys_per_core: int,
    request_length: int,
    miss_count: int,
    token_bytes: int,
    address_mode: str = "lookup",
    batch_get_mode: str | None = None,
) -> None:
    if not samples or any(len(sample) != npu_core_count for sample in samples):
        raise AssertionError("BatchGet kernel profile sample/core count mismatch")
    post_stages = ("wqe_fill", "mte_copy", "doorbell") if batch_get_mode == "no-pipeline" else ()
    if post_stages and any(
        profile.get("post_stage_profile_enabled") != 1 or
        any(f"{stage}_cycles" not in profile for stage in post_stages)
        for sample in samples for profile in sample
    ):
        raise AssertionError(
            "No-pipeline stage profiling is unavailable; rebuild the host, Python binding "
            "and V4 kernels together, and select --batch-get-mode no-pipeline"
        )

    def elapsed_ms(profile: dict[str, object], begin: str, end: str) -> float:
        return (int(profile[end]) - int(profile[begin])) / clock_mhz / 1000.0

    stage_samples: dict[str, list[float]] = {
        "kernel_wall": [],
        "kernel_total": [],
        "kernel_setup": [],
        "kernel_compact": [],
        "kernel_post": [],
        "kernel_transfer_window": [],
        "kernel_completion_wait": [],
        "kernel_finalize": [],
        **{f"kernel_{stage}": [] for stage in post_stages},
    }
    per_core: list[dict[str, float]] = [
        {
            "total_ms": 0.0,
            "setup_ms": 0.0,
            "compact_ms": 0.0,
            "post_ms": 0.0,
            "completion_wait_ms": 0.0,
            "finalize_ms": 0.0,
            "active_requests": 0.0,
            "total_posts": 0.0,
            "submitted_group_count": 0.0,
            "completed_group_count": 0.0,
            "completion_count": 0.0,
            "peak_submitted_not_drained": 0.0,
            **{f"{stage}_ms": 0.0 for stage in post_stages},
            **{
                name: 0.0
                for jetty in range(MAX_JETTYS_PER_CORE)
                for name in (f"jetty{jetty}_wait_ms", f"jetty{jetty}_polls")
            },
        }
        for _ in range(npu_core_count)
    ]
    for sample in samples:
        stage_samples["kernel_wall"].append(
            (
                max(int(profile["kernel_finish_cycle"]) for profile in sample)
                - min(int(profile["kernel_start_cycle"]) for profile in sample)
            )
            / clock_mhz
            / 1000.0
        )
        critical: dict[str, float] = {
            "kernel_total": 0.0,
            "kernel_setup": 0.0,
            "kernel_compact": 0.0,
            "kernel_post": 0.0,
            "kernel_transfer_window": 0.0,
            "kernel_completion_wait": 0.0,
            "kernel_finalize": 0.0,
            **{f"kernel_{stage}": 0.0 for stage in post_stages},
        }
        for profile in sample:
            core = int(profile["core"])
            total = elapsed_ms(profile, "kernel_start_cycle", "kernel_finish_cycle")
            setup = elapsed_ms(profile, "kernel_start_cycle", "setup_done_cycle")
            compact = elapsed_ms(profile, "setup_done_cycle", "compact_done_cycle")
            post = int(profile["post_active_cycles"]) / clock_mhz / 1000.0
            completion = int(profile["drain_wait_cycles"]) / clock_mhz / 1000.0
            finalize = elapsed_ms(
                profile, "completion_done_cycle", "kernel_finish_cycle"
            )
            first_doorbell = int(profile["first_doorbell_cycle"])
            transfer = (
                0.0
                if first_doorbell == 0
                else (
                    int(profile["completion_done_cycle"]) - first_doorbell
                )
                / clock_mhz
                / 1000.0
            )
            critical["kernel_total"] = max(critical["kernel_total"], total)
            critical["kernel_setup"] = max(critical["kernel_setup"], setup)
            critical["kernel_compact"] = max(critical["kernel_compact"], compact)
            critical["kernel_post"] = max(critical["kernel_post"], post)
            critical["kernel_transfer_window"] = max(
                critical["kernel_transfer_window"], transfer
            )
            critical["kernel_completion_wait"] = max(
                critical["kernel_completion_wait"], completion
            )
            critical["kernel_finalize"] = max(
                critical["kernel_finalize"], finalize
            )
            accumulator = per_core[core]
            for stage in post_stages:
                value = int(profile[f"{stage}_cycles"]) / clock_mhz / 1000.0
                critical[f"kernel_{stage}"] = max(critical[f"kernel_{stage}"], value)
                accumulator[f"{stage}_ms"] += value
            accumulator["total_ms"] += total
            accumulator["setup_ms"] += setup
            accumulator["compact_ms"] += compact
            accumulator["post_ms"] += post
            accumulator["completion_wait_ms"] += completion
            accumulator["finalize_ms"] += finalize
            accumulator["active_requests"] += int(profile["active_requests"])
            accumulator["total_posts"] += int(profile["total_posts"])
            accumulator["submitted_group_count"] += int(profile["submitted_group_count"])
            accumulator["completed_group_count"] += int(profile["completed_group_count"])
            accumulator["completion_count"] += int(profile["completion_count"])
            accumulator["peak_submitted_not_drained"] += int(profile["peak_submitted_not_drained"])
            waits = list(profile["jetty_completion_wait_cycles"])
            polls = list(profile["jetty_completion_polls"])
            for jetty in range(MAX_JETTYS_PER_CORE):
                accumulator[f"jetty{jetty}_wait_ms"] += (
                    int(waits[jetty]) / clock_mhz / 1000.0
                )
                accumulator[f"jetty{jetty}_polls"] += int(polls[jetty])
        for stage, value in critical.items():
            stage_samples[stage].append(value)

    print(
        "KERNEL_PROFILE_CONTRACT "
        f"clock_mhz={clock_mhz} unit=ms samples={len(samples)} "
        f"address_mode={address_mode} "
        f"aiv_cores={npu_core_count} jettys_per_core={jettys_per_core} "
        f"request_length={request_length} miss_count={miss_count} "
        "setup=entity_queue_and_miss_count compact=removed_direct_prefix "
        "post=sum_group_capacity_check_wqe_fill_sq_copy_and_doorbell "
        "transfer_window=first_doorbell_to_all_cq_complete "
        "completion_wait=sum_all_group_drains "
        "peak_submitted_not_drained=per_core_conservative_software_count "
        "profile_overhead_included=1 d2h_outside_host_metric=1"
    )
    if post_stages:
        print(
            "KERNEL_POST_STAGE_CONTRACT mode=no-pipeline "
            "aggregation=max_per_core_sum_over_posts "
            "wqe_fill=fill_and_vector_completion_wait "
            "mte_copy=sq_copy_and_mte3_completion_wait "
            "doorbell=scalar_pi_update_and_st_dev "
            "cq_wait_excluded=1 synchronization_and_timer_overhead_included=1 "
            "independent_stage_maxima_not_additive=1"
        )
    total_useful_bytes = len(samples) * miss_count * token_bytes
    for stage, values in stage_samples.items():
        cycle_values = [value * clock_mhz * 1000.0 for value in values]
        useful_gibps = ""
        if stage == "kernel_total":
            useful_gibps = (
                " useful_GiBps="
                f"{total_useful_bytes / (1024.0**3) / (sum(values) / 1000.0):.3f}"
            )
        print(
            "PERF_RESULT "
            f"operation=batch_get_{stage} samples={len(values)} "
            f"warmup_excluded={warmup} "
            f"mean_ms={statistics.mean(values):.3f} "
            f"p50_ms={percentile(values, 0.50):.3f} "
            f"p90_ms={percentile(values, 0.90):.3f} "
            f"p99_ms={percentile(values, 0.99):.3f} "
            f"mean_us={statistics.mean(values) * 1000.0:.3f} "
            f"p50_us={percentile(values, 0.50) * 1000.0:.3f} "
            f"p90_us={percentile(values, 0.90) * 1000.0:.3f} "
            f"p99_us={percentile(values, 0.99) * 1000.0:.3f} "
            f"mean_cycles={statistics.mean(cycle_values):.3f} "
            f"p50_cycles={percentile(cycle_values, 0.50):.3f} "
            f"p90_cycles={percentile(cycle_values, 0.90):.3f} "
            f"p99_cycles={percentile(cycle_values, 0.99):.3f} "
            f"units={request_length}_requests_per_call{useful_gibps} "
            f"address_mode={address_mode} "
            f"batch_get_mode={batch_get_mode or address_mode}"
        )
    divisor = float(len(samples))
    for core, accumulator in enumerate(per_core):
        print(
            "KERNEL_PROFILE_CORE "
            f"core={core} samples={len(samples)} "
            + " ".join(
                f"{name}_mean={value / divisor:.6f}"
                for name, value in accumulator.items()
            )
        )


def batch_get_kernel_path(batch_get_mode: str) -> Path:
    kernel = Path(os.environ.get(
        "KVAF_BATCH_GET_KERNEL", os.environ.get("KVAF_AIV_BATCH_GET_KERNEL", "")
    ))
    if batch_get_mode == "no-pipeline":
        kernel = kernel.parent / "kvaf_execute_batch_get_no_pipeline_kernels.o"
    return kernel


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    error = validate_args(args)
    if error is not None:
        print(
            f"FAIL: invalid BatchGet performance configuration: {error}",
            file=sys.stderr,
        )
        return 2

    backend = detect_torch_backend()
    if backend.name != "npu":
        print(
            "SKIP: NPU benchmark requires torch_npu; "
            f"detected {backend.name} ({backend.reason})."
        )
        return 0
    import kvaf

    if not hasattr(kvaf, "NpuRuntime"):
        print(
            "SKIP: rebuild KVAF with KVAF_ENABLE_AIV_BATCH_GET=ON "
            "and KVAF_BUILD_PYTHON=ON."
        )
        return 0
    if args.kernel_profile and not all(
        hasattr(kvaf.NpuRuntime, method)
        for method in (
            "start_batch_get_kernel_profile",
            "batch_get_kernel_profile",
            "stop_batch_get_kernel_profile",
        )
    ):
        print(
            "FAIL: the loaded KVAF binding does not expose BatchGet kernel profiling",
            file=sys.stderr,
        )
        return 2
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the A5/CANN target")
        return 0

    kernel = batch_get_kernel_path(args.batch_get_mode)
    if not kernel.is_file():
        print(
            f"FAIL: missing BatchGet kernel for {args.batch_get_mode}: {kernel}; rebuild with ./build.sh npu",
            file=sys.stderr,
        )
        return 2
    batch_put_kernel = Path(
        os.environ.get(
            "KVAF_BATCH_PUT_KERNEL",
            str(kernel.with_name("kvaf_execute_batch_put_kernels.o")),
        )
    )
    if not batch_put_kernel.is_file():
        print(
            "FAIL: KVAF_BATCH_PUT_KERNEL must name the built BatchPut kernel object",
            file=sys.stderr,
        )
        return 2
    if not args.peer_service_addr:
        print(
            "FAIL: set --peer-service-addr or KVAF_PEER_SERVICE_ADDR "
            "(legacy: KVAF_CPU_SERVICE_ADDR) to the peer host:port",
            file=sys.stderr,
        )
        return 2

    torch = backend.torch
    assert torch is not None
    device_id = int(os.environ.get("DEVICE_ID", "0"))
    request_length = args.request_length
    token_bytes = args.token_bytes
    tokens_per_block = args.tokens_per_block
    miss_count_value = args.miss_count
    block_bytes = token_bytes * tokens_per_block
    table_capacity = args.table_capacity
    ddr_bytes = table_capacity * block_bytes
    total_tokens = table_capacity * tokens_per_block

    torch.npu.set_device(device_id)
    torch.manual_seed(args.seed)
    device = f"npu:{device_id}"

    runtime = kvaf.NpuRuntime(
        str(kernel),
        batch_get_token_bytes=token_bytes,
        batch_get_tokens_per_block=tokens_per_block,
        device_id=device_id,
        npu_core_count=args.aiv_cores,
        batch_put_kernel_object_path=str(batch_put_kernel),
        jettys_per_core=args.jettys_per_core,
        batch_get_max_inflight_total=args.max_inflight_total,
        npu_eids=args.npu_eids,
        adaptive_ub_device=args.adaptive_ub_device,
        adaptive_ub_miss_count=args.adaptive_ub_miss_count,
        peer_ub_device=args.peer_ub_device,
        adaptive_peer_ub_devices=args.adaptive_peer_ub_devices,
    )
    runtime.link_to(args.peer_service_addr, peer_device=args.peer_device)
    table = None
    device_table = None
    kernel_profile_started = False
    try:
        print(
            "BATCH_GET_PERF_CONFIG "
            f"address_mode={'raw' if args.raw_addresses else 'lookup'} "
            f"batch_get_mode={args.batch_get_mode} kernel={kernel} "
            f"peer_device={args.peer_device} peer_service_addr={args.peer_service_addr} "
            f"request_length={request_length} table_capacity={table_capacity} "
            f"tokens_per_block={tokens_per_block} token_bytes={token_bytes} "
            f"block_bytes={block_bytes} miss_count={miss_count_value} "
            f"source_access_pattern={args.source_access_pattern} "
            f"output_access_pattern={args.output_access_pattern} "
            f"ddr_bytes={ddr_bytes} aiv_cores={args.aiv_cores} "
            f"max_inflight_total={args.max_inflight_total} "
            f"jettys_per_core={args.jettys_per_core} eid_paths={len(args.npu_eids) or 1} "
            f"npu_eids={','.join(args.npu_eids) or 'auto'} warmup={args.warmup} "
            f"adaptive_ub_device={int(args.adaptive_ub_device)} "
            f"adaptive_ub_miss_count={args.adaptive_ub_miss_count} "
            f"peer_ub_device={args.peer_ub_device or 'auto'} "
            f"iterations={args.iterations} seed={args.seed} "
            f"sort_input_indices={int(args.sort_input_indices)} "
            f"kernel_profile={int(args.kernel_profile)} "
            f"kernel_profile_clock_mhz={args.kernel_profile_clock_mhz} "
            "timing=call_to_stream_completion"
        )

        print("BATCH_GET_PERF_PHASE_BEGIN phase=initialize", flush=True)
        runtime.initialize(ddr_bytes, block_bytes)
        if runtime.remote_region["device"] != args.peer_device:
            raise AssertionError("Runtime selected the wrong remote memory device")
        print("BATCH_GET_PERF_PHASE_DONE phase=initialize", flush=True)

        # Every table entry contains the same deterministic token pattern. Reusing
        # a bounded source Tensor keeps setup memory independent of table capacity;
        # each BatchPut still allocates a distinct remote entry for every key.
        token_rows = torch.arange(
            tokens_per_block, dtype=torch.int64, device=device
        ).unsqueeze(1)
        token_byte_indices = torch.arange(
            token_bytes, dtype=torch.int64, device=device
        ).unsqueeze(0)
        entry_pattern = (
            (
                (token_rows * 131 + token_byte_indices * 17 + (token_byte_indices >> 3))
                & 0xFF
            )
            .to(torch.uint8)
            .contiguous()
        )
        populate_entries = min(args.populate_chunk_entries, table_capacity)
        populate_source = entry_pattern.reshape(1, block_bytes).repeat(populate_entries, 1)
        populate_offsets = [row * block_bytes for row in range(populate_entries)]
        populate_lengths = [block_bytes] * populate_entries
        output = torch.empty(
            (request_length, token_bytes), dtype=torch.uint8, device=device
        )
        miss_count = torch.tensor([miss_count_value], dtype=torch.int32, device=device)

        print("BATCH_GET_PERF_PHASE_BEGIN phase=populate_table")
        populated = 0
        while populated < table_capacity:
            count = min(populate_entries, table_capacity - populated)
            keys = list(range(populated + 1, populated + count + 1))
            runtime.batch_put(
                keys,
                populate_source[:count],
                populate_offsets[:count],
                populate_lengths[:count],
            )
            populated += count
        torch.npu.synchronize()
        print(
            "BATCH_GET_PERF_PHASE_DONE phase=populate_table "
            f"entries={table_capacity}"
        )

        print("BATCH_GET_PERF_PHASE_BEGIN phase=prepare_access_table")
        prepare_keys = torch.arange(1, table_capacity + 1, dtype=torch.int64)
        table = runtime.prepare_access_table(prepare_keys)
        device_table = runtime.access_table_view(
            table, device="npu", capacity=table_capacity
        )
        torch.npu.synchronize()
        print(
            "BATCH_GET_PERF_PHASE_DONE phase=prepare_access_table "
            f"entries={table_capacity}"
        )

        def prepare_submission(indices: object, offsets: object, rows: object) -> tuple:
            if args.raw_addresses:
                # These allocations/gathers run BEFORE synchronization and timing.
                remote = device_table.addresses[indices.to(torch.int64)] + offsets.to(torch.int64)
                local = rows.to(torch.int64) * token_bytes + int(output.data_ptr())
                return remote.contiguous(), local.contiguous(), miss_count, output
            return device_table, indices, offsets, miss_count, rows, output

        submit_batch_get = runtime.batch_get_raw if args.raw_addresses else runtime.batch_get

        # Execute one deterministic request plan and validate both the copied
        # prefix and the untouched suffix before collecting timing samples.
        print("BATCH_GET_PERF_PHASE_BEGIN phase=correctness_preflight")
        request_ids = torch.arange(request_length, dtype=torch.int64, device=device)
        preflight_token_ids = request_ids % total_tokens
        preflight_indices = (preflight_token_ids // tokens_per_block).to(torch.int32)
        preflight_token_slots = preflight_token_ids % tokens_per_block
        preflight_offsets = (preflight_token_slots * token_bytes).to(torch.int32)
        preflight_output_indices = request_ids.to(torch.int32)
        if args.sort_input_indices:
            (
                preflight_indices,
                preflight_offsets,
                preflight_output_indices,
            ) = sort_request_plan_prefix(
                torch,
                device_table.addresses,
                preflight_indices,
                preflight_offsets,
                preflight_output_indices,
                miss_count_value,
            )
        output.zero_()
        preflight_submission = prepare_submission(
            preflight_indices, preflight_offsets, preflight_output_indices
        )
        submit_batch_get(*preflight_submission)
        torch.npu.synchronize()
        if miss_count_value > 0 and not torch.equal(
            output[:miss_count_value],
            entry_pattern[preflight_token_slots[:miss_count_value]],
        ):
            raise AssertionError("BatchGet preflight payload validation failed")
        if miss_count_value < request_length and bool(
            torch.count_nonzero(output[miss_count_value:]).item()
        ):
            raise AssertionError("BatchGet modified rows outside miss_count prefix")
        print(
            "BATCH_GET_PERF_CORRECTNESS_PASS "
            f"request_length={request_length} miss_count={miss_count_value}"
        )
        print("BATCH_GET_PERF_PHASE_DONE phase=correctness_preflight")

        # The table is populated in key order and MetaManager allocates fixed
        # slots FIFO, so linear token IDs map to adjacent remote addresses.
        # shuffled-contiguous changes only their submission order and keeps the
        # exact same remote working set as contiguous.
        contiguous_source_token_ids = request_ids % total_tokens

        def create_request_plan() -> tuple[object, object, object]:
            if args.source_access_pattern == "random":
                # Preserve the established benchmark's full-table random source
                # distribution as the default behavior.
                indices = torch.randint(
                    0,
                    table_capacity,
                    (request_length,),
                    dtype=torch.int32,
                    device=device,
                )
                token_slots = torch.randint(
                    0,
                    tokens_per_block,
                    (request_length,),
                    dtype=torch.int32,
                    device=device,
                )
            else:
                source_token_ids = contiguous_source_token_ids
                if args.source_access_pattern == "shuffled-contiguous":
                    source_token_ids = source_token_ids[
                        torch.randperm(request_length, device=device)
                    ]
                indices = (source_token_ids // tokens_per_block).to(torch.int32)
                token_slots = (source_token_ids % tokens_per_block).to(torch.int32)
            offsets = token_slots * token_bytes
            output_indices = (
                request_ids.to(torch.int32)
                if args.output_access_pattern == "contiguous"
                else torch.randperm(request_length, device=device).to(torch.int32)
            )
            if args.sort_input_indices:
                indices, offsets, output_indices = sort_request_plan_prefix(
                    torch,
                    device_table.addresses,
                    indices,
                    offsets,
                    output_indices,
                    miss_count_value,
                )
            return indices, offsets, output_indices

        if args.kernel_profile:
            runtime.start_batch_get_kernel_profile()
            kernel_profile_started = True
            print("BATCH_GET_KERNEL_PROFILE_ENABLED units=cycles")

        print("BATCH_GET_PERF_PHASE_BEGIN phase=warmup")
        for _ in range(args.warmup):
            indices, offsets, output_indices = create_request_plan()
            submission = prepare_submission(indices, offsets, output_indices)
            torch.npu.synchronize()
            submit_batch_get(*submission)
            # Keep caller-owned request Tensors alive through completion.
            torch.npu.synchronize()
        print("BATCH_GET_PERF_PHASE_DONE phase=warmup " f"iterations={args.warmup}")

        print("BATCH_GET_PERF_PHASE_BEGIN phase=batch_get")
        submit_samples: list[float] = []
        synchronize_samples: list[float] = []
        batch_get_samples: list[float] = []
        kernel_profile_samples: list[list[dict[str, object]]] = []
        for _ in range(args.iterations):
            indices, offsets, output_indices = create_request_plan()
            submission = prepare_submission(indices, offsets, output_indices)
            # Request generation and Tensor work are deliberately excluded.
            torch.npu.synchronize()

            begin = time.perf_counter()
            submit_batch_get(*submission)
            submitted = time.perf_counter()
            torch.npu.synchronize()
            completed = time.perf_counter()
            submit_samples.append((submitted - begin) * 1000.0)
            synchronize_samples.append((completed - submitted) * 1000.0)
            batch_get_samples.append((completed - begin) * 1000.0)
            if args.kernel_profile:
                kernel_profile_samples.append(
                    list(runtime.batch_get_kernel_profile())
                )

        print_samples(
            "batch_get_submit",
            submit_samples,
            args.warmup,
            request_length,
            miss_count_value,
            token_bytes,
        )
        print_samples(
            "batch_get_synchronize",
            synchronize_samples,
            args.warmup,
            request_length,
            miss_count_value,
            token_bytes,
        )
        print_samples(
            "batch_get",
            batch_get_samples,
            args.warmup,
            request_length,
            miss_count_value,
            token_bytes,
        )
        if args.kernel_profile:
            print_kernel_profile(
                kernel_profile_samples,
                args.warmup,
                args.kernel_profile_clock_mhz,
                args.aiv_cores,
                args.jettys_per_core,
                request_length,
                miss_count_value,
                token_bytes,
                address_mode="raw" if args.raw_addresses else "lookup",
                batch_get_mode=args.batch_get_mode,
            )
            runtime.stop_batch_get_kernel_profile()
            kernel_profile_started = False
            print("BATCH_GET_KERNEL_PROFILE_DISABLED")
        print("BATCH_GET_PERF_PHASE_DONE phase=batch_get")
        return 0
    except Exception as error:
        print(
            "BATCH_GET_PERF_UNHANDLED_ERROR "
            f"type={type(error).__name__} message={error}",
            file=sys.stderr,
        )
        raise
    finally:
        print("BATCH_GET_PERF_PHASE_BEGIN phase=shutdown")
        if kernel_profile_started and runtime.initialized:
            try:
                runtime.stop_batch_get_kernel_profile()
            except Exception as error:
                print(
                    "BATCH_GET_KERNEL_PROFILE_DISABLE_ERROR "
                    f"type={type(error).__name__} message={error}",
                    file=sys.stderr,
                )
        if table is not None and runtime.initialized:
            runtime.release_access_table(table)
        device_table = None
        if runtime.initialized:
            runtime.shutdown()
        print("BATCH_GET_PERF_PHASE_DONE phase=shutdown")


if __name__ == "__main__":
    raise SystemExit(main())
