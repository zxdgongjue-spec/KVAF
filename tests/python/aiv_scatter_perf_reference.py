"""Original miss-count workload CLI shared by the AIV Scatter benchmarks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.npu_eids import parse_npu_eids


DEFAULT_REQUEST_LENGTH = 32_768
DEFAULT_TABLE_CAPACITY = 16_384 * 78
DEFAULT_TOKENS_PER_BLOCK = 128
DEFAULT_TOKEN_BYTES = 656
DEFAULT_AIV_CORES = 2
DEFAULT_JETTYS_PER_CORE = 2
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
        description=(
            "Measure NPU BatchGet performance with an independently configurable "
            "request length, table capacity, block layout and exact miss count."
        )
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
            "remote CPU DDR read order: random samples the full table, "
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
            "patterns to isolate remote CPU DDR access behavior"
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
        "--adaptive-calibration-only",
        action="store_true",
        help=argparse.SUPPRESS,
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
        "--ubmem-profile-detail", action="store_true",
        help="measure UBmem address preparation, MTE submission and final drain; adds timing overhead",
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
    parser.add_argument("--transport", choices=("urma", "ubmem"), default="urma")
    parser.set_defaults(
        adaptive_peer_ub_devices=configured_adaptive_ub_devices(),
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
    if args.miss_count is None:
        args.miss_count = args.request_length
    args.adaptive_ub_miss_count = environment_int(
        "KVAF_ADAPTIVE_UB_MISS_COUNT", DEFAULT_ADAPTIVE_UB_MISS_COUNT
    )
    return args


def validate_args(args: argparse.Namespace) -> str | None:
    if args.transport == "ubmem" and (args.adaptive_ub_device or args.peer_ub_device):
        return "adaptive-ub-device and peer-ub-device apply only to URMA"
    if args.ubmem_profile_detail and (args.transport != "ubmem" or not args.kernel_profile):
        return "ubmem-profile-detail requires --transport ubmem and --kernel-profile"
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
    if args.adaptive_calibration_only and not args.adaptive_ub_device:
        return "adaptive-calibration-only requires adaptive-ub-device"
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
