#!/usr/bin/env python3
"""Configurable NPU/AIV BatchPut kernel performance benchmark."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kvaf

from backend import detect_torch_backend


DEFAULT_ROW_COUNT = 4096
DEFAULT_ENTRY_BYTES = 1024
DEFAULT_AIV_CORES = 2
DEFAULT_JETTYS_PER_CORE = 2
DEFAULT_WARMUP = 3
DEFAULT_ITERATIONS = 100
MAX_INT32 = (1 << 31) - 1
MAX_UINT32 = (1 << 32) - 1
MAX_UINT64 = (1 << 64) - 1
HOST_PROFILE_DURATION_FIELDS = (
    "request_validation_ns",
    "prepare_put_ns",
    "metadata_reserve_ns",
    "metadata_pack_ns",
    "metadata_h2d_ns",
    "local_buffer_acquire_ns",
    "execute_sync_ns",
    "local_buffer_release_ns",
    "commit_put_ns",
    "runtime_non_execute_ns",
    "runtime_e2e_ns",
)


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


def uint32(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 0 or parsed > MAX_UINT32:
        raise argparse.ArgumentTypeError("value must fit uint32")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure end-to-end BatchPut latency and useful remote-write "
            "bandwidth, with optional host and AIV kernel breakdowns."
        )
    )
    parser.add_argument(
        "--row-count",
        type=int,
        default=environment_int("KVAF_BATCH_PUT_PERF_ROW_COUNT", DEFAULT_ROW_COUNT),
        help="number of KV entries written by each BatchPut",
    )
    parser.add_argument(
        "--entry-bytes",
        type=int,
        default=environment_int(
            "KVAF_BATCH_PUT_PERF_ENTRY_BYTES", DEFAULT_ENTRY_BYTES
        ),
        help="bytes written for each KV entry/WQE",
    )
    parser.add_argument(
        "--aiv-cores",
        type=int,
        default=environment_int("KVAF_BATCH_PUT_PERF_AIV_CORES", DEFAULT_AIV_CORES),
    )
    parser.add_argument(
        "--jettys-per-core",
        type=int,
        choices=(1, 2),
        default=environment_int(
            "KVAF_BATCH_PUT_PERF_JETTYS_PER_CORE", DEFAULT_JETTYS_PER_CORE
        ),
    )
    parser.add_argument(
        "--cpu-peer-candidate",
        type=uint32,
        default=os.environ.get("KVAF_BATCH_PUT_CPU_PEER_CANDIDATE") or None,
        help=(
            "CPU Runtime UMDK/EID candidate id; empty probes candidates "
            "in discovery order"
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=environment_int("KVAF_BATCH_PUT_PERF_WARMUP", DEFAULT_WARMUP),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=environment_int(
            "KVAF_BATCH_PUT_PERF_ITERATIONS", DEFAULT_ITERATIONS
        ),
    )
    kernel_profile = parser.add_mutually_exclusive_group()
    kernel_profile.add_argument(
        "--kernel-profile",
        dest="kernel_profile",
        action="store_true",
        help="collect AIV kernel execute and non-execute cycle summaries",
    )
    kernel_profile.add_argument(
        "--no-kernel-profile",
        dest="kernel_profile",
        action="store_false",
        help="disable AIV kernel cycle profiling",
    )
    parser.set_defaults(
        kernel_profile=environment_bool("KVAF_BATCH_PUT_PERF_KERNEL_PROFILE")
    )
    host_profile = parser.add_mutually_exclusive_group()
    host_profile.add_argument(
        "--host-profile",
        dest="host_profile",
        action="store_true",
        help="collect Runtime and host-side BatchPut stage timings per sample",
    )
    host_profile.add_argument(
        "--no-host-profile",
        dest="host_profile",
        action="store_false",
        help="disable Runtime and host-side BatchPut stage timing",
    )
    parser.set_defaults(
        host_profile=environment_bool("KVAF_BATCH_PUT_PERF_HOST_PROFILE")
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> str | None:
    payload_bytes = args.row_count * args.entry_bytes
    if args.row_count <= 0 or args.row_count > MAX_INT32:
        return "row-count must be in [1, INT32_MAX]"
    if args.entry_bytes <= 0 or args.entry_bytes > MAX_INT32:
        return "entry-bytes must be in [1, INT32_MAX]"
    if payload_bytes <= 0 or payload_bytes > MAX_UINT64:
        return "payload size must fit uint64"
    if args.aiv_cores <= 0 or args.aiv_cores > 48:
        return "aiv-cores must be in [1, 48]"
    if args.aiv_cores > args.row_count:
        return "aiv-cores must not exceed row-count"
    if args.warmup < 0:
        return "warmup must be non-negative"
    if args.iterations <= 0:
        return "iterations must be positive"
    return None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def print_runtime_samples(
    samples: list[float], warmup: int, row_count: int, entry_bytes: int
) -> None:
    mean_ms = statistics.mean(samples)
    payload_bytes = row_count * entry_bytes
    useful_gbps = payload_bytes / (mean_ms / 1000.0) / 1_000_000_000.0
    print(
        "PERF_RESULT "
        f"operation=batch_put samples={len(samples)} warmup_excluded={warmup} "
        f"row_count={row_count} entry_bytes={entry_bytes} "
        f"payload_bytes_per_call={payload_bytes} "
        f"mean_ms={mean_ms:.3f} "
        f"p50_ms={percentile(samples, 0.50):.3f} "
        f"p90_ms={percentile(samples, 0.90):.3f} "
        f"p99_ms={percentile(samples, 0.99):.3f} "
        f"useful_GBps={useful_gbps:.3f}"
    )


def print_host_profile_samples(
    samples: list[dict[str, object]], runtime_samples: list[float], warmup: int
) -> None:
    if not samples:
        raise AssertionError("BatchPut host profile snapshot is empty")
    mean_ns = {
        name: statistics.mean(float(sample[name]) for sample in samples)
        for name in HOST_PROFILE_DURATION_FIELDS
    }
    mean_caller_us = statistics.mean(runtime_samples) * 1000.0
    mean_runtime_e2e_us = mean_ns["runtime_e2e_ns"] / 1000.0
    print(
        "PERF_RESULT "
        f"operation=batch_put_host samples={len(samples)} "
        f"warmup_excluded={warmup} "
        f"mean_caller_e2e_us={mean_caller_us:.3f} "
        f"mean_runtime_e2e_us={mean_runtime_e2e_us:.3f} "
        "mean_caller_overhead_us="
        f"{mean_caller_us - mean_runtime_e2e_us:.3f} "
        "mean_runtime_non_execute_us="
        f"{mean_ns['runtime_non_execute_ns'] / 1000.0:.3f} "
        f"mean_request_validation_us={mean_ns['request_validation_ns'] / 1000.0:.3f} "
        f"mean_prepare_put_us={mean_ns['prepare_put_ns'] / 1000.0:.3f} "
        f"mean_metadata_reserve_us={mean_ns['metadata_reserve_ns'] / 1000.0:.3f} "
        f"mean_metadata_pack_us={mean_ns['metadata_pack_ns'] / 1000.0:.3f} "
        f"mean_metadata_h2d_us={mean_ns['metadata_h2d_ns'] / 1000.0:.3f} "
        "mean_local_buffer_acquire_us="
        f"{mean_ns['local_buffer_acquire_ns'] / 1000.0:.3f} "
        f"mean_execute_sync_us={mean_ns['execute_sync_ns'] / 1000.0:.3f} "
        "mean_local_buffer_release_us="
        f"{mean_ns['local_buffer_release_ns'] / 1000.0:.3f} "
        f"mean_commit_put_us={mean_ns['commit_put_ns'] / 1000.0:.3f} "
        f"mean_metadata_bytes={statistics.mean(float(sample['metadata_bytes']) for sample in samples):.3f}"
    )


def kernel_cycle_summary(profiles: list[dict[str, object]]) -> tuple[int, int]:
    if not profiles:
        raise AssertionError("BatchPut kernel profile snapshot is empty")
    slowest_profile = max(
        profiles,
        key=lambda profile: int(profile["execute_cycles"])
        + int(profile["non_execute_cycles"]),
    )
    return (
        int(slowest_profile["execute_cycles"]),
        int(slowest_profile["non_execute_cycles"]),
    )


def print_kernel_profile_samples(
    execute_samples: list[int],
    non_execute_samples: list[int],
    warmup: int,
    row_count: int,
    entry_bytes: int,
) -> None:
    execute_float_samples = [float(sample) for sample in execute_samples]
    non_execute_float_samples = [float(sample) for sample in non_execute_samples]
    print(
        "PERF_RESULT "
        f"operation=batch_put_kernel samples={len(execute_samples)} "
        f"warmup_excluded={warmup} row_count={row_count} "
        f"entry_bytes={entry_bytes} "
        f"payload_bytes_per_call={row_count * entry_bytes} "
        f"mean_execute_cycles={statistics.mean(execute_float_samples):.3f} "
        f"p50_execute_cycles={percentile(execute_float_samples, 0.50):.3f} "
        f"p90_execute_cycles={percentile(execute_float_samples, 0.90):.3f} "
        f"p99_execute_cycles={percentile(execute_float_samples, 0.99):.3f} "
        "mean_non_execute_cycles="
        f"{statistics.mean(non_execute_float_samples):.3f} "
        "p50_non_execute_cycles="
        f"{percentile(non_execute_float_samples, 0.50):.3f} "
        "p90_non_execute_cycles="
        f"{percentile(non_execute_float_samples, 0.90):.3f} "
        "p99_non_execute_cycles="
        f"{percentile(non_execute_float_samples, 0.99):.3f} "
        "aggregation=slowest_total_elapsed_core"
    )


def validate_payload(
    runtime: object,
    torch: object,
    keys: list[int],
    source: object,
    row_count: int,
    entry_bytes: int,
    device: str,
) -> None:
    table = runtime.prepare_access_table(keys)
    device_table = None
    try:
        device_table = runtime.access_table_view(
            table, device="npu", capacity=row_count
        )
        indices = torch.arange(row_count, dtype=torch.int32, device=device)
        offsets = torch.zeros(row_count, dtype=torch.int32, device=device)
        mask = torch.ones(row_count, dtype=torch.uint8, device=device)
        output = torch.empty(
            (row_count, entry_bytes), dtype=torch.uint8, device=device
        )
        runtime.batch_get(
            device_table,
            indices,
            offsets,
            mask,
            indices,
            output,
        )
        torch.npu.synchronize()
        if not torch.equal(output, source):
            raise AssertionError("BatchPut payload validation through BatchGet failed")
    finally:
        device_table = None
        runtime.release_access_table(table)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    error = validate_args(args)
    if error is not None:
        print(f"FAIL: invalid BatchPut performance configuration: {error}", file=sys.stderr)
        return 2

    backend = detect_torch_backend()
    if backend.name != "npu":
        print(
            "SKIP: NPU benchmark requires torch_npu; "
            f"detected {backend.name} ({backend.reason})."
        )
        return 0
    required_methods = (
        "start_batch_put_kernel_profile",
        "batch_put_kernel_profile",
        "stop_batch_put_kernel_profile",
    )
    required_host_methods = (
        "start_batch_put_host_profile",
        "batch_put_host_profile",
        "stop_batch_put_host_profile",
    )
    if not hasattr(kvaf, "NpuRuntime") or (
        args.kernel_profile
        and not all(hasattr(kvaf.NpuRuntime, method) for method in required_methods)
    ) or (
        args.host_profile
        and not all(hasattr(kvaf.NpuRuntime, method) for method in required_host_methods)
    ):
        print(
            "FAIL: rebuild KVAF with requested BatchPut profiling and Python binding",
            file=sys.stderr,
        )
        return 2
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the A5/CANN target")
        return 0

    batch_put_kernel = Path(os.environ.get("KVAF_BATCH_PUT_KERNEL", ""))
    batch_get_kernel = Path(os.environ.get("KVAF_BATCH_GET_KERNEL", ""))
    if not batch_put_kernel.is_file() or not batch_get_kernel.is_file():
        print(
            "FAIL: KVAF_BATCH_PUT_KERNEL and KVAF_BATCH_GET_KERNEL must name built objects",
            file=sys.stderr,
        )
        return 2
    remote_cpu_service = os.environ.get("KVAF_CPU_SERVICE_ADDR")
    if not remote_cpu_service:
        print(
            "FAIL: KVAF_CPU_SERVICE_ADDR must name the separately running CPU service",
            file=sys.stderr,
        )
        return 2

    torch = backend.torch
    assert torch is not None
    device_id = int(os.environ.get("DEVICE_ID", "0"))
    device = f"npu:{device_id}"
    torch.npu.set_device(device_id)

    row_count = args.row_count
    entry_bytes = args.entry_bytes
    payload_bytes = row_count * entry_bytes
    byte_indices = torch.arange(entry_bytes, dtype=torch.int64, device=device)
    entry_pattern = ((byte_indices * 17 + (byte_indices >> 3)) & 0xFF).to(
        torch.uint8
    )
    source = entry_pattern.reshape(1, entry_bytes).repeat(row_count, 1).contiguous()
    keys = list(range(1, row_count + 1))
    offsets = [row * entry_bytes for row in range(row_count)]
    lengths = [entry_bytes] * row_count
    torch.npu.synchronize()

    runtime = kvaf.NpuRuntime(
        str(batch_get_kernel),
        batch_get_token_bytes=entry_bytes,
        batch_get_tokens_per_block=1,
        device_id=device_id,
        npu_core_count=args.aiv_cores,
        batch_put_kernel_object_path=str(batch_put_kernel),
        jettys_per_core=args.jettys_per_core,
        cpu_peer_candidate=args.cpu_peer_candidate,
    )
    runtime.link_to(remote_cpu_service)
    committed = False
    kernel_profile_started = False
    host_profile_started = False
    try:
        print(
            "BATCH_PUT_PERF_CONFIG "
            f"row_count={row_count} entry_bytes={entry_bytes} "
            f"payload_bytes={payload_bytes} aiv_cores={args.aiv_cores} "
            f"jettys_per_core={args.jettys_per_core} warmup={args.warmup} "
            "cpu_peer_candidate="
            f"{args.cpu_peer_candidate if args.cpu_peer_candidate is not None else 'auto'} "
            f"iterations={args.iterations} kernel_profile={int(args.kernel_profile)} "
            f"host_profile={int(args.host_profile)} "
            "backend=aiv_direct_hccp runtime_timing=synchronous_call_return"
        )
        runtime.initialize(payload_bytes, entry_bytes)

        if args.kernel_profile:
            runtime.start_batch_put_kernel_profile()
            kernel_profile_started = True
            print("BATCH_PUT_KERNEL_PROFILE_ENABLED units=cycles")
        if args.host_profile:
            runtime.start_batch_put_host_profile()
            host_profile_started = True
            print("BATCH_PUT_HOST_PROFILE_ENABLED units=ns")

        # This call both validates payload and primes the cached input-HBM MR.
        print("BATCH_PUT_PERF_PHASE_BEGIN phase=correctness_preflight")
        runtime.batch_put(keys, source, offsets, lengths)
        committed = True
        validate_payload(runtime, torch, keys, source, row_count, entry_bytes, device)
        runtime.batch_free(keys)
        committed = False
        print(
            "BATCH_PUT_PERF_CORRECTNESS_PASS "
            f"row_count={row_count} entry_bytes={entry_bytes}"
        )
        print("BATCH_PUT_PERF_PHASE_DONE phase=correctness_preflight")

        print("BATCH_PUT_PERF_PHASE_BEGIN phase=warmup")
        for _ in range(args.warmup):
            runtime.batch_put(keys, source, offsets, lengths)
            committed = True
            runtime.batch_free(keys)
            committed = False
        print(f"BATCH_PUT_PERF_PHASE_DONE phase=warmup iterations={args.warmup}")

        print("BATCH_PUT_PERF_PHASE_BEGIN phase=batch_put")
        runtime_samples: list[float] = []
        kernel_execute_cycle_samples: list[int] = []
        kernel_non_execute_cycle_samples: list[int] = []
        host_profile_samples: list[dict[str, object]] = []
        for iteration in range(args.iterations):
            begin = time.perf_counter()
            runtime.batch_put(keys, source, offsets, lengths)
            completed = time.perf_counter()
            committed = True
            runtime_samples.append((completed - begin) * 1000.0)
            if args.kernel_profile:
                kernel_execute_cycles, kernel_non_execute_cycles = kernel_cycle_summary(
                    list(runtime.batch_put_kernel_profile())
                )
                kernel_execute_cycle_samples.append(kernel_execute_cycles)
                kernel_non_execute_cycle_samples.append(kernel_non_execute_cycles)
            if args.host_profile:
                host_profile_samples.append(dict(runtime.batch_put_host_profile()))
            if iteration + 1 < args.iterations:
                runtime.batch_free(keys)
                committed = False

        print_runtime_samples(runtime_samples, args.warmup, row_count, entry_bytes)
        if args.kernel_profile:
            print_kernel_profile_samples(
                kernel_execute_cycle_samples,
                kernel_non_execute_cycle_samples,
                args.warmup,
                row_count,
                entry_bytes,
            )
            runtime.stop_batch_put_kernel_profile()
            kernel_profile_started = False
            print("BATCH_PUT_KERNEL_PROFILE_DISABLED")
        if args.host_profile:
            print_host_profile_samples(
                host_profile_samples, runtime_samples, args.warmup
            )
            runtime.stop_batch_put_host_profile()
            host_profile_started = False
            print("BATCH_PUT_HOST_PROFILE_DISABLED")
        print("BATCH_PUT_PERF_PHASE_DONE phase=batch_put")

        print("BATCH_PUT_PERF_PHASE_BEGIN phase=post_perf_validation")
        validate_payload(runtime, torch, keys, source, row_count, entry_bytes, device)
        runtime.batch_free(keys)
        committed = False
        print("BATCH_PUT_PERF_PHASE_DONE phase=post_perf_validation")
        return 0
    except Exception as error:
        print(
            "BATCH_PUT_PERF_UNHANDLED_ERROR "
            f"type={type(error).__name__} message={error}",
            file=sys.stderr,
        )
        raise
    finally:
        if kernel_profile_started and runtime.initialized:
            try:
                runtime.stop_batch_put_kernel_profile()
            except Exception as error:
                print(
                    "BATCH_PUT_KERNEL_PROFILE_DISABLE_ERROR "
                    f"type={type(error).__name__} message={error}",
                    file=sys.stderr,
                )
        if host_profile_started and runtime.initialized:
            try:
                runtime.stop_batch_put_host_profile()
            except Exception as error:
                print(
                    "BATCH_PUT_HOST_PROFILE_DISABLE_ERROR "
                    f"type={type(error).__name__} message={error}",
                    file=sys.stderr,
                )
        if committed and runtime.initialized:
            runtime.batch_free(keys)
        if runtime.initialized:
            runtime.shutdown()
        print("BATCH_PUT_PERF_PHASE_DONE phase=shutdown")


if __name__ == "__main__":
    raise SystemExit(main())
