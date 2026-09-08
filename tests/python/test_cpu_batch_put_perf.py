#!/usr/bin/env python3
"""Measure CPU Runtime BatchPut from packed CPU DRAM into remote NPU HBM.

The benchmark uses the same compact-prefix shape as the latest AIV BatchGet
benchmark: ``request_length`` is the fixed descriptor/buffer capacity and
``miss_count`` is the number of active rows. Full-capacity inputs are passed to
BatchPut, which consumes only the active prefix.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Sequence

DEFAULT_REQUEST_LENGTH = 8192
DEFAULT_TOKEN_BYTES = 656
DEFAULT_WARMUP = 5
DEFAULT_ITERATIONS = 100
MAX_INT32 = (1 << 31) - 1


def environment_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure CPU DRAM to NPU HBM Runtime BatchPut performance."
    )
    parser.add_argument(
        "--request-length",
        type=int,
        default=environment_int(
            "KVAF_CPU_BATCH_PUT_REQUEST_LENGTH", DEFAULT_REQUEST_LENGTH
        ),
        help="fixed request capacity used to size the remote HBM region",
    )
    parser.add_argument(
        "--miss-count",
        type=int,
        default=(
            int(os.environ["KVAF_CPU_BATCH_PUT_MISS_COUNT"])
            if "KVAF_CPU_BATCH_PUT_MISS_COUNT" in os.environ
            else None
        ),
        help="active compact-prefix rows; defaults to request-length",
    )
    parser.add_argument(
        "--token-bytes",
        type=int,
        default=environment_int(
            "KVAF_CPU_BATCH_PUT_TOKEN_BYTES", DEFAULT_TOKEN_BYTES
        ),
        help="bytes transferred by each active row",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=environment_int("KVAF_CPU_BATCH_PUT_WARMUP", DEFAULT_WARMUP),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=environment_int(
            "KVAF_CPU_BATCH_PUT_ITERATIONS", DEFAULT_ITERATIONS
        ),
    )
    parser.add_argument(
        "--jettys",
        type=int,
        default=environment_int("KVAF_CPU_URMA_JETTYS", 1),
    )
    parser.add_argument(
        "--omp-threads",
        type=int,
        default=environment_int("KVAF_CPU_URMA_OMP_THREADS", 1),
    )
    parser.add_argument(
        "--doorbell-wqes",
        type=int,
        default=environment_int("KVAF_CPU_URMA_DOORBELL_WQES", 0),
    )
    parser.add_argument(
        "--device-id", type=int, default=environment_int("DEVICE_ID", 0)
    )
    parser.add_argument(
        "--peer",
        default=(
            os.environ.get("KVAF_NPU_SERVICE_ADDR")
            or os.environ.get("KVAF_CPU_SERVICE_ADDR")
            or "127.0.0.1:28999"
        ),
    )
    parser.add_argument("--ub-device", default=os.environ.get("KVAF_UB_DEVICE", ""))
    parser.add_argument(
        "--eid-index",
        type=int,
        default=(int(os.environ["EID_INDEX"]) if "EID_INDEX" in os.environ else None),
    )
    args = parser.parse_args(argv)
    if args.miss_count is None:
        args.miss_count = args.request_length
    return args


def validate_args(args: argparse.Namespace) -> str | None:
    if args.request_length <= 0 or args.request_length > MAX_INT32:
        return "request-length must be in [1, INT32_MAX]"
    if args.miss_count <= 0 or args.miss_count > args.request_length:
        return "miss-count must be in [1, request-length] for transport measurements"
    if args.token_bytes <= 0 or args.token_bytes > MAX_INT32:
        return "token-bytes must be in [1, INT32_MAX]"
    if args.warmup < 0 or args.iterations <= 0:
        return "warmup must be non-negative and iterations must be positive"
    if args.jettys <= 0 or args.omp_threads <= 0:
        return "jettys and omp-threads must be positive"
    if args.doorbell_wqes < 0 or args.doorbell_wqes > 1024:
        return "doorbell-wqes must be in [0, 1024]"
    return None


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def print_samples(
    operation: str,
    samples_ns: list[int],
    warmup: int,
    request_length: int,
    miss_count: int,
    token_bytes: int,
) -> None:
    if not samples_ns or any(value < 0 for value in samples_ns):
        raise ValueError("timing samples must be non-empty and non-negative")
    if operation in {"batch_put", "batch_put_transport_total"} and any(
        value == 0 for value in samples_ns
    ):
        raise ValueError("call/transport timing samples must be positive")
    mean_ns = statistics.mean(samples_ns)
    p50_ns = percentile(samples_ns, 0.50)
    p90_ns = percentile(samples_ns, 0.90)
    p99_ns = percentile(samples_ns, 0.99)
    result = (
        "PERF_RESULT "
        f"operation={operation} samples={len(samples_ns)} "
        f"warmup_excluded={warmup} request_length={request_length} "
        f"miss_count={miss_count} token_bytes={token_bytes} "
        f"mean_ns={mean_ns:.0f} p50_ns={p50_ns:.0f} "
        f"p90_ns={p90_ns:.0f} p99_ns={p99_ns:.0f} "
        f"mean_us={mean_ns / 1000.0:.3f} p50_us={p50_ns / 1000.0:.3f} "
        f"p90_us={p90_ns / 1000.0:.3f} p99_us={p99_ns / 1000.0:.3f}"
    )
    if operation in {"batch_put", "batch_put_transport_total"}:
        payload_bytes = miss_count * token_bytes
        useful_gbps = payload_bytes / mean_ns if mean_ns > 0 else 0.0
        # Compute before formatting latency: rounded microseconds can distort
        # bandwidth for small transfers. This is throughput at median latency,
        # not the median of per-sample reciprocals.
        p50_gbps = payload_bytes / p50_ns
        result += (
            f" payload_bytes_per_call={payload_bytes} "
            f"useful_GBps={useful_gbps:.3f} p50_GBps={p50_gbps:.3f}"
        )
    print(result)


def validate_timing(timing: dict, call_ns: int) -> dict[str, int]:
    if timing.get("total_scope") != "transport_call":
        raise ValueError(
            "transport wall-clock timing is unavailable; rebuild the CPU Python binding"
        )
    names = ("prepare_ns", "post_ns", "poll_ns", "total_ns")
    for name in names:
        value = timing.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"invalid transport timing {name}={value!r}")
    total = timing["total_ns"]
    if total <= 0 or total > call_ns:
        raise ValueError("transport total must be positive and within the Runtime call")
    if sum(timing[name] for name in names[:3]) > total:
        raise ValueError("transport stages exceed the measured transport total")
    return {name: timing[name] for name in names}


def make_source(total_bytes: int, token_bytes: int) -> bytes:
    pattern = bytes((index * 17 + 29) & 0xFF for index in range(256))
    source = bytearray((pattern * ((total_bytes + len(pattern) - 1) // len(pattern)))[
        :total_bytes
    ])
    # Repeated 256-byte patterns hide row/addressing swaps for common token
    # sizes. Tag each row so the readback preflight can detect those errors.
    tag_bytes = min(8, token_bytes)
    for row, offset in enumerate(range(0, total_bytes, token_bytes)):
        source[offset : offset + tag_bytes] = row.to_bytes(8, "little")[:tag_bytes]
    return bytes(source)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    error = validate_args(args)
    if error is not None:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    import kvaf

    if not hasattr(kvaf, "CpuRuntime"):
        print("SKIP: CPU Runtime Python binding was not built")
        return 0

    slot_size = args.token_bytes
    hbm_bytes = args.request_length * slot_size
    payload_bytes = args.miss_count * args.token_bytes
    keys = list(range(1, args.request_length + 1))
    offsets = [row * args.token_bytes for row in range(args.request_length)]
    lengths = [args.token_bytes] * args.request_length
    source = make_source(args.request_length * args.token_bytes, args.token_bytes)
    active_keys = keys[: args.miss_count]

    runtime = kvaf.CpuRuntime()
    print(
        "CPU_BATCH_PUT_CONFIG "
        f"peer={args.peer} device_id={args.device_id} "
        f"request_length={args.request_length} miss_count={args.miss_count} "
        f"token_bytes={args.token_bytes} payload_bytes={payload_bytes} "
        f"jettys={args.jettys} omp_threads={args.omp_threads} "
        f"doorbell_wqes={args.doorbell_wqes} warmup={args.warmup} "
        f"iterations={args.iterations}"
    )
    print("CPU_BATCH_PUT_TIMING_SCOPE total=transport_call bandwidth=payload_over_latency bytes_unit=decimal_GB")
    try:
        runtime.initialize(
            args.peer,
            hbm_bytes,
            slot_size,
            device_id=args.device_id,
            ub_device=args.ub_device,
            eid_index=args.eid_index,
            jetty_count=args.jettys,
            omp_thread_count=args.omp_threads,
            wqes_per_doorbell=args.doorbell_wqes,
        )

        def execute() -> None:
            runtime.batch_put(
                [keys],
                [source],
                [offsets],
                [lengths],
                miss_count=args.miss_count,
            )

        def free_active() -> None:
            if active_keys:
                runtime.batch_free(active_keys)

        print("CPU_BATCH_PUT_PERF_PHASE_BEGIN phase=correctness_preflight")
        execute()
        if active_keys:
            table = runtime.prepare_access_table(active_keys)
            output = bytearray(payload_bytes)
            try:
                rows = list(range(args.miss_count))
                runtime.batch_get(
                    table,
                    rows,
                    [0] * args.miss_count,
                    rows,
                    output,
                    args.token_bytes,
                )
                if bytes(output) != source[:payload_bytes]:
                    raise AssertionError("CPU BatchPut payload validation failed")
            finally:
                runtime.release_access_table(table)
        free_active()
        print(
            "CPU_BATCH_PUT_CORRECTNESS_PASS "
            f"miss_count={args.miss_count} token_bytes={args.token_bytes}"
        )
        print("CPU_BATCH_PUT_PERF_PHASE_DONE phase=correctness_preflight")

        print("CPU_BATCH_PUT_PERF_PHASE_BEGIN phase=warmup")
        for _ in range(args.warmup):
            execute()
            free_active()
        print(
            "CPU_BATCH_PUT_PERF_PHASE_DONE "
            f"phase=warmup iterations={args.warmup}"
        )

        call_samples: list[int] = []
        prepare_samples: list[int] = []
        post_samples: list[int] = []
        poll_samples: list[int] = []
        total_samples: list[int] = []
        print("CPU_BATCH_PUT_PERF_PHASE_BEGIN phase=batch_put")
        for _ in range(args.iterations):
            begin_ns = time.perf_counter_ns()
            execute()
            completed_ns = time.perf_counter_ns()
            call_ns = completed_ns - begin_ns
            timing = validate_timing(dict(runtime.last_batch_timing), call_ns)
            call_samples.append(call_ns)
            prepare_samples.append(timing["prepare_ns"])
            post_samples.append(timing["post_ns"])
            poll_samples.append(timing["poll_ns"])
            total_samples.append(timing["total_ns"])
            free_active()

        for operation, samples in (
            ("batch_put", call_samples),
            ("batch_put_transport_total", total_samples),
            ("batch_put_transport_prepare", prepare_samples),
            ("batch_put_transport_post", post_samples),
            ("batch_put_transport_completion_wait", poll_samples),
        ):
            print_samples(
                operation,
                samples,
                args.warmup,
                args.request_length,
                args.miss_count,
                args.token_bytes,
            )
        print("CPU_BATCH_PUT_PERF_PHASE_DONE phase=batch_put")
        return 0
    except Exception as caught:
        print(
            "CPU_BATCH_PUT_PERF_ERROR "
            f"type={type(caught).__name__} message={caught}",
            file=sys.stderr,
        )
        raise
    finally:
        if runtime.initialized:
            runtime.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
