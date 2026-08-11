#!/usr/bin/env python3
"""Direct pybind + torch_npu integration test for Runtime BatchPut."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kvaf

from backend import detect_torch_backend


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_batch(
    runtime: object,
    keys: list[int],
    expected: bytes,
    rows: int,
    block_bytes: int,
) -> None:
    table = runtime.prepare_access_table(keys)
    try:
        view = runtime.access_table_view(table)
        addresses = list(view["addresses"])
        if list(view["lengths"]) != [block_bytes] * rows:
            raise AssertionError("committed BatchPut lengths do not match")
        if list(view["valid"]) != [1] * rows:
            raise AssertionError("committed BatchPut entries are not valid")
        for row, address in enumerate(addresses):
            begin = row * block_bytes
            actual = ctypes.string_at(address, block_bytes)
            if actual != expected[begin : begin + block_bytes]:
                raise AssertionError(f"BatchPut payload mismatch row={row}")
    finally:
        runtime.release_access_table(table)


def main() -> int:
    backend = detect_torch_backend()
    if backend.name != "npu":
        print(
            "SKIP: NPU integration requires torch_npu and an available NPU; "
            f"detected {backend.name} ({backend.reason})."
        )
        return 0
    if not hasattr(kvaf, "NpuRuntime"):
        print(
            "SKIP: kvaf was not built with -DKVAF_ENABLE_AIV_BATCH_GET=ON, "
            "-DKVAF_ENABLE_AIV_BATCH_PUT=ON, and -DKVAF_BUILD_PYTHON=ON."
        )
        return 0
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the CANN target")
        return 0

    batch_put_kernel = Path(os.environ.get("KVAF_BATCH_PUT_KERNEL", ""))
    if not batch_put_kernel.is_file():
        print(
            "FAIL: KVAF_BATCH_PUT_KERNEL must name the built BatchPut kernel object",
            file=sys.stderr,
        )
        return 2
    batch_get_kernel = Path(
        os.environ.get(
            "KVAF_BATCH_GET_KERNEL",
            os.environ.get(
                "KVAF_AIV_BATCH_GET_KERNEL",
                str(batch_put_kernel.with_name("kvaf_execute_batch_get_kernels.o")),
            ),
        )
    )
    if not batch_get_kernel.is_file():
        print(
            "FAIL: KVAF_BATCH_GET_KERNEL must name the built BatchGet kernel object",
            file=sys.stderr,
        )
        return 2

    device_id = int(os.environ.get("DEVICE_ID", "0"))
    rows = int(os.environ.get("KVAF_NPU_REQUESTS", "4096"))
    block_bytes = int(os.environ.get("KVAF_NPU_BLOCK_BYTES", "1024"))
    npu_core_count = int(
        os.environ.get("KVAF_NPU_CORE_COUNT", os.environ.get("KVAF_NPU_AIV_CORES", "32"))
    )
    jettys_per_core = int(os.environ.get("KVAF_NPU_JETTYS_PER_CORE", "2"))
    warmup = int(os.environ.get("KVAF_NPU_WARMUP", "1"))
    iterations = int(os.environ.get("KVAF_NPU_ITERATIONS", "5"))
    if not (
        rows > 0
        and block_bytes > 0
        and 0 < npu_core_count <= rows
        and 1 <= jettys_per_core <= 2
        and warmup >= 0
        and iterations > 0
    ):
        print("FAIL: invalid KVAF_NPU_* configuration", file=sys.stderr)
        return 2

    torch = backend.torch
    assert torch is not None
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    source_rows = torch.arange(rows, dtype=torch.int64).unsqueeze(1)
    source_bytes = torch.arange(block_bytes, dtype=torch.int64).unsqueeze(0)
    source_host = (
        (source_rows * 131 + source_bytes * 17 + (source_bytes >> 3)) & 0xFF
    ).to(torch.uint8).contiguous()
    expected = source_host.numpy().tobytes()
    source = source_host.to(device)
    offsets = [row * block_bytes for row in range(rows)]
    lengths = [block_bytes] * rows
    keys = list(range(1, rows + 1))
    torch.npu.synchronize()

    runtime = kvaf.NpuRuntime(
        str(batch_get_kernel),
        device_id,
        npu_core_count,
        str(batch_put_kernel),
        jettys_per_core,
    )
    committed = False
    try:
        print(
            "RUNTIME_NPU_BATCH_PUT_CONFIG "
            f"rows={rows} block_bytes={block_bytes} npu_cores={npu_core_count} "
            f"jettys_per_core={jettys_per_core} warmup={warmup} "
            f"iterations={iterations} backend=aiv_direct_hccp"
        )
        runtime.initialize(rows * block_bytes, block_bytes)

        def execute() -> None:
            nonlocal committed
            runtime.batch_put(keys, source, offsets, lengths)
            committed = True

        def release() -> None:
            nonlocal committed
            runtime.batch_free(keys)
            committed = False

        print("RUNTIME_NPU_PHASE_BEGIN phase=runtime_batch_put")
        execute()
        torch.npu.synchronize()
        validate_batch(runtime, keys, expected, rows, block_bytes)
        print(f"RUNTIME_NPU_BATCH_PUT_VALIDATION_PASS rows={rows}")
        release()

        for _ in range(warmup):
            execute()
            release()
        torch.npu.synchronize()

        samples = []
        for iteration in range(iterations):
            begin = time.perf_counter()
            execute()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - begin) * 1000.0)
            if iteration + 1 < iterations:
                release()
        print("RUNTIME_NPU_PHASE_DONE phase=runtime_batch_put")

        print("RUNTIME_NPU_PHASE_BEGIN phase=post_perf_validation")
        validate_batch(runtime, keys, expected, rows, block_bytes)
        release()
        print("RUNTIME_NPU_PHASE_DONE phase=post_perf_validation")

        mean_ms = statistics.mean(samples)
        useful_gbps = rows * block_bytes / 1_000_000_000.0 / (mean_ms / 1000.0)
        print(
            "PERF_RESULT metric=runtime_batch_put_call_to_stream_completion "
            f"samples={iterations} warmup_excluded={warmup} rows={rows} "
            f"block_bytes={block_bytes} npu_cores={npu_core_count} "
            f"mean_ms={mean_ms:.3f} "
            f"p50_ms={percentile(samples, 0.50):.3f} "
            f"p90_ms={percentile(samples, 0.90):.3f} "
            f"p99_ms={percentile(samples, 0.99):.3f} "
            f"useful_GBps={useful_gbps:.3f}"
        )
        return 0
    except Exception as error:
        print(
            "RUNTIME_NPU_BATCH_PUT_UNHANDLED_ERROR "
            f"type={type(error).__name__} message={error}",
            file=sys.stderr,
        )
        raise
    finally:
        if runtime.initialized:
            if committed:
                runtime.batch_free(keys)
            runtime.shutdown()
        print("RUNTIME_NPU_PHASE_DONE phase=teardown")


if __name__ == "__main__":
    raise SystemExit(main())
