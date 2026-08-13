#!/usr/bin/env python3
"""NPU Runtime steady-state performance benchmark.

This is intentionally a direct Python/pybind benchmark, not a CTest case.
It measures Runtime calls only: caller inputs are created before the timed
loops, and NPU calls end each sample with an external stream synchronize.
"""

from __future__ import annotations

import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kvaf

from backend import detect_torch_backend


AIV_CORES = 32
JETTYS_PER_CORE = 2
PUT_BATCH_SIZE = 128
WARMUP = 3
ITERATIONS = 100
ACTIVE_PERCENT = 5
BLOCK_BYTES = 656
WQE_BYTES = 656
ROWS_PER_ENTRY = 128
ENTRY_BYTES = ROWS_PER_ENTRY * WQE_BYTES


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def print_samples(operation: str, samples: list[float], warmup: int, units: str) -> None:
    print(
        "PERF_RESULT "
        f"operation={operation} samples={len(samples)} warmup_excluded={warmup} "
        f"mean_ms={statistics.mean(samples):.3f} "
        f"p50_ms={percentile(samples, 0.50):.3f} "
        f"p90_ms={percentile(samples, 0.90):.3f} "
        f"p99_ms={percentile(samples, 0.99):.3f} "
        f"units={units}"
    )


def main(batch: int = 16) -> int:
    backend = detect_torch_backend()
    if backend.name != "npu":
        print(f"SKIP: NPU benchmark requires torch_npu; detected {backend.name}.")
        return 0
    if not hasattr(kvaf, "NpuRuntime"):
        print("SKIP: rebuild KVAF with KVAF_ENABLE_AIV_BATCH_GET=ON and KVAF_BUILD_PYTHON=ON.")
        return 0
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the A5/CANN target")
        return 0

    torch = backend.torch
    assert torch is not None
    device_id = int(os.environ.get("DEVICE_ID", "0"))
    batch = int(os.environ.get("KVAF_RUNTIME_PERF_BATCH", str(batch)))
    npu_core_count = int(os.environ.get("KVAF_RUNTIME_PERF_AIV_CORES", str(AIV_CORES)))
    jettys_per_core = int(
        os.environ.get(
            "KVAF_RUNTIME_PERF_JETTYS_PER_CORE",
            os.environ.get("KVAF_NPU_JETTYS_PER_CORE", str(JETTYS_PER_CORE)),
        )
    )
    warmup = int(os.environ.get("KVAF_RUNTIME_PERF_WARMUP", str(WARMUP)))
    iterations = int(os.environ.get("KVAF_RUNTIME_PERF_ITERATIONS", str(ITERATIONS)))
    active_percent = int(
        os.environ.get("KVAF_RUNTIME_PERF_ACTIVE_PERCENT", str(ACTIVE_PERCENT))
    )
    block_bytes = int(os.environ.get("KVAF_RUNTIME_PERF_BLOCK_BYTES", str(BLOCK_BYTES)))
    table_capacity = batch * PUT_BATCH_SIZE * 8
    requests = batch * 2048
    source_stride = ENTRY_BYTES
    total_put_batches = table_capacity // PUT_BATCH_SIZE
    if not (
        batch > 0
        and npu_core_count > 0
        and 1 <= jettys_per_core <= 2
        and warmup >= 0
        and iterations > 0
        and 1 <= active_percent <= 100
        and block_bytes == WQE_BYTES
        and total_put_batches >= warmup + iterations
    ):
        print("FAIL: invalid KVAF_RUNTIME_PERF_* configuration", file=sys.stderr)
        return 2

    kernel = Path(
        os.environ.get(
            "KVAF_BATCH_GET_KERNEL",
            os.environ.get("KVAF_AIV_BATCH_GET_KERNEL", ""),
        )
    )
    if not kernel.is_file():
        print("FAIL: KVAF_BATCH_GET_KERNEL must name the built kernel object", file=sys.stderr)
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

    # All BatchPut and BatchFree inputs are created before their measured loops.
    # Key batches are distinct, so Runtime never sees a duplicate Put or Free.
    all_keys = list(range(1, table_capacity + 1))
    prepare_keys = torch.arange(1, table_capacity + 1, dtype=torch.int64)
    key_batches = [
        all_keys[batch_index * PUT_BATCH_SIZE : (batch_index + 1) * PUT_BATCH_SIZE]
        for batch_index in range(total_put_batches)
    ]
    put_offsets = [row * ENTRY_BYTES for row in range(PUT_BATCH_SIZE)]
    lengths = [ENTRY_BYTES] * PUT_BATCH_SIZE

    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    kv_data = torch.empty((PUT_BATCH_SIZE, ENTRY_BYTES), dtype=torch.uint8, device=device)
    offsets = torch.randint(
        0, ROWS_PER_ENTRY, (requests,), dtype=torch.int32, device=device
    ) * WQE_BYTES
    output_indices = torch.arange(requests, dtype=torch.int32, device=device)
    output = torch.empty((requests, block_bytes), dtype=torch.uint8, device=device)
    runtime = kvaf.NpuRuntime(
        str(kernel), device_id, npu_core_count, str(batch_put_kernel),
        jettys_per_core,
    )
    table = None
    device_table = None
    try:
        print(
            "RUNTIME_PERF_CONFIG "
            f"table_capacity={table_capacity} put_batch_size={PUT_BATCH_SIZE} "
            f"requests={requests} block_bytes={block_bytes} aiv_cores={npu_core_count} "
            f"jettys_per_core={jettys_per_core} "
            f"active_percent={active_percent} warmup={warmup} iterations={iterations} "
            "batch_put_backend=aiv_direct_hccp"
        )

        # Initialize and Shutdown run once: they are lifecycle operations, not
        # a steady-state dataplane metric.
        print("RUNTIME_PERF_PHASE_BEGIN phase=initialize")
        runtime.initialize(table_capacity * ENTRY_BYTES, source_stride)
        print("RUNTIME_PERF_PHASE_DONE phase=initialize")

        print("RUNTIME_PERF_PHASE_BEGIN phase=batch_put")
        for key_batch in key_batches[:warmup]:
            runtime.batch_put(key_batch, kv_data, put_offsets, lengths)
        torch.npu.synchronize()
        batch_put_samples: list[float] = []
        for key_batch in key_batches[warmup : warmup + iterations]:
            begin = time.perf_counter()
            runtime.batch_put(key_batch, kv_data, put_offsets, lengths)
            torch.npu.synchronize()
            batch_put_samples.append((time.perf_counter() - begin) * 1000.0)
        print_samples(
            "batch_put", batch_put_samples, warmup,
            f"{PUT_BATCH_SIZE}_kv_blocks_per_call"
        )

        # Finish the one-time data population. It is deliberately excluded
        # from BatchPut samples, but is necessary for a full AccessTable and
        # valid random BatchGet indices.
        for key_batch in key_batches[warmup + iterations :]:
            runtime.batch_put(key_batch, kv_data, put_offsets, lengths)
        torch.npu.synchronize()
        print("RUNTIME_PERF_PHASE_DONE phase=batch_put")

        # PrepareAccessTable is CPU-only. Measure the requested full table once.
        print("RUNTIME_PERF_PHASE_BEGIN phase=prepare_access_table")
        begin = time.perf_counter()
        table = runtime.prepare_access_table(prepare_keys)
        prepare_ms = (time.perf_counter() - begin) * 1000.0
        print(
            "PERF_RESULT "
            "operation=prepare_access_table samples=1 "
            f"keys={table_capacity} elapsed_ms={prepare_ms:.3f}"
        )
        print("RUNTIME_PERF_PHASE_DONE phase=prepare_access_table")

        # Materialize the full device table once. This is the caller-visible
        # Runtime H2D path; it is intentionally separate from BatchGet timing.
        print("RUNTIME_PERF_PHASE_BEGIN phase=access_table_view_npu")
        torch.npu.synchronize()
        begin = time.perf_counter()
        device_table = runtime.access_table_view(
            table, device="npu", capacity=table_capacity
        )
        torch.npu.synchronize()
        access_view_ms = (time.perf_counter() - begin) * 1000.0
        print(
            "PERF_RESULT "
            "operation=access_table_view_npu samples=1 "
            f"entries={table_capacity} elapsed_ms={access_view_ms:.3f}"
        )
        print("RUNTIME_PERF_PHASE_DONE phase=access_table_view_npu")

        # Pre-create different random request tensors. This setup is excluded
        # from BatchGet samples; each sample measures Runtime enqueue through
        # completion of its caller-owned current stream.
        probability = active_percent / 100.0
        indices_batches = [
            torch.randint(0, table_capacity, (requests,), dtype=torch.int32, device=device)
            for _ in range(warmup + iterations)
        ]
        mask_batches = [
            (torch.rand(requests, device=device) < probability).to(torch.uint8)
            for _ in range(warmup + iterations)
        ]
        torch.npu.synchronize()

        print("RUNTIME_PERF_PHASE_BEGIN phase=batch_get")
        for index, mask in zip(indices_batches[:warmup], mask_batches[:warmup]):
            runtime.batch_get(device_table, index, offsets, mask, output_indices, output)
        torch.npu.synchronize()
        batch_get_samples: list[float] = []
        for index, mask in zip(
            indices_batches[warmup:], mask_batches[warmup:]
        ):
            begin = time.perf_counter()
            runtime.batch_get(device_table, index, offsets, mask, output_indices, output)
            torch.npu.synchronize()
            batch_get_samples.append((time.perf_counter() - begin) * 1000.0)
        print_samples(
            "batch_get", batch_get_samples, warmup,
            f"{requests}_requests_per_call"
        )
        print("RUNTIME_PERF_PHASE_DONE phase=batch_get")

        # The full-table lock must be released before BatchFree can reclaim its
        # entries. DeviceAccessTable is no longer used after BatchGet.
        print("RUNTIME_PERF_PHASE_BEGIN phase=release_access_table")
        begin = time.perf_counter()
        runtime.release_access_table(table)
        release_ms = (time.perf_counter() - begin) * 1000.0
        table = None
        device_table = None
        print(
            "PERF_RESULT "
            "operation=release_access_table samples=1 "
            f"keys={table_capacity} elapsed_ms={release_ms:.3f}"
        )
        print("RUNTIME_PERF_PHASE_DONE phase=release_access_table")

        # Free different 128-key batches only after BatchGet, so it cannot
        # invalidate the AccessTable used by the dataplane benchmark.
        print("RUNTIME_PERF_PHASE_BEGIN phase=batch_free")
        free_batches = key_batches[-(warmup + iterations) :]
        for key_batch in free_batches[:warmup]:
            runtime.batch_free(key_batch)
        batch_free_samples: list[float] = []
        for key_batch in free_batches[warmup:]:
            begin = time.perf_counter()
            runtime.batch_free(key_batch)
            batch_free_samples.append((time.perf_counter() - begin) * 1000.0)
        print_samples(
            "batch_free", batch_free_samples, warmup,
            f"{PUT_BATCH_SIZE}_keys_per_call"
        )
        print("RUNTIME_PERF_PHASE_DONE phase=batch_free")
        return 0
    finally:
        print("RUNTIME_PERF_PHASE_BEGIN phase=shutdown")
        if table is not None and runtime.initialized:
            runtime.release_access_table(table)
        if runtime.initialized:
            runtime.shutdown()
        print("RUNTIME_PERF_PHASE_DONE phase=shutdown")


if __name__ == "__main__":
    raise SystemExit(main())
