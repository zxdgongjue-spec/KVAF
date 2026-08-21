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
ITERATIONS = 1000
ACTIVE_PERCENT = 5
BLOCK_BYTES = 512 + 64
MAX_ACTIVE_REQUESTS_PER_AIV = 16384


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def print_samples(
    operation: str,
    samples: list[float],
    warmup: int,
    units: str,
    payload_bytes_per_call: int = 0,
) -> None:
    result = (
        "PERF_RESULT "
        f"operation={operation} samples={len(samples)} warmup_excluded={warmup} "
        f"mean_ms={statistics.mean(samples):.3f} "
        f"p50_ms={percentile(samples, 0.50):.3f} "
        f"p90_ms={percentile(samples, 0.90):.3f} "
        f"p99_ms={percentile(samples, 0.99):.3f} "
        f"units={units}"
    )
    if payload_bytes_per_call > 0:
        useful_gibps = (
            payload_bytes_per_call
            / (statistics.mean(samples) / 1000.0)
            / (1024.0**3)
        )
        result += (
            f" payload_bytes_per_call={payload_bytes_per_call} "
            f"useful_GiBps={useful_gibps:.3f}"
        )
    print(result)


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
    source_stride = int(
        os.environ.get("KVAF_RUNTIME_PERF_SOURCE_STRIDE", str(block_bytes))
    )
    table_capacity = int(
        os.environ.get(
            "KVAF_RUNTIME_PERF_TABLE_CAPACITY", str(batch * PUT_BATCH_SIZE * 1024)
        )
    )
    requests = int(os.environ.get("KVAF_RUNTIME_PERF_REQUESTS", str(batch * 2048)))
    total_put_batches = table_capacity // PUT_BATCH_SIZE
    full_validation_capacity = npu_core_count * MAX_ACTIVE_REQUESTS_PER_AIV
    if not (
        batch > 0
        and npu_core_count > 0
        and 1 <= jettys_per_core <= 2
        and warmup >= 0
        and iterations > 0
        and 1 <= active_percent <= 100
        and block_bytes > 0
        and source_stride >= block_bytes
        and requests > 0
        and requests <= full_validation_capacity
        and table_capacity >= PUT_BATCH_SIZE
        and table_capacity % PUT_BATCH_SIZE == 0
        and total_put_batches >= warmup + iterations
    ):
        print(
            "FAIL: invalid KVAF_RUNTIME_PERF_* configuration "
            f"(requests={requests}, aiv_cores={npu_core_count}, "
            f"full_validation_capacity={full_validation_capacity})",
            file=sys.stderr,
        )
        return 2
    active_requests = max(1, requests * active_percent // 100)
    ddr_bytes = table_capacity * source_stride

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
    offsets = [row * block_bytes for row in range(PUT_BATCH_SIZE)]
    lengths = [block_bytes] * PUT_BATCH_SIZE

    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    row_pattern = torch.arange(block_bytes, dtype=torch.int64, device=device).to(torch.uint8)
    kv_data = row_pattern.repeat(PUT_BATCH_SIZE, 1).contiguous()
    output_indices = torch.arange(requests, dtype=torch.int32, device=device)
    output = torch.empty((requests, block_bytes), dtype=torch.uint8, device=device)
    runtime = kvaf.NpuRuntime(
        str(kernel), device_id, npu_core_count, str(batch_put_kernel),
        jettys_per_core,
    )
    remote_cpu_service = os.environ.get("KVAF_CPU_SERVICE_ADDR")
    if not remote_cpu_service:
        print("FAIL: KVAF_CPU_SERVICE_ADDR must name the separately running CPU service", file=sys.stderr)
        return 2
    runtime.link_to(remote_cpu_service)
    table = None
    device_table = None
    preflight_table = None
    try:
        print(
            "RUNTIME_PERF_CONFIG "
            f"transport=cpu_service_link batch={batch} table_capacity={table_capacity} "
            f"ddr_bytes={ddr_bytes} put_batch_size={PUT_BATCH_SIZE} "
            f"requests={requests} active_percent={active_percent} "
            f"active_requests={active_requests} block_bytes={block_bytes} source_stride={source_stride} "
            f"aiv_cores={npu_core_count} jettys_per_core={jettys_per_core} "
            f"warmup={warmup} iterations={iterations} "
            "batch_put_backend=aiv_direct_hccp timing=call_to_stream_completion"
        )

        print("RUNTIME_PERF_PHASE_BEGIN phase=initialize")
        begin = time.perf_counter()
        runtime.initialize(table_capacity * block_bytes, source_stride)
        initialize_ms = (time.perf_counter() - begin) * 1000.0
        print(f"PERF_RESULT operation=initialize samples=1 elapsed_ms={initialize_ms:.3f} kind=lifecycle")
        print("RUNTIME_PERF_PHASE_DONE phase=initialize")

        # Verify the complete CPU-DDR data path before accepting steady-state samples.
        print("RUNTIME_PERF_PHASE_BEGIN phase=correctness_preflight")
        preflight_keys = list(range(table_capacity + 1, table_capacity + PUT_BATCH_SIZE + 1))
        runtime.batch_put(preflight_keys, kv_data, offsets, lengths)
        torch.npu.synchronize()
        preflight_table = runtime.prepare_access_table(preflight_keys)
        preflight_device_table = runtime.access_table_view(
            preflight_table, device="npu", capacity=PUT_BATCH_SIZE
        )
        preflight_indices = torch.arange(PUT_BATCH_SIZE, dtype=torch.int32, device=device)
        preflight_mask = torch.ones(PUT_BATCH_SIZE, dtype=torch.uint8, device=device)
        preflight_output = torch.empty(
            (PUT_BATCH_SIZE, block_bytes), dtype=torch.uint8, device=device
        )
        runtime.batch_get(
            preflight_device_table,
            preflight_indices,
            preflight_mask,
            preflight_indices,
            preflight_output,
        )
        torch.npu.synchronize()
        if not torch.equal(preflight_output, kv_data):
            raise AssertionError("Runtime performance preflight BatchPut/BatchGet validation failed")
        runtime.release_access_table(preflight_table)
        preflight_table = None
        runtime.batch_free(preflight_keys)
        print(
            "RUNTIME_PERF_CORRECTNESS_PASS "
            f"phase=preflight requests={PUT_BATCH_SIZE} block_bytes={block_bytes}"
        )
        print("RUNTIME_PERF_PHASE_DONE phase=correctness_preflight")

        print("RUNTIME_PERF_PHASE_BEGIN phase=batch_put")
        for key_batch in key_batches[:warmup]:
            runtime.batch_put(key_batch, kv_data, offsets, lengths)
        torch.npu.synchronize()
        batch_put_samples: list[float] = []
        for key_batch in key_batches[warmup : warmup + iterations]:
            begin = time.perf_counter()
            runtime.batch_put(key_batch, kv_data, offsets, lengths)
            torch.npu.synchronize()
            batch_put_samples.append((time.perf_counter() - begin) * 1000.0)
        print_samples(
            "batch_put", batch_put_samples, warmup,
            f"{PUT_BATCH_SIZE}_kv_blocks_per_call",
            PUT_BATCH_SIZE * block_bytes,
        )

        # Finish the one-time data population. It is deliberately excluded
        # from BatchPut samples, but is necessary for a full AccessTable and
        # valid random BatchGet indices.
        for key_batch in key_batches[warmup + iterations :]:
            runtime.batch_put(key_batch, kv_data, offsets, lengths)
        torch.npu.synchronize()
        print("RUNTIME_PERF_PHASE_DONE phase=batch_put")

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

        validation_requests = min(requests, table_capacity)
        validation_indices = torch.arange(validation_requests, dtype=torch.int32, device=device)
        validation_mask = torch.ones(validation_requests, dtype=torch.uint8, device=device)
        validation_output = torch.empty(
            (validation_requests, block_bytes), dtype=torch.uint8, device=device
        )
        runtime.batch_get(
            device_table,
            validation_indices,
            validation_mask,
            validation_indices,
            validation_output,
        )
        torch.npu.synchronize()
        if not torch.equal(validation_output, row_pattern.expand(validation_requests, -1)):
            raise AssertionError("Runtime performance BatchGet validation failed")
        print(
            "RUNTIME_PERF_CORRECTNESS_PASS "
            f"phase=full_table requests={validation_requests} block_bytes={block_bytes}"
        )

        # Pre-create random accesses with an exact active count. This setup is
        # excluded from BatchGet samples; every sample ends at stream completion.
        indices_batches = [
            torch.randint(0, table_capacity, (requests,), dtype=torch.int32, device=device)
            for _ in range(warmup + iterations)
        ]
        mask_batches = []
        for _ in range(warmup + iterations):
            mask = torch.zeros(requests, dtype=torch.uint8, device=device)
            if active_requests == requests:
                mask.fill_(1)
            else:
                mask[torch.randperm(requests, device=device)[:active_requests]] = 1
            mask_batches.append(mask)
        torch.npu.synchronize()

        print("RUNTIME_PERF_PHASE_BEGIN phase=batch_get")
        for index, mask in zip(indices_batches[:warmup], mask_batches[:warmup]):
            runtime.batch_get(device_table, index, mask, output_indices, output)
        torch.npu.synchronize()
        batch_get_samples: list[float] = []
        for index, mask in zip(
            indices_batches[warmup:], mask_batches[warmup:]
        ):
            begin = time.perf_counter()
            runtime.batch_get(device_table, index, mask, output_indices, output)
            torch.npu.synchronize()
            batch_get_samples.append((time.perf_counter() - begin) * 1000.0)
        print_samples(
            "batch_get", batch_get_samples, warmup,
            f"{active_requests}_active_of_{requests}_requests_per_call",
            active_requests * block_bytes,
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
        if preflight_table is not None and runtime.initialized:
            runtime.release_access_table(preflight_table)
        if table is not None and runtime.initialized:
            runtime.release_access_table(table)
        if runtime.initialized:
            begin = time.perf_counter()
            runtime.shutdown()
            shutdown_ms = (time.perf_counter() - begin) * 1000.0
            print(f"PERF_RESULT operation=shutdown samples=1 elapsed_ms={shutdown_ms:.3f} kind=lifecycle")
        print("RUNTIME_PERF_PHASE_DONE phase=shutdown")


if __name__ == "__main__":
    raise SystemExit(main())
