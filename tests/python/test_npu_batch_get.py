#!/usr/bin/env python3
"""Direct pybind + torch_npu integration test for NPU Runtime BatchGet.

The test owns every caller buffer. Output, status and queue tensors are passed
to each ``batch_get`` call; no buffer is bound during Runtime initialization.
No native runner process is launched.
"""

from __future__ import annotations

import os
from pathlib import Path
import statistics
import struct
import sys
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kvaf

from backend import detect_torch_backend


def pattern(slot: int, block_bytes: int) -> bytes:
    return bytes((slot * 131 + byte * 17 + (byte >> 3)) & 0xFF for byte in range(block_bytes))


def tensor_bytes(tensor: object) -> int:
    return int(tensor.numel() * tensor.element_size())


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_output(
    output: bytes,
    indices: Iterable[int],
    mask: Iterable[int],
    output_indices: Iterable[int],
    block_bytes: int,
) -> None:
    for request, (table_index, enabled, row) in enumerate(
        zip(indices, mask, output_indices)
    ):
        offset = row * block_bytes
        actual = output[offset : offset + block_bytes]
        expected = bytes(block_bytes) if not enabled else pattern(table_index, block_bytes)
        if actual != expected:
            raise AssertionError(f"output mismatch for request={request}, row={row}")


def validate_completion(
    status: bytes,
    queue: bytes,
    requests: int,
    mask: list[int],
    npu_core_count: int,
    expected_batches: int,
) -> None:
    for core in range(npu_core_count):
        begin = requests * core // npu_core_count
        end = requests * (core + 1) // npu_core_count
        active = sum(mask[begin:end])
        status_offset = core * kvaf.NPU_BATCH_GET_STATUS_BYTES
        magic, processed, result, completed = struct.unpack_from(
            "<QQQQ", status, status_offset
        )
        queue_offset = core * kvaf.NPU_BATCH_GET_QUEUE_STATE_BYTES
        queue_magic, initialized, _, _, queue_completed, queue_error = struct.unpack_from(
            "<IIQQQQ", queue, queue_offset
        )
        if (
            magic != kvaf.NPU_BATCH_GET_STATUS_MAGIC
            or result != 0
            or processed != active
            or completed != (0 if active == 0 else expected_batches)
            or queue_magic != 0x4B564151
            or initialized != 1
            or queue_error != 0
            or queue_completed != (0 if active == 0 else expected_batches)
        ):
            raise AssertionError(
                "completion state mismatch "
                f"core={core} result={result} queue_error={queue_error}"
            )


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
            "SKIP: kvaf was not built with -DKVAF_ENABLE_AIV_BATCH_GET=ON "
            "and -DKVAF_BUILD_PYTHON=ON."
        )
        return 0
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the A5/CANN target")
        return 0

    torch = backend.torch
    assert torch is not None
    kernel = Path(
        os.environ.get(
            "KVAF_BATCH_GET_KERNEL",
            os.environ.get("KVAF_AIV_BATCH_GET_KERNEL", ""),
        )
    )
    if not kernel.is_file():
        print("FAIL: KVAF_BATCH_GET_KERNEL must name the built kernel object", file=sys.stderr)
        return 2

    device_id = int(os.environ.get("DEVICE_ID", "0"))
    table_capacity = int(os.environ.get("KVAF_NPU_TABLE_CAPACITY", "8"))
    requests = int(os.environ.get("KVAF_NPU_REQUESTS", "8"))
    output_rows = int(os.environ.get("KVAF_NPU_OUTPUT_ROWS", "8"))
    block_bytes = int(os.environ.get("KVAF_NPU_BLOCK_BYTES", "1024"))
    source_stride = int(os.environ.get("KVAF_NPU_SOURCE_STRIDE", "4096"))
    npu_core_count = int(
        os.environ.get("KVAF_NPU_CORE_COUNT", os.environ.get("KVAF_NPU_AIV_CORES", "1"))
    )
    mask_every = int(os.environ.get("KVAF_NPU_MASK_EVERY", "0"))
    warmup = int(os.environ.get("KVAF_NPU_WARMUP", "1"))
    iterations = int(os.environ.get("KVAF_NPU_ITERATIONS", "5"))
    if not (
        table_capacity > 0
        and requests > 0
        and requests <= output_rows
        and block_bytes > 0
        and source_stride >= block_bytes
        and npu_core_count > 0
        and npu_core_count <= requests
        and iterations > 0
    ):
        print("FAIL: invalid KVAF_NPU_* shape configuration", file=sys.stderr)
        return 2

    # torch_npu owns ACL process initialization.  Select the same device before
    # creating the private ACL stream in the pybind Runtime.
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    runtime = kvaf.NpuRuntime(str(kernel), device_id, npu_core_count)
    table = None
    try:
        print("RUNTIME_NPU_PHASE_BEGIN phase=runtime_initialize")
        runtime.initialize(table_capacity * source_stride, source_stride)
        print(
            "RUNTIME_NPU_PHASE_DONE phase=runtime_initialize "
            f"ddr_bytes={table_capacity * source_stride}"
        )

        print("RUNTIME_NPU_PHASE_BEGIN phase=seed_fixed_blocks")
        keys = list(range(1, table_capacity + 1))
        runtime.seed_fixed_pattern_blocks(keys, block_bytes)
        print(
            "RUNTIME_NPU_PHASE_DONE phase=seed_fixed_blocks "
            f"blocks={table_capacity} block_bytes={block_bytes}"
        )

        print("RUNTIME_NPU_PHASE_BEGIN phase=prepare_access_table")
        table = runtime.prepare_access_table(keys)
        del keys
        print(
            "RUNTIME_NPU_PHASE_DONE phase=prepare_access_table "
            f"entries={table_capacity}"
        )

        indices_values = [table_capacity - 1 - (request % table_capacity) for request in range(requests)]
        mask_values = [
            0 if mask_every and request % mask_every == 0 else 1
            for request in range(requests)
        ]
        output_indices_values = [output_rows - 1 - request for request in range(requests)]

        print("RUNTIME_NPU_PHASE_BEGIN phase=caller_buffer_setup")
        address_table = torch.empty(table_capacity, dtype=torch.int64, device=device)
        valid_table = torch.empty(table_capacity, dtype=torch.uint8, device=device)
        print("RUNTIME_NPU_PHASE_BEGIN phase=copy_access_table_to_device")
        runtime.copy_access_table_to_device(
            table,
            int(address_table.data_ptr()),
            tensor_bytes(address_table),
            int(valid_table.data_ptr()),
            tensor_bytes(valid_table),
        )
        print(
            "RUNTIME_NPU_PHASE_DONE phase=copy_access_table_to_device "
            f"entries={table_capacity}"
        )
        indices = torch.tensor(indices_values, dtype=torch.int32, device=device)
        mask = torch.tensor(mask_values, dtype=torch.uint8, device=device)
        output_indices = torch.tensor(output_indices_values, dtype=torch.int32, device=device)
        output_a = torch.zeros(output_rows * block_bytes, dtype=torch.uint8, device=device)
        output_b = torch.zeros_like(output_a)
        status = torch.zeros(
            npu_core_count * kvaf.NPU_BATCH_GET_STATUS_BYTES, dtype=torch.uint8, device=device
        )
        queue = torch.zeros(
            npu_core_count * kvaf.NPU_BATCH_GET_QUEUE_STATE_BYTES, dtype=torch.uint8, device=device
        )
        torch.npu.synchronize()
        print("RUNTIME_NPU_PHASE_DONE phase=caller_buffer_setup")

        def execute(output: object) -> float:
            return runtime.batch_get(
                table,
                int(indices.data_ptr()),
                tensor_bytes(indices),
                int(mask.data_ptr()),
                tensor_bytes(mask),
                int(output_indices.data_ptr()),
                tensor_bytes(output_indices),
                int(output.data_ptr()),
                tensor_bytes(output),
                int(status.data_ptr()),
                tensor_bytes(status),
                int(queue.data_ptr()),
                tensor_bytes(queue),
                block_bytes,
                int(address_table.data_ptr()),
                tensor_bytes(address_table),
                int(valid_table.data_ptr()),
                tensor_bytes(valid_table),
            )

        print("RUNTIME_NPU_PHASE_BEGIN phase=runtime_batch_get")
        execute(output_a)
        torch.npu.synchronize()
        validate_output(bytes(output_a.cpu().tolist()), indices_values, mask_values, output_indices_values, block_bytes)
        validate_completion(
            bytes(status.cpu().tolist()),
            bytes(queue.cpu().tolist()),
            requests,
            mask_values,
            npu_core_count,
            1,
        )
        print(f"RUNTIME_NPU_BATCH_GET_VALIDATION_PASS requests={requests}")

        # Runtime must acquire A and B independently, then reuse A's cached
        # local HBM registration on the third call. The transport is owned by
        # the operator and remains unchanged throughout this sequence.
        execute(output_b)
        torch.npu.synchronize()
        validate_output(bytes(output_b.cpu().tolist()), indices_values, mask_values, output_indices_values, block_bytes)
        validate_completion(
            bytes(status.cpu().tolist()),
            bytes(queue.cpu().tolist()),
            requests,
            mask_values,
            npu_core_count,
            2,
        )
        execute(output_a)
        torch.npu.synchronize()
        validate_output(bytes(output_a.cpu().tolist()), indices_values, mask_values, output_indices_values, block_bytes)
        validate_completion(
            bytes(status.cpu().tolist()),
            bytes(queue.cpu().tolist()),
            requests,
            mask_values,
            npu_core_count,
            3,
        )
        print("RUNTIME_NPU_DYNAMIC_OUTPUT_VALIDATION_PASS sequence=A_B_A")

        for _ in range(warmup):
            execute(output_a)
        samples = [execute(output_a) for _ in range(iterations)]
        torch.npu.synchronize()
        validate_output(bytes(output_a.cpu().tolist()), indices_values, mask_values, output_indices_values, block_bytes)
        validate_completion(
            bytes(status.cpu().tolist()),
            bytes(queue.cpu().tolist()),
            requests,
            mask_values,
            npu_core_count,
            3 + warmup + iterations,
        )

        active_requests = sum(mask_values)
        mean_ms = statistics.mean(samples)
        useful_gibps = (
            active_requests * block_bytes / (1024.0**3) / (mean_ms / 1000.0)
        )
        print(
            "PERF_RESULT metric=runtime_batch_get_call_to_stream_completion "
            f"samples={iterations} warmup_excluded={warmup} "
            f"active_requests={active_requests} block_bytes={block_bytes} "
            f"npu_cores={npu_core_count} mean_ms={mean_ms:.3f} "
            f"p50_ms={percentile(samples, 0.50):.3f} "
            f"p90_ms={percentile(samples, 0.90):.3f} "
            f"p99_ms={percentile(samples, 0.99):.3f} "
            f"useful_GiBps={useful_gibps:.3f}"
        )
        return 0
    finally:
        print("RUNTIME_NPU_PHASE_BEGIN phase=teardown")
        if table is not None and runtime.initialized:
            runtime.release_access_table(table)
        if runtime.initialized:
            runtime.shutdown()
        print("RUNTIME_NPU_PHASE_DONE phase=teardown status=0")


if __name__ == "__main__":
    raise SystemExit(main())
