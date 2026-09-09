#!/usr/bin/env python3
"""Variable-length single-Tensor versus Tensor-list BatchPut benchmark."""

from __future__ import annotations

import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kvaf

from backend import detect_torch_backend

DEFAULT_BATCH_GET_TOKEN_BYTES = 656
DEFAULT_BATCH_GET_TOKENS_PER_BLOCK = 128


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def make_lengths(rows: int, minimum: int, maximum: int) -> list[int]:
    span = maximum - minimum + 1
    return [minimum + ((row * 131 + 17) % span) for row in range(rows)]


def validate(runtime: object, keys: list[int], expected_lengths: list[int]) -> None:
    table = runtime.prepare_access_table(keys)
    try:
        view = runtime.access_table_view(table)
        if list(view["lengths"]) != expected_lengths:
            raise AssertionError("BatchPut committed lengths do not match")
        if list(view["valid"]) != [1] * len(keys):
            raise AssertionError("BatchPut committed valid entries do not match")
    finally:
        runtime.release_access_table(table)


def main() -> int:
    backend = detect_torch_backend()
    if backend.name != "npu":
        print(f"SKIP: NPU required; detected {backend.name} ({backend.reason})")
        return 0
    if not hasattr(kvaf, "NpuRuntime") or os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: build the NPU binding and set KVAF_RUN_NPU_TESTS=1")
        return 0

    put_kernel = Path(os.environ.get("KVAF_BATCH_PUT_KERNEL", ""))
    get_kernel = Path(os.environ.get(
        "KVAF_BATCH_GET_KERNEL",
        str(put_kernel.with_name("kvaf_execute_batch_get_kernels.o")),
    ))
    cpu_service = os.environ.get("KVAF_CPU_SERVICE_ADDR")
    if not put_kernel.is_file() or not get_kernel.is_file() or not cpu_service:
        print("FAIL: kernel paths and KVAF_CPU_SERVICE_ADDR are required", file=sys.stderr)
        return 2

    torch = backend.torch
    assert torch is not None
    device_id = int(os.environ.get("DEVICE_ID", "0"))
    rows = int(os.environ.get("KVAF_NPU_REQUESTS", "4096"))
    group_count = int(os.environ.get("KVAF_NPU_INPUT_GROUPS", "4"))
    minimum = int(os.environ.get("KVAF_NPU_MIN_BLOCK_BYTES", "512"))
    maximum = int(os.environ.get("KVAF_NPU_MAX_BLOCK_BYTES", "2048"))
    cores = int(os.environ.get("KVAF_NPU_CORE_COUNT", "32"))
    jettys = int(os.environ.get("KVAF_NPU_JETTYS_PER_CORE", "2"))
    warmup = int(os.environ.get("KVAF_NPU_WARMUP", "1"))
    iterations = int(os.environ.get("KVAF_NPU_ITERATIONS", "5"))
    batch_get_token_bytes = int(os.environ.get(
        "KVAF_NPU_TOKEN_BYTES", str(DEFAULT_BATCH_GET_TOKEN_BYTES)))
    batch_get_tokens_per_block = int(os.environ.get(
        "KVAF_NPU_TOKENS_PER_BLOCK", str(DEFAULT_BATCH_GET_TOKENS_PER_BLOCK)))
    if not (rows > 0 and 0 < group_count <= rows and 0 < minimum <= maximum and
            0 < cores <= rows and 1 <= jettys <= 2 and warmup >= 0 and
            iterations > 0 and batch_get_token_bytes > 0 and
            batch_get_tokens_per_block > 0):
        print("FAIL: invalid KVAF_NPU_* configuration", file=sys.stderr)
        return 2

    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    row_lengths = make_lengths(rows, minimum, maximum)
    sources: list[object] = []
    offsets: list[list[int]] = []
    key_groups: list[list[int]] = []
    length_groups: list[list[int]] = []
    for group in range(group_count):
        begin = rows * group // group_count
        end = rows * (group + 1) // group_count
        group_lengths = row_lengths[begin:end]
        group_offsets: list[int] = []
        cursor = 0
        for length in group_lengths:
            group_offsets.append(cursor)
            cursor += length
        byte_index = torch.arange(cursor, dtype=torch.int64, device=device)
        source = ((byte_index * 17 + begin * 131 + group) & 0xFF).to(torch.uint8).contiguous()
        sources.append(source)
        offsets.append(group_offsets)
        key_groups.append(list(range(begin + 1, end + 1)))
        length_groups.append(group_lengths)

    runtime = kvaf.NpuRuntime(
        str(get_kernel),
        batch_get_token_bytes=batch_get_token_bytes,
        batch_get_tokens_per_block=batch_get_tokens_per_block,
        device_id=device_id, npu_core_count=cores,
        batch_put_kernel_object_path=str(put_kernel), jettys_per_core=jettys,
    )
    runtime.link_to(cpu_service)
    total_bytes = sum(row_lengths)
    # MetaManager allocates one fixed-size remote slot per row.  The peer
    # allocator requires ddr_bytes to be an exact multiple of slot_size;
    # variable payload bytes cannot be used directly as the region size.
    runtime.initialize(rows * maximum, maximum)
    try:
        print(f"BATCH_PUT_CONFIG rows={rows} groups={group_count} min_bytes={minimum} max_bytes={maximum} total_bytes={total_bytes} cores={cores} jettys_per_core={jettys}")
        single_source = [torch.cat(sources)]
        single_offsets = [[sum(row_lengths[:row]) for row in range(rows)]]
        plans = [("single_tensor", [list(range(1, rows + 1))], single_source, single_offsets, [row_lengths])]
        plans.append(("tensor_list", key_groups, sources, offsets, length_groups))
        for name, keys, inputs, input_offsets, lengths in plans:
            flat_keys = [key for group in keys for key in group]

            def execute() -> None:
                runtime.batch_put(keys, inputs, input_offsets, lengths)

            execute()
            torch.npu.synchronize()
            validate(runtime, flat_keys, row_lengths)
            runtime.batch_free(flat_keys)
            for _ in range(warmup):
                execute()
                runtime.batch_free(flat_keys)
            torch.npu.synchronize()
            samples: list[float] = []
            for _ in range(iterations):
                begin = time.perf_counter()
                execute()
                torch.npu.synchronize()
                samples.append((time.perf_counter() - begin) * 1000.0)
                runtime.batch_free(flat_keys)
            mean = statistics.mean(samples)
            gbps = total_bytes / 1_000_000_000.0 / (mean / 1000.0)
            print(f"PERF_RESULT metric=batch_put layout={name} samples={iterations} warmup={warmup} rows={rows} total_bytes={total_bytes} mean_ms={mean:.3f} p50_ms={percentile(samples, .5):.3f} p90_ms={percentile(samples, .9):.3f} p99_ms={percentile(samples, .99):.3f} useful_GBps={gbps:.3f}")
        return 0
    finally:
        if runtime.initialized:
            runtime.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
