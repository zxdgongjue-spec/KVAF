#!/usr/bin/env python3
"""CPU Runtime -> NPU HBM BatchPut and BatchGet benchmarks.

The NPU service must already be listening. BatchPut follows the variable-size
single-buffer versus buffer-list design used by test_npu_batch_put.py.
BatchGet follows the fixed AccessTable, masked requests, warmup, measurement,
and post-measurement validation design used by test_npu_batch_get.py.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from typing import Iterable

import kvaf


DEFAULT_TOKEN_BYTES = 656
DEFAULT_TOKENS_PER_BLOCK = 128


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def make_lengths(rows: int, minimum: int, maximum: int) -> list[int]:
    span = maximum - minimum + 1
    return [minimum + ((row * 131 + 17) % span) for row in range(rows)]


def pattern(slot: int, token_bytes: int) -> bytes:
    return bytes(
        (slot * 131 + byte * 17 + (byte >> 3)) & 0xFF
        for byte in range(token_bytes)
    )


def runtime_options() -> tuple[str, int, str, int | None]:
    peer = (
        os.environ.get("KVAF_NPU_SERVICE_ADDR")
        or os.environ.get("KVAF_CPU_SERVICE_ADDR")
        or "127.0.0.1:28999"
    )
    device_id = int(os.environ.get("DEVICE_ID", "0"))
    ub_device = os.environ.get("KVAF_UB_DEVICE", "")
    eid_index_text = os.environ.get("EID_INDEX")
    eid_index = None if eid_index_text is None else int(eid_index_text)
    return peer, device_id, ub_device, eid_index


def initialize_runtime(hbm_bytes: int, slot_size: int) -> object:
    peer, device_id, ub_device, eid_index = runtime_options()
    runtime = kvaf.CpuRuntime()
    runtime.initialize(
        peer,
        hbm_bytes,
        slot_size,
        device_id=device_id,
        ub_device=ub_device,
        eid_index=eid_index,
        wqes_per_doorbell=int(
            os.environ.get("KVAF_CPU_URMA_DOORBELL_WQES", "0")
        ),
        remote_sgl_size=int(os.environ.get("KVAF_CPU_URMA_REMOTE_SGL_SIZE", "1")),
    )
    return runtime


def configured_region_bytes(required: int, slot_size: int, phase: str) -> int:
    phase_name = f"KVAF_HBM_{phase.upper()}_BYTES"
    configured = os.environ.get(phase_name, os.environ.get("KVAF_HBM_BYTES"))
    result = required if configured is None else int(configured)
    if result < required or result % slot_size:
        raise ValueError(
            f"{phase_name} must be at least {required} and a multiple of "
            f"the {slot_size}-byte slot size"
        )
    return result


def validate_put_metadata(
    runtime: object, keys: list[int], expected_lengths: list[int]
) -> None:
    table = runtime.prepare_access_table(keys)
    try:
        view = runtime.access_table_view(table)
        if list(view["lengths"]) != expected_lengths:
            raise AssertionError("BatchPut committed lengths do not match")
        if list(view["valid"]) != [1] * len(keys):
            raise AssertionError("BatchPut committed valid entries do not match")
    finally:
        runtime.release_access_table(table)


def run_batch_put(warmup: int, iterations: int) -> None:
    rows = int(os.environ.get("KVAF_HBM_ROWS", "4096"))
    group_count = int(os.environ.get("KVAF_HBM_INPUT_GROUPS", "4"))
    minimum = int(os.environ.get("KVAF_HBM_MIN_BLOCK_BYTES", "512"))
    maximum = int(os.environ.get("KVAF_HBM_MAX_BLOCK_BYTES", "2048"))
    slot_size = int(os.environ.get("KVAF_HBM_PUT_SLOT_SIZE", str(maximum)))
    if not (
        rows > 0
        and 0 < group_count <= rows
        and 0 < minimum <= maximum <= slot_size
    ):
        raise ValueError("invalid KVAF_HBM_* BatchPut shape configuration")

    hbm_bytes = configured_region_bytes(rows * slot_size, slot_size, "put")
    row_lengths = make_lengths(rows, minimum, maximum)
    sources: list[bytes] = []
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
        sources.append(
            bytes((byte * 17 + begin * 131 + group) & 0xFF for byte in range(cursor))
        )
        offsets.append(group_offsets)
        key_groups.append(list(range(begin + 1, end + 1)))
        length_groups.append(group_lengths)

    all_keys = list(range(1, rows + 1))
    single_offsets: list[int] = []
    cursor = 0
    for length in row_lengths:
        single_offsets.append(cursor)
        cursor += length
    plans = [
        ("single_buffer", [all_keys], [b"".join(sources)], [single_offsets], [row_lengths]),
        ("buffer_list", key_groups, sources, offsets, length_groups),
    ]
    total_bytes = sum(row_lengths)
    runtime = initialize_runtime(hbm_bytes, slot_size)
    try:
        print(
            "CPU_HBM_BATCH_PUT_CONFIG "
            f"rows={rows} groups={group_count} min_bytes={minimum} "
            f"max_bytes={maximum} total_bytes={total_bytes} "
            f"hbm_bytes={hbm_bytes} slot_size={slot_size}"
        )
        for name, keys, inputs, input_offsets, lengths in plans:
            flat_keys = [key for group in keys for key in group]

            def execute() -> None:
                runtime.batch_put(keys, inputs, input_offsets, lengths)

            execute()
            validate_put_metadata(runtime, flat_keys, row_lengths)
            runtime.batch_free(flat_keys)
            for _ in range(warmup):
                execute()
                runtime.batch_free(flat_keys)
            samples: list[float] = []
            for _ in range(iterations):
                begin = time.perf_counter()
                execute()
                samples.append((time.perf_counter() - begin) * 1000.0)
                runtime.batch_free(flat_keys)
            mean_ms = statistics.mean(samples)
            useful_gbps = total_bytes / 1_000_000_000.0 / (mean_ms / 1000.0)
            print(
                "PERF_RESULT metric=batch_put "
                f"layout={name} samples={iterations} warmup_excluded={warmup} "
                f"rows={rows} total_bytes={total_bytes} mean_ms={mean_ms:.3f} "
                f"p50_ms={percentile(samples, 0.50):.3f} "
                f"p90_ms={percentile(samples, 0.90):.3f} "
                f"p99_ms={percentile(samples, 0.99):.3f} "
                f"useful_GBps={useful_gbps:.3f}"
            )
    finally:
        if runtime.initialized:
            runtime.shutdown()


def validate_get_output(
    output: bytes,
    mask: Iterable[int],
    output_indices: Iterable[int],
    payload_slots: Iterable[int | None],
    token_bytes: int,
) -> None:
    zero_block = bytes(token_bytes)
    payload_cache: dict[int, bytes] = {}
    for request, (enabled, row, payload_slot) in enumerate(
        zip(mask, output_indices, payload_slots)
    ):
        offset = row * token_bytes
        actual = output[offset : offset + token_bytes]
        if not enabled:
            expected = zero_block
        elif payload_slot is None:
            raise AssertionError(f"missing expected payload for request={request}")
        else:
            expected = payload_cache.get(payload_slot)
            if expected is None:
                expected = pattern(payload_slot, token_bytes)
                payload_cache[payload_slot] = expected
        if actual != expected:
            raise AssertionError(f"BatchGet output mismatch for request={request}, row={row}")


def run_batch_get(warmup: int, iterations: int) -> None:
    table_capacity = int(os.environ.get("KVAF_HBM_TABLE_CAPACITY", "8"))
    requests = int(os.environ.get("KVAF_HBM_REQUESTS", "8"))
    output_rows = int(os.environ.get("KVAF_HBM_OUTPUT_ROWS", str(requests)))
    token_bytes = int(os.environ.get("KVAF_HBM_TOKEN_BYTES", str(DEFAULT_TOKEN_BYTES)))
    tokens_per_block = int(
        os.environ.get("KVAF_HBM_TOKENS_PER_BLOCK", str(DEFAULT_TOKENS_PER_BLOCK))
    )
    active_percent = int(os.environ.get("KVAF_HBM_ACTIVE_PERCENT", "100"))
    block_bytes = token_bytes * tokens_per_block
    slot_size = int(os.environ.get("KVAF_HBM_GET_SLOT_SIZE", str(block_bytes)))
    if not (
        table_capacity > 0
        and requests > 0
        and requests <= output_rows
        and token_bytes > 0
        and tokens_per_block > 0
        and block_bytes <= slot_size
        and 1 <= active_percent <= 100
    ):
        raise ValueError("invalid KVAF_HBM_* BatchGet shape configuration")

    mask_values = [
        1 if request % 100 < active_percent else 0 for request in range(requests)
    ]
    active_positions = [
        request for request, enabled in enumerate(mask_values) if enabled
    ]
    active_requests = len(active_positions)
    if active_requests > table_capacity:
        raise ValueError(
            "KVAF_HBM_TABLE_CAPACITY must be at least the number of active requests"
        )

    indices = [0] * requests
    offsets = [0] * requests
    payload_slots: list[int | None] = [None] * requests
    for table_index, request in enumerate(active_positions):
        indices[request] = table_index
        offsets[request] = (request % tokens_per_block) * token_bytes
        payload_slots[request] = table_index * tokens_per_block + request % tokens_per_block
    output_indices = [output_rows - 1 - request for request in range(requests)]
    active_keys = list(range(1_000_001, 1_000_001 + active_requests))
    source = b"".join(
        pattern(block * tokens_per_block + token, token_bytes)
        for block in range(active_requests)
        for token in range(tokens_per_block)
    )
    source_offsets = [block * block_bytes for block in range(active_requests)]
    source_lengths = [block_bytes] * active_requests
    hbm_bytes = configured_region_bytes(table_capacity * slot_size, slot_size, "get")
    runtime = initialize_runtime(hbm_bytes, slot_size)
    table = None
    populated = False
    try:
        print(
            "CPU_HBM_BATCH_GET_CONFIG "
            f"table_capacity={table_capacity} requests={requests} "
            f"active_percent={active_percent} active_requests={active_requests} "
            f"output_rows={output_rows} token_bytes={token_bytes} "
            f"tokens_per_block={tokens_per_block} block_bytes={block_bytes} "
            f"hbm_bytes={hbm_bytes} slot_size={slot_size}"
        )
        runtime.batch_put(
            [active_keys], [source], [source_offsets], [source_lengths]
        )
        populated = True
        table = runtime.prepare_access_table(active_keys)

        output = bytearray(output_rows * token_bytes)

        def execute() -> None:
            runtime.batch_get(
                table,
                indices,
                offsets,
                output_indices,
                output,
                token_bytes,
                mask_values,
            )

        execute()
        validate_get_output(
            bytes(output), mask_values, output_indices, payload_slots, token_bytes
        )
        print(f"CPU_HBM_BATCH_GET_VALIDATION_PASS requests={requests}")
        for _ in range(warmup):
            execute()
        samples: list[float] = []
        for _ in range(iterations):
            begin = time.perf_counter()
            execute()
            samples.append((time.perf_counter() - begin) * 1000.0)
        validate_get_output(
            bytes(output), mask_values, output_indices, payload_slots, token_bytes
        )
        mean_ms = statistics.mean(samples)
        useful_gibps = (
            active_requests * token_bytes / (1024.0**3) / (mean_ms / 1000.0)
        )
        print(
            "PERF_RESULT metric=runtime_batch_get_call_to_completion "
            f"samples={iterations} warmup_excluded={warmup} "
            f"active_requests={active_requests} token_bytes={token_bytes} "
            f"mean_ms={mean_ms:.3f} "
            f"p50_ms={percentile(samples, 0.50):.3f} "
            f"p90_ms={percentile(samples, 0.90):.3f} "
            f"p99_ms={percentile(samples, 0.99):.3f} "
            f"useful_GiBps={useful_gibps:.3f}"
        )
    finally:
        if table is not None:
            runtime.release_access_table(table)
        if populated:
            runtime.batch_free(active_keys)
        if runtime.initialized:
            runtime.shutdown()


def main() -> int:
    if not hasattr(kvaf, "CpuRuntime"):
        print("SKIP: CPU Runtime Python binding was not built")
        return 0

    warmup = int(os.environ.get("KVAF_HBM_WARMUP", "1"))
    iterations = int(os.environ.get("KVAF_HBM_ITERATIONS", "5"))
    if warmup < 0 or iterations <= 0:
        print("FAIL: invalid KVAF_HBM_WARMUP or KVAF_HBM_ITERATIONS", file=sys.stderr)
        return 2

    peer, device_id, _, eid_index = runtime_options()
    print(
        f"CPU_HBM_TEST peer={peer} device_id={device_id} "
        f"eid_index={'auto' if eid_index is None else eid_index}"
    )
    try:
        run_batch_put(warmup, iterations)
        run_batch_get(warmup, iterations)
    except (ValueError, AssertionError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    print("CPU_HBM_TEST PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
