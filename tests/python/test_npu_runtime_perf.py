#!/usr/bin/env python3
"""NPU Runtime steady-state performance benchmark.

This is intentionally a direct Python/pybind benchmark, not a CTest case.
It measures Runtime calls only: source request plans are created before the
timed loops, BatchGet tensors are derived immediately before each call, and
NPU calls end each sample with an external stream synchronize.
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
KERNEL_PROFILE_CLOCK_MHZ = 1000
BLOCK_BYTES = 656
WQE_BYTES = 656
ROWS_PER_ENTRY = 128
ENTRY_BYTES = ROWS_PER_ENTRY * WQE_BYTES
REQUESTS_PER_BATCH = 2048


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
    useful_gibps: float | None = None,
) -> None:
    bandwidth = (
        "" if useful_gibps is None else f" useful_GiBps={useful_gibps:.3f}"
    )
    print(
        "PERF_RESULT "
        f"operation={operation} samples={len(samples)} warmup_excluded={warmup} "
        f"mean_ms={statistics.mean(samples):.3f} "
        f"p50_ms={percentile(samples, 0.50):.3f} "
        f"p90_ms={percentile(samples, 0.90):.3f} "
        f"p99_ms={percentile(samples, 0.99):.3f} "
        f"units={units}{bandwidth}"
    )


def print_kernel_profile(
    samples: list[list[dict[str, object]]],
    warmup: int,
    clock_mhz: int,
    npu_core_count: int,
    jettys_per_core: int,
    requests: int,
    total_useful_bytes: int,
) -> None:
    if not samples or any(len(sample) != npu_core_count for sample in samples):
        raise AssertionError("BatchGet kernel profile sample/core count mismatch")

    def elapsed_ms(profile: dict[str, object], begin: str, end: str) -> float:
        return (
            int(profile[end]) - int(profile[begin])
        ) / clock_mhz / 1000.0

    stage_samples: dict[str, list[float]] = {
        "kernel_wall": [],
        "kernel_total": [],
        "kernel_setup": [],
        "kernel_compact": [],
        "kernel_post": [],
        "kernel_wqe_fill": [],
        "kernel_wqe_lookup_address": [],
        "kernel_wqe_control": [],
        "kernel_wqe_encode": [],
        "kernel_wqe_ub_store": [],
        "kernel_wqe_fill_overhead": [],
        "kernel_sq_copy": [],
        "kernel_doorbell": [],
        "kernel_post_overhead": [],
        "kernel_transfer_window": [],
        "kernel_completion_wait": [],
        "kernel_finalize": [],
    }
    per_core: list[dict[str, float]] = [
        {
            "total_ms": 0.0,
            "setup_ms": 0.0,
            "compact_ms": 0.0,
            "post_ms": 0.0,
            "wqe_fill_ms": 0.0,
            "wqe_lookup_address_ms": 0.0,
            "wqe_control_ms": 0.0,
            "wqe_encode_ms": 0.0,
            "wqe_ub_store_ms": 0.0,
            "wqe_fill_overhead_ms": 0.0,
            "sq_copy_ms": 0.0,
            "doorbell_ms": 0.0,
            "post_overhead_ms": 0.0,
            "completion_wait_ms": 0.0,
            "finalize_ms": 0.0,
            "active_requests": 0.0,
            "total_posts": 0.0,
            "jetty0_wait_ms": 0.0,
            "jetty0_polls": 0.0,
            "jetty1_wait_ms": 0.0,
            "jetty1_polls": 0.0,
        }
        for _ in range(npu_core_count)
    ]
    for sample in samples:
        stage_samples["kernel_wall"].append(
            (
                max(int(profile["kernel_finish_cycle"]) for profile in sample)
                - min(int(profile["kernel_start_cycle"]) for profile in sample)
            ) / clock_mhz / 1000.0
        )
        critical: dict[str, float] = {
            "kernel_total": 0.0,
            "kernel_setup": 0.0,
            "kernel_compact": 0.0,
            "kernel_post": 0.0,
            "kernel_wqe_fill": 0.0,
            "kernel_wqe_lookup_address": 0.0,
            "kernel_wqe_control": 0.0,
            "kernel_wqe_encode": 0.0,
            "kernel_wqe_ub_store": 0.0,
            "kernel_wqe_fill_overhead": 0.0,
            "kernel_sq_copy": 0.0,
            "kernel_doorbell": 0.0,
            "kernel_post_overhead": 0.0,
            "kernel_transfer_window": 0.0,
            "kernel_completion_wait": 0.0,
            "kernel_finalize": 0.0,
        }
        for profile in sample:
            core = int(profile["core"])
            total = elapsed_ms(profile, "kernel_start_cycle", "kernel_finish_cycle")
            setup = elapsed_ms(profile, "kernel_start_cycle", "setup_done_cycle")
            compact = elapsed_ms(profile, "setup_done_cycle", "compact_done_cycle")
            post = elapsed_ms(profile, "compact_done_cycle", "post_done_cycle")
            completion = elapsed_ms(
                profile, "post_done_cycle", "completion_done_cycle"
            )
            finalize = elapsed_ms(
                profile, "completion_done_cycle", "kernel_finish_cycle"
            )
            wqe_fill = int(profile["wqe_fill_cycles"]) / clock_mhz / 1000.0
            wqe_lookup_address = (
                int(profile["wqe_lookup_address_cycles"])
                / clock_mhz
                / 1000.0
            )
            wqe_control = (
                int(profile["wqe_control_cycles"]) / clock_mhz / 1000.0
            )
            wqe_encode = (
                int(profile["wqe_encode_cycles"]) / clock_mhz / 1000.0
            )
            wqe_ub_store = (
                int(profile["wqe_ub_store_cycles"]) / clock_mhz / 1000.0
            )
            wqe_fill_overhead = max(
                0.0,
                wqe_fill
                - wqe_lookup_address
                - wqe_control
                - wqe_encode
                - wqe_ub_store,
            )
            sq_copy = int(profile["sq_copy_cycles"]) / clock_mhz / 1000.0
            doorbell = int(profile["doorbell_cycles"]) / clock_mhz / 1000.0
            post_overhead = max(0.0, post - wqe_fill - sq_copy - doorbell)
            first_doorbell = int(profile["first_doorbell_cycle"])
            transfer = (
                0.0
                if first_doorbell == 0
                else (
                    int(profile["completion_done_cycle"]) - first_doorbell
                ) / clock_mhz / 1000.0
            )
            critical["kernel_total"] = max(critical["kernel_total"], total)
            critical["kernel_setup"] = max(critical["kernel_setup"], setup)
            critical["kernel_compact"] = max(critical["kernel_compact"], compact)
            critical["kernel_post"] = max(critical["kernel_post"], post)
            critical["kernel_wqe_fill"] = max(
                critical["kernel_wqe_fill"], wqe_fill
            )
            critical["kernel_wqe_lookup_address"] = max(
                critical["kernel_wqe_lookup_address"], wqe_lookup_address
            )
            critical["kernel_wqe_control"] = max(
                critical["kernel_wqe_control"], wqe_control
            )
            critical["kernel_wqe_encode"] = max(
                critical["kernel_wqe_encode"], wqe_encode
            )
            critical["kernel_wqe_ub_store"] = max(
                critical["kernel_wqe_ub_store"], wqe_ub_store
            )
            critical["kernel_wqe_fill_overhead"] = max(
                critical["kernel_wqe_fill_overhead"], wqe_fill_overhead
            )
            critical["kernel_sq_copy"] = max(
                critical["kernel_sq_copy"], sq_copy
            )
            critical["kernel_doorbell"] = max(
                critical["kernel_doorbell"], doorbell
            )
            critical["kernel_post_overhead"] = max(
                critical["kernel_post_overhead"], post_overhead
            )
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
            accumulator["total_ms"] += total
            accumulator["setup_ms"] += setup
            accumulator["compact_ms"] += compact
            accumulator["post_ms"] += post
            accumulator["wqe_fill_ms"] += wqe_fill
            accumulator["wqe_lookup_address_ms"] += wqe_lookup_address
            accumulator["wqe_control_ms"] += wqe_control
            accumulator["wqe_encode_ms"] += wqe_encode
            accumulator["wqe_ub_store_ms"] += wqe_ub_store
            accumulator["wqe_fill_overhead_ms"] += wqe_fill_overhead
            accumulator["sq_copy_ms"] += sq_copy
            accumulator["doorbell_ms"] += doorbell
            accumulator["post_overhead_ms"] += post_overhead
            accumulator["completion_wait_ms"] += completion
            accumulator["finalize_ms"] += finalize
            accumulator["active_requests"] += int(profile["active_requests"])
            accumulator["total_posts"] += int(profile["total_posts"])
            waits = list(profile["jetty_completion_wait_cycles"])
            polls = list(profile["jetty_completion_polls"])
            for jetty in range(2):
                accumulator[f"jetty{jetty}_wait_ms"] += (
                    int(waits[jetty]) / clock_mhz / 1000.0
                )
                accumulator[f"jetty{jetty}_polls"] += int(polls[jetty])
        for stage, value in critical.items():
            stage_samples[stage].append(value)

    print(
        "KERNEL_PROFILE_CONTRACT "
        f"clock_mhz={clock_mhz} unit=ms samples={len(samples)} "
        f"aiv_cores={npu_core_count} jettys_per_core={jettys_per_core} "
        "schedule=serial_stages_no_overlap wqe_fill=simt_32_lanes "
        "wqe_fill_detail=lookup_address_plus_control_plus_encode_plus_ub_store "
        "wqe_fill_detail_clock=simt_lane0 "
        "sq_copy=waited_mte3 doorbell=per_post_after_sq_copy "
        "stage_sync=pipe_all barrier_overhead_included=1 "
        "setup=entity_queue_and_miss_count compact=removed_direct_prefix "
        "post=wqe_fill_plus_sq_copy_plus_doorbell_plus_loop_overhead "
        "transfer_window=first_doorbell_to_all_cq_complete "
        "completion_wait=cq_poll_until_udma_complete "
        "profile_overhead_included=1 d2h_outside_host_metric=1"
    )
    for stage, values in stage_samples.items():
        useful_gibps = None
        if stage == "kernel_total":
            useful_gibps = (
                total_useful_bytes
                / (1024.0**3)
                / (sum(values) / 1000.0)
            )
        print_samples(
            f"batch_get_{stage}", values, warmup,
            f"{requests}_requests_per_call", useful_gibps=useful_gibps
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
    kernel_profile = int(
        os.environ.get("KVAF_RUNTIME_PERF_KERNEL_PROFILE", "0")
    )
    kernel_profile_clock_mhz = int(
        os.environ.get(
            "KVAF_RUNTIME_PERF_KERNEL_PROFILE_CLOCK_MHZ",
            str(KERNEL_PROFILE_CLOCK_MHZ),
        )
    )
    block_bytes = int(os.environ.get("KVAF_RUNTIME_PERF_BLOCK_BYTES", str(BLOCK_BYTES)))
    table_capacity = batch * PUT_BATCH_SIZE * 8
    requests = batch * REQUESTS_PER_BATCH
    total_tokens = table_capacity * ROWS_PER_ENTRY
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
        and kernel_profile in (0, 1)
        and kernel_profile_clock_mhz > 0
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
            f"total_tokens={total_tokens} requests={requests} "
            f"block_bytes={block_bytes} aiv_cores={npu_core_count} "
            f"jettys_per_core={jettys_per_core} "
            f"active_percent={active_percent} warmup={warmup} iterations={iterations} "
            f"kernel_profile={kernel_profile} "
            f"kernel_profile_clock_mhz={kernel_profile_clock_mhz} "
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

        # Pre-create only the source data for one token-level request plan per
        # warmup/measured sample. source_token_ids select one of
        # table_capacity * 128 remote tokens, destination_slots select random
        # output rows, and each call contributes one random miss_count smaller
        # than the request capacity. BatchGet tensors are derived from these
        # immediately before the corresponding sample call.
        probability = active_percent / 100.0
        sample_count = warmup + iterations
        source_token_id_batches = []
        destination_slot_batches = []
        miss_count_batches = []
        for _ in range(sample_count):
            source_token_ids = torch.randint(
                0, total_tokens, (requests,), dtype=torch.int32, device=device
            )
            torch.npu.synchronize()

            destination_slots = torch.randperm(requests, device=device).to(
                torch.int32
            )
            torch.npu.synchronize()

            miss_count = (
                torch.rand(
                    (requests - 1,), device=device
                )
                < probability
            ).sum().reshape(1).to(torch.int32)
            torch.npu.synchronize()

            source_token_id_batches.append(source_token_ids)
            destination_slot_batches.append(destination_slots)
            miss_count_batches.append(miss_count)

        miss_count_host = torch.stack(miss_count_batches).cpu()
        if bool((miss_count_host >= requests).any().item()):
            raise AssertionError("miss_count must be smaller than request capacity")
        mean_active_requests = float(miss_count_host.sum().item()) / sample_count
        target_active_requests = requests * probability
        print(
            "RUNTIME_PERF_BATCH_GET_INPUTS_READY "
            f"samples={sample_count} total_tokens={total_tokens} "
            f"requests_per_sample={requests} "
            f"mean_active_requests={mean_active_requests:.3f} "
            f"target_active_requests={target_active_requests:.3f} "
            "source_token_ids=random destination_slots=random_permutation "
            "request_layout=miss_count_prefix"
        )
        print("RUNTIME_PERF_PHASE_BEGIN phase=batch_get")
        if kernel_profile:
            runtime.start_batch_get_kernel_profile()
        warmup_inputs = []
        for source_token_ids, destination_slots, miss_count in zip(
            source_token_id_batches[:warmup],
            destination_slot_batches[:warmup],
            miss_count_batches[:warmup],
        ):
            indices = source_token_ids // ROWS_PER_ENTRY
            torch.npu.synchronize()
            offsets = (source_token_ids % ROWS_PER_ENTRY) * WQE_BYTES
            torch.npu.synchronize()
            output_indices = destination_slots.clone()
            torch.npu.synchronize()
            warmup_inputs.append((indices, offsets, output_indices, miss_count))
            runtime.batch_get(
                device_table, indices, offsets, miss_count, output_indices, output
            )
        torch.npu.synchronize()
        warmup_inputs.clear()
        batch_get_samples: list[float] = []
        kernel_profile_samples: list[list[dict[str, object]]] = []
        host_profile_samples: dict[str, list[float]] = {
            "submit": [],
            "synchronize": [],
        }
        for source_token_ids, destination_slots, miss_count in zip(
            source_token_id_batches[warmup:],
            destination_slot_batches[warmup:],
            miss_count_batches[warmup:],
        ):
            indices = source_token_ids // ROWS_PER_ENTRY
            torch.npu.synchronize()

            offsets = (source_token_ids % ROWS_PER_ENTRY) * WQE_BYTES
            torch.npu.synchronize()

            output_indices = destination_slots.clone()
            torch.npu.synchronize()

            begin = time.perf_counter()
            runtime.batch_get(
                device_table, indices, offsets, miss_count, output_indices, output
            )
            submitted = time.perf_counter()
            torch.npu.synchronize()
            completed = time.perf_counter()
            host_profile_samples["submit"].append(
                (submitted - begin) * 1000.0
            )
            host_profile_samples["synchronize"].append(
                (completed - submitted) * 1000.0
            )
            batch_get_samples.append((completed - begin) * 1000.0)
            if kernel_profile:
                kernel_profile_samples.append(
                    list(runtime.batch_get_kernel_profile())
                )
        for stage in ("submit", "synchronize"):
            print_samples(
                f"batch_get_{stage}", host_profile_samples[stage], warmup,
                f"{requests}_requests_per_call"
            )
        measured_active_requests = int(miss_count_host[warmup:].sum().item())
        total_useful_bytes = measured_active_requests * block_bytes
        useful_gibps = (
            total_useful_bytes
            / (1024.0**3)
            / (sum(batch_get_samples) / 1000.0)
        )
        print_samples(
            "batch_get", batch_get_samples, warmup,
            f"{requests}_requests_per_call", useful_gibps=useful_gibps
        )
        if kernel_profile:
            print_kernel_profile(
                kernel_profile_samples,
                warmup,
                kernel_profile_clock_mhz,
                npu_core_count,
                jettys_per_core,
                requests,
                total_useful_bytes,
            )
            runtime.stop_batch_get_kernel_profile()
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
