#!/usr/bin/env python3
"""Single-layer BatchPut bandwidth test; optional KVAF_DSA_* geometry.

KVAF_PEER_UB_DEVICE selects the CPU UB device; KVAF_NPU_EIDS selects local
NPU EIDs. BatchPut uses the first EID when an explicit list is provided.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kvaf

from backend import detect_torch_backend
from batch_put_profile import profile_options, print_profile, summarize_sample
from npu_eids import parse_npu_eids

DEFAULT_BATCH_GET_TOKEN_BYTES = 656
DEFAULT_BATCH_GET_TOKENS_PER_BLOCK = 128
DDR_SLOT_ALIGNMENT = 4096  # vLLM-DSA M0_DDR_LAYER_ALIGNMENT_BYTES.
MAX_AIV_CORES = 48  # batch_put_abi.h.


def build_keys(
    rows: int, pool_rows: int, n_layers: int, q_capacity: int, layer: int
) -> list[int]:
    if min(rows, pool_rows, n_layers, q_capacity) <= 0:
        raise ValueError("block count and DSA dimensions must be positive")
    if not 0 <= layer < n_layers:
        raise ValueError("layer ordinal must be in [0, n_layers)")
    if pool_rows * n_layers * q_capacity > (1 << 31) - 1:
        raise ValueError("DSA directory exceeds its int32 index range")
    if rows > pool_rows * q_capacity:
        raise ValueError("one layer's block count exceeds pool_rows * q_capacity")
    # Spread one layer's jobs evenly across pool rows; q_capacity stays fixed.
    return [
        pool_row * (n_layers * q_capacity) + layer * q_capacity + block_q
        for pool_row in range(min(pool_rows, rows))
        for block_q in range(rows // pool_rows + (pool_row < rows % pool_rows))
    ]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_batch(
    runtime: object,
    keys: list[int],
    rows: int,
    block_bytes: int,
) -> None:
    table = runtime.prepare_access_table(keys)
    try:
        view = runtime.access_table_view(table)
        if list(view["lengths"]) != [block_bytes] * rows:
            raise AssertionError("committed BatchPut lengths do not match")
        if list(view["valid"]) != [1] * rows:
            raise AssertionError("committed BatchPut entries are not valid")
        base = runtime.remote_region["base"]
        slot_bytes = runtime.remote_region["slot_size"]
        if list(view["addresses"]) != [base + key * slot_bytes for key in keys]:
            raise AssertionError("BatchPut addresses must follow DSA key_id == slot_id")
        # DDR belongs to the CPU service process. The companion BatchGet
        # integration test validates the bytes through AIV UB rather than
        # dereferencing a remote process address from Python.
        print("RUNTIME_NPU_REMOTE_CPU_DIRECT_DEREFERENCE_SKIPPED")
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

    try:
        device_id = int(os.environ.get("DEVICE_ID", "0"))
        peer_ub_device = os.environ.get("KVAF_PEER_UB_DEVICE", "").strip()
        npu_eids = parse_npu_eids(os.environ.get("KVAF_NPU_EIDS"))
        rows = int(os.environ.get("KVAF_NPU_REQUESTS", "4096"))
        token_bytes = int(
            os.environ.get("KVAF_NPU_TOKEN_BYTES", str(DEFAULT_BATCH_GET_TOKEN_BYTES))
        )
        tokens_per_block = int(
            os.environ.get(
                "KVAF_NPU_TOKENS_PER_BLOCK", str(DEFAULT_BATCH_GET_TOKENS_PER_BLOCK)
            )
        )
        block_bytes = int(os.environ.get(
            "KVAF_NPU_BLOCK_BYTES", str(token_bytes * tokens_per_block)
        ))
        npu_core_count = int(os.environ.get(
            "KVAF_NPU_CORE_COUNT", os.environ.get("KVAF_NPU_AIV_CORES", "32")
        ))
        jettys_per_core = int(os.environ.get("KVAF_NPU_JETTYS_PER_CORE", "2"))
        warmup = int(os.environ.get("KVAF_NPU_WARMUP", "20"))
        iterations = int(os.environ.get("KVAF_NPU_ITERATIONS", "500"))
        profiling, profile_clock_mhz, profile_path = profile_options(os.environ)
        if profiling and not all(hasattr(kvaf.NpuRuntime, name) for name in (
            "start_batch_put_profile", "batch_put_profile", "stop_batch_put_profile",
        )):
            raise ValueError("rebuild the NPU binding and BatchPut kernel with profiling support")
        if not (
            device_id >= 0 and 0 < rows <= (1 << 32) - 1
            and 0 < block_bytes <= (1 << 32) - 1
            and token_bytes > 0 and tokens_per_block > 0
            and block_bytes == token_bytes * tokens_per_block
            and 0 < npu_core_count <= min(rows, MAX_AIV_CORES)
            and 1 <= jettys_per_core <= 2 and warmup >= 0 and iterations > 0
        ):
            raise ValueError("invalid shape, core count or sampling configuration; "
                             "block_bytes must equal token_bytes * tokens_per_block")

        geometry_names = ("KVAF_DSA_POOL_ROWS", "KVAF_DSA_N_LAYERS", "KVAF_DSA_Q_CAPACITY")
        geometry_options = geometry_names + (
            "KVAF_DSA_LAYER_ORDINAL", "KVAF_DSA_CATALOG_CAPACITY_BLOCKS"
        )
        configured = any(name in os.environ for name in geometry_options)
        if configured:
            if any(name not in os.environ for name in geometry_names):
                raise ValueError("set all of " + ", ".join(geometry_names))
            pool_rows, n_layers, q_capacity = (int(os.environ[name]) for name in geometry_names)
        else:
            # A bandwidth baseline, without assuming a real model's geometry.
            pool_rows, n_layers, q_capacity = 1, 1, rows
        layer = int(os.environ.get("KVAF_DSA_LAYER_ORDINAL", "0"))
        catalog_blocks = int(os.environ.get("KVAF_DSA_CATALOG_CAPACITY_BLOCKS", "0"))
        keys = build_keys(rows, pool_rows, n_layers, q_capacity, layer)
        slot_count = (pool_rows * q_capacity + catalog_blocks) * n_layers
        slot_bytes = (
            (block_bytes + DDR_SLOT_ALIGNMENT - 1)
            // DDR_SLOT_ALIGNMENT * DDR_SLOT_ALIGNMENT
        )
        ddr_bytes = slot_count * slot_bytes
        if catalog_blocks < 0 or slot_count > (1 << 63) - 1 or ddr_bytes > (1 << 64) - 1:
            raise ValueError("invalid catalog capacity or DDR size overflow")
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2

    torch = backend.torch
    assert torch is not None
    torch.npu.set_device(device_id)
    device = f"npu:{device_id}"
    source_rows = torch.arange(rows, dtype=torch.int64).unsqueeze(1)
    source_bytes = torch.arange(block_bytes, dtype=torch.int64).unsqueeze(0)
    # Cast the small row/byte patterns first, avoiding a rows*block_bytes int64
    # temporary (2.56 GiB for the default 4096-block payload).
    row_pattern = (source_rows * 131).to(torch.uint8)
    byte_pattern = (source_bytes * 17 + (source_bytes >> 3)).to(torch.uint8)
    source_host = torch.empty((rows, block_bytes), dtype=torch.uint8)
    torch.add(row_pattern, byte_pattern, out=source_host)
    source = source_host.to(device)
    offsets = [row * block_bytes for row in range(rows)]
    lengths = [block_bytes] * rows
    torch.npu.synchronize()
    stream = torch.npu.current_stream()

    runtime = kvaf.NpuRuntime(
        str(batch_get_kernel),
        batch_get_token_bytes=token_bytes,
        batch_get_tokens_per_block=tokens_per_block,
        device_id=device_id,
        npu_core_count=npu_core_count,
        batch_put_kernel_object_path=str(batch_put_kernel),
        jettys_per_core=jettys_per_core,
        peer_ub_device=peer_ub_device,
        npu_eids=npu_eids,
    )
    remote_cpu_service = os.environ.get("KVAF_CPU_SERVICE_ADDR")
    if not remote_cpu_service:
        print("FAIL: KVAF_CPU_SERVICE_ADDR must name the separately running CPU service", file=sys.stderr)
        return 2
    runtime.link_to(remote_cpu_service)
    committed = False
    profile_log = None
    try:
        print(
            "RUNTIME_NPU_BATCH_PUT_CONFIG "
            f"rows={rows} block_bytes={block_bytes} npu_cores={npu_core_count} "
            f"scenario={'dsa_single_layer' if configured else 'bandwidth_baseline'} "
            f"pool_rows={pool_rows} n_layers={n_layers} q_capacity={q_capacity} "
            f"layer={layer} catalog_blocks={catalog_blocks} "
            f"slot_bytes={slot_bytes} ddr_bytes={ddr_bytes} "
            f"peer_ub_device={peer_ub_device or 'auto'} "
            f"npu_eids={','.join(npu_eids) or 'auto'} "
            f"batch_get_token_bytes={token_bytes} "
            f"batch_get_tokens_per_block={tokens_per_block} "
            f"jettys_per_core={jettys_per_core} warmup={warmup} "
            f"iterations={iterations} backend=aiv_direct_hccp"
            f" profiling={int(profiling)}"
        )
        min_rows_per_core = rows // npu_core_count
        max_rows_per_core = (rows + npu_core_count - 1) // npu_core_count
        print(
            "RUNTIME_NPU_BATCH_PUT_COVERAGE "
            f"requests_per_core={min_rows_per_core}..{max_rows_per_core} "
            f"active_jettys_per_core={min(jettys_per_core, (min_rows_per_core + 31) // 32)}"
            f"..{min(jettys_per_core, (max_rows_per_core + 31) // 32)}"
        )
        runtime.initialize(ddr_bytes, slot_bytes)

        def execute() -> None:
            nonlocal committed
            runtime.batch_put(keys, source, offsets, lengths, stream)
            committed = True

        def release() -> None:
            nonlocal committed
            runtime.batch_free(keys)
            committed = False

        print("RUNTIME_NPU_PHASE_BEGIN phase=runtime_batch_put")
        execute()
        torch.npu.synchronize()
        validate_batch(runtime, keys, rows, block_bytes)
        print(f"RUNTIME_NPU_BATCH_PUT_VALIDATION_PASS kind=metadata rows={rows}")
        release()

        for _ in range(warmup):
            execute()
            release()
        torch.npu.synchronize()

        profile_records = []
        if profiling:
            log_path = Path(profile_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            profile_log = log_path.open("w", encoding="utf-8")
            runtime.start_batch_put_profile()
            print(f"BATCH_PUT_PROFILE_LOG path={log_path.resolve()} counter_clock_mhz={profile_clock_mhz}")
        samples = []
        for iteration in range(iterations):
            begin = time.perf_counter()
            execute()
            # BatchPut already waits for its stream and checks completion status.
            samples.append((time.perf_counter() - begin) * 1000.0)
            if profiling:
                snapshot = runtime.batch_put_profile()
                record = {"iteration": iteration, "clock_mhz": profile_clock_mhz,
                          "profile": snapshot,
                          "derived": summarize_sample(snapshot, samples[-1], profile_clock_mhz)}
                profile_records.append(record)
                profile_log.write(json.dumps(record) + "\n")
                profile_log.flush()
            if iteration + 1 < iterations:
                release()
        print("RUNTIME_NPU_PHASE_DONE phase=runtime_batch_put")
        if profiling:
            runtime.stop_batch_put_profile()
            print_profile(profile_records, warmup, profile_clock_mhz)

        print("RUNTIME_NPU_PHASE_BEGIN phase=post_perf_validation")
        validate_batch(runtime, keys, rows, block_bytes)
        release()
        print("RUNTIME_NPU_PHASE_DONE phase=post_perf_validation")

        mean_ms = statistics.mean(samples)
        useful_gbps = rows * block_bytes / 1_000_000_000.0 / (mean_ms / 1000.0)
        print(
            "PERF_RESULT metric=runtime_batch_put_sync_call_to_return "
            f"samples={iterations} warmup_excluded={warmup} rows={rows} "
            f"block_bytes={block_bytes} npu_cores={npu_core_count} "
            f"mean_ms={mean_ms:.3f} "
            f"p50_ms={percentile(samples, 0.50):.3f} "
            f"p90_ms={percentile(samples, 0.90):.3f} "
            f"p99_ms={percentile(samples, 0.99):.3f} "
            f"useful_GBps={useful_gbps:.3f}"
            f" profiling={int(profiling)}"
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
        if profile_log is not None:
            profile_log.close()
        if runtime.initialized:
            if committed:
                runtime.batch_free(keys)
            runtime.shutdown()
        print("RUNTIME_NPU_PHASE_DONE phase=teardown")


if __name__ == "__main__":
    raise SystemExit(main())
