#!/usr/bin/env python3
"""Small end-to-end test for explicit RoCE HBM registration handles.

Run on an A5/CANN client with a separately running kvaf_roce_cpu_service.
Set KVAF_LOCAL_CPU_SERVICE_ADDR and KVAF_RUN_NPU_TESTS=1 to also exercise
local AICPU GatherRead in the same process with the RoCE session alive.
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


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5))
    return ordered[index]


def report(operation: str, samples_ms: list[float], batch_bytes: int) -> None:
    mean_ms = statistics.mean(samples_ms)
    gib_per_second = batch_bytes / (mean_ms / 1000.0) / (1024**3)
    print(
        f"PERF_RESULT operation={operation} samples={len(samples_ms)} "
        f"bytes_per_batch={batch_bytes} mean_ms={mean_ms:.3f} "
        f"p50_ms={percentile(samples_ms, 0.50):.3f} "
        f"p90_ms={percentile(samples_ms, 0.90):.3f} "
        f"GiBps={gib_per_second:.3f}"
    )


def main() -> int:
    backend = detect_torch_backend()
    if backend.name != "npu" or backend.torch is None:
        print(f"FAIL: this test requires an available torch_npu device ({backend.reason})",
              file=sys.stderr)
        return 2
    if not hasattr(kvaf, "NpuRuntime") or not hasattr(kvaf, "HbmRegion"):
        print("FAIL: rebuild the Python binding with RoCE support", file=sys.stderr)
        return 2

    service = os.environ.get("KVAF_CPU_SERVICE_ADDR")
    rdma_device = os.environ.get("KVAF_ROCE_DEVICE")
    gid_index = os.environ.get("KVAF_ROCE_GID_INDEX")
    if not service or not rdma_device or gid_index is None:
        print("FAIL: set KVAF_CPU_SERVICE_ADDR, KVAF_ROCE_DEVICE, and "
              "KVAF_ROCE_GID_INDEX", file=sys.stderr)
        return 2

    torch = backend.torch
    device_id = int(os.environ.get("DEVICE_ID", "0"))
    blocks = int(os.environ.get("KVAF_HBM_TEST_BLOCKS", "8"))
    token_bytes = int(os.environ.get("KVAF_HBM_TEST_TOKEN_BYTES", "4096"))
    tokens_per_block = int(os.environ.get("KVAF_HBM_TEST_TOKENS_PER_BLOCK", "1"))
    warmup = int(os.environ.get("KVAF_HBM_TEST_WARMUP", "1"))
    iterations = int(os.environ.get("KVAF_HBM_TEST_ITERATIONS", "5"))
    if min(blocks, token_bytes, tokens_per_block, iterations) <= 0 or warmup < 0:
        print("FAIL: block, token, and iteration counts must be positive", file=sys.stderr)
        return 2

    block_bytes = token_bytes * tokens_per_block
    ddr_bytes = blocks * block_bytes
    keys = list(range(blocks))
    input_offsets = [row * block_bytes for row in range(blocks)]
    lengths = [block_bytes] * blocks
    device = f"npu:{device_id}"

    torch.npu.set_device(device_id)
    expected_source = bytes(
        (row * 29 + byte * 7 + (byte >> 3)) & 0xFF
        for row in range(blocks)
        for byte in range(block_bytes)
    )
    source = torch.tensor(list(expected_source), dtype=torch.uint8).to(device).contiguous()
    output = torch.zeros((blocks, token_bytes), dtype=torch.uint8, device=device)
    indices = torch.arange(blocks, dtype=torch.int32, device=device)
    offsets = torch.zeros((blocks,), dtype=torch.int32, device=device)
    miss_count = torch.tensor([blocks], dtype=torch.int32, device=device)
    output_indices = torch.arange(blocks, dtype=torch.int32, device=device)
    expected_output = b"".join(
        expected_source[row * block_bytes:row * block_bytes + token_bytes]
        for row in range(blocks)
    )
    torch.npu.synchronize()

    runtime = kvaf.NpuRuntime(
        "",
        transport="roce",
        batch_get_token_bytes=token_bytes,
        batch_get_tokens_per_block=tokens_per_block,
        device_id=device_id,
        npu_core_count=1,
        rdma_device=rdma_device,
        rdma_port=int(os.environ.get("KVAF_ROCE_PORT", "1")),
        gid_index=int(gid_index),
    )
    runtime.link_to(service)

    source_region = None
    output_region = None
    table = None
    keys_committed = False
    try:
        runtime.initialize(ddr_bytes, block_bytes)
        source_region = runtime.register_hbm(source)
        output_region = runtime.register_hbm(output)
        print(
            f"HBM_REGION_REGISTERED source_bytes={source_region.bytes} "
            f"output_bytes={output_region.bytes} device={source_region.device_id}"
        )
        local_service = os.environ.get("KVAF_LOCAL_CPU_SERVICE_ADDR")
        if local_service:
            from test_npu_aicpu_gather_ranges import main as run_gather_ranges

            status = run_gather_ranges(["--peer", local_service, "--device", str(device_id)])
            if status != 0:
                return status

        write_samples: list[float] = []
        read_samples: list[float] = []
        total_calls = warmup + iterations
        for call in range(total_calls):
            started = time.perf_counter()
            runtime.batch_put(
                keys, source, input_offsets, lengths, hbm_region=source_region
            )
            write_ms = (time.perf_counter() - started) * 1000.0
            keys_committed = True

            table = runtime.prepare_access_table(keys)
            device_table = runtime.access_table_view(
                table, device="npu", capacity=blocks
            )
            started = time.perf_counter()
            runtime.batch_get(
                device_table,
                indices,
                offsets,
                miss_count,
                output_indices,
                output,
                synchronize=True,
                hbm_region=output_region,
            )
            read_ms = (time.perf_counter() - started) * 1000.0

            actual = output.cpu().contiguous().numpy().tobytes()
            if actual != expected_output:
                raise AssertionError(f"BatchGet payload mismatch on call {call}")

            runtime.release_access_table(table)
            table = None
            runtime.batch_free(keys)
            keys_committed = False

            if call >= warmup:
                write_samples.append(write_ms)
                read_samples.append(read_ms)

        print(f"HBM_REGION_DATA_VALIDATION_PASS blocks={blocks} block_bytes={block_bytes}")
        report("WRITE", write_samples, blocks * block_bytes)
        report("READ", read_samples, blocks * token_bytes)

        runtime.unregister_hbm(source_region)
        source_region = None
        runtime.unregister_hbm(output_region)
        output_region = None
        print("HBM_REGION_UNREGISTER_PASS")
        if local_service:
            print("AICPU_ROCE_COEXISTENCE_PASS")
        return 0
    finally:
        if table is not None:
            runtime.release_access_table(table)
        if keys_committed:
            runtime.batch_free(keys)
        if source_region is not None and not source_region.closed:
            runtime.unregister_hbm(source_region)
        if output_region is not None and not output_region.closed:
            runtime.unregister_hbm(output_region)
        if runtime.initialized:
            runtime.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
