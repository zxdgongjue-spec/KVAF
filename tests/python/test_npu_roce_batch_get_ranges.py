#!/usr/bin/env python3
"""Hardware smoke test for RoCE reads and writes using one registered HBM allocation."""

import os
import sys
from time import perf_counter_ns


def report_latency(operation: str, started_ns: int, bytes_: int) -> None:
    elapsed_ms = (perf_counter_ns() - started_ns) / 1_000_000
    print(f"ROCE_RANGE_LATENCY op={operation} bytes={bytes_} elapsed_ms={elapsed_ms:.3f}",
          flush=True)


def main() -> int:
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("ROCE_RANGE_SMOKE_SKIP set KVAF_RUN_NPU_TESTS=1 to run")
        return 77

    required = ("KVAF_CPU_SERVICE_ADDR", "KVAF_ROCE_DEVICE", "KVAF_ROCE_GID_INDEX")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        print("ROCE_RANGE_SMOKE_FAIL missing " + ", ".join(missing), file=sys.stderr)
        return 2

    import kvaf
    import torch
    import torch_npu  # noqa: F401 — registers the NPU backend

    device_id = int(os.environ.get("DEVICE_ID", "0"))
    device = f"npu:{device_id}"
    torch.npu.set_device(device_id)
    stream = torch.npu.current_stream()

    def npu(values, dtype):
        return torch.tensor(values, dtype=dtype).to(device)

    runtime = kvaf.NpuRuntime(
        batch_get_token_bytes=656,
        batch_get_tokens_per_block=1,
        device_id=device_id,
        transport="roce",
        rdma_device=os.environ["KVAF_ROCE_DEVICE"],
        rdma_port=int(os.environ.get("KVAF_ROCE_PORT", "1")),
        gid_index=int(os.environ["KVAF_ROCE_GID_INDEX"]),
    )
    print("ROCE_RANGE_SMOKE_MODULE", kvaf.__file__, flush=True)
    handle = region = None
    try:
        runtime.link_to(os.environ["KVAF_CPU_SERVICE_ADDR"])
        block_bytes = 4 * 1024 * 1024
        runtime.initialize(8 * block_bytes, block_bytes)
        hbm_bytes = 64 * 1024 * 1024
        hbm = torch.empty(hbm_bytes, dtype=torch.uint8, device=device)
        started = perf_counter_ns()
        region = runtime.register_hbm(hbm)
        report_latency("HBM_REGISTER", started, hbm_bytes)
        print(f"ROCE_RANGE_HBM_REGISTERED bytes={region.bytes}", flush=True)

        # Use contiguous views at distinct nonzero offsets in the registered storage.
        source = hbm[4096:4096 + 2 * block_bytes]
        sync_output = hbm[hbm_bytes // 2:hbm_bytes // 2 + 12]
        async_output = hbm[hbm_bytes - 4096:hbm_bytes - 4096 + 12]
        consumed = hbm[hbm_bytes - 2048:hbm_bytes - 2048 + 12]
        source.zero_()
        source[block_bytes - 8:block_bytes].copy_(
            torch.tensor(list(range(10, 18)), dtype=torch.uint8))
        source[-8:].copy_(torch.tensor(list(range(20, 28)), dtype=torch.uint8))
        stream.synchronize()
        started = perf_counter_ns()
        runtime.batch_put([3, 7], source, [0, block_bytes], [block_bytes, block_bytes],
                          stream=stream, hbm_region=region)
        report_latency("BATCH_PUT", started, 2 * block_bytes)

        handle = runtime.prepare_access_table([7, 3])
        table = runtime.access_table_view(handle, device="npu")
        # Read the packages in reverse table order. The invalid suffix must be ignored.
        indices = npu([1, 0, -1], torch.int32)
        starts = npu([block_bytes - 6, block_bytes - 7, -1], torch.int64)
        lengths = npu([3, 4, -1], torch.int64)
        count = npu([2], torch.int32)
        targets = npu([6, 0, -1], torch.int64)
        expected = [21, 22, 23, 24, 165, 165, 12, 13, 14, 165, 165, 165]
        args = (table, indices, starts, lengths, count, targets)

        sync_output.fill_(0xA5)
        stream.synchronize()
        started = perf_counter_ns()
        runtime.batch_get_ranges(*args, sync_output, stream=stream, synchronize=True,
                                 hbm_region=region)
        report_latency("BATCH_GET_RANGES_SYNC", started, 7)
        actual = sync_output.cpu().tolist()
        if actual != expected:
            raise AssertionError(f"synchronous read: expected {expected}, got {actual}")
        print("ROCE_RANGE_SYNC_PASS", flush=True)

        async_output.fill_(0xA5)
        stream.synchronize()
        started = perf_counter_ns()
        runtime.batch_get_ranges(
            *args, async_output, stream=stream, synchronize=False, read_chunk_bytes=2,
            hbm_region=region,
        )
        consumed.copy_(async_output)
        runtime.check_batch_get_ranges_status(stream)
        report_latency("BATCH_GET_RANGES_ASYNC_CHUNKED", started, 7)
        actual = consumed.cpu().tolist()
        if actual != expected:
            raise AssertionError(f"async chunked read: expected {expected}, got {actual}")
        print("ROCE_RANGE_ASYNC_CHUNKED_HBM_PASS", flush=True)
    finally:
        try:
            if handle is not None:
                runtime.release_access_table(handle)
            if region is not None:
                runtime.unregister_hbm(region)
        finally:
            runtime.shutdown()

    print("ROCE_RANGE_SMOKE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
