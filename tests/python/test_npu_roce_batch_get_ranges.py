#!/usr/bin/env python3
"""Hardware smoke test for Host CPU RoCE variable-length reads."""

import os
import sys


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
        runtime.initialize(8 * 4096, 4096)
        source = npu(list(range(10, 18)) + list(range(20, 28)), torch.uint8)
        runtime.batch_put([3, 7], source, [0, 8], [8, 8], stream=stream)

        handle = runtime.prepare_access_table([7, 3])
        table = runtime.access_table_view(handle, device="npu")
        # Read the packages in reverse table order. The invalid suffix must be ignored.
        indices = npu([1, 0, -1], torch.int32)
        starts = npu([2, 1, -1], torch.int64)
        lengths = npu([3, 4, -1], torch.int64)
        count = npu([2], torch.int32)
        targets = npu([6, 0, -1], torch.int64)
        output = torch.empty(12, dtype=torch.uint8, device=device)
        expected = [21, 22, 23, 24, 165, 165, 12, 13, 14, 165, 165, 165]
        args = (table, indices, starts, lengths, count, targets, output)

        output.fill_(0xA5)
        runtime.batch_get_ranges(*args, stream=stream, synchronize=True)
        actual = output.cpu().tolist()
        if actual != expected:
            raise AssertionError(f"synchronous read: expected {expected}, got {actual}")
        print("ROCE_RANGE_SYNC_PASS", flush=True)

        region = runtime.register_hbm(output)
        output.fill_(0xA5)
        runtime.batch_get_ranges(
            *args, stream=stream, synchronize=False, read_chunk_bytes=2,
            hbm_region=region,
        )
        consumed = output.clone()
        runtime.check_batch_get_ranges_status(stream)
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
