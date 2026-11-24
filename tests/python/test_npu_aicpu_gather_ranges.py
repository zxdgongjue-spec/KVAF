#!/usr/bin/env python3
"""Exercise shared-slot range Get and the per-worker staging limit on NPU."""

import argparse
import os
import re

from npu_eids import parse_npu_eids
from test_npu_batch_get_ranges import pattern

MIB = 1024 * 1024
SLOT = 64 * MIB
TOKEN_BYTES = 656


def expect_value_error(action, message):
    try:
        action()
    except ValueError as error:
        if not re.search(message, str(error)):
            raise AssertionError(f"unexpected range Get error: {error}") from error
    else:
        raise AssertionError("expected range Get to raise ValueError")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=int(os.environ.get("DEVICE_ID", "0")))
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--peer", default=os.environ.get("KVAF_CPU_SERVICE_ADDR"))
    args = parser.parse_args(argv)
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("AICPU_GATHER_RANGES_SKIP: enable KVAF_RUN_NPU_TESTS=1 on the NPU target")
        return 77
    if not args.peer:
        parser.error("--peer or KVAF_CPU_SERVICE_ADDR is required")

    import torch
    import torch_npu  # noqa: F401
    import kvaf

    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    stream = torch.npu.current_stream()
    rt = kvaf.NpuRuntime("", TOKEN_BYTES, 1, device_id=args.device, npu_core_count=args.workers,
                         transport="urma", aicpu_batch_get_mode="gather_read",
                         npu_eids=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")))
    rt.link_to(args.peer)
    handle = fixed_handle = None
    # Reuse the registered allocations and keep them alive through shutdown.
    source = None
    output = None
    fixed_output = None
    request_tensors = []
    try:
        assert rt.allocation_mode == "append_only"
        rt.initialize(2 * SLOT, SLOT)
        range_keys = [101, 202, 303]
        fixed_key = 404
        fixed_payload = pattern(4, TOKEN_BYTES)
        payloads = [pattern(1, MIB + 17), pattern(2, 32 * MIB), pattern(3, 32 * MIB),
                    fixed_payload]
        sizes = [len(p) for p in payloads]
        offsets = [0, sizes[0], sizes[0] + sizes[1], sum(sizes[:3])]
        source = torch.frombuffer(bytearray(b"".join(payloads)), dtype=torch.uint8).to(device)
        rt.batch_put([*range_keys, fixed_key], source, offsets, sizes, stream=stream)
        handle = rt.prepare_access_table(range_keys)
        view = rt.access_table_view(handle)
        base = rt.remote_region["base"]
        assert list(view["addresses"]) == [base, base + sizes[0], base + SLOT]
        assert list(view["lengths"]) == sizes[:3]
        table = rt.access_table_view(handle, device="npu")
        output = torch.empty((SLOT + MIB,), dtype=torch.uint8, device=device)

        def tensor(values, dtype):
            return torch.tensor(values, dtype=dtype, device=device)

        def submit(requests, *, sync=True, active=None, chunk=0):
            # Reverse output order and leave guards so the byte-offset scatter
            # is exercised independently of the order of the source list.
            targets = [0] * len(requests)
            used = 7
            for i in reversed(range(len(requests))):
                targets[i] = used
                used += requests[i][2] + 3
            indices, offsets, lengths = zip(*requests)
            metadata = (tensor([*indices, -1], torch.int32),
                        tensor([*offsets, -1], torch.int64),
                        tensor([*lengths, -1], torch.int64),
                        tensor([len(requests) if active is None else active], torch.int32),
                        tensor([*targets, -1], torch.int64))
            request_tensors.append(metadata)
            output.fill_(0xa5)
            rt.batch_get_ranges(table, *metadata, output, stream=stream,
                                synchronize=sync, read_chunk_bytes=chunk)
            return targets, used

        def verify(requests, targets, used):
            expected = bytearray(b"\xa5") * used
            for (index, offset, length), target in zip(requests, targets):
                expected[target:target + length] = payloads[index][offset:offset + length]
            assert output[:used].cpu().numpy().tobytes() == expected

        # Exact capacity in the last worker. Other workers receive only three
        # bytes each to exercise uneven bytes with the same request count.
        small = [(0, i, 1) for i in range(3 * (args.workers - 1))]
        exact = small + [(1, 0, MIB), (1, MIB, 31 * MIB), (2, 0, 32 * MIB)]
        targets, used = submit(exact, sync=False)
        rt.check_batch_get_ranges_status(stream)
        verify(exact, targets, used)
        print("AICPU_GATHER_RANGES_CASE_PASS exact_lane_capacity", flush=True)

        overflow = small + [(1, 0, 32 * MIB), (2, 0, 32 * MIB), (0, 0, 1)]
        submit(overflow, sync=False)
        over_limit = (rf"staging capacity exceeded: worker={args.workers - 1} .*"
                      rf"required_at_least={SLOT + 1} capacity={SLOT}")
        expect_value_error(lambda: rt.check_batch_get_ranges_status(stream), over_limit)
        expect_value_error(lambda: rt.check_batch_get_ranges_status(stream), over_limit)
        print("AICPU_GATHER_RANGES_CASE_PASS lane_capacity_overflow", flush=True)

        # This range would read the first byte of the next Key in slot 0.
        expect_value_error(lambda: submit([(0, sizes[0], 1)]),
                           "invalid GatherRead range request")
        expect_value_error(lambda: submit([(0, 0, 1)], active=-1),
                           "invalid request_count")

        # A valid async launch after a failed synchronous call must work.
        # One logical range also spans multiple READs at 16 KiB.
        recovered = [(1, 7, MIB + 13), (0, 0, MIB + 17), (0, 2, 13)]
        targets, used = submit(recovered, sync=False, chunk=16 * 1024)
        rt.check_batch_get_ranges_status(stream)
        verify(recovered, targets, used)
        targets, used = submit([(0, 0, 1)], active=0)
        verify([], targets, used)
        print("AICPU_GATHER_RANGES_CASE_PASS bounds_empty_and_recovery", flush=True)

        # Fixed Get uses the same session, with a table of 656-byte entries.
        fixed_handle = rt.prepare_access_table([fixed_key])
        fixed_table = rt.access_table_view(fixed_handle, device="npu")
        zero = tensor([0], torch.int32)
        count = tensor([1], torch.int32)
        fixed_output = torch.empty((1, TOKEN_BYTES), dtype=torch.uint8, device=device)
        expect_value_error(lambda: rt.batch_get(table, zero, zero, count, zero, fixed_output,
                           stream=stream, synchronize=True), "entries do not match the configured layout")
        rt.batch_get(fixed_table, zero, zero, count, zero, fixed_output,
                     stream=stream, synchronize=True)
        assert fixed_output.cpu().numpy().tobytes() == fixed_payload
        print("AICPU_GATHER_RANGES_CASE_PASS fixed_get_after_ranges", flush=True)
    finally:
        stream.synchronize()
        if fixed_handle is not None:
            rt.release_access_table(fixed_handle)
        if handle is not None:
            rt.release_access_table(handle)
        rt.shutdown()
    print(f"AICPU_GATHER_RANGES_HARDWARE_PASS workers={args.workers}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
