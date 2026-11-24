#!/usr/bin/env python3
"""Hardware acceptance of default append_only and explicit fixed allocation."""
import argparse
import os
from pathlib import Path
import sys

from npu_eids import parse_npu_eids
from test_npu_batch_get_ranges import pattern


def expect(error_type, action, message):
    try:
        action()
    except error_type as error:
        if message not in str(error):
            raise AssertionError(f"unexpected {type(error).__name__}: {error}") from error
    else:
        raise AssertionError(f"expected {error_type.__name__}: {message}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=int, default=int(os.environ.get("DEVICE_ID", "0")))
    args = p.parse_args()
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("APPEND_ONLY_SKIP: enable KVAF_RUN_NPU_TESTS=1 on the NPU target")
        return 77
    import torch
    import torch_npu  # noqa: F401
    import kvaf
    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable for enabled hardware acceptance")
    get, put = Path(os.environ["KVAF_BATCH_GET_KERNEL"]), Path(os.environ["KVAF_BATCH_PUT_KERNEL"])
    if not get.is_file() or not put.is_file():
        raise FileNotFoundError("matching Get/Put objects required")
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    slot = 1024*1024
    stream = torch.npu.current_stream()

    def runtime(mode=None):
        options = {} if mode is None else dict(allocation_mode=mode)
        rt = kvaf.NpuRuntime(str(get), 656, 1, device_id=args.device,
            npu_core_count=2, jettys_per_core=2, transport="urma",
            batch_put_kernel_object_path=str(put),
            npu_eids=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")), **options)
        rt.link_to(os.environ["KVAF_CPU_SERVICE_ADDR"])
        rt.initialize(2*slot, slot)
        return rt

    def put_objects(rt, keys, payloads):
        offsets, cursor = [], 0
        for data in payloads:
            offsets.append(cursor)
            cursor += len(data)
        source = torch.frombuffer(bytearray(b"".join(payloads)), dtype=torch.uint8).to(device)
        rt.batch_put(keys, source, offsets, [len(data) for data in payloads], stream=stream)

    def read_objects(rt, keys, payloads, expected_addresses=None):
        handle = rt.prepare_access_table(keys)
        try:
            view = rt.access_table_view(handle)
            if list(view["lengths"]) != [len(data) for data in payloads]:
                raise AssertionError("actual Key lengths mismatch")
            if expected_addresses is not None and list(view["addresses"]) != expected_addresses:
                raise AssertionError(f"allocation address mismatch: {view['addresses']}")
            table = rt.access_table_view(handle, device="npu")
            def tensor(values, dtype):
                return torch.tensor(values, dtype=dtype, device=device)
            targets, cursor = [], 0
            for data in payloads:
                targets.append(cursor)
                cursor += len(data)
            indices = tensor(list(range(len(keys))), torch.int32)
            starts = tensor([0]*len(keys), torch.int64)
            lengths = tensor([len(data) for data in payloads], torch.int64)
            count = tensor([len(keys)], torch.int32)
            destinations = tensor(targets, torch.int64)
            output = torch.full((cursor+19,), 0xa5, dtype=torch.uint8, device=device)
            rt.batch_get_ranges(table, indices, starts, lengths, count, destinations, output,
                                stream=stream, synchronize=True)
            if output.cpu().numpy().tobytes() != b"".join(payloads) + b"\xa5"*19:
                raise AssertionError("range Get payload/guard mismatch")
        finally:
            rt.release_access_table(handle)

    # Explicit fixed mode retains the bounded DSA Key/slot mapping.
    rt = runtime("fixed")
    try:
        assert rt.allocation_mode == "fixed"
        data = pattern(1, 656)
        put_objects(rt, [1], [data])
        read_objects(rt, [1], [data], [rt.remote_region["base"]+slot])
        rt.batch_free([1])
        expect(ValueError, lambda: put_objects(rt, [1000], [data]), "slot ID")
        print("APPEND_ONLY_CASE_PASS explicit_fixed", flush=True)
    finally:
        stream.synchronize()
        rt.shutdown()

    rt = runtime()
    pinned = None
    try:
        assert rt.allocation_mode == "append_only"
        base = rt.remote_region["base"]
        small = [pattern(21,656), pattern(22,1968), pattern(23,37)]
        put_objects(rt, [10001,10002,10003], small)
        read_objects(rt, [10002,10003,10001], [small[1],small[2],small[0]],
                     [base+656,base+2624,base])
        rt.batch_free([10001,10002,10003])
        print("APPEND_ONLY_CASE_PASS dense_unaligned_independent_keys", flush=True)
        a, b, c = pattern(11, slot//4), pattern(12, slot//2), pattern(13, slot*3//8)
        put_objects(rt, [101,202,303], [a,b,c])
        read_objects(rt, [202,101,303], [b,a,c], [base+len(a),base,base+slot])
        print("APPEND_ONLY_CASE_PASS multiple_keys_and_rollover", flush=True)
        pinned = rt.prepare_access_table([101,202])
        expect(ValueError, lambda: rt.batch_free([202]), "pinned")
        # The first item could fit, the second cannot. Neither reservation may
        # remain after PreparePut returns OOM.
        expect(RuntimeError, lambda: put_objects(rt, [404,505],
            [pattern(14,slot//2), pattern(15,slot//4)]), "no empty slot")
        expect(KeyError, lambda: rt.prepare_access_table([404]), "Key not found")
        d = pattern(16, slot-len(c))
        put_objects(rt, [404], [d])
        read_objects(rt, [404], [d], [base+slot+len(c)])
        rt.release_access_table(pinned)
        pinned = None
        print("APPEND_ONLY_CASE_PASS atomic_oom_and_pinned_append", flush=True)
        rt.batch_free([202])
        expect(KeyError, lambda: rt.prepare_access_table([202]), "Key not found")
        expect(RuntimeError, lambda: put_objects(rt, [505], [b"x"]), "no empty slot")
        read_objects(rt, [101,303,404], [a,c,d])
        print("APPEND_ONLY_CASE_PASS deleted_hole_not_reused", flush=True)
        rt.batch_free([101])
        e = pattern(17,slot)
        put_objects(rt, [606], [e])
        read_objects(rt, [606,303,404], [e,c,d], [base,base+slot,base+slot+len(c)])
        rt.batch_free([303])
        expect(RuntimeError, lambda: put_objects(rt, [707], [b"x"]), "no empty slot")
        rt.batch_free([404])
        f = pattern(18,slot)
        put_objects(rt, [707], [f])
        read_objects(rt, [606,707], [e,f], [base,base+slot])
        rt.batch_free([606,707])
        print("APPEND_ONLY_CASE_PASS whole_slot_reuse_neighbor_payload_intact", flush=True)
    finally:
        stream.synchronize()
        if pinned is not None:
            rt.release_access_table(pinned)
        rt.shutdown()
    print("APPEND_ONLY_HARDWARE_PASS fixed_explicit=1 append_only_default=1", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"APPEND_ONLY_HARDWARE_FAIL {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
