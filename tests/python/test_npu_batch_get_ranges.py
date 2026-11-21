#!/usr/bin/env python3
"""Scheme B acceptance: packed variable tensors -> AIV Put -> AIV range Get.

Explicit opt-in is required. Exit 77 means not run, never hardware success.
See ../README_batch_get_ranges.md for setup, API and result interpretation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys


@dataclass(frozen=True)
class Entry:
    offset: int
    payload: bytes

    @property
    def length(self) -> int:
        return len(self.payload)


def pack(payloads: list[bytes], alignment: int) -> tuple[bytes, list[Entry]]:
    """Reference adapter: preserve payload lengths, pad only storage gaps."""
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    source = bytearray()
    entries = []
    for payload in payloads:
        if not payload:
            raise ValueError("zero-length tensors are not supported by V1")
        start = (len(source) + alignment - 1) // alignment * alignment
        source.extend(b"\xcc" * (start - len(source)))
        entries.append(Entry(start, payload))
        source.extend(payload)
    return bytes(source), entries


def pattern(identity: int, size: int) -> bytes:
    seed = bytes((identity * 131 + i * 17 + (i >> 3)) & 255 for i in range(4096))
    payload = bytearray(seed * (size // len(seed)) + seed[:size % len(seed)])
    # Distinct page headers catch reusing a prior chunk's source/output offset.
    for start in range(0, size, 4096):
        header = identity.to_bytes(4, "little") + (start // 4096).to_bytes(8, "little")
        count = min(len(header), size - start)
        payload[start:start + count] = header[:count]
    return bytes(payload)


def require_error(error_type, action, name: str) -> None:
    try:
        action()
    except error_type as error:
        print(f"RANGE_GET_CASE_PASS {name} expected={type(error).__name__}: {error}", flush=True)
    else:
        raise AssertionError(f"{name}: expected {error_type.__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=int(os.environ.get("DEVICE_ID", "0")))
    parser.add_argument("--cores", type=int, default=2)
    parser.add_argument("--jettys", type=int, default=2)
    parser.add_argument("--inflight", type=int, default=2048)
    parser.add_argument("--alignments", type=int, nargs="+", default=[1, 64, 512])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--read-chunk-bytes", type=int, default=0,
                        help="physical READ size; 0 removes the software split (device limits still apply)")
    parser.add_argument("--large-tensors", action="store_true",
                        help="also test 1MiB and 32MiB tensors; reserves 1GiB remote DDR")
    args = parser.parse_args()
    if args.iterations < 1 or any(a < 1 for a in args.alignments):
        parser.error("iterations and alignments must be positive")
    if not 0 <= args.read_chunk_bytes <= 0xffffffff:
        parser.error("read-chunk-bytes must be in [0, UINT32_MAX]")
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("RANGE_GET_SKIP: set KVAF_RUN_NPU_TESTS=1 on an NPU host")
        return 77

    # An opted-in run fails on missing dependencies or hardware; it cannot skip
    # silently and look like a passing hardware validation.
    import torch
    import torch_npu  # noqa: F401
    import kvaf
    from npu_eids import parse_npu_eids

    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable on an explicitly enabled hardware run")
    get_kernel = Path(os.environ["KVAF_BATCH_GET_KERNEL"])
    put_kernel = Path(os.environ["KVAF_BATCH_PUT_KERNEL"])
    if not get_kernel.is_file() or not put_kernel.is_file():
        raise FileNotFoundError("Get/Put kernel objects are required")
    peer = os.environ["KVAF_CPU_SERVICE_ADDR"]
    torch.npu.set_device(args.device)
    device = f"npu:{args.device}"
    runtime = kvaf.NpuRuntime(
        str(get_kernel), 656, 1, device_id=args.device, npu_core_count=args.cores,
        batch_put_kernel_object_path=str(put_kernel), jettys_per_core=args.jettys,
        transport="urma", aicpu_batch_get_mode="",
        batch_get_max_inflight_total=args.inflight,
        npu_eids=parse_npu_eids(os.environ.get("KVAF_NPU_EIDS")),
    )
    runtime.link_to(peer)
    slot_size = (64 if args.large_tensors else 1) * 1024 * 1024
    slot_count = 16
    table = None
    cases = 0

    def passed(name: str) -> None:
        nonlocal cases
        cases += 1
        print(f"RANGE_GET_CASE_PASS {name}", flush=True)

    def npu(values, dtype):
        return torch.tensor(values, dtype=dtype, device=device)

    def byte_tensor(payload: bytes):
        # Avoid millions of Python integers for MiB-scale input buffers.
        return torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(device)

    def raw(tensor) -> bytes:
        return tensor.cpu().contiguous().numpy().tobytes()

    try:
        runtime.initialize(slot_count * slot_size, slot_size)
        print(f"RANGE_GET_SETUP backend=aiv+urma peer={peer} cores={args.cores} "
              f"jettys={args.jettys} slot_size={slot_size} "
              f"read_chunk_bytes={args.read_chunk_bytes} large_tensors={args.large_tensors}", flush=True)
        oversized = torch.empty((slot_size + 1,), dtype=torch.uint8, device=device)
        require_error(ValueError, lambda: runtime.batch_put([12], oversized, [0], [slot_size + 1]),
                      "put_exceeds_slot")
        for alignment in args.alignments:
            sizes = [1, 7, 656, 1968, 65535, 65536, 65537, 150 * 1024]
            if args.large_tensors:
                sizes += [1024 * 1024, 32 * 1024 * 1024]
            packet_a, entries_a = pack([pattern(i + 1, s) for i, s in enumerate(sizes)], alignment)
            packet_b, entries_b = pack([pattern(i + 21, s) for i, s in enumerate(reversed(sizes))], alignment)
            if max(len(packet_a), len(packet_b)) > slot_size:
                raise ValueError("alignment makes the packet larger than one slot")
            source = byte_tensor(packet_a + packet_b)
            # Two packet Keys. Put offsets describe packets in source; Get
            # offsets below describe individual tensors inside each packet.
            runtime.batch_put([3, 7], source, [0, len(packet_a)], [len(packet_a), len(packet_b)])
            table = runtime.prepare_access_table([7, 3])
            host_table = runtime.access_table_view(table)
            if list(host_table["lengths"]) != [len(packet_b), len(packet_a)]:
                raise AssertionError("committed packet lengths differ from Put")
            device_table = runtime.access_table_view(table, device="npu", capacity=4)
            if device_table.lengths.cpu().tolist() != [len(packet_b), len(packet_a), 0, 0]:
                raise AssertionError("device lengths or padded table capacity incorrect")
            wi, ws, wl, wc, wd = (
                npu([0, 1], torch.int32), npu([0, 0], torch.int64),
                npu([len(packet_b), len(packet_a)], torch.int64),
                npu([2], torch.int32), npu([0, len(packet_b)], torch.int64))
            whole_output = torch.empty((len(packet_b) + len(packet_a),), dtype=torch.uint8, device=device)
            runtime.batch_get_ranges(device_table, wi, ws, wl, wc, wd, whole_output,
                                     synchronize=True, read_chunk_bytes=args.read_chunk_bytes)
            if raw(whole_output) != packet_b + packet_a:
                raise AssertionError("whole-packet Put/Get mismatch including padding")
            passed(f"put_and_metadata alignment={alignment}")

            # Whole logical tensors, deliberately out of order; repeat a tensor
            # to prove repeated indices and independent output positions work.
            reads = [(1, e) for e in reversed(entries_a)] + [(0, e) for e in entries_b] + [(1, entries_a[2])]
            destinations = [0] * len(reads)
            cursor = 13
            for i in reversed(range(len(reads))):
                destinations[i] = cursor
                cursor += reads[i][1].length + 11
            # Unused suffix contains invalid metadata and must be ignored.
            indices = npu([i for i, _ in reads] + [-1, -1], torch.int32)
            offsets = npu([e.offset for _, e in reads] + [-1, -1], torch.int64)
            lengths = npu([e.length for _, e in reads] + [-1, -1], torch.int64)
            targets = npu(destinations + [-1, -1], torch.int64)
            output_size = cursor + 19
            stream = torch.npu.current_stream()

            def buffers(active: int):
                count = npu([active], torch.int32)
                output = torch.full((output_size,), 0xa5, dtype=torch.uint8, device=device)
                expected = bytearray(b"\xa5" * output_size)
                for i in range(active):
                    payload = reads[i][1].payload
                    expected[destinations[i]:destinations[i] + len(payload)] = payload
                return count, output, bytes(expected)

            def submit(count, output, sync=False, on_stream=stream, **overrides):
                kwargs = dict(device_access_table=device_table, indices=indices,
                              source_offsets=offsets, lengths=lengths, request_count=count,
                              output_offsets=targets, output=output, stream=on_stream,
                              synchronize=sync, read_chunk_bytes=args.read_chunk_bytes)
                kwargs.update(overrides)
                return runtime.batch_get_ranges(**kwargs)

            for iteration in range(args.iterations):
                for active in [0, 1, 5, len(reads)]:
                    count, output, expected = buffers(active)
                    submit(count, output, sync=(iteration % 2 == 0))
                    # A Torch consumer is enqueued on the very same stream.
                    consumed = output.clone()
                    runtime.check_batch_get_ranges_status(stream)
                    if raw(consumed) != expected:
                        raise AssertionError(f"payload/guard mismatch alignment={alignment} active={active}")
                    passed(f"prefix_and_same_stream alignment={alignment} iteration={iteration} active={active}")

            # Producer -> transfer -> consumer across three streams. Table and
            # every tensor remain live until transfer status AND consumer end.
            transfer = torch.npu.Stream(device=args.device)
            consumer = torch.npu.Stream(device=args.device)
            count, output, expected = buffers(len(reads))
            ready, done = torch.npu.Event(), torch.npu.Event()
            ready.record(stream)
            transfer.wait_event(ready)
            submit(count, output, on_stream=transfer)
            done.record(transfer)
            consumer.wait_event(done)
            with torch.npu.stream(consumer):
                consumed = output.clone()
            runtime.check_batch_get_ranges_status(transfer)
            consumer.synchronize()
            if raw(consumed) != expected:
                raise AssertionError("cross-stream event dependency mismatch")
            passed(f"cross_stream_events alignment={alignment}")

            count, output, _ = buffers(1)
            require_error(ValueError, lambda: submit(count, output, sync=True, lengths=lengths.to(torch.int32)), "dtype")
            require_error(ValueError, lambda: submit(count, output, sync=True, lengths=lengths[:-1]), "capacity")
            require_error(ValueError, lambda: submit(count, output, sync=True, profile=True), "unsupported_profile")
            for invalid_chunk in [-1, 1 << 32]:
                require_error(ValueError, lambda: submit(count, output, sync=True, read_chunk_bytes=invalid_chunk),
                              f"invalid_chunk_{invalid_chunk}")
            for name, field, value in [
                ("negative_source", "source_offsets", -1),
                ("source_overflow", "source_offsets", (1 << 63) - 1),
                ("zero_length", "lengths", 0),
                ("length_overflow", "lengths", (1 << 63) - 1),
                ("negative_target", "output_offsets", -1),
                ("output_overflow", "output_offsets", output_size),
                ("invalid_index", "indices", 4),
                ("invalid_padded_entry", "indices", 3),
            ]:
                base = {"source_offsets": offsets, "lengths": lengths,
                        "output_offsets": targets, "indices": indices}[field]
                bad = base.clone()
                bad[0] = value
                # Device-side checking must surface through the explicit
                # completion check when synchronize=False.
                submit(count, output, **{field: bad})
                require_error(ValueError, lambda: runtime.check_batch_get_ranges_status(stream), name)
            for active in [-1, len(reads) + 3]:
                bad_count = npu([active], torch.int32)
                require_error(ValueError, lambda: submit(bad_count, output, sync=True), f"invalid_count_{active}")
            count, output, expected = buffers(len(reads))
            submit(count, output, sync=True)
            if raw(output) != expected:
                raise AssertionError("valid execution did not recover after argument errors")
            passed(f"argument_errors_and_recovery alignment={alignment}")

            # The table pins both packet Keys even after Get is complete.
            require_error(ValueError, lambda: runtime.batch_free([3]), "free_pinned_key")
            runtime.release_access_table(table)
            table = None
            require_error(KeyError, lambda: submit(count, output, sync=True), "released_table")
            runtime.batch_free([3, 7])
            # Reuse the same slot/Key with changed content, obtain a fresh table.
            replacement = pattern(77, 1968)
            replacement_source = byte_tensor(replacement)
            runtime.batch_put([3], replacement_source, [0], [len(replacement)])
            table = runtime.prepare_access_table([3])
            reused = runtime.access_table_view(table, device="npu")
            ri, ro, rl, rc, rd = (npu([0], torch.int32), npu([0], torch.int64),
                                  npu([len(replacement)], torch.int64), npu([1], torch.int32),
                                  npu([0], torch.int64))
            reuse_output = torch.empty((len(replacement),), dtype=torch.uint8, device=device)
            runtime.batch_get_ranges(reused, ri, ro, rl, rc, rd, reuse_output,
                                     synchronize=True, read_chunk_bytes=args.read_chunk_bytes)
            if raw(reuse_output) != replacement:
                raise AssertionError("free/reuse read stale content")
            runtime.release_access_table(table)
            table = None
            runtime.batch_free([3])
            passed(f"pin_release_free_reuse alignment={alignment}")

        # Legacy fixed-token API regression: its original layout remains valid.
        legacy = pattern(91, 656)
        legacy_source = byte_tensor(legacy)
        runtime.batch_put([9], legacy_source, [0], [656])
        table = runtime.prepare_access_table([9])
        legacy_table = runtime.access_table_view(table, device="npu")
        li, lo, lc, ld = (npu([0], torch.int32), npu([0], torch.int32),
                          npu([1], torch.int32), npu([0], torch.int32))
        legacy_output = torch.zeros((1, 656), dtype=torch.uint8, device=device)
        runtime.batch_get(legacy_table, li, lo, lc, ld, legacy_output, synchronize=True)
        if raw(legacy_output) != legacy:
            raise AssertionError("legacy fixed-token Get regression")
        runtime.release_access_table(table)
        table = None
        runtime.batch_free([9])
        passed("legacy_fixed_get")
    finally:
        # A failure still drains streams before releasing table/Runtime.
        torch.npu.synchronize()
        if table is not None:
            runtime.release_access_table(table)
        runtime.shutdown()
    print(f"RANGE_GET_HARDWARE_PASS cases={cases} alignments={args.alignments} "
          f"iterations={args.iterations} read_chunk_bytes={args.read_chunk_bytes} "
          f"large_tensors={args.large_tensors}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"RANGE_GET_HARDWARE_FAIL {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
