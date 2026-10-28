#!/usr/bin/env python3
"""Prefill/Decode roles for the real URMA shared-allocation experiment."""
from __future__ import annotations
import argparse
import ctypes as C
import ipaddress
import json
import os
import socket
import sys
import time
from pathlib import Path

GUARD = 4096
INFO, ARM, EXPORT, PULL, ACK, CLOSE, FAIL = range(1, 8)


class Message(C.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [(n, C.c_uint32) for n in ("magic", "version", "op")] + [
        ("status", C.c_int32)] + [(n, C.c_uint64) for n in
        ("base", "bytes", "round", "ack", "tp")] + [(n, C.c_uint32) for n in
        ("psn", "token", "key_bytes", "segment_bytes")] + [
        ("eid", C.c_uint8 * 16), ("key", C.c_uint8 * 64),
        ("segment", C.c_uint8 * 64), ("error", C.c_char * 192)]

    def __init__(self, op=INFO, **kwargs):
        super().__init__(magic=0x44525042, version=1, op=op, **kwargs)


assert C.sizeof(Message) == 408


def rpc(args, message):
    host, port = args.control.rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=args.timeout) as sock:
        sock.settimeout(args.timeout)
        sock.sendall(bytes(message))
        data = bytearray()
        while len(data) < C.sizeof(Message):
            chunk = sock.recv(C.sizeof(Message) - len(data))
            if not chunk:
                raise ConnectionError("K closed the control connection")
            data.extend(chunk)
    reply = Message.from_buffer_copy(data)
    if (reply.magic, reply.version, reply.op) != (message.magic, message.version, message.op):
        raise RuntimeError("control protocol mismatch")
    if reply.status < 0:
        raise RuntimeError(bytes(reply.error).decode(errors="replace"))
    return reply


def wait_for(args, condition):
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        reply = rpc(args, Message())
        if condition(reply):
            return reply
        time.sleep(0.1)
    raise TimeoutError("waiting for peer state; K resources remain allocated")


def payload(size, round_id):
    import numpy as np
    with np.errstate(over="ignore"):
        x = np.arange(size, dtype=np.uint64) ^ np.uint64(
            (round_id * 0x9e3779b97f4a7c15) & ((1 << 64) - 1))
        x = (x ^ (x >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> 27)) * np.uint64(0x94d049bb133111eb)
        return (x ^ (x >> 31)).astype(np.uint8)


def eid(value):
    compact = value.replace("0x", "").replace(":", "")
    raw = bytes.fromhex(compact) if len(compact) == 32 else ipaddress.IPv6Address(value).packed
    if len(raw) != 16 or not any(raw):
        raise ValueError("EID must contain 16 non-all-zero bytes")
    return (C.c_uint8 * 16).from_buffer_copy(raw)


def endpoint(value, option):
    """Validate endpoints without connecting to or changing the K service."""
    try:
        host, port_text = value.rsplit(":", 1)
        port = int(port_text)
        if not host or not 1 <= port <= 65535:
            raise ValueError
    except (AttributeError, ValueError):
        raise ValueError(f"{option} must be host:port with a port in 1..65535") from None
    host = host.strip("[]").lower().rstrip(".")
    if host in ("localhost", "127.0.0.1", "::1"):
        host = "loopback"
    return host, port


def input_file(value, option):
    if not value:
        raise ValueError(f"{option} is required")
    filename = Path(value).expanduser().resolve()
    if not filename.is_file():
        raise ValueError(f"{option} file does not exist: {filename}")
    if not os.access(filename, os.R_OK):
        raise ValueError(f"{option} file is not readable: {filename}")
    return str(filename)


def decode_runtime_options(args):
    extra = {}
    if args.runtime_options:
        extra = json.loads(Path(args.runtime_options).read_text())
        if not isinstance(extra, dict):
            raise ValueError("--runtime-options must contain a JSON object")
    for name in ("batch_get_token_bytes", "batch_get_tokens_per_block"):
        if name in extra:
            raise ValueError(f"--runtime-options cannot override probe layout field {name}")
    # Explicit CLI paths override JSON, which overrides environment defaults.
    get_default = extra.pop("batch_get_kernel_object_path", os.environ.get("KVAF_BATCH_GET_KERNEL"))
    put_default = extra.pop("batch_put_kernel_object_path", os.environ.get("KVAF_BATCH_PUT_KERNEL"))
    args.get_kernel = input_file(args.get_kernel if args.get_kernel is not None else get_default,
                                 "--get-kernel / KVAF_BATCH_GET_KERNEL")
    args.put_kernel = input_file(args.put_kernel if args.put_kernel is not None else put_default,
                                 "--put-kernel / KVAF_BATCH_PUT_KERNEL (required for RMA link validation)")
    kwargs = dict(device_id=args.device, npu_core_count=1, jettys_per_core=1,
                  npu_eids=[bytes(eid(args.eid)).hex()], peer_ub_device=args.peer_ub_device)
    kwargs.update(extra)
    kwargs["batch_put_kernel_object_path"] = args.put_kernel
    return kwargs


def prefill(args, torch, device):
    info = wait_for(args, lambda m: m.base != 0)
    if info.bytes != args.allocation_bytes:
        raise ValueError("K/Prefill allocation sizes differ")
    lib = C.CDLL(str(Path(args.exporter).resolve()))
    lib.probe_hbm_error.restype = C.c_char_p
    lib.probe_hbm_export.argtypes = [C.c_uint64, C.c_uint64, C.POINTER(C.c_uint8),
                                    C.POINTER(C.c_uint8), C.POINTER(Message)]
    lib.probe_hbm_export.restype = C.c_void_p
    lib.probe_hbm_bind.argtypes = [C.c_void_p, C.POINTER(Message)]
    lib.probe_hbm_bind.restype = C.c_int
    lib.probe_hbm_close.argtypes = [C.c_void_p]
    lib.probe_hbm_close.restype = C.c_int
    # Keep the allocation and exporter alive, including in the failure handler.
    source = torch.empty(args.allocation_bytes, dtype=torch.uint8, device=device)
    args._retained = [source, lib]
    source.fill_(0xA5)
    torch.npu.synchronize()
    descriptor = Message(EXPORT)
    handle = lib.probe_hbm_export(source.data_ptr(), source.numel(), eid(args.eid),
                                 info.eid, C.byref(descriptor))
    if not handle:
        raise RuntimeError(lib.probe_hbm_error().decode())
    args._retained.append(handle)
    # Keep the full keys as well as their EID prefixes: the prefixes follow the
    # expected UMDK layout, while raw bytes also help identify a deployed ABI
    # difference. Do not rewrite either key to make an identity check pass.
    segment_key = bytes(descriptor.segment)[:descriptor.segment_bytes]
    jetty_key = bytes(descriptor.key)[:descriptor.key_bytes]
    print(json.dumps({"role": "prefill", "event": "HBM_EXPORT",
                      "logical_device": args.device,
                      "configured_eid": bytes(descriptor.eid).hex(),
                      "segment_eid_prefix": segment_key[:16].hex(),
                      "jetty_eid_prefix": jetty_key[:16].hex(),
                      "segment_bytes": descriptor.segment_bytes,
                      "jetty_key_bytes": descriptor.key_bytes,
                      "segment_key": segment_key.hex(),
                      "jetty_key": jetty_key.hex()}), flush=True)
    reply = rpc(args, descriptor)
    if lib.probe_hbm_bind(handle, C.byref(reply)):
        raise RuntimeError(lib.probe_hbm_error().decode())
    for round_id in range(1, args.rounds + 1):
        source[GUARD:-GUARD].copy_(torch.from_numpy(payload(args.payload_bytes, round_id)).to(device))
        # Producer completion precedes publishing the source for K's READ.
        torch.npu.synchronize()
        rpc(args, Message(PULL, round=round_id))
        wait_for(args, lambda m: m.ack == round_id)
    # K first destroys its queues and imported HBM segment, then source can go.
    rpc(args, Message(CLOSE))
    if lib.probe_hbm_close(handle):
        raise RuntimeError(lib.probe_hbm_error().decode())
    print(json.dumps({"role": "prefill", "result": "PASS", "rounds": args.rounds}), flush=True)


def decode(args, torch, device):
    import numpy as np
    import kvaf
    runtime = kvaf.NpuRuntime(args.get_kernel, args.token_bytes, 1, **args.runtime_kwargs)
    args._retained = [runtime]
    runtime.link_to(args.cpu)
    # 4096-byte slots keep guarded allocation exact even for 656-byte tokens.
    runtime.initialize(args.allocation_bytes, 4096)
    region = dict(runtime.remote_region)
    info = rpc(args, Message(ARM, base=region["base"], bytes=args.allocation_bytes))
    if info.base != region["base"]:
        raise AssertionError("probe and Decode do not use the same exported allocation")
    rows = args.payload_bytes // args.token_bytes
    order = np.random.default_rng(42).permutation(rows)
    destinations = np.arange(rows - 1, -1, -1, dtype=np.int32) + 1
    # A single table base plus byte offsets includes every token, including page crossings.
    addresses = torch.tensor([info.base + GUARD], dtype=torch.int64, device=device)
    valid = torch.ones(1, dtype=torch.uint8, device=device)
    indices = torch.zeros(rows, dtype=torch.int32, device=device)
    offsets = torch.tensor(order * args.token_bytes, dtype=torch.int32, device=device)
    output_indices = torch.tensor(destinations, dtype=torch.int32, device=device)
    output = torch.empty((rows + 2, args.token_bytes), dtype=torch.uint8, device=device)
    # Keep these alive if a failed kernel still holds their addresses.
    args._retained.extend([addresses, valid, indices, offsets, output_indices, output])
    for round_id in range(1, args.rounds + 1):
        wait_for(args, lambda m: m.round == round_id)
        expected_source = payload(args.payload_bytes, round_id).reshape(rows, args.token_bytes)
        # Full, empty and partial active prefixes, all on the same DDR round.
        for active in (rows, 0, max(1, rows // 3)):
            expected = np.full((rows + 2, args.token_bytes), 0xD3, dtype=np.uint8)
            expected[destinations[:active]] = expected_source[order[:active]]
            output.fill_(0xD3)
            count = torch.tensor([active], dtype=torch.int32, device=device)
            args._retained.append(count)
            runtime.batch_get(addresses, valid, indices, offsets, count,
                              output_indices, output, synchronize=True)
            # Device-side consumer checks visibility before the D2H comparison.
            consumed = torch.bitwise_xor(output, 0x5A)
            args._retained.append(consumed)
            actual = consumed.cpu().numpy() ^ np.uint8(0x5A)
            wrong = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
            if wrong.size:
                row, byte = divmod(int(wrong[0]), args.token_bytes)
                raise AssertionError(f"round={round_id} active={active} row={row} byte={byte} "
                    f"expected={expected[row, byte]} actual={actual[row, byte]} bad_bytes={wrong.size}")
            args._retained.pop()
            args._retained.pop()
        # Only now does K read its CPU mapping, so it cannot conceal a first-read
        # DMA visibility problem by touching the payload before BatchGet.
        rpc(args, Message(ACK, round=round_id))
        print(json.dumps({"role": "decode", "round": round_id, "result": "PASS",
                          "ddr_base": hex(info.base), "payload_bytes": args.payload_bytes}), flush=True)
    wait_for(args, lambda m: m.status == 1)
    runtime.shutdown()
    print(json.dumps({"role": "decode", "result": "ALL_PASS", "rounds": args.rounds}), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["prefill", "decode"])
    parser.add_argument("--control", required=True, help="K diagnostic IPv4:port")
    parser.add_argument("--cpu", help="K KVAF IPv4:port (Decode)")
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--eid", required=True, help="this NPU's reachable EID")
    parser.add_argument("--peer-ub-device", default="")
    parser.add_argument("--exporter", help="libkvaf_probe_hbm.so (Prefill)")
    parser.add_argument("--get-kernel", help="BatchGet kernel path; defaults to KVAF_BATCH_GET_KERNEL")
    parser.add_argument("--put-kernel", help="BatchPut kernel for link validation; defaults to KVAF_BATCH_PUT_KERNEL")
    parser.add_argument("--runtime-options", help="JSON object of extra NpuRuntime constructor arguments")
    parser.add_argument("--token-bytes", type=int, default=656)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    args.payload_bytes = args.token_bytes * args.rows
    args.allocation_bytes = args.payload_bytes + 2 * GUARD
    if (args.rows < 3 or args.token_bytes <= 0 or args.rounds < 1 or args.timeout <= 0 or
            args.allocation_bytes % 4096 or args.allocation_bytes > 1 << 30):
        parser.error("rows>=3, positive sizes/rounds/timeout, total allocation multiple of 4096 and <=1GiB required")
    if args.role == "prefill" and not args.exporter:
        parser.error("Prefill requires --exporter")
    if args.role == "decode" and not args.cpu:
        parser.error("Decode requires --cpu")
    if sys.byteorder != "little":
        parser.error("the deployed probe ABI requires little-endian hosts")
    # Local argument/file errors must exit before NPU allocation or any RPC.
    # They must not send FAIL and poison another client's K probe session.
    try:
        control = endpoint(args.control, "--control")
        eid(args.eid)
        if args.role == "decode":
            if endpoint(args.cpu, "--cpu") == control:
                raise ValueError("--cpu and --control must use different service endpoints: "
                                 "normally --cpu K_IP:29990 and --control K_IP:29991")
            args.runtime_kwargs = decode_runtime_options(args)
        else:
            args.exporter = input_file(args.exporter, "--exporter")
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    return args


def main():
    args = parse_args()
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_device(args.device)
    try:
        (prefill if args.role == "prefill" else decode)(args, torch, f"npu:{args.device}")
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            rpc(args, Message(FAIL))
        except Exception:
            pass
        print("FAILED: resources retained. Stop the K probe first, then interrupt this process.", flush=True)
        # Do not let Tensor destruction race a timed-out DMA/kernel operation.
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            os._exit(2)


if __name__ == "__main__":
    main()
