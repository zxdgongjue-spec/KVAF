#!/usr/bin/env python3
"""Measure NpuRuntime initialization without loading a model or filling the DDR pool."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import time


GIB = 1024**3
TOKEN_BYTES = 656
TOKENS_PER_BLOCK = 128
SLOT_ALIGNMENT = 4096
DEFAULT_SLOT_SIZE = (
    (TOKEN_BYTES * TOKENS_PER_BLOCK + SLOT_ALIGNMENT - 1) // SLOT_ALIGNMENT
) * SLOT_ALIGNMENT


def log(event: str, device: int, **fields: object) -> None:
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": event,
        "pid": os.getpid(),
        "device": device,
        **fields,
    }
    print("DDR_INIT_PROBE " + json.dumps(record, sort_keys=True), flush=True)


def timed_call(operation: str, device: int, callback):
    log("begin", device, operation=operation)
    started = time.perf_counter()
    try:
        result = callback()
    except BaseException:
        elapsed = time.perf_counter() - started
        log("failed", device, operation=operation, elapsed_s=round(elapsed, 6))
        raise
    elapsed = time.perf_counter() - started
    log("done", device, operation=operation, elapsed_s=round(elapsed, 6))
    return result


def interrupt(_signum, _frame) -> None:
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True, help="Matching KVAF NPU build directory")
    parser.add_argument("--address", default="127.0.0.1:34555")
    parser.add_argument("--device", type=int, default=0, help="Logical NPU device ID")
    parser.add_argument("--ddr-gib", type=int, required=True, help="Requested GiB per process, rounded up to whole slots")
    parser.add_argument("--slot-size", type=int, default=DEFAULT_SLOT_SIZE, help="Bytes per slot; default 86016 matches the local vLLM packed layout")
    parser.add_argument("--aiv-cores", type=int, default=1)
    parser.add_argument("--jettys-per-core", type=int, choices=(1, 2), default=2)
    parser.add_argument("--peer-ub-device", default="", help="Optional known-working CPU UB device")
    parser.add_argument("--npu-eids", default="", help="Optional comma-separated known-working NPU EIDs")
    parser.add_argument("--batch-get-kernel", type=Path)
    parser.add_argument("--batch-put-kernel", type=Path)
    parser.add_argument("--hold-seconds", type=float, default=0, help="Keep successful regions alive after timing, then release")
    parser.add_argument("--hold", action="store_true", help="Keep the region until Enter; use in a foreground terminal")
    parser.add_argument("--dry-run", action="store_true", help="Print sizing/configuration without importing KVAF or using an NPU")
    args = parser.parse_args()
    if args.device < 0 or args.ddr_gib <= 0 or args.aiv_cores <= 0:
        parser.error("device must be nonnegative; ddr-gib and aiv-cores must be positive")
    if args.slot_size < TOKEN_BYTES * TOKENS_PER_BLOCK or args.slot_size % SLOT_ALIGNMENT:
        parser.error("slot-size must fit one 656 x 128 block and be a multiple of 4096")
    if not 0 <= args.hold_seconds < float("inf"):
        parser.error("hold-seconds must be finite and nonnegative")
    if args.hold and args.hold_seconds:
        parser.error("choose either --hold or --hold-seconds")

    build = args.build.resolve()
    get_kernel = (args.batch_get_kernel or build / "bin/kvaf_execute_batch_get_kernels.o").resolve()
    put_kernel = (args.batch_put_kernel or build / "bin/kvaf_execute_batch_put_kernels.o").resolve()
    requested = args.ddr_gib * GIB
    ddr_bytes = ((requested + args.slot_size - 1) // args.slot_size) * args.slot_size
    npu_eids = [eid.strip() for eid in args.npu_eids.split(",") if eid.strip()]
    log(
        "config", args.device,
        address=args.address, requested_gib=args.ddr_gib,
        requested_bytes=requested, allocated_bytes=ddr_bytes,
        allocated_gib=ddr_bytes / GIB, slot_size=args.slot_size,
        slot_count=ddr_bytes // args.slot_size,
        aiv_cores=args.aiv_cores, jettys_per_core=args.jettys_per_core,
        peer_ub_device=args.peer_ub_device or "auto", npu_eids=npu_eids,
        build=str(build), batch_get_kernel=str(get_kernel), batch_put_kernel=str(put_kernel),
    )
    if args.dry_run:
        return 0
    for kernel in (get_kernel, put_kernel):
        if not kernel.is_file():
            parser.error(f"kernel not found: {kernel}")
    if not (build / "python/kvaf").is_dir():
        parser.error(f"KVAF Python package not found in {build / 'python'}")
    sys.path.insert(0, str(build / "python"))
    import torch
    import torch_npu  # noqa: F401 -- registers torch.npu
    import kvaf

    if not hasattr(kvaf, "NpuRuntime"):
        parser.error("the selected KVAF build does not provide NpuRuntime")
    signal.signal(signal.SIGTERM, interrupt)
    runtime = None
    interrupted = False
    try:
        timed_call("set_device", args.device, lambda: torch.npu.set_device(args.device))
        runtime = timed_call("construct", args.device, lambda: kvaf.NpuRuntime(
            str(get_kernel), batch_get_token_bytes=TOKEN_BYTES,
            batch_get_tokens_per_block=TOKENS_PER_BLOCK,
            device_id=args.device, npu_core_count=args.aiv_cores,
            batch_put_kernel_object_path=str(put_kernel),
            jettys_per_core=args.jettys_per_core,
            peer_ub_device=args.peer_ub_device, npu_eids=npu_eids,
        ))
        timed_call("link_to", args.device, lambda: runtime.link_to(args.address))
        timed_call("initialize", args.device, lambda: runtime.initialize(ddr_bytes, args.slot_size))
        region = runtime.remote_region
        log("ready", args.device, remote_region=region)
        if int(region["length"]) != ddr_bytes:
            raise RuntimeError(f"unexpected DDR length: {region['length']} != {ddr_bytes}")
        if args.hold:
            log("hold", args.device, until="Enter")
            input("Region is live. Press Enter to release it: ")
        elif args.hold_seconds:
            log("hold", args.device, seconds=args.hold_seconds)
            time.sleep(args.hold_seconds)
    except KeyboardInterrupt:
        interrupted = True
        log("interrupted", args.device)
    finally:
        if runtime is not None:
            timed_call("shutdown", args.device, runtime.shutdown)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
