#!/usr/bin/env python3
"""Post-build import and ABI check for the in-tree KVAF Python package."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-aiv", action="store_true")
    parser.add_argument("--require-offload", action="store_true")
    options = parser.parse_args()

    import kvaf

    if options.require_offload:
        if not hasattr(kvaf, "OffloadRuntime") or any(
            not hasattr(kvaf.OffloadRuntime, name)
            for name in ("offload", "resolve_offload", "pending_offload", "last_checksum")
        ):
            raise RuntimeError("KVAF Offload binding is incomplete")
        for owner, names in (("HbmSource", ("register", "ready", "release_batch", "shutdown")),
                             ("NpuRuntime", ("offload", "offload_from", "offload_region", "resolve_offload"))):
            if not hasattr(kvaf, owner) or any(not hasattr(getattr(kvaf, owner), name) for name in names):
                raise RuntimeError(f"KVAF shared DDR binding is incomplete: {owner}")

    if options.require_aiv or options.require_offload:
        for owner, names in (("HbmReadSource", ("register", "listen", "ready", "release_batch", "shutdown")),
                             ("NpuRuntime", ("attach_hbm", "batch_get_raw_addr", "resolve_raw_addr", "detach_hbm"))):
            if not hasattr(kvaf, owner) or any(not hasattr(getattr(kvaf, owner), name) for name in names):
                raise RuntimeError(f"KVAF raw HBM READ binding is incomplete: {owner}")

    if options.require_aiv:
        if (
            not hasattr(kvaf, "NpuRuntime")
            or not hasattr(kvaf, "DeviceAccessTable")
            or not hasattr(kvaf, "AclGraph")
            or not hasattr(kvaf.NpuRuntime, "batch_free")
            or not hasattr(kvaf.NpuRuntime, "start_batch_get_kernel_profile")
            or not hasattr(kvaf.NpuRuntime, "batch_get_kernel_profile")
            or not hasattr(kvaf.NpuRuntime, "stop_batch_get_kernel_profile")
        ):
            raise RuntimeError(
                "KVAF AIV binding does not export NpuRuntime, DeviceAccessTable, "
                "AclGraph, BatchFree, and BatchGet kernel profiling"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
