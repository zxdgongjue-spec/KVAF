#!/usr/bin/env python3
"""Post-build import and ABI check for the in-tree KVAF Python package."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-aiv", action="store_true")
    parser.add_argument("--require-cpu-urma", action="store_true")
    options = parser.parse_args()

    import kvaf

    if options.require_aiv:
        if (
            not hasattr(kvaf, "NpuRuntime")
            or not hasattr(kvaf, "DeviceAccessTable")
            or not hasattr(kvaf, "AclGraph")
            or not hasattr(kvaf.NpuRuntime, "batch_free")
        ):
            raise RuntimeError(
                "KVAF AIV binding does not export NpuRuntime, DeviceAccessTable, "
                "AclGraph, and NpuRuntime.batch_free"
            )
    if options.require_cpu_urma:
        if (
            not hasattr(kvaf, "CpuRuntime")
            or not hasattr(kvaf.CpuRuntime, "batch_put")
            or not hasattr(kvaf.CpuRuntime, "last_batch_timing")
        ):
            raise RuntimeError(
                "KVAF CPU binding does not export CpuRuntime.batch_put and "
                "CpuRuntime.last_batch_timing"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
