#!/usr/bin/env python3
"""Post-build import and ABI check for the in-tree KVAF Python package."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-aiv", action="store_true")
    options = parser.parse_args()

    import kvaf

    if not all(hasattr(kvaf.Runtime, name) for name in (
        "initialize_pools", "create_pool", "bind_layer", "get_layer", "pool_info", "destroy_pool"
    )) or not hasattr(kvaf, "LayerHandle") or not hasattr(kvaf, "PoolHandle"):
        raise RuntimeError("KVAF binding is missing the multi-pool API")

    if options.require_aiv:
        if (
            not hasattr(kvaf, "NpuRuntime")
            or not hasattr(kvaf.NpuRuntime, "initialize_pools")
            or not hasattr(kvaf.NpuRuntime, "create_pool")
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
