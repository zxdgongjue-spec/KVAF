#!/usr/bin/env python3
"""Post-build import and ABI check for the in-tree KVAF Python package."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-npu", "--require-aiv", dest="require_npu", action="store_true")
    parser.add_argument("--require-offload", action="store_true")
    parser.add_argument("--require-raw", action="store_true")
    options = parser.parse_args()

    import kvaf

    if options.require_npu:
        if (
            not hasattr(kvaf, "NpuRuntime")
            or not hasattr(kvaf, "DeviceAccessTable")
            or not hasattr(kvaf, "AclGraph")
            or not hasattr(kvaf.NpuRuntime, "batch_free")
            or not hasattr(kvaf.NpuRuntime, "batch_get")
            or not hasattr(kvaf.NpuRuntime, "graph_resources")
            or not hasattr(kvaf.NpuRuntime, "check_batch_get_status")
            or not hasattr(kvaf.NpuRuntime, "start_batch_get_kernel_profile")
            or not hasattr(kvaf.NpuRuntime, "batch_get_kernel_profile")
            or not hasattr(kvaf.NpuRuntime, "stop_batch_get_kernel_profile")
            or not hasattr(kvaf.NpuRuntime, "prepare_scatter_batch_get")
            or not hasattr(kvaf.NpuRuntime, "start_scatter_profile")
            or not hasattr(kvaf.NpuRuntime, "scatter_profile")
            or not hasattr(kvaf.NpuRuntime, "stop_scatter_profile")
        ):
            raise RuntimeError(
                "KVAF NPU binding does not export NpuRuntime, DeviceAccessTable, "
                "AclGraph, graph resources, BatchFree, BatchGet status checking and kernel profiling"
            )
    for required, exports, methods in (
        (options.require_offload, ("CpuPullOptions", "CpuPullHbmSource", "ReadyToken", "PutHandle"),
         ("initialize_ddr", "batch_cpu_pull", "resolve_cpu_pull", "prepare_put", "commit_put")),
        (options.require_raw, ("NpuPullOptions", "NpuPullHbmSource", "ReadyToken"),
         ("listen_hbm", "batch_get_raw_addr", "cancel_raw_addr", "resolve_raw_addr")),
    ):
        if required and (any(not hasattr(kvaf, name) for name in exports) or
                         any(not hasattr(kvaf.NpuRuntime, name) for name in methods)):
            raise RuntimeError("KVAF Pull binding is missing required public APIs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
