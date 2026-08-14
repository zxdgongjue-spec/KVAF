#!/usr/bin/env python3
"""Post-build import and ABI check for the in-tree KVAF Python package."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-aiv", action="store_true")
    parser.add_argument("--require-dsa-row", action="store_true")
    options = parser.parse_args()

    import kvaf

    if not hasattr(kvaf, "DsaRowMockRuntime"):
        raise RuntimeError(
            "KVAF binding does not export the required DsaRowMockRuntime "
            "CPU reference storage API"
        )

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
    if options.require_dsa_row:
        if (
            not options.require_aiv
            or not hasattr(kvaf, "DsaRowRuntime")
            or kvaf.DsaRowRuntime.probe_dsa_row_payload_abi()
            != "torch_npu_packed_v1"
            or kvaf.DsaRowRuntime.probe_dsa_row_graph_direct_get_abi()
            != "graph_direct_get_v1"
        ):
            raise RuntimeError(
                "KVAF Direct-HCCP binding does not export the production "
                "DsaRowRuntime torch_npu_packed_v1 payload ABI and "
                "graph_direct_get_v1 graph ABI"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
