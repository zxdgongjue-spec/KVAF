#!/usr/bin/env python3
"""Post-build import and ABI check for the in-tree KVAF Python package."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-aiv", action="store_true")
    options = parser.parse_args()

    import kvaf

    if options.require_aiv:
        if not hasattr(kvaf, "AclStream"):
            raise RuntimeError("KVAF AIV binding does not export AclStream")
        constructor_doc = getattr(kvaf.NpuRuntime.__init__, "__doc__", "")
        if "retain_unsynchronized_launch_args" not in constructor_doc:
            raise RuntimeError(
                "KVAF AIV binding has a stale NpuRuntime constructor ABI"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
