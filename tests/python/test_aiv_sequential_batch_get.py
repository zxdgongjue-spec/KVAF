#!/usr/bin/env python3
"""Compatibility suite: run each mode in a separate process, even after failure."""
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    if os.environ.get("KVAF_RUN_NPU_TESTS") != "1":
        print("SKIP: set KVAF_RUN_NPU_TESTS=1 on the A5/CANN target")
        return 77
    scripts = {"cpu_push_scatter": "test_aiv_scatter_batch_get.py",
               "gather_read": "test_aiv_gather_batch_get.py"}
    modes = os.environ.get("KVAF_AIV_BATCH_GET_MODES", "cpu_push_scatter,gather_read").split(",")
    if not all(mode in scripts for mode in modes):
        print("FAIL: unknown KVAF_AIV_BATCH_GET_MODES", file=sys.stderr)
        return 2
    failed = False
    for mode in modes:
        script = Path(__file__).with_name(scripts[mode])
        result = subprocess.run([sys.executable, str(script)], check=False)
        print(f"[KVAF_TEST] mode={mode} exit_code={result.returncode}", flush=True)
        failed |= result.returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
