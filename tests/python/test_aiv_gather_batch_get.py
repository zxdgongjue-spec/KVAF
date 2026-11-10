#!/usr/bin/env python3
"""Independent gather hardware regression; requires KVAF_RUN_NPU_TESTS=1."""
from aiv_batch_get_test_common import run

if __name__ == "__main__":
    raise SystemExit(run("gather_read"))
