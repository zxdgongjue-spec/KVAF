#!/usr/bin/env python3
"""Independent scatter hardware regression; honors KVAF_AIV_SCATTER_PROBE."""
from aiv_batch_get_test_common import run

if __name__ == "__main__":
    raise SystemExit(run("cpu_push_scatter"))
