#!/usr/bin/env python3
"""Functional attempt: raw ACL stream plus retained kernel host arguments."""

from __future__ import annotations

import os

from test_npu_batch_get import main


def configure() -> None:
    defaults = {
        "KVAF_NPU_TABLE_CAPACITY": "8192",
        "KVAF_NPU_REQUESTS": "8192",
        "KVAF_NPU_OUTPUT_ROWS": "8192",
        "KVAF_NPU_BLOCK_BYTES": "512",
        "KVAF_NPU_SOURCE_STRIDE": "1024",
        "KVAF_NPU_CORE_COUNT": "4",
        "KVAF_NPU_ACTIVE_PERCENT": "5",
        "KVAF_NPU_WARMUP": "0",
        "KVAF_NPU_ITERATIONS": "1",
        "KVAF_NPU_GRAPH_REPLAYS": "1",
        "KVAF_NPU_GRAPH_WARMUP": "0",
        "KVAF_NPU_GRAPH_ITERATIONS": "1",
        "KVAF_NPU_GRAPH_STREAM_KIND": "acl",
        "KVAF_NPU_GRAPH_INPUT_PREFLIGHT_SYNC": "0",
        "KVAF_NPU_GRAPH_RETAIN_HOST_ARGS": "1",
    }
    for name, value in defaults.items():
        os.environ[name] = value


if __name__ == "__main__":
    configure()
    raise SystemExit(main())
