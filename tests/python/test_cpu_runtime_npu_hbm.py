#!/usr/bin/env python3
"""CPU Runtime -> NPU HBM Python integration test.

The NPU service must already be listening. This exercises the same Runtime
order as the remote CPU E2E test: BatchPut, AccessTable lookup, BatchGet,
validation, release and BatchFree.
"""

from __future__ import annotations

import os
import sys
import kvaf


def main() -> int:
    if not hasattr(kvaf, "CpuRuntime"):
        print("SKIP: CPU Runtime Python binding was not built")
        return 0

    peer = (os.environ.get("KVAF_NPU_SERVICE_ADDR") or
            os.environ.get("KVAF_CPU_SERVICE_ADDR") or "127.0.0.1:28999")
    device_id = int(os.environ.get("DEVICE_ID", "0"))
    eid_index = os.environ.get("EID_INDEX")
    hbm_bytes = int(os.environ.get("KVAF_HBM_BYTES", "16384"))
    slot_size = int(os.environ.get("KVAF_HBM_SLOT_SIZE", "4096"))
    if hbm_bytes <= 0 or slot_size <= 0 or hbm_bytes % slot_size:
        print("FAIL: KVAF_HBM_BYTES must be a positive multiple of KVAF_HBM_SLOT_SIZE", file=sys.stderr)
        return 2

    runtime = kvaf.CpuRuntime()
    runtime.initialize(peer, hbm_bytes, slot_size, device_id=device_id,
                       eid_index=None if eid_index is None else int(eid_index))
    keys = [101, 102, 103]
    blocks = [bytes((key + i) & 0xFF for i in range(32)) for key in keys]
    packed = b"".join(blocks)
    offsets = [0, len(blocks[0]), len(blocks[0]) + len(blocks[1])]
    lengths = [len(block) for block in blocks]
    try:
        print(f"CPU_HBM_TEST peer={peer} device_id={device_id} bytes={hbm_bytes}")
        print("CPU_HBM_PHASE_BEGIN batch_put")
        runtime.batch_put([keys], [packed], [offsets], [lengths])
        print("CPU_HBM_PHASE_DONE batch_put")

        table = runtime.prepare_access_table(keys)
        try:
            view = runtime.access_table_view(table)
            if list(view["lengths"]) != lengths or list(view["valid"]) != [1, 1, 1]:
                print("FAIL: AccessTable metadata mismatch", file=sys.stderr)
                return 1
            output = bytearray(len(blocks) * 32)
            print("CPU_HBM_PHASE_BEGIN batch_get")
            runtime.batch_get(table, [2, 0, 1], [0, 0, 0], [0, 1, 2], output, 32)
            expected = blocks[2] + blocks[0] + blocks[1]
            if bytes(output) != expected:
                print("FAIL: BatchGet data mismatch", file=sys.stderr)
                return 1
            print("CPU_HBM_PHASE_DONE batch_get")
        finally:
            runtime.release_access_table(table)
        runtime.batch_free(keys)
        print(f"CPU_HBM_TEST PASS bytes={hbm_bytes}")
        return 0
    finally:
        if runtime.initialized:
            runtime.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
