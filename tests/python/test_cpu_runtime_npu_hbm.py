#!/usr/bin/env python3
"""CPU Runtime -> NPU HBM Python integration test.

The NPU service must already be listening. This exercises the same Runtime
order as the remote CPU E2E test: BatchPut, AccessTable lookup, BatchGet,
validation, release and BatchFree.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
import kvaf


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


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
    warmup = int(os.environ.get("KVAF_HBM_WARMUP", "1"))
    iterations = int(os.environ.get("KVAF_HBM_ITERATIONS", "5"))
    if (hbm_bytes <= 0 or slot_size <= 0 or hbm_bytes % slot_size or
            warmup < 0 or iterations <= 0):
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
        put_samples: list[float] = []
        get_samples: list[float] = []
        for sample in range(warmup + iterations):
            sample_keys = [key + 1000 * (sample + 1) for key in keys]
            sample_blocks = [bytes((key + i + sample) & 0xFF for i in range(32))
                             for key in sample_keys]
            sample_packed = b"".join(sample_blocks)
            sample_offsets = [0, len(sample_blocks[0]),
                              len(sample_blocks[0]) + len(sample_blocks[1])]
            begin = time.perf_counter()
            runtime.batch_put([sample_keys], [sample_packed], [sample_offsets], [lengths])
            put_ms = (time.perf_counter() - begin) * 1000.0
            sample_table = runtime.prepare_access_table(sample_keys)
            sample_output = bytearray(len(sample_blocks) * 32)
            begin = time.perf_counter()
            runtime.batch_get(sample_table, [2, 0, 1], [0, 0, 0], [0, 1, 2],
                               sample_output, 32)
            get_ms = (time.perf_counter() - begin) * 1000.0
            expected_sample = sample_blocks[2] + sample_blocks[0] + sample_blocks[1]
            runtime.release_access_table(sample_table)
            runtime.batch_free(sample_keys)
            if bytes(sample_output) != expected_sample:
                print("FAIL: performance BatchGet data mismatch", file=sys.stderr)
                return 1
            if sample >= warmup:
                put_samples.append(put_ms)
                get_samples.append(get_ms)
        total_bytes = sum(lengths)
        for metric, samples in (("batch_put", put_samples), ("batch_get", get_samples)):
            mean = statistics.mean(samples)
            gbps = total_bytes / 1_000_000_000.0 / (mean / 1000.0)
            print(f"PERF_RESULT metric={metric} samples={iterations} warmup={warmup} "
                  f"bytes={total_bytes} mean_ms={mean:.3f} "
                  f"p50_ms={percentile(samples, .5):.3f} "
                  f"p90_ms={percentile(samples, .9):.3f} "
                  f"p99_ms={percentile(samples, .99):.3f} useful_GBps={gbps:.3f}")
        print(f"CPU_HBM_TEST PASS bytes={hbm_bytes}")
        return 0
    finally:
        if runtime.initialized:
            runtime.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
