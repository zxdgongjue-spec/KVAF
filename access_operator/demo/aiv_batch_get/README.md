# A5 AIV ExecuteBatchGet (minimal)

This directory is the smallest standalone implementation of the NPU-side
`ExecuteBatchGet` path: an AIV kernel looks up fixed-shape device tensors and
submits UDMA READ WQEs that move selected Host DDR blocks into a scattered HBM
output buffer. It does not include historical throughput experiments, SHMEM,
or the optional Python/Torch adapter.

## Operator contract

All lookup tensors are device-resident.  The transport owns the Host-DDR
registration; each HBM output is registered separately as a local UDMA range.

| Tensor | Type / shape | Meaning |
| --- | --- | --- |
| `address_table` | `uint64[table_capacity]` | Absolute Host DDR UBAs, produced by the meta layer. |
| `valid_table` | `uint8[table_capacity]` | A zero entry makes its lookup invalid. |
| `indices` | `int32[request_capacity]` | Logical entries requested by this batch. |
| `offsets` | `int32[request_capacity]` | Byte offset inside the selected KV block. |
| `miss_count` | `int32[1]` | Number of valid descriptors in the compact request prefix. |
| `output_indices` | `int32[request_capacity]` | Destination row in `output`. |
| `output` | `uint8[output_rows * token_bytes]` | Contiguous Device HBM output. |

For an active request `i`, the kernel reads exactly `token_bytes` from
`address_table[indices[i]] + offsets[i]` and stores it at
`output + output_indices[i] * token_bytes`. Each address-table entry spans
`tokens_per_block * token_bytes`, so the requested token must stay within that
block. This is the
fixed-block, single-output-scatter scope; duplicate output rows are not
rejected and have last-completion-wins semantics.

The local-memory lifetime is intentionally separate from the shared
transport:

```cpp
KvafBatchGetTransport *transport = nullptr;
KvafBatchGetLocalBuffer *output_binding = nullptr;
uint64_t entity = 0;
uint32_t jettys_per_core = 2;
KvafBatchGetTransportCreateWithPeerLink(host_ddr, host_ddr_bytes, aiv_cores,
                                        jettys_per_core, peer_link, npu_local_eid,
                                        &transport);
KvafBatchGetLocalBufferAcquire(transport, output_hbm, output_bytes,
                               &output_binding, &entity);
args.output = reinterpret_cast<uint64_t>(output_hbm);
args.channel_entity = entity;
args.jettys_per_core = jettys_per_core;
// args.queue_state points to a zero-initialized
// KvafBatchGetQueueState[aiv_cores * jettys_per_core] array.
// launch and synchronize the stream
KvafBatchGetLocalBufferRelease(output_binding);
KvafBatchGetTransportDestroy(transport);
```

## Dual-stream BatchGet

Set `NpuRuntime(..., npu_core_count=C, jettys_per_core=J,
batch_get_stream_count=2)` to run two kernels sharing one transport. Each
stream launches `C` AIV blocks; the transport contains `2*C*J` independent
SQ/CQ pairs. The current total block limit is 48. The default stream count is
1. BatchPut continues to launch `C` blocks on one stream.

The caller stream records a ready event, the internal stream waits for it,
and the two kernels read disjoint halves of the live `[0, miss_count)` prefix.
Each kernel further divides its half among its `C` blocks. Both keep global
request/output pointers and write through the original `output_indices`.
Use non-overlapping output destinations. Resource arrays are stream-major:
`[stream][block][Jetty]` for entity/queue state, `[stream][block]` for status.
The internal stream records done, and the caller stream waits for done before
any subsequent work. No host synchronization or device-to-host miss-count
copy is added to the normal enqueue path. Queue indices persist across calls
and output changes.

Streams/events are created before capture. The ready/done dependency chain
captures the side kernel and joins it back into the originating graph stream.
Warm output registration before capture, order all calls/replays using the
same Operator, and finish work/destroy graphs before shutting it down. Actual
kernel overlap and graph behavior must be verified on the target A5/CANN.

The kernel argument ABI is now 120 bytes, with `shard_id` and `shard_count`
appended. A zero-initialized `shard_count` means one shard. The loader requires
`KvafExecuteBatchGetAivKernelV2`, so an old binary fails to load instead of
silently reading the entire prefix twice. **Rebuild both host and AIV kernel.**

The benchmark accepts `--stream-count 2` (or
`KVAF_NPU_BATCH_GET_STREAM_COUNT=2`); `--aiv-cores` is per stream. Compare equal
resources with `--stream-count 1 --aiv-cores 2 --jettys-per-core 1` versus
`--stream-count 2 --aiv-cores 1 --jettys-per-core 1`. Multi-kernel profile
bandwidth uses `kernel_wall` (earliest start to latest finish), not the maximum
individual block duration. End-to-end call-to-completion remains the main
latency metric.

Hardware integration coverage:

```bash
KVAF_RUN_NPU_TESTS=1 KVAF_NPU_CORE_COUNT=1 \
KVAF_NPU_JETTYS_PER_CORE=1 KVAF_NPU_BATCH_GET_STREAM_COUNT=2 \
KVAF_NPU_GRAPH_REPLAYS=4 python3 tests/python/test_npu_batch_get.py
```

Also set `PYTHONPATH`, `KVAF_BATCH_GET_KERNEL`, `KVAF_BATCH_PUT_KERNEL`, and
`KVAF_CPU_SERVICE_ADDR` as for the existing NPU tests. This checks output
switching, caller-stream-only synchronization, zero/one/odd prefix sizes,
profile coverage and replay with changing device-side miss counts.

For simultaneous local-EID paths, call
`KvafBatchGetTransportCreateWithPeerLinkMultiEid` with a packed array of
16-byte EIDs. It creates one RaCtx and one local/remote MR registration per
EID, then assigns the existing core/Jetty workers to paths in jetty-major
round-robin order. The single-EID function above is retained as an ABI wrapper.

An exact repeat of `Acquire(transport, output_hbm, output_bytes, ...)`
returns the cached binding/entity and increments its use reference. New HBM
addresses receive one local MR per EID context and a new entity table but reuse
the transport.

## What is included

- `kernel/execute_batch_get_aiv.asc`: SIMT-compacts requests,
  caches each active request's table index and request ID in UB, generates 32
  UDMA READ WQEs in parallel, and overlaps vector fill, MTE3 UB-to-SQ copies,
  and scalar doorbells with two UB banks. It never uses MTE to dereference Host
  DDR.
- `host/batch_get_host.cpp`: CPU/device EID discovery, Direct-HCCP transport,
  Host-DDR registration, per-output HBM-MR cache, and up to two private
  USER_CTL SQ/CQ/Jettys per AIV.
- `host/execute_batch_get_loader.cpp`: ACL binary load and AIV launch.
- Runtime/Python integration coverage lives in
  `tests/python/test_npu_batch_get.py` and
  `tests/python/test_npu_runtime_perf.py`.

The checked Direct-HCCP declarations in `include/direct_hccp_abi.h` are the
minimum private driver ABI needed to create the queue resources. They must be
rebuilt against the target CANN/driver combination.

## Pipelined kernel boundary

The data path keeps the public KVAF ABI and the verified Direct-HCCP WQE token
fields. It does not copy the AIV_URMA benchmark's server/client handshake,
control-region writes, fixed 64K request shape, or per-launch CQ reset.

Each AIV consumes its assigned slice of the compact `[0, miss_count)` request
prefix directly. Before touching the SQ, the kernel rejects an invalid
descriptor or a per-Jetty assignment that fills the entire SQ.
The phase-two scheduler assigns complete 32-WQE posts to the core's Jettys in
round-robin order, with one to sixty-four Jettys supported per core. Each Jetty has an
independent persistent SQ producer and CQ consumer index. Only the last READ
assigned to each used Jetty requests a CQE;
the kernel drains one CQE from every used Jetty before reporting the batch
complete. This is required because a CQE on one Jetty does not fence READs
submitted through another Jetty. A core with one post uses only Jetty 0, while
the one-Jetty phase-one path remains selectable with `JETTYS_PER_CORE=1`.

The ACL launcher explicitly reserves 160 KiB of AIV local memory for the two
WQE banks, compacted table-index/request-ID arrays and SIMT validation scratch.

## Build and validate on A5

```bash
./build.sh npu

KVAF_CPU_SERVICE_ADDR=141.61.92.71:28999 \
DEVICE_ID=0 \
KVAF_NPU_TOKEN_BYTES=656 \
KVAF_NPU_TOKENS_PER_BLOCK=128 \
KVAF_NPU_ACTIVE_PERCENT=5 \
tests/run_remote_cpu_e2e.sh "$PWD/build-npu"
```

The kernel uses the Ascend C SIMT API. The build automatically checks the
toolkit's top-level `include/simt_api` directory and the older tikcpp layouts.
If asc-devkit is installed separately, point the build at the directory that
contains `simt_api/device_atomic_functions.h` before invoking the root build:

```bash
ASCEND_SIMT_API_DIR=/path/to/asc-devkit/include ./build.sh npu
```

If that header is absent from both the CANN toolkit and asc-devkit, install the
asc-devkit package matching the target CANN version before building; mixing
headers from a different CANN release is not supported. The example above uses
656 bytes per token, 128 tokens per block, and a 5% active probability.

## Graph boundary

`KvafBatchGetTransportCreateWithPeerLink(host_ddr, ..., peer_link, npu_local_eid, ...)`
creates only the reusable peer-link, DDR import and queue resources. Before capture, the caller acquires the fixed
graph output through `KvafBatchGetLocalBufferAcquire(transport, output, ...)`.
That call returns an opaque local-buffer handle and a device entity table whose
local HBM address/token match `args.output`; capture contains only the AIV
kernel launch. One graph therefore serves replay-time changes to addresses,
validity, request indices, offsets, miss count and output rows. A different output allocation
uses its own acquire result and graph instance; eager launches may select any
cached acquire result at runtime. A capacity, block-size, AIV-count or
Jetty-count change also needs a new session/captured graph. After all stream
work completes,
release local buffers before destroying the transport.

## Status

The Direct-HCCP A5 data path and the fixed-shape eager/ACL graph tests were
validated on the target environment before this minimal extraction. This
branch is intentionally a source reduction of that known route; it still
needs one remote build and smoke run after extraction before it can be called
independently revalidated.
