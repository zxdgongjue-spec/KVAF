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
| `mask` | `uint8[request_capacity]` | Zero skips that request without changing the graph shape. |
| `output_indices` | `int32[request_capacity]` | Destination row in `output`. |
| `output` | `uint8[output_rows * block_bytes]` | Contiguous Device HBM output. |

For an active request `i`, the kernel reads `block_bytes` from
`address_table[indices[i]]` and stores it at
`output + output_indices[i] * block_bytes`. This is exactly the initial
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

An exact repeat of `Acquire(transport, output_hbm, output_bytes, ...)`
returns the cached binding/entity and increments its use reference. New HBM
addresses receive a new local MR and entity table but reuse the transport.

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
- `test/execute_batch_get_main.cpp`: eager correctness/performance test and
  ACL graph capture/replay test. Every graph functional replay changes fixed
  tensor contents and checks output bytes, status and queues.
- Three scripts: build, eager run, graph run.

The checked Direct-HCCP declarations in `include/direct_hccp_abi.h` are the
minimum private driver ABI needed to create the queue resources. They must be
rebuilt against the target CANN/driver combination.

## Pipelined kernel boundary

The data path keeps the public KVAF ABI and the verified Direct-HCCP WQE token
fields. It does not copy the AIV_URMA benchmark's server/client handshake,
control-region writes, fixed 64K request shape, or per-launch CQ reset.

Each AIV compacts its complete request partition once into parallel 16384-entry
UB arrays containing the active table indices and original request IDs. The
caller must guarantee that each AIV owns at most 16384 mask entries whose value
is nonzero. Before touching the SQ, the kernel rejects an active count beyond
that capacity or a per-Jetty assignment that fills the entire SQ.
The phase-two scheduler assigns complete 32-WQE posts to the core's Jettys in
round-robin order. Each Jetty has an independent persistent SQ producer and CQ
consumer index. Only the last READ assigned to each used Jetty requests a CQE;
the kernel drains one CQE from every used Jetty before reporting the batch
complete. This is required because a CQE on one Jetty does not fence READs
submitted through another Jetty. A core with one post uses only Jetty 0, while
the one-Jetty phase-one path remains selectable with `JETTYS_PER_CORE=1`.

The ACL launcher explicitly reserves 160 KiB of AIV local memory for the two
WQE banks, compacted table-index/request-ID arrays and SIMT validation scratch.

## Build and run on A5

```bash
cd access_operator/demo/aiv_batch_get
bash scripts/build.sh
DEVICE_ID=0 bash scripts/run_execute_batch_get.sh
DEVICE_ID=0 bash scripts/run_execute_batch_get_graph.sh
```

The kernel uses the Ascend C SIMT API. The build automatically checks the
toolkit's top-level `include/simt_api` directory and the older tikcpp layouts.
If asc-devkit is installed separately, point the build at the directory that
contains `simt_api/device_atomic_functions.h`:

```bash
ASCEND_SIMT_API_DIR=/path/to/asc-devkit/include bash scripts/build.sh
```

If that header is absent from both the CANN toolkit and asc-devkit, install the
asc-devkit package matching the target CANN version before building; mixing
headers from a different CANN release is not supported.

The eager default is 4096 requests of 1024 bytes, 4 KiB source stride, four
AIVs, two Jettys per AIV, and a 5% active probability. As in
`tests/python/test_npu_runtime_perf.py`, every warmup and measured sample uses
its own random indices/mask pair. All pairs are generated and copied to the
device before the benchmark starts. Results are written beneath `results/`;
`feedback.txt` contains
the outcome, key phase markers, run log, and any locally copied PLOG files.

Useful overrides:

```bash
DEVICE_ID=0 TABLE_CAPACITY=256 REQUESTS=10 OUTPUT_ROWS=200 \
BLOCK_BYTES=1024 SOURCE_STRIDE=4096 AIV_CORES=1 JETTYS_PER_CORE=2 ACTIVE_PERCENT=20 \
WARMUP=1 ITERATIONS=1 bash scripts/run_execute_batch_get.sh

# Proves TransportCreate -> Acquire(A) -> Acquire(B) -> re-Acquire(A), then
# releases both local-buffer references without rebuilding EID/CTP/Jettys.
DEVICE_ID=0 DYNAMIC_OUTPUT_SMOKE=1 WARMUP=1 ITERATIONS=1 \
bash scripts/run_execute_batch_get.sh

DEVICE_ID=0 GRAPH_REPLAYS=4 GRAPH_WARMUP=1 GRAPH_ITERATIONS=5 \
bash scripts/run_execute_batch_get_graph.sh
```

For the eager benchmark, `ACTIVE_PERCENT=N` controls the independent Bernoulli
mask probability for every sample; random generation and the one-time H2D
upload are excluded from timing. `MASK_EVERY=N` and `ACTIVE_EVERY=N` remain
deterministic functional controls for the dynamic-output smoke test and graph
replay: the former masks request IDs divisible by `N`, while the latter
activates only request IDs divisible by `N`; the two options are mutually
exclusive.

The graph performance metric is
`graph_input_update_to_stream_completion`: updates into the fixed lookup
tensors, captured graph execution, and stream completion are included;
capture, HCCP setup and final D2H validation are excluded. The normal metric
is `host_submit_to_stream_completion`; it includes only launch through stream
completion and excludes input generation, H2D upload, warmup and validation.

## Graph boundary

`KvafBatchGetTransportCreateWithPeerLink(host_ddr, ..., peer_link, npu_local_eid, ...)`
creates only the reusable peer-link, DDR import and queue resources. Before capture, the caller acquires the fixed
graph output through `KvafBatchGetLocalBufferAcquire(transport, output, ...)`.
That call returns an opaque local-buffer handle and a device entity table whose
local HBM address/token match `args.output`; capture contains only the AIV
kernel launch. One graph therefore serves replay-time changes to addresses,
validity, request indices, mask and output rows. A different output allocation
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
