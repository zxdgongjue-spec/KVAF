# A5 AIV ExecuteBatchGet (serial SIMT baseline)

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
| `valid_table` | `uint8[table_capacity]` | Fixed ABI validity tensor; active entries must be prevalidated in this baseline. |
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
KvafBatchGetTransportCreate(host_ddr, host_ddr_bytes, aiv_cores, 1, &transport);
KvafBatchGetLocalBufferAcquire(transport, output_hbm, output_bytes,
                               &output_binding, &entity);
args.output = reinterpret_cast<uint64_t>(output_hbm);
args.channel_entity = entity;
// launch and synchronize the stream
KvafBatchGetLocalBufferRelease(output_binding);
KvafBatchGetTransportDestroy(transport);
```

An exact repeat of `Acquire(transport, output_hbm, output_bytes, ...)`
returns the cached binding/entity and increments its use reference. New HBM
addresses receive a new local MR and entity table but reuse the transport.

## What is included

- `kernel/execute_batch_get_aiv.asc`: SIMT mask compaction and 32-lane UDMA
  READ WQE generation, followed by CQ owner polling. It never uses MTE to
  dereference Host DDR.
- `host/batch_get_host.cpp`: CPU/device EID discovery, Direct-HCCP transport,
  Host-DDR registration, per-output HBM-MR cache, and one private USER_CTL
  SQ/CQ/Jetty per AIV.
- `host/execute_batch_get_loader.cpp`: ACL binary load and AIV launch.
- `test/execute_batch_get_main.cpp`: eager correctness/performance test and
  ACL graph capture/replay test. Every graph functional replay changes fixed
  tensor contents and checks output bytes, status and queues.
- Three scripts: build, eager run, graph run.

The checked Direct-HCCP declarations in `include/direct_hccp_abi.h` are the
minimum private driver ABI needed to create the queue resources. They must be
rebuilt against the target CANN/driver combination.

## Baseline boundary

This version keeps the same SIMT WQE format, 32-WQE post size, active-request
capacity, low-level UB-to-SQ copy, and 128 KiB local-memory configuration as
`simt-batch-get`. It deliberately removes both optimized scheduling axes:

- Every AIV owns exactly one Jetty and one persistent SQ/CQ state.
- One UB WQE bank is used. For every post, Vector WQE generation completes,
  then MTE3 copies that post to SQ, then Scalar updates the producer index and
  rings the doorbell. Only after all three stages finish does the next post
  start.

Each AIV compacts its full request partition once into a 16384-entry active-ID
array. A batch is rejected before SQ submission if that capacity is exceeded
or if the active request count would fill the single Jetty's SQ. As in the
referenced performance path, active table indices, addresses, and output rows
must already be valid; `valid_table` remains in the fixed public ABI.

## Build and run on A5

```bash
cd access_operator/demo/aiv_batch_get
bash scripts/build.sh
DEVICE_ID=0 bash scripts/run_execute_batch_get.sh
DEVICE_ID=0 bash scripts/run_execute_batch_get_graph.sh
```

The kernel uses the Ascend C SIMT API. The build automatically checks the
toolkit's top-level `include/simt_api` directory and older tikcpp layouts. If
asc-devkit is installed separately, point the build at the directory that
contains `simt_api/device_atomic_functions.h`:

```bash
ASCEND_SIMT_API_DIR=/path/to/asc-devkit/include bash scripts/build.sh
```

Use the asc-devkit package matching the target CANN version.

The eager default is 4096 requests of 1024 bytes, 4 KiB source stride, four
AIVs, and a 5% active probability. Every warmup and measured sample uses its
own random indices/mask pair. All pairs are generated and copied to the device
before timing. Results are written beneath `results/`; `feedback.txt` contains
the outcome, key phase markers, run log, and any locally copied PLOG files.

Useful overrides:

```bash
DEVICE_ID=0 TABLE_CAPACITY=256 REQUESTS=10 OUTPUT_ROWS=200 \
BLOCK_BYTES=1024 SOURCE_STRIDE=4096 AIV_CORES=1 ACTIVE_PERCENT=20 \
WARMUP=1 ITERATIONS=1 bash scripts/run_execute_batch_get.sh

# Proves TransportCreate -> Acquire(A) -> Acquire(B) -> re-Acquire(A), then
# releases both local-buffer references without rebuilding EID/CTP/Jettys.
DEVICE_ID=0 DYNAMIC_OUTPUT_SMOKE=1 WARMUP=1 ITERATIONS=1 \
bash scripts/run_execute_batch_get.sh

DEVICE_ID=0 GRAPH_REPLAYS=4 GRAPH_WARMUP=1 GRAPH_ITERATIONS=5 \
bash scripts/run_execute_batch_get_graph.sh
```

For eager runs, `ACTIVE_PERCENT=N` controls the independent Bernoulli
probability that each request's mask is 1; it defaults to 5. Random generation
and the one-time H2D upload are excluded from the performance interval.
`MASK_EVERY=N` and `ACTIVE_EVERY=N` remain deterministic controls for the
dynamic-output and graph paths: the former clears request IDs divisible by
`N`, while the latter sets only request IDs divisible by `N`. They are mutually
exclusive.

The graph performance metric is
`graph_input_update_to_stream_completion`: updates into the fixed lookup
tensors, captured graph execution, and stream completion are included;
capture, HCCP setup and final D2H validation are excluded. The normal metric
is `host_submit_to_stream_completion` and excludes warmup.

## Graph boundary

`KvafBatchGetTransportCreate(host_ddr, ...)` creates only the reusable EID,
Host-DDR and queue resources. Before capture, the caller acquires the fixed
graph output through `KvafBatchGetLocalBufferAcquire(transport, output, ...)`.
That call returns an opaque local-buffer handle and a device entity table whose
local HBM address/token match `args.output`; capture contains only the AIV
kernel launch. One graph therefore serves replay-time changes to addresses,
validity, request indices, mask and output rows. A different output allocation
uses its own acquire result and graph instance; eager launches may select any
cached acquire result at runtime. A capacity, block-size or AIV-count change
also needs a new session/captured graph. After all stream work completes,
release local buffers before destroying the transport.

## Status

The baseline is implemented against the established Direct-HCCP A5 data path
and fixed-shape eager/ACL graph harness. It still needs a build and smoke run
on an A5 environment with the matching CANN asc-devkit before it can be called
hardware-validated.
