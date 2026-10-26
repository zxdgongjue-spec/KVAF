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
args.batch_get_max_inflight_total = 2048; // 0 selects the baseline
// args.queue_state points to a zero-initialized
// KvafBatchGetQueueState[aiv_cores * jettys_per_core] array.
// launch and synchronize the stream
KvafBatchGetLocalBufferRelease(output_binding);
KvafBatchGetTransportDestroy(transport);
```

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

- `kernel/execute_batch_get_aiv.asc`: consumes the compact request prefix,
  generates up to 128 UDMA READ WQEs in parallel, and overlaps vector fill,
  MTE3 UB-to-SQ copies, and scalar doorbells with two UB banks. Larger batches
  use two completion-tracked groups per core under a shared total budget.
  It never uses MTE to dereference Host DDR.
- `host/batch_get_host.cpp`: CPU/device EID discovery, Direct-HCCP transport,
  Host-DDR registration, per-output HBM-MR cache, and up to sixty-four private
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

Each AIV consumes its assigned slice of the compact `[0, miss_count)` prefix.
The scheduler assigns posts of up to 128 WQEs to one to sixty-four Jettys in
round-robin order. Each Jetty has independent persistent SQ/CQ indices.

Runtime defaults to a total `batch_get_max_inflight_total` of 2048 READs across
all AIV cores. Small batches retain the whole-slice path. Larger batches split
that budget across cores and keep at most two groups pending per core. Every
group signals its last READ on each participating Jetty; all those CQEs must
succeed before reusing the group's credit. Jetty rotation continues across
groups. Zero disables the limit. See [design, configuration and validation](../../../docs/batch_get_inflight_control.md).

Capacity checks use the currently pending groups. Failed submissions poison
the affected queues until transport rebuild, preventing stale completions from
being consumed by a later replay. The baseline's descriptor validation remains
temporarily disabled in this branch.

The ACL launcher reserves 160 KiB of AIV local memory for the two WQE banks and
SIMT validation scratch. Rebuild Host, Python binding and Kernel together: the
launcher now requires the `KvafExecuteBatchGetAivKernelV2` symbol.

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

## Per-Jetty priority experiments

Set `KVAF_BATCH_GET_JETTY_PRIORITIES` in the NPU test process to configure
BatchGet's local Jettys. No Python, Runtime, C API, wire protocol or kernel ABI
change is required. Host Jettys and BatchPut Jettys retain their existing
configuration; the original Direct-HCCP creation entry point still uses 15.

The value is a comma-separated list of decimal integers in `[0, 15]`.
Whitespace around each integer is accepted. An unset variable or an empty
string selects `15` for all Jettys. Invalid values (including empty entries,
trailing commas, signs and out-of-range integers) fail BatchGet transport
creation before it allocates device resources; they do not fall back to 15.

Assignment follows the existing core-major worker index:

```text
entity_index = core * jettys_per_core + jetty_slot
priority = priorities[entity_index % priority_count]
```

There is no required relationship between list length and core/Jetty count.
A single value applies to every Jetty; a shorter list repeats; a longer list
leaves its unused suffix unapplied. Repeated entries are allowed. Existing
core and Jetty limits still apply, and EID path assignment is unchanged.
For example, `3,10,13,15` with three cores and two Jettys per core gives:

| Core | Jetty 0 | Jetty 1 |
| --- | --- | --- |
| 0 | 3 | 10 |
| 1 | 13 | 15 |
| 2 | 3 | 10 |

The list is read once at transport creation, and each transport starts at its
first entry. Changing the environment does not change existing Jettys or a
captured graph; start a new test process to recreate the transport.

After `./build.sh npu`, with the existing CPU service listening on port 29000:

```bash
mkdir -p logs
set -o pipefail
KVAF_CPU_SERVICE_ADDR=127.0.0.1:29000 \
DEVICE_ID=7 \
KVAF_BATCH_GET_JETTY_PRIORITIES=3,10,13,15 \
tests/run_remote_cpu_batch_get_perf.sh "$PWD/build-npu" \
    --aiv-cores 3 --jettys-per-core 2 \
    --token-bytes 656 --request-length 4096 --miss-count 4096 \
    --max-inflight-total 2048 --warmup 20 --iterations 100 \
    --kernel-profile 2>&1 | tee logs/batch_get_priority_3_10_13_15.log
```

Use the CPU endpoint and NPU device appropriate to the server. The same
environment variable also passes through
`tests/run_remote_cpu_batch_get_miss_count_perf.sh` without script changes;
its existing sweep determines core/Jetty counts. For controlled comparisons,
use the single-configuration runner above and keep device/EID selection,
workload and inflight budget fixed. Compare `15`, `13`, `15,13`, and `13,15`
in interleaved repeated runs before adding more priority values.

`BATCH_GET_JETTY_PRIORITY_CONFIG` logs the normalized list, assignment order,
worker count and number of unused entries. `BATCH_GET_JETTY_PRIORITY` logs
`core`, `jetty_slot`, `entity_index`, `eid_path`, `jetty_id`, and the priority
passed for each successfully created BatchGet Jetty. `DIRECT_JETTY_READY`
also includes priority, including 15 for callers of the original entry point.
Inspect the mapping with:

```bash
grep -E 'BATCH_GET_JETTY_PRIORITY|DIRECT_JETTY_READY' \
    logs/batch_get_priority_3_10_13_15.log
```

These logs confirm the requested priority, not the hardware's effective SL.
NPU priority-to-SL mapping must be verified for the actual NPU device. Compare
overall throughput and latency; existing per-Jetty completion wait counters
measure sequential polling time and are not independent transfer latencies.

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
Jetty-count or inflight-budget change also needs a new session/captured graph. After all stream
work completes,
release local buffers before destroying the transport.

## Status

The Direct-HCCP A5 data path and the fixed-shape eager/ACL graph tests were
validated on the target environment before this minimal extraction. This
branch is intentionally a source reduction of that known route; it still
needs one remote build and smoke run after extraction before it can be called
independently revalidated.
