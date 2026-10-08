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

### Raw addresses (V3)

Set `remote_addresses` and `local_addresses` to device-resident
`uint64[request_capacity]` arrays containing final remote UBAs and local HBM
addresses. Set `address_table`, `valid_table`, `indices`, `offsets`,
`output_indices`, `table_capacity`, `output_rows`, and `tokens_per_block` to
zero. `miss_count` still selects the active prefix, and each READ transfers
exactly `token_bytes`. `output` and `channel_entity` identify the acquired
local registration containing the destinations; destinations can have arbitrary
byte offsets and the registration size need not be a multiple of `token_bytes`.

The raw WQE fill only loads `remote_addresses[i]` and `local_addresses[i]`.
It performs no AccessTable lookup, offset addition, or output-row multiplication.
Raw and lookup modes use separate compile-time posting specializations and
share the existing multi-Jetty, inflight, completion and profiling behavior.

Descriptor contents are trusted, as in the existing lookup path with request
validation disabled. Callers must ensure every active source range is inside
the linked remote MR and every destination range is inside the acquired local
MR. All entries in a launch use that peer and local registration. Overlapping
destinations have no guaranteed completion order. No metadata lock is acquired
by raw BatchGet; retain any needed AccessTable lock or otherwise guarantee the
remote allocations remain valid through stream/graph completion.

Python exposes `runtime.batch_get_raw(remote_addresses, local_addresses,
miss_count, output, stream=None, synchronize=False)`. Address tensors are
contiguous one-dimensional `torch.int64`; `miss_count` is `torch.int32[1]`.
All tensors must reside on the Runtime NPU. `output` is a contiguous backing
buffer for registration, and the transfer size is `batch_get_token_bytes`.
The C++ `BatchGetRequest` accepts the same two address buffer fields; leave its
lookup fields empty and `tokens_per_block=0`.

```python
# Caller-side preparation, outside the BatchGet kernel/timed region.
remote_addresses = (device_table.addresses[indices.long()] + offsets.long()).contiguous()
local_addresses = (output_indices.long() * token_bytes + output.data_ptr()).contiguous()
runtime.batch_get_raw(remote_addresses, local_addresses, miss_count, output)
torch.npu.synchronize()  # keep all tensors and remote allocations alive until here
```

For graph use, warm up the output registration before capture. Keep the address
arrays, output allocation and their addresses fixed, and update array contents
and `miss_count` on the capture stream before replay. Raw mode needs no device
AccessTable. Host, binding and kernel must be rebuilt together: the argument
record is now 128 bytes and the launcher requires `KvafExecuteBatchGetAivKernelV3`.

The local-memory lifetime is intentionally separate from the shared
transport:

```cpp
KvafBatchGetTransport *transport = nullptr;
KvafBatchGetLocalBuffer *output_binding = nullptr;
uint64_t entity = 0;
uint32_t jettys_per_core = 1;
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
launcher now requires the `KvafExecuteBatchGetAivKernelV3` symbol.

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

Set `KVAF_BATCH_GET_RAW_ADDRESSES=1` to run the same eager, A/B/A output,
inflight and graph integration checks through raw addresses. This also works
with `tests/run_remote_npu_e2e.sh` for an NPU HBM peer. The performance script
`tests/python/test_npu_batch_get_perf.py` accepts `--raw-addresses`; address
preparation is excluded from its timing and it reports `address_mode=raw`.

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

## Raw-address miss-count performance sweep

The existing `tests/run_remote_cpu_batch_get_miss_count_perf.sh` accepts
`--raw-addresses` (or `KVAF_BATCH_GET_RAW_ADDRESSES=1`). It enables kernel
profiling automatically and keeps the same statistics as lookup BatchGet.
Rebuild the V3 host, Python binding and kernel first (`./build.sh npu`), start
the CPU peer service as usual, then run on the NPU client:

```bash
KVAF_CPU_SERVICE_ADDR=CPU_HOST_IP:28999 \
DEVICE_ID=0 \
KVAF_BATCH_GET_SWEEP_TOKEN_BYTES="656" \
KVAF_BATCH_GET_SWEEP_MISS_COUNTS="32 128 512 2048 8192" \
KVAF_BATCH_GET_SWEEP_AIV_CORES="2" \
KVAF_BATCH_GET_SWEEP_JETTYS_PER_CORE="1" \
KVAF_BATCH_GET_KERNEL_CLOCK_MHZ=1000 \
bash tests/run_remote_cpu_batch_get_miss_count_perf.sh "$PWD/build-npu" \
  --raw-addresses --request-length 8192 --table-capacity 8192 \
  --tokens-per-block 128 --warmup 3 --iterations 100
```

The four `KVAF_BATCH_GET_SWEEP_*` variables accept space-separated values.
Omit them to retain the original 2646-shape sweep. Set
`KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR` to choose a persistent output directory;
otherwise its temporary path is printed at startup. Results also go to
`batch_get_miss_sweep_summary.csv` in that directory. Configuration, kernel
profile records, sweep results and CSV identify `address_mode=raw` or `lookup`.

For each sample, `kernel_total` uses the maximum elapsed kernel time across
AIV cores. The summary reports `mean_us`, `p50_us`, `p90_us`, `p99_us` and
`p50_GBps = miss_count * token_bytes / p50_us / 1000` (decimal GB/s).
The `batch_get_kernel_total` profile record also reports `useful_GiBps`, based
on the mean kernel time (binary GiB/s). `kernel_post` and
`kernel_completion_wait` separately report submission and accumulated CQ wait
times; their independently aggregated maxima need not add up to total time.
Address preparation, remote-data population and profile copies to Host are
outside the timed region. Kernel profiling overhead is included.

`KVAF_BATCH_GET_KERNEL_CLOCK_MHZ` is the **GetSystemCycle counter frequency**,
which must match the target platform. The existing default is 1000 MHz;
an incorrect value scales both reported time and bandwidth. Raw cycle counts
are retained in per-shape logs.

The host-only regression `python3 tests/python/test_batch_get_miss_sweep.py`
checks both shell wrappers with synthetic profiles, including raw CLI/env
selection, cycle conversion, bandwidth and CSV output. It does not measure
hardware performance.

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
