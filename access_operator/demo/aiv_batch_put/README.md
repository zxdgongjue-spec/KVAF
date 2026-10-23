# A5 AIV BatchPut V1

This demo provides the synchronous eager Direct-HCCP BatchPut data plane used
by KVAF Runtime. It does not call MetaManager and does not support graph mode.

Runtime owns the transaction boundary:

```text
PreparePut(keys, lengths)
-> upload destination addresses
-> KvafExecuteBatchPutSync
-> success: CommitPut
-> failure: AbortPut
```

## Data contract

All request arrays are device-resident and contain exactly `row_count`
entries:

| Field | Layout | Meaning |
| --- | --- | --- |
| `input` | `uint8[input_bytes]` | Packed NPU HBM input. |
| `input_offsets` | `uint64[row_count]` | Source byte offset per block. |
| `lengths` | `uint64[row_count]` | Actual byte length per block. |
| `destination_addresses` | `uint64[row_count]` | Host DDR destination returned by `PreparePut`. |
| `core_status` | `KvafBatchPutCoreStatus[aiv_core_count]` | Device diagnostic output. |

Request `i` is always:

```cpp
source = input + input_offsets[i];
target = destination_addresses[i];
bytes = lengths[i];
UDMA_WRITE(source, target, bytes);
```

The source payload must already be visible in Global Memory before launch.
Inputs produced by DMA or `DataCopy` satisfy this requirement. A Scalar
producer must clean its Data Cache before handing the buffer to BatchPut; the
BatchPut kernel does not perform source-payload cache maintenance.

There are no indices, masks, validity tables, table capacities or graph
capture/replay semantics in V1.

The kernel trusts per-request metadata and does not validate it before
publishing WQEs. The caller must ensure that every length is nonzero and fits
the 32-bit WQE byte-count field, every source range is inside `input_bytes`,
and every destination range is inside the Host DDR region passed to
`KvafBatchPutTransportCreateWithPeerLink`. Invalid metadata is unsupported and may result
in a transport or completion error.

## Resource lifecycle

```cpp
KvafBatchPutTransportCreateWithPeerLink(host_ddr, host_ddr_bytes, aiv_core_count,
                                        jettys_per_core, peer_link, npu_local_eid,
                                        &transport);
KvafBatchPutLocalBufferAcquire(transport, input_hbm, input_bytes, &local_buffer);

KvafExecuteBatchPutSync(local_buffer, &args, stream, statuses, aiv_core_count,
                        nullptr);

KvafBatchPutLocalBufferRelease(local_buffer);
KvafBatchPutTransportDestroy(transport);
```

`TransportCreate` owns EIDs, contexts, CTPs, Jettys, SQ/CQ, Host DDR
registration/import and persistent queue state. Each AIV core owns one or two
independent Jetty slots, each with its own SQ, CQ, doorbells and persistent
producer/consumer indices. It never registers input HBM.

`LocalBufferAcquire` registers only input HBM. Repeated Acquire with the same
`(address, bytes)` returns the same opaque handle and reuses its MR/entity.
Release decrements its use reference; cached
MRs are unregistered by `TransportDestroy`. Inputs using the same transport
must not execute concurrently because each AIV queue is single-producer; the
synchronous API serializes launches on that transport.

Each post contains at most 32 WQEs. A core assigns complete posts to its Jettys
in round-robin order. Only the last WQE assigned to each used Jetty requests a
CQE, and the kernel waits for one CQE from every used Jetty before reporting
success. A core with at most one post uses only Jetty 0.

## Completion and errors

`KvafExecuteBatchPutSync` performs kernel launch, stream synchronization,
device-to-host status copy, and checking of every core status. It returns
success only when all cores report success.

Host return codes:

| Code | Meaning |
| --- | --- |
| `0` | Entire batch completed successfully. |
| `-1` | Invalid pointer, size, core count or buffer association. |
| `-2` | ACL allocation, launch, synchronization or status-copy failure. |
| `-3` | Device/EID discovery failure. |
| `-4` | Direct-HCCP setup, registration or import failure. |
| `-5` | At least one AIV core reported a data-plane failure. |

Core `result` values:

| Code | Meaning |
| --- | --- |
| `0` | Success. |
| `1` | Invalid kernel arguments. |
| `2` | Invalid local/remote entity, token, SQ or CQ metadata. |
| `3` | Corrupt persistent queue state. |
| `4` | Reserved compatibility value; this kernel does not validate source ranges. |
| `5` | Reserved compatibility value; this kernel does not validate destination ranges. |
| `6` | Core shard cannot fit the SQ. |
| `7` | CQ completion timeout. |
| `8` | CQ status or sub-status reports an error. |
| `9` | Reserved compatibility value; this kernel does not validate request lengths. |

The kernel does not produce a request-level error index; for a normally written
status, `failed_request_index` is `UINT64_MAX`. `sq_producer_index`,
`cq_consumer_index`,
`cq_status`, `cq_sub_status`, `last_wqe_protocol` and `completed_batch_count`
preserve queue diagnostics. The per-core status reports Jetty 0
producer/consumer indices; each Jetty still maintains its own persistent queue
state internally.

## Logging

Normal BatchPut execution prints only errors and the Runtime-level transport
ready marker. The successful Direct-HCCP setup details (EID selection,
contexts, tokens, MR registration, CQ/Jetty, CTP, imports, and local-buffer
cache activity) are disabled by default.

Set the following only while investigating a Direct-HCCP bring-up or teardown
failure:

```bash
KVAF_AIV_VERBOSE_DIRECT_HCCP=1
```

This restores the complete successful resource-lifecycle diagnostics. Failures
are always printed, regardless of this setting.

## A5 eager test

```bash
cd access_operator/demo/aiv_batch_put
bash scripts/build.sh
bash scripts/run_execute_batch_put.sh
```

Small variable-length run:

```bash
ROWS=10 MIN_BYTES=128 MAX_BYTES=1024 DESTINATION_STRIDE=1024 \
AIV_CORES=1 ITERATIONS=1 bash scripts/run_execute_batch_put.sh
```

Select one or two Jettys per AIV core with `JETTYS_PER_CORE`. The default is 2;
use 1 for the single-Jetty comparison path:

```bash
JETTYS_PER_CORE=1 AIV_CORES=1 ITERATIONS=1 bash scripts/run_execute_batch_put.sh
```

A passing run proves packed input HBM, Direct-HCCP setup, variable-length
UDMA WRITE, per-block Host DDR correctness, all-core success, A-to-B-to-A input
MR cache reuse,
normal teardown and process exit code zero.

The declarations in `include/direct_hccp_abi.h` mirror CANN private
`libra.so` ABI and must match the target CANN/driver installation.
