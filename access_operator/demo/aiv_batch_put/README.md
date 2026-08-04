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

There are no indices, masks, validity tables, table capacities or graph
capture/replay semantics in V1.

Before publishing any WQE in its shard, each AIV core validates that the
length is nonzero and fits the 32-bit WQE byte-count field, the source range is
inside `input_bytes`, and the destination range is inside the exact Host DDR
range passed to `KvafBatchPutTransportCreate`.

## Resource lifecycle

```cpp
KvafBatchPutTransportCreate(host_ddr, host_ddr_bytes, aiv_core_count, &transport);
KvafBatchPutLocalBufferAcquire(transport, input_hbm, input_bytes, &local_buffer);

KvafExecuteBatchPutSync(local_buffer, &args, stream, statuses, aiv_core_count);

KvafBatchPutLocalBufferRelease(local_buffer);
KvafBatchPutTransportDestroy(transport);
```

`TransportCreate` owns EIDs, contexts, CTPs, Jettys, SQ/CQ, Host DDR
registration/import and persistent queue state. It never registers input HBM.

`LocalBufferAcquire` registers only input HBM. Repeated Acquire with the same
`(address, bytes)` returns the same opaque handle and reuses its MR/entity.
Release decrements its use reference; cached
MRs are unregistered by `TransportDestroy`. Inputs using the same transport
must not execute concurrently because each AIV queue is single-producer; the
synchronous API serializes launches on that transport.

## Completion and errors

`KvafExecuteBatchPutSync` performs kernel launch, stream synchronization,
device-to-host status copy, and validation of every core status. It returns
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
| `4` | `input_offsets[i] + lengths[i]` exceeds `input_bytes`. |
| `5` | Destination range is outside registered Host DDR. |
| `6` | Core shard cannot fit the SQ. |
| `7` | CQ completion timeout. |
| `8` | CQ status or sub-status reports an error. |
| `9` | Zero length or length larger than the WQE byte-count field. |

`failed_request_index` identifies the invalid row for per-request validation
errors. `sq_producer_index`, `cq_consumer_index`, `cq_status`,
`cq_sub_status`, `last_wqe_protocol` and `completed_batch_count` preserve queue
diagnostics.

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

## A5 eager validation

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

A passing run proves packed input HBM, Direct-HCCP setup, variable-length
UDMA WRITE, per-block Host DDR correctness, all-core success, A-to-B-to-A input
MR cache reuse,
normal teardown and process exit code zero.

The declarations in `include/direct_hccp_abi.h` mirror CANN private
`libra.so` ABI and must match the target CANN/driver installation.
