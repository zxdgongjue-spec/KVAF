# A5 AIV ExecuteBatchGet (minimal)

This directory is the smallest standalone implementation of the NPU-side
`ExecuteBatchGet` path: an AIV kernel looks up fixed-shape device tensors and
submits UDMA READ WQEs that move selected Host DDR blocks into a scattered HBM
output buffer. It does not include historical throughput experiments, SHMEM,
or the optional Python/Torch adapter.

## Operator contract

All lookup tensors are device-resident and retain their shapes and addresses
through graph capture:

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

## What is included

- `kernel/execute_batch_get_aiv.asc`: AIV UDMA READ WQE producer and CQ owner
  polling. It never uses MTE to dereference Host DDR.
- `host/batch_get_host.cpp`: CPU/device EID discovery, Direct-HCCP context,
  Host DDR/HBM registrations, and one private USER_CTL SQ/CQ/Jetty per AIV.
- `host/execute_batch_get_loader.cpp`: ACL binary load and AIV launch.
- `test/execute_batch_get_main.cpp`: eager correctness/performance test and
  ACL graph capture/replay test. Every graph functional replay changes fixed
  tensor contents and checks output bytes, status and queues.
- Three scripts: build, eager run, graph run.

The checked Direct-HCCP declarations in `include/direct_hccp_abi.h` are the
minimum private driver ABI needed to create the queue resources. They must be
rebuilt against the target CANN/driver combination.

## Build and run on A5

```bash
cd access_operator/demo/aiv_batch_get
bash scripts/build.sh
DEVICE_ID=0 bash scripts/run_execute_batch_get.sh
DEVICE_ID=0 bash scripts/run_execute_batch_get_graph.sh
```

The eager default is 4096 requests of 1024 bytes, 4 KiB source stride, and
four AIVs. Results are written beneath `results/`; `feedback.txt` contains
the outcome, key phase markers, run log, and any locally copied PLOG files.

Useful overrides:

```bash
DEVICE_ID=0 TABLE_CAPACITY=256 REQUESTS=10 OUTPUT_ROWS=200 \
BLOCK_BYTES=1024 SOURCE_STRIDE=4096 AIV_CORES=1 MASK_EVERY=3 \
WARMUP=1 ITERATIONS=1 bash scripts/run_execute_batch_get.sh

DEVICE_ID=0 GRAPH_REPLAYS=4 GRAPH_WARMUP=1 GRAPH_ITERATIONS=5 \
bash scripts/run_execute_batch_get_graph.sh
```

The graph performance metric is
`graph_input_update_to_stream_completion`: updates into the fixed lookup
tensors, captured graph execution, and stream completion are included;
capture, HCCP setup and final D2H validation are excluded. The normal metric
is `host_submit_to_stream_completion` and excludes warmup.

## Graph boundary

Resource creation and Host DDR registration occur before capture. Capture
contains only the fixed-address AIV kernel launch. Therefore one graph
session serves replay-time changes to addresses, validity, request indices,
mask and output rows, but a change to any capacity, block size, output shape,
or AIV count requires a new session/captured graph.

## Status

The Direct-HCCP A5 data path and the fixed-shape eager/ACL graph tests were
validated on the target environment before this minimal extraction. This
branch is intentionally a source reduction of that known route; it still
needs one remote build and smoke run after extraction before it can be called
independently revalidated.
