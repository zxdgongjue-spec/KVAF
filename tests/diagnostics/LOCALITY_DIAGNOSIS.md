# Random READ bottleneck discrimination

Use the same independent branch as random656_sweep. No kernel/WQE order bit is
changed. Existing V9-compatible build-npu and THP service can be reused.

## Client: two short commands

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:29876 DEVICE_ID=0 \
python3 tests/diagnostics/locality_sweep.py --build build-npu \
  --suite locality --output locality-results

KVAF_CPU_SERVICE_ADDR=127.0.0.1:29876 DEVICE_ID=0 \
python3 tests/diagnostics/locality_sweep.py --build build-npu \
  --suite size --output size-results
```

First command: 7 configurations x 2 passes. All register 688 MB and read 7740
656B requests. Random and logically contiguous reads each sample 16.1/86.0/687.9
MB working ranges. An additional sorted case uses exactly the same random token
set/seed as the 86 MB random case. Tokens within blocks are contiguous; physical
contiguity across runtime allocations is NOT presumed. The separate profiled
launch records actual remote virtual start-address page counts and adjacent
address fraction. These are not measured translation misses.

Second command: 4 configurations x 2 passes. Random 256/656/1024/4096B requests
keep useful bytes approximately 5.077 MB and registered range approximately
86 MB. Request counts necessarily change. Larger requests are DIAGNOSTIC,
not a claim that production 656B requests can be inflated without cost.

Each case has three warmups and twenty uninstrumented Graph event samples,
changing inputs and fully verifying payload outside timing. One separate eager
instrumented launch gives post/wait ticks. All configurations use 30 AIV and two
Jettys. No SQ-depth or fence setting is silently changed. CPU service memory
population can take longer than the measured test; allow at least 688 MB free
registered host memory plus normal runtime headroom. No automatic timeout kills
an in-flight DMA test. Failure stops further cases and retains the log.

## CPU-service host: optional concurrent evidence collection

In another terminal, before starting clients (replace PID):

```bash
python3 tests/diagnostics/collect_path_counters.py --pid <CPU_SERVICE_PID> \
  --seconds 1800 --output path-counters.json
```

Add `--interface <actual-interface>` for netdev/ethtool counters. Ctrl-C after
clients finish writes the final snapshot/deltas. This is read-only: it does not
restart the service, alter affinity/page mode, or enable privileged PMUs.
It records THP backing via smaps_rollup, NUMA residency counts, available RDMA
statistics and sysfs counters. Missing interfaces/tools/vendor counters are
reported as unavailable, never as evidence that congestion/misses are zero.
Snapshots include initialization and all cases; they are NOT per-case PMU data.
NUMA page counts can use different page sizes and are not reported as bytes.

## What can actually be concluded

| Observation | Supported direction | Still not proved |
|---|---|---|
| Fixed registration, wider random range slows reads | Working-set sensitivity | NIC translation vs memory/cache latency |
| Sorted same addresses improves throughput | Submission order/locality matters | Specific protocol fence is the cause |
| Contiguous helps but random does not | Random access path limits throughput | CPU DRAM alone is saturated |
| Larger requests raise GB/s while request/s plateaus | Per-request costs merit investigation | NIC is the sole limiting engine |
| Link counters show sustained physical utilization near capacity | Link capacity is a candidate bound | Payload GB/s equals physical bandwidth |

To separate translation from memory, rerun the SAME client configurations with
existing memory_service.sh base/THP modes and fixed CPU/memory node. Actual
AnonHugePages must differ while NUMA placement remains comparable. This is a
controlled page-mode test, not proof of a particular NIC TLB implementation.
If needed, repeat with fixed CPU node and alternate memory node using the already
validated service launcher; do not change CPU and memory affinity together.

Definitive NIC request-engine, NIC/IOMMU translation or DRAM-controller attribution
requires model-specific counters/event definitions. Generic process perf cache
misses do not measure DMA traffic. The collector intentionally does not invent
vendor counter names. Protocol-order attribution additionally requires documented
WQE ordering/error semantics before a safe fence-count A/B can be implemented.
This package does NOT claim that such a fence A/B has been implemented.

## Return only compact output

Send locality-results/summary.json and size-results/summary.json first. Keep
path-counters.json for the next narrowing step; it includes evidence of page
backing and counter availability. If a case fails, send that case's log instead.
Local validation covers plan invariants, same-set sorting, bounds, variable-size
payloads and summary arithmetic (7 tests); NPU execution is pending.
