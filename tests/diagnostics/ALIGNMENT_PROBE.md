# Fixed 656B alignment and page-crossing probe

No physical link-capacity test is included. No WQE ordering bit is changed.
Use the existing V9-compatible build; only Python diagnostics change.

## Run in this order

1. On the CPU-service host, start a separate terminal BEFORE the client:

```bash
pgrep -af kvaf_cpu_service
python3 tests/diagnostics/collect_path_counters.py --pid <PID_FOR_PORT_29876> \
  --seconds 1800 --output alignment-counters.json
```

Optional: add `--interface <actual-device-interface>` to include driver and
ethtool statistics. Do not guess an interface. The collector also records
PCI IDs, kernel, RDMA/URMA device inventory and available perf event names.
It does not enable privileged PMUs or assume CPU process misses describe DMA.
Missing commands/counters are recorded explicitly; no installation is required.

2. In another terminal, from the updated KVAF repository:

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:29876 DEVICE_ID=0 \
python3 tests/diagnostics/locality_sweep.py --build build-npu \
  --suite alignment --output alignment-results
```

Six configurations x two shuffled passes, each three warmups and twenty timed
Graph replays, plus an independent profiled launch. All use 30 AIV, two Jettys,
7740 distinct random source pages, random destination slots and 656-byte READs.
Registered source size is 85,983,232 bytes. Each block contributes the same safe
page pool for every offset, based on its ACTUAL returned remote virtual address.
Page offsets: 0, 16, 64, 3392, 3456, 3840. The first four do not cross 4 KiB;
the last two do. In particular 3392 vs 3456 retain 64-byte alignment while
changing the page crossing. Relative offsets are checked inside each valid
block. READs can span packed token boundaries; expected payload is computed
byte-by-byte from the original populated blocks, not assumed to be one token.
No larger production fetch or output layout change is proposed by this test.

3. Once the client finishes, Ctrl-C the collector to preserve final deltas.
Send alignment-results/summary.json first. Preserve alignment-counters.json;
it identifies whether suitable vendor counters exist and actual THP residency.

## Interpretation and limits

Compare non-crossing offsets to each other first, then 3392 vs 3456/3840. A
large reproducible crossing penalty supports boundary-related processing cost,
not a unique diagnosis of NIC TLB, IOMMU or physical DRAM access. With actual THP
backing, crossing 4 KiB is not necessarily crossing a translation page. Kernel
profile ticks are from a separate instrumented run, not the bandwidth sample.

The current KVAF WQE builder selects place_order=1 or 2 along with completion
notification. The checked local source documents no authoritative mapping of
these encodings to cross-post READ fencing and error propagation. Therefore no
ordering A/B, terminal no-op or skipped completion is introduced. Device model,
driver and protocol definitions are prerequisites for that change.

Host validation: eight tests cover page selection, block bounds, crossing
classification, arbitrary-byte payloads and prior sweep behavior. NPU execution
and performance are pending; no zero-copy DMA/transport simulator result is
claimed.
