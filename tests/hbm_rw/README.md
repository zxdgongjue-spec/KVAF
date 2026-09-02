# CPU -> NPU HBM read/write test

This is an isolated two-process hardware test. It does not use `RuntimeCore`,
the KV metadata path, or the production `PeerLink` protocol.

Build on a host that has both CANN and UMDK/liburma:

```bash
./build.sh hbm-rw --build-dir build-hbm-rw
```

On the NPU host, the server listens for a CPU `AllocateHbmRegionRequest`.
The CPU request carries the HBM size, slot size, remote `device_id`, and EID
list index; the NPU then calls `aclrtMalloc`, registers the segment, and
returns the descriptor. The server-side `--device-id`/`--bytes` options are
optional compatibility checks (the CPU request is authoritative):

```bash
tests/run_hbm_rw_server.sh build-hbm-rw \
  --port 29888
```

`--eid-index` is the context EID index. For diagnostics, bypass pair-info
matching with `--npu-eid-list-index N`; `--eid HEX` remains an explicit EID
override.

On the CPU host, `local_eid` is read from `pair_info`; pass `--host-eid HEX`
only when overriding discovery:

```bash
tests/run_hbm_rw_client.sh build-hbm-rw <npu-host> 29888 \
  --blocks 1024 --block-bytes 656 --device-id 0 --eid-index 0
```

The client registers a local host buffer, imports the NPU HBM segment, creates
the CTP/Jetty pair, performs one CPU-originated WRITE and one READ, then checks
CQ completion and byte equality. The server keeps the HBM registration alive
until the client has completed the transfers.

## Random 656-byte block benchmark

The CPU client also benchmarks `n` independently allocated, page-aligned host
blocks. Their transfer order is shuffled (remote offsets remain contiguous as
`i * block_bytes`), exercising scattered CPU source addresses writing to one
contiguous HBM allocation:

```bash
tests/run_hbm_rw_client.sh build-hbm-rw <npu-host> 29888 \
  --blocks 10000 --block-bytes 656 --iterations 10 --warmup 1 --seed 1
```

The server allocates the matching total HBM size from the CPU request; no
matching `--bytes` argument is required on the NPU host.

Bandwidth is calculated from the wall-clock interval covering all WR
submissions and matching CQ completions. Per-operation latency is measured from
immediately before `urma_post_jetty_send_wr` until its CQ completion. Allocation,
registration, import, connection setup, and warmup are excluded. This is the
achieved CPU-initiated URMA payload rate, not raw HBM silicon bandwidth; with
656-byte messages it includes per-WR protocol and CQ overhead.

The test accepts explicit EIDs for diagnostics, but the default path follows
the pair-info local/remote EID rule used by the H2D script.
