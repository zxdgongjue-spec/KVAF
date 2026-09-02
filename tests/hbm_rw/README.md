# CPU -> NPU HBM read/write test

This is a two-process hardware test. The benchmark loop is standalone, but
its control plane and URMA data plane are the production `PeerLink` and
`CpuUrmaTransport` used by `CpuRuntime`.

Build on a host that has both CANN and UMDK/liburma:

```bash
./build.sh hbm-rw --build-dir build-hbm-rw

# Equivalent split builds are also supported:
# ./build.sh cpu --build-dir build-cpu
# ./build.sh npu --build-dir build-npu
```

On the NPU host, run the production `kvaf_npu_service`. It listens for the
CPU `AllocateHbmRegionRequest`; the CPU request carries the HBM size, slot
size, remote `device_id`, and EID list index. The NPU service then calls
`aclrtMalloc`, registers the segment, and returns the descriptor:

```bash
tests/run_hbm_rw_server.sh --port 29888
```

`--eid-index` is the NPU EID list index. The CPU-side URMA device/EID is
discovered by `CpuUrmaTransport`; use `--ub-device` or `--eid-index` only to
constrain that local discovery.

```bash
tests/run_hbm_rw_client.sh <npu-host> 29888 \
  --blocks 1024 --block-bytes 656 --device-id 0 --eid-index 0
```

The client calls `CpuUrmaTransport::Initialize`, which sends the allocation
request, imports the segment, and creates the CTP/Jetty pair through
`PeerLink`. Each benchmark operation uses the direct registered host-memory
path (`Write`/`Read`), so the block address itself is the URMA SGE and no
bounce-buffer memcpy is performed.

## Random 656-byte block benchmark

The CPU client allocates one page-aligned host region and registers it as one
URMA segment. It places `n` 656-byte blocks at shuffled offsets inside that
region; transfer order is also shuffled while remote offsets remain contiguous
as `i * block_bytes`:

```bash
tests/run_hbm_rw_client.sh <npu-host> 29888 \
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
