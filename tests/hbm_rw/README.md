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

The CPU client reserves one page-aligned host virtual address space and
registers it as one URMA segment. It touches only `n` non-overlapping 656-byte
blocks at random offsets inside that space; transfer order is also shuffled
while remote HBM offsets remain contiguous as `i * block_bytes`. The host
address-space size is independent from the HBM allocation size:

```bash
tests/run_hbm_rw_client.sh <npu-host> 29888 \
  --blocks 1024 --block-bytes 656 --host-space-bytes 100G \
  --iterations 10 --warmup 1 --seed 1
```

The CPU `AllocateHbmRegionRequest` still carries only
`1024 * 656 = 671744` bytes, so the NPU allocates that amount of contiguous
HBM. `--host-space-bytes` controls only the CPU-side virtual address range;
it does not increase the remote HBM allocation.

Bandwidth is calculated from the wall-clock interval covering all WR
submissions and matching CQ completions. Per-operation latency is measured from
immediately before `urma_post_jetty_send_wr` until its CQ completion. Allocation,
registration, import, connection setup, and warmup are excluded. This is the
achieved CPU-initiated URMA payload rate, not raw HBM silicon bandwidth; with
656-byte messages it includes per-WR protocol and CQ overhead.

The test accepts explicit EIDs for diagnostics, but the default path follows
the pair-info local/remote EID rule used by the H2D script.

Each block is one single-SGE WR. With `--jettys N`, the client creates `N`
independent local JFC/JFR/Jetty/CTP resources. An OpenMP `parallel for` builds
the disjoint WR/SGE entries, then the calling thread posts every Jetty list and
polls one final CQE per Jetty. `--omp-threads N` controls only WQE preparation;
it is independent of `--jettys N` and defaults to 1. The current resource limit
is 16 CPU Jettys per channel and 1024 requests per logical batch.

Use `--sgl-size N` to combine `N` blocks into one WR with `N` local SGEs. The
default `N=1` uses the OpenMP batch path above. Values greater than one use the
transport's `WriteSgl`/`ReadSgl` path and must not exceed the device's reported
`max_jfs_sge` value.

For example, this sends 1024 blocks as 512 two-SGE WRs:

```bash
tests/run_hbm_rw_client.sh <npu-host> 29888 \
  --blocks 1024 --block-bytes 656 --sgl-size 2 \
  --iterations 10 --warmup 1
```

`--doorbell-wqes N` optionally splits each Jetty list into groups of `N` WRs.
Zero, the default, submits one linked list per Jetty. This is an API-level
approximation: `urma_post_jetty_send_wr` may coalesce or implement doorbells
internally.

For example:

```bash
tests/run_hbm_rw_client.sh <npu-host> 29888 \
  --blocks 1024 --block-bytes 656 --jettys 4 --omp-threads 4 \
  --doorbell-wqes 32 \
  --iterations 10 --warmup 1
```
