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

Per-batch transport logs are disabled so terminal I/O does not distort the
measurement. For the single-SGE OpenMP batch path, the final summary also
reports average `prepare`, `post`, and `poll` time. The explicitly registered
host address space is cached as one URMA segment, so batch transfers reuse its
`tseg` directly instead of searching the registration list for every block.

The test accepts explicit EIDs for diagnostics, but the default path follows
the pair-info local/remote EID rule used by the H2D script.

Each block is one single-SGE WR. The transport uses two fixed Jettys and two
OpenMP workers. Each worker owns one Jetty, prepares its WR/SGE entries and
submits each chunk directly. The caller polls the final CQE per Jetty after
the workers join. Each Jetty supports 1024 outstanding WRs; this standalone
single-SGE probe accepts at most 2048 blocks per batch.

`--doorbell-wqes N` controls the maximum WRs in each worker's submission.
Zero, the default, submits the complete list for that Jetty. With `N=32`,
each worker independently prepares and posts up to 32 WRs at a time. This
is an API-level batch size; the provider implements the actual doorbell.

`--window-bytes N` is an experimental completion window for diagnosing
large-batch stalls. Zero keeps the normal behavior. A non-zero value splits
one logical transfer into byte-bounded sub-batches and waits for completion
between them; for example, `--window-bytes 4M` compares an 8 MiB transfer as
two completed 4 MiB windows against the default single 8 MiB submission.

```bash
tests/run_hbm_rw_client.sh <npu-host> 29888 \
  --blocks 1024 --block-bytes 656 --doorbell-wqes 32 \
  --iterations 10 --warmup 1
```

For packed CPU data to multiple remote addresses, use the production
`run_cpu_batch_put_perf.sh` script with `--remote-sgl-size N`.
