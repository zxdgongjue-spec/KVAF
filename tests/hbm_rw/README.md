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
path (`WriteSglBatch`/`ReadSglBatch`), so the block address itself is the URMA
SGE and no bounce-buffer memcpy is performed.

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
submissions and matching CQ completions. Per-batch latency covers the entire
`WriteSglBatch`/`ReadSglBatch` call and timing retrieval. Allocation,
registration, import, connection setup, and warmup are excluded. This is the
achieved CPU-initiated URMA payload rate, not raw HBM silicon bandwidth; with
656-byte messages it includes per-WR protocol and CQ overhead.

Per-batch transport logs are disabled so terminal I/O does not distort the
measurement. For the single-SGE OpenMP batch path, the final summary also
reports average worker `prepare`/`post` and caller `poll` time. A separate
`_WALL` line reports `setup`, `submit_wall`, `poll_wait`, `other`, and `total`.
Worker timings are contained within `submit_wall`; do not add them to it.
The explicitly registered host address space is cached as one URMA segment,
so batch transfers reuse its
`tseg` directly instead of searching the registration list for every block.

Each block is one single-SGE WR. `--jettys N` selects 1 to 16 Jettys, with
the same number of OpenMP workers; the default is 4. Each worker owns one
Jetty, prepares its WR/SGE entries and submits each chunk directly. The caller
polls the final CQE per Jetty after
the workers join. Each Jetty supports 8192 outstanding WRs, so a batch accepts
at most `8192 * jettys` blocks. The CPU HBM client fixes `OMP_WAIT_POLICY=ACTIVE`
before OpenMP initialization, overriding any inherited policy. It does not
set CPU affinity; `OMP_PROC_BIND` and `OMP_PLACES` remain caller-controlled.

`--doorbell-wqes N` controls the maximum WRs in each worker's submission.
Zero, the default, submits the complete list for that Jetty. With `N=32`,
each worker independently prepares and posts up to 32 WRs at a time. This
is an API-level batch size; the provider implements the actual doorbell.

```bash
tests/run_hbm_rw_client.sh <npu-host> 29888 \
  --blocks 1024 --block-bytes 656 --doorbell-wqes 32 \
  --iterations 10 --warmup 1
```

For packed CPU data to multiple remote addresses, use the production
`run_cpu_batch_put_perf.sh` script with `--remote-sgl-size N`.
`CpuRuntime.initialize(..., remote_sgl_size=N)` and
`kvaf_cpu_urma_eid_perf --remote-sgl-size N` also configure this WRITE grouping.

The CPU HBM client accepts `--remote-sgl-size N` (default 1). Values above 1
use `WriteBatch` to group contiguous local source blocks into one local SGE
and up to N remote SGEs per WR. The transport clamps N to device capacity.
The WRITE request order is sorted by local address once per pass, preserving
each block's original HBM offset; this changes the posting order, not the
memory placement or contents. Gaps in sparse host memory still split groups.
READ remains single-SGE and uses the original shuffled order. The summary
reports actual `wrs`, which may exceed `ceil(blocks / N)` due to gaps and
device limits. Default N=1 preserves the random baseline.

```bash
./build-cpu/tests/kvaf_cpu_hbm_client HOST:PORT \
  --device-id 4 --jettys 4 --blocks 2048 --block-bytes 4096 \
  --remote-sgl-size 8 --warmup 20 --iterations 200
```
