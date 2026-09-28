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

The client discovers CPU and NPU EID candidates and probes each pair with an
actual WRITE, using the same selection logic as `CpuRuntime`. It releases a
failed attempt before retrying, and prints `LINK_SELECTED` and `CHANNEL_READY`
only after a probe succeeds. Discovery and probing are outside benchmark timing.

`--ub-device` and `--eid-index` constrain the local CPU endpoint only.
`--device-id` selects the remote NPU device; `--remote-eid-index` optionally pins
that device's EID candidate list index. Local EID indices and remote candidate
indices are independent. Omitting `--remote-eid-index` tries all remote
candidates; supplying it never falls back to a different remote candidate.

```bash
tests/run_hbm_rw_client.sh <npu-host> 29888 \
  --blocks 1024 --block-bytes 656 --device-id 0 --eid-index 0
```

For each candidate pair, the client allocates HBM through `PeerLink`, passes
the region to a fresh `CpuUrmaTransport::Initialize`, and calls `ValidateLink`.
The transport imports the segment and creates the CTP/Jetty pair through
`PeerLink`. Each benchmark operation uses the direct registered host-memory
path (`WriteBatch`/`ReadSglBatch`), so the block address itself is the URMA
SGE and no bounce-buffer memcpy is performed.

For example, to pin CPU EID 0 and NPU device 6's EID candidate 1:

```bash
./build-cpu/tests/kvaf_cpu_hbm_client 127.0.0.1:28111 \
  --ub-device udmac0d1e2 --eid-index 0 \
  --device-id 6 --remote-eid-index 1 \
  --jettys 4 --prepare-threads 4 --remote-sgl-size 12 \
  --blocks 4096 --block-bytes 656 --warmup 50 --iterations 1000
```

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
measurement. The final summary separates non-contiguous batch `plan`,
per-sub-batch `setup`, WR `prepare`, hardware `post`, and CQ `poll` wall time.
The contiguous READ path has no separate plan, so its validation remains in
`setup`. A separate `_WALL` line
keeps the aggregate `submit_wall`, `other`, and `total`; `submit_wall` already
contains WR preparation and post, so do not add those fields to it.
The explicitly registered host address space is cached as one URMA segment,
so batch transfers reuse its
`tseg` directly instead of searching the registration list for every block.

Each WRITE WR has one contiguous remote SGE and up to the effective local
source-SGE limit; READ keeps one local and one remote SGE per WR. `--jettys N`
selects 1 to 16 Jettys; the default is 4. `--prepare-threads N` controls
parallel grouping and WR preparation. Preparation completes before submission.
One owner then posts each
Jetty and polls that Jetty's final CQE; different Jettys submit and poll
concurrently. Each Jetty queue holds at most 8192 WRs; gathered source blocks
do not each consume a WR. The CPU HBM client fixes `OMP_WAIT_POLICY=ACTIVE`
before OpenMP initialization, overriding any inherited policy. It does not
set CPU affinity; `OMP_PROC_BIND` and `OMP_PLACES` remain caller-controlled.

`--doorbell-wqes N` controls the maximum WRs in each Jetty owner's submission.
Zero, the default, submits the complete list for that Jetty. With `N=32`,
each owner independently posts up to 32 prepared WRs at a time. This
is an API-level batch size; the provider implements the actual doorbell.

```bash
tests/run_hbm_rw_client.sh <npu-host> 29888 \
  --blocks 1024 --block-bytes 656 --doorbell-wqes 32 \
  --iterations 10 --warmup 1
```

For the production CPU BatchPut path, use `run_cpu_batch_put_perf.sh` with the
legacy `--remote-sgl-size N` option.
`CpuRuntime.initialize(..., remote_sgl_size=N)` and
`kvaf_cpu_urma_eid_perf --remote-sgl-size N` configure the same WRITE gather.

The CPU HBM client accepts `--remote-sgl-size N` (default 1). Despite its legacy
name, N requests the maximum number of local source SGEs in one WRITE WR; the
effective value is clamped by the device's `max_jfs_sge`. The benchmark source
blocks remain scattered in the registered CPU address space, while their
destination offsets are contiguous in HBM. A WR may therefore gather unrelated
CPU SGEs into one contiguous remote SGE. Every source row is kept as a separate
local SGE even if two CPU addresses happen to be adjacent. A remote gap, the effective local-SGE
cap, or `max_msg_size` starts a new WR; this path does not construct remote
scatter. Validation reads the contiguous HBM range into an independent packed
CPU buffer with one local and one remote SGE per WR, so a WRITE mapping error
cannot be hidden by reusing the same scattered layout. The summary reports
actual `wrs`, which may exceed `ceil(blocks / N)`.
Default N=1 preserves the one-source-SGE-per-WR baseline.

```bash
./build-cpu/tests/kvaf_cpu_hbm_client HOST:PORT \
  --device-id 4 --jettys 4 --blocks 2048 --block-bytes 4096 \
  --remote-sgl-size 8 --warmup 20 --iterations 200
```
