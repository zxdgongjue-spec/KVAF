# DDR initialization probe

Run these commands from the KVAF root on the NPU host, using matching CPU/NPU
builds. The probe loads no model and does not fill the DDR pool with benchmark
data. It retains KVAF's initialization-time connectivity probes.

Start one local CPU service in a separate terminal:

```bash
./build-cpu/runtime/kvaf_cpu_service \
  --listen 127.0.0.1 --port 34555 --memory-page-mode thp \
  > cpu-34555.log 2>&1
```

Use the actual service binary path if your build directory differs. Wait for
`LISTEN` in its log before running clients. `127.0.0.1` assumes the clients and
service share the host network namespace. Fix the service's NUMA placement
across experiments when placement matters; check node capacity as well as
total free memory.

Measure one device at a time, freeing each region before the next sample:

```bash
for gib in 1 8 16 32; do
  python3 -u tests/diagnostics/ddr_init_probe.py \
    --build build-npu --address 127.0.0.1:34555 \
    --device 0 --ddr-gib "$gib" \
    > "ddr-device0-${gib}g.log" 2>&1 || break
done
```

`--ddr-gib` controls the first argument to `initialize(ddr_bytes, slot_size)`.
It is rounded up to a whole number of slots. The default 86,016-byte slot
matches the local vLLM layout: 656 bytes/token x 128 tokens, aligned to 4,096.
The initial `config` record prints requested and actual bytes; `ready` prints
the returned remote region. Slot size must stay fixed for capacity comparisons.
Set `--slot-size` to match the deployed layout if it differs. The kernel paths
default to `build-npu/bin/kvaf_execute_batch_{get,put}_kernels.o`; override them
with `--batch-get-kernel` and `--batch-put-kernel` if needed.

Read the JSON log line with `event=done`, `operation=initialize`, `elapsed_s`:

```text
DDR_INIT_PROBE {"device": 0, "elapsed_s": 12.345, "event": "done", "operation": "initialize", ...}
```

This is elapsed wall time around the synchronous Python/native initialize
call. It includes CPU service queuing, allocation, pre-touch (in thp mode),
registration, metadata/channel setup and connectivity probes. It excludes
constructor, link_to, holding and shutdown, which have separate records. The
number is NOT the isolated malloc time or steady-state BatchGet time. No
external NPU synchronization is added to this scope.

For sequential multi-device allocation with earlier regions retained, start
one foreground terminal per device and use `--hold`:

```bash
python3 -u tests/diagnostics/ddr_init_probe.py \
  --build build-npu --device 0 --ddr-gib 32 --hold
```

Wait for `ready`/`hold`, then start device 1 in another terminal with
`--device 1`, and so on. Keep every earlier terminal at its hold prompt. Press
Enter only after all measurements are complete. Eight live 32-GiB requests
require about 256 GiB for the regions plus runtime/system memory. Device IDs
are logical IDs under the current visibility configuration.

For overlapping launches, `--hold-seconds 3600` retains completed regions
for an hour without requiring terminal input. Background processes launched
in a shell loop are overlapping, not a strict simultaneous barrier. On this
version, one CPU service processes requests serially regardless of client count.
Compare each client's initialize time with server stage times before concluding
there is parallel allocation contention.

To isolate the server work, add monotonic BEGIN/END timing around
`posix_memalign`, `madvise`, `memset`, `urma_alloc_token_id`, and
`urma_register_seg` in `runtime/src/cpu_runtime.cc`, plus request-handling
BEGIN/END. The existing `REGION_ALLOCATED` message is emitted after registration
and has no elapsed duration. Do not infer malloc duration from that message.
Correlate request attempts: candidate fallback can allocate the full region
more than once. Optional `--peer-ub-device` and `--npu-eids` allow a known-working
path to be held fixed. Keep these and the AIV configuration fixed across runs.

For THP diagnosis, the existing thp/origin comparison is only a comparison of
two practical code paths: thp also pre-touches all bytes, while origin does
not and does not explicitly forbid THP. A causal THP comparison needs matched
alignment and pre-touch in both paths.

Local validation (no NPU required):

```bash
python3 tests/diagnostics/ddr_init_probe.py \
  --build build-npu --ddr-gib 32 --dry-run
```

Actual initialization, NUMA placement and service timings require an NPU host.
