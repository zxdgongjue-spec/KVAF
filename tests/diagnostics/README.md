# Paired bandwidth timing scopes

Run from KVAF on ak-batchget-test, using its matching build-npu. Existing
benchmark and kernel files are not modified. The runner instruments a copy of
the benchmark source in memory and refuses unknown source layouts.

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:29876 DEVICE_ID=0 \
python3 tests/diagnostics/bandwidth_scopes.py \
  --build build-npu --output bandwidth-5000.json --clock-mhz 1000 \
  --backend aiv --execution eager --get-timing async-drain \
  --token-bytes 656 --request-length 32768 --miss-count 5000 \
  --aiv-cores 2 --jettys-per-core 1 --warmup 3 --iterations 30
```

Repeat with miss-count 2048 and 8192, using different output names. Keep the
same machine, runtime build and network configuration for V9. Clock MHz is an
explicit assumption, not autodetection; retain raw ticks and confirm frequency.

Each JSON sample pairs host elapsed, external NPU event elapsed, kernel wall
elapsed, and first-doorbell-to-last-completion elapsed with payload bytes.
Host/event instrumentation itself has overhead. Kernel wall uses cross-core
min/max timestamps, so it requires a common clock. Transfer overlaps posting:
do not add those durations. Original correctness preflight remains active.
The independent upstream script remains the uninstrumented control.

Instrumentation follows the current eager sampling loop, including its
`async-drain` / `sync-call` completion branches. It exports paired samples after
the loop, independently of the optional submit-only output. The Event end is
waited outside the measured Host interval before reading elapsed time; input
generation, output validation, and profile snapshots remain in their existing
locations. Use `--get-timing sync-call` to diagnose the synchronous call path.
This diagnostic pairs eager samples; use the main benchmark for Graph timing.

Validation: python3 -m unittest discover -s tests/diagnostics -p 'test_*.py'.
Actual device execution must be validated on an NPU.

## Compact AICPU gap report

After exporting msprof JSON/CSVs, extract only core diagnostic information locally:

```bash
python3 tests/diagnostics/summarize_aicpu_gap.py \
  --msprof-dir results/cann-gap-gather \
  --benchmark-dir results/gap-gather \
  --last 50 --samples 5 --max-chars 60000 \
  --output results/gather-gap-report.txt
```

Copy the contents of `gather-gap-report.txt`. It contains selected benchmark
gap/Host metrics, critical API/task duration statistics and up to five linked
launch/task/copy/sync timelines, with counts and P50/P99. It does not export
source paths, payloads, addresses, EIDs or arbitrary event arguments. Unknown
operator names are replaced by category labels. The same ASCII JSON is written
to stdout and the output file, within the requested character limit (up to
65535); omitted detail is identified in `omitted_sections`.

The script uses only Python's standard library and reads the existing files;
it does not compile, run a workload or change source reports. If the root has
multiple `PROF_*` captures, point `--msprof-dir` to one exact capture instead.
JSON may be a Chrome trace array or a `traceEvents` object; split exports and
CSV time units are supported. `displayTimeUnit` is a viewer preference, not a
conversion of Chrome trace microseconds.

Host/device pairing requires flows or matching correlation/connection IDs or
Stream/Task IDs. Unlinked data stays unlinked. `--last` selects linked launch
groups, rather than individual worker events; without links, all target-task
statistics are retained with a warning. A missing known kernel name triggers
an explicitly marked all-AICPU fallback; `--kernel-pattern` can restrict the
selection to a locally known device kernel name. All-trace API and CSV totals
include setup/warmup/readback and must not be treated as measured-round latency.
Relative Host/device gaps use msprof's clock alignment and report invalid time
ordering; benchmark function-body time is not substituted for a paired CANN
task duration. CSVs alone cannot resolve the full dispatch/completion timeline.
`csv_supplement.selected_aicpu_rows` filters only unique selected Task IDs;
reused or missing IDs are excluded with coverage counts. These CSV statistics
are per exported row (possibly per worker), not a replacement for the full task
envelope or an additive partition of it.

### AICPU binding and graph gap comparisons

Update the Host library, device SO and kernel JSON together (`./build.sh npu`
installs the package), then restart the NPU process. The Get entries are V4;
the CPU push wire protocol stays V3. These changes have not been compiled or
run locally.

Eager binding A/B uses the same fixed request storage, updating its contents
outside measurement. The prepared callable reads Tensor metadata once and keeps
the Tensor/stream owners; do not resize or replace any captured storage. Every
submission still reads `npu_stream` once and uses the normal execution, lease
and completion path. Compare both backends without msprof first:

```bash
for backend in gather_read cpu_push_scatter; do
  for binding in ordinary prepared; do
    bash tests/run_remote_cpu_batch_get_perf.sh build-npu \
      --peer 127.0.0.1:28990 --device-id 7 --backend "$backend" \
      --aicpu-workers 4 --execution eager --get-binding "$binding" \
      --request-length 32768 --table-capacity 4096 --miss-count 4096 \
      --token-bytes 656 --tokens-per-block 128 --warmup 20 --iterations 100 \
      --kernel-profile --output-dir "results/binding-$backend-$binding"
  done
done
```

Graph defaults remain one Get per replay with Torch capture. Diagnose this
single-Get path before trying any amortization experiment:

```bash
msprof --output=results/cann-gap-gather-graph \
  --ascendcl=on --runtime-api=on --task-time=on --aicpu=on \
  --application="bash tests/run_remote_cpu_batch_get_perf.sh build-npu --peer 127.0.0.1:28990 --device-id 7 --backend gather_read --aicpu-workers 4 --execution graph --graph-api torch --request-length 32768 --table-capacity 4096 --miss-count 4096 --token-bytes 656 --tokens-per-block 128 --warmup 20 --iterations 50 --kernel-profile --output-dir results/gap-gather-graph"

python3 tests/diagnostics/summarize_aicpu_gap.py \
  --msprof-dir results/cann-gap-gather-graph \
  --benchmark-dir results/gap-gather-graph --last 50 --samples 5 \
  --max-chars 60000 --output results/gather-graph-gap-core.txt
```

Repeat with `--backend cpu_push_scatter` and separate result directories.
Export msprof JSON/CSVs before extraction, as with the eager capture. If replay
flows/IDs are absent, the report keeps tasks unlinked; capture-time launch IDs
alone cannot establish replay ownership. Missing Event-record links also remain
missing, rather than being treated as zero overhead.

Read the following intervals together; inclusive subspans must not be added:

| Source | Metric | Meaning |
|---|---|---|
| `profile.csv` | `host_stream.select`, `host_stream.handle` | Current-stream selection versus stream getter/queue boundary |
| `profile.csv` | `device_boundary.entry_to_body`, `body_to_return` | Instrumented device entry/setup and exit/unlock/profile publication |
| `profile.csv` | `device_boundary.entry_span` | Earliest instrumented entry to latest instrumented return |
| `profile.csv` | `gap.entry_outside_body` | Entry span minus the original kernel body span |
| `profile.csv` | `gap.e2e_outside_entry` | Remaining E2E gap outside that entry span |
| `profile.csv` | `gap.graph_host_outside_event` | Host replay-and-wait minus Event elapsed; not part of Event E2E |
| msprof report | `aligned_launch_to_task_begin` | Model execute start to first linked AICPU task |
| msprof report | `aligned_task_end_to_sync_return` | Last task end to Event/stream sync return |
| msprof report | `device_event_start_to_task`, `device_task_to_end_event` | Device gaps between linked Event record tasks and AICPU tasks, when exported |
| msprof report | `task_envelope`, `task_union`, `inter_task_gaps` | Full target-task span, union of task intervals, and gaps between tasks |

The body metric is unchanged: `kernel_mean_us` still excludes setup before its
old entry timestamp and cleanup after its old exit timestamp. Compare CANN's
full task envelope with the new entry span to locate the previously uncovered
~26 us. This remaining difference includes framework code outside the exported
entry and is an aggregate scope comparison, not a per-call paired timer.
The return timestamp is immediately before its final store/C return, not after
the CANN task finishes. Host and device absolute clocks must not be mixed.

If the gap is mainly before/after the task, compare `--graph-api acl` with
`--graph-api torch`, keeping the workload and single Get unchanged. Use
unprofiled runs to assess gains; msprof itself changes Host timings. If task
envelope minus entry span dominates, additional Python binding optimizations
will not remove that device framework overhead.

Optional `--graph-get-count 4` or `16` captures sequential Get nodes in one
replay to test amortization of fixed replay/Event costs. The default is `1`.
With multiple nodes, `summary.csv` explicitly reports `amortized_per_get`:
Event duration is divided by node count; kernel duration is the mean of each
node's body span, excluding inter-node gaps. All nodes' status/profiles are read
outside timing. `host_graph` stays per replay and `host_graph_per_get` is divided
by node count. This experiment does not reduce single-Get latency or establish
that per-task AICPU framework costs are eliminated.

## Get recovery fault injection (Host only)

`get_recovery.cc` uses the real Dispatcher, RuntimeCore and MetaManager with a
fault-injecting operator. Run from the repository root without CANN or URMA:

```bash
c++ -std=c++17 -pthread \
  -Icommon/include -Iaccess_operator/include -Iruntime/include -Imeta_manager/include \
  tests/diagnostics/get_recovery.cc \
  runtime/execution/execution_config.cc \
  runtime/execution/execution_dispatcher.cc \
  runtime/execution/execution_resource_manager.cc \
  runtime/src/runtime.cc meta_manager/src/in_memory_meta_manager.cc \
  -o /tmp/kvaf-get-recovery
/tmp/kvaf-get-recovery
```

It checks persistent synchronous/asynchronous device failures, close and rebuild
retries, table pin retention/release, failed output unbinding, unrelated pending
requests and live-graph recovery guards. Add `-fsanitize=address,undefined -g`
with a supporting compiler for sanitizer checks. This standalone regression does
not test real DMA completion or restore the removed full C++ test suite.
