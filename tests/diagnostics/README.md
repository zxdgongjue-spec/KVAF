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

Direct Host/device pairing requires flows or matching correlation/connection
IDs or Stream/Task IDs. In eager mode, `--last` selects linked launch groups,
rather than individual worker events; without links, all target-task statistics
are retained with a warning. Graph mode selects validated timed replay Event
brackets, described below. A missing known kernel name in eager mode triggers
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
single-Get path before trying any amortization experiment. The following capture
keeps the measured device-1/six-worker/8192-row configuration. Keep the CPU
service placement and settings unchanged:

```bash
numactl --cpunodebind=1 --membind=1 ./build-cpu/runtime/kvaf_cpu_service \
  --listen 127.0.0.1 --port 28990 --gather-threads 12 --gather-mode active
```

In the NPU terminal:

```bash
msprof --output=results/cann-gap-gather-d1-w6 \
  --ascendcl=on --runtime-api=on --task-time=on --aicpu=on \
  --application="bash tests/run_remote_cpu_batch_get_perf.sh build-npu --peer 127.0.0.1:28990 --device-id 1 --backend gather_read --aicpu-workers 6 --execution graph --graph-api torch --graph-get-count 1 --graph-replays 5 --request-length 32768 --table-capacity 16384 --miss-count 8192 --token-bytes 656 --tokens-per-block 128 --warmup 100 --iterations 1000 --kernel-profile --output-dir results/gap-gather-d1-w6"

python3 tests/diagnostics/summarize_aicpu_gap.py \
  --execution graph --msprof-dir results/cann-gap-gather-d1-w6 \
  --benchmark-dir results/gap-gather-d1-w6 --last 1000 --samples 5 \
  --max-chars 60000 --output results/gather-d1-w6-gap-core.txt
```

Repeat with `--backend cpu_push_scatter` and separate result directories.
If JSON/CSV export did not run automatically, run
`msprof --export=on --output=results/cann-gap-gather-d1-w6` before extraction.
Use fresh directories per capture, or pass the exact `PROF_*` directory.
Updating only the extractor requires no compilation; existing exports can be
reprocessed. No workload or tests were run locally.

The script infers execution and Get count from `summary.csv`; `--execution graph`
is explicit and also handles a benchmark directory containing both modes.
Capture-time launch IDs do not identify a replay. Graph grouping uses the
benchmark's record/model/record/Event-sync Host order, explicitly linked device
Events, the same device Stream track and the expected Get count. The report
labels same-stream containment `event_bracket_same_stream`, separately from
direct model/task ID links. Physical and logical Stream IDs stay separate;
matching physical IDs can span different display tracks. Graph-internal streams
can be grouped as `event_bracket_captured_get_internal_stream` only when the
exact expected task set lies within the Event window and each task has model
or captured-kernel provenance from the same Host process. This inference relies
on this benchmark's serialized replay contract, not arbitrary concurrent models.
Ambiguous brackets and unrelated work on the Event stream are rejected.
Missing associations never fall back to eager preparation/warmup chains.

For a fully associated capture above, expect `timeline.execution=graph`,
`selected_replays=1000`, `complete_chains=1000`, `expected_gets_per_replay=1`.
`model_execute_calls` includes validation/warmup. Inspect `replay_rejections` and
`linked_device_event_records` if selection is incomplete.
Unique Event connection/correlation IDs are paired independently of cross-clock
timestamp order. Clock inversions are flagged; invalid aligned Host/device gaps
are excluded, while device-only intervals remain available. Reused/ambiguous
identities still require disambiguation. Task-minus-body/entry mean comparisons
are omitted when the selected trace does not cover the complete timed selection.
`graph_match_diagnostics` gives actual same-stream, same-device-contained and
overlapping Get counts. `graph_match_examples` emits bounded relative intervals
and identity-equality flags, without original Stream/Task IDs. These distinguish
an internal stream from missing tasks or a record-window boundary mismatch.
`CAPTURE_LAUNCH_NOT_REPLAY_LINK` can still count rejected capture-ID links even
when Event-based grouping succeeds; it does not measure remaining replay coverage.

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
| msprof report | `task_sum_per_get`, `task_overlap` | Mean CANN task duration per Get and overlapping task time |
| msprof report | `aligned_start_record_begin_to_device_event_end` | Host start-record API begin to device start-record end |
| msprof report | `aligned_end_event_end_to_sync_return` | Device end-record end to Host Event-sync return |
| msprof report | `scope_mean_comparison_us` | Selected CANN task mean minus benchmark body/entry means; not paired per-call timers |

For the supplied baseline, Event E2E minus body is `228.313 - 178.787 = 49.526 us`;
Host replay/wait minus Event E2E is another `287.062 - 228.313 = 58.750 us`.
The device-record inner/outer windows approximate the hardware Event interval;
trace record-task boundaries are not exact Torch Event timestamps. The inner
window equals start-to-task + task union + inter-task gaps + task-to-end-record.
Host record-to-sync time includes aligned start-record delay, that inner window,
end-record duration and aligned return tail. Keep Host and device clock domains
separate. A zero selected-replay count means missing association, not zero overhead.

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

For submission A/B tests use `--graph-api acl --graph-submit python|native`,
keeping `--graph-get-count 1` first, then try `2` or `4` with native submission.
The native call fuses record/replay/record/wait and reuses timing Events; it
preserves the trace pattern above. The extractor includes `graph_submit`,
`host_replay_mean_us`/`host_replay_p99_us` and native Host phases. Compare Host
latency as well as Event-minus-body: stream resolution is before the native
start Event. See [the comparison commands](../../docs/batch_get_performance.md#acl-原生提交与同图多-get).

### Split CANN task-to-entry and return-to-task-end

Graph AICPU runs with `--kernel-profile` now also save `aicpu_samples.json`
beside the CSVs. Version 3 records all AICPU entry/body/return timestamps using
`clock_gettime(CLOCK_MONOTONIC_RAW)`, matching the device raw-clock anchor in
`dev_start.log`. Host profiling continues to use `steady_clock`.
Host call bounds, Event elapsed time and warmup metadata identify formal rounds.
Sample serialization and file writes happen after measured calls.
Reusing an output directory replaces this file; runs without raw sample export
remove a previous `aicpu_samples.json` to avoid analyzing stale timing samples.
Rebuild the NPU binding and AICPU package, reinstall the package and restart the
NPU process. Get entry symbols are now `ExecutionV6`, guarding the clock source
and profile layout against an older installed kernel. Raw-clock reads run only
with profiling enabled. The local sample file contains absolute timing values;
share the bounded summary, not this raw file.

Version-2 samples with mismatched CNTFRQ/HWTS frequencies cannot be repaired by
rescaling their counter values: the clock origins are not known to coincide.
Keep the old `results/can-native-empty-entry-v2` capture and collect a new v3
capture after rebuilding. Keep the CPU service unchanged and run:

```bash
bash build.sh npu

msprof --output=results/can-native-empty-entry-v3 \
  --ascendcl=on --runtime-api=on --task-time=on --aicpu=on \
  --application="bash tests/run_remote_cpu_batch_get_perf.sh build-npu --peer 127.0.0.1:28990 --device-id 1 --backend gather_read --aicpu-workers 6 --execution graph --graph-api acl --graph-submit native --graph-get-count 1 --graph-replays 8 --request-length 32768 --table-capacity 16384 --miss-count 0 --token-bytes 656 --tokens-per-block 128 --warmup 20 --iterations 50 --kernel-profile --output-dir results/native-empty-entry-v3"

python3 tests/diagnostics/summarize_aicpu_gap.py \
  --execution graph --msprof-dir results/can-native-empty-entry-v3 \
  --benchmark-dir results/native-empty-entry-v3 --last 50 --samples 5 \
  --max-chars 60000 --output results/native-empty-entry-v3-gap.txt
```

Use new directories for each capture. Repeat with `--miss-count 8192` and
separate directories. The command and workload above are for the server;
no compilation, tests or workloads were run locally.

Read `entry_boundary` in the resulting report:

| Field | Meaning |
|---|---|
| `status=paired` | Formal replay count/order, node/worker layout, Event windows and Host call cadence agree. No nearest-task matching is used. |
| `stats_us.task_outside_entry` | Per-node CANN task duration minus that node's entry span, paired before aggregation. |
| `stats_us.worker_start_skew` | Last worker entry minus first entry, independent of the trace clock. |
| `stats_us.worker_wall_mean/max` | Individual worker entry-to-return durations, distinguishing work from worker-start skew. |
| `clock.status=msprof_monotonic_raw_aligned`, `split_available=true` | Version-3 raw timestamps were aligned using this capture's device/Host raw-clock anchors. |
| `stats_us.task_begin_to_first_entry` | CANN task start to the first worker's raw-clock entry timestamp. |
| `stats_us.last_return_to_task_end` | Last worker's raw-clock return timestamp to CANN task end. |
| `stats_us.calibrated_entry_span` | Earliest entry to latest return; the two boundary gaps plus this span equal the task duration. |
| `dominant_boundary` | `before_entry`, `after_return`, or `similar_within_uncertainty`. |
| `clock.status=sync_metadata_*` | Preserve the original capture's device directory and pass its PROF root; copied JSON/CSVs alone cannot align counters. |
| `clock.metadata_diagnostics` | Failed file pattern and match counts in the counter directory and entire capture; identifies missing, relocated or duplicate metadata without exporting paths or clock values. |
| `clock.status=sync_counter_frequency_mismatch` | Legacy v2 samples use a different counter frequency; diagnostics show both Hz values and request raw-clock samples. Rebuild and recapture v3. |
| `clock.status=calibration_inconsistent_with_task_containment` | Converted entry/return timestamps fall outside their associated task; no split is emitted. |
| `clock.status=counter_and_steady_durations_disagree` | Legacy v2 hardware and steady spans disagree beyond the clock-read tolerance; no split is emitted. |
| `clock.status=bounds_only_no_independent_calibration` | Only conditional bounds are available; no measured before/after point split. |
| `clock.task_begin_to_first_entry_mean_bounds_us` | Possible mean time before user entry, under a common unit-rate clock-offset assumption. |
| `clock.last_return_to_task_end_mean_bounds_us` | Possible mean time after user return, under the same unknown offset. |
| `clock.status=no_consistent_unit_rate_offset` | A constant-offset, unit-rate mapping cannot fit all samples; check clock drift, timing domains or run association. |

All timing-shaped native warmups have the same trace pattern as measured
replays. The sample metadata identifies the `warmup + 1` native prefix (including
Event initialization replay) before formal iterations. Thus 71 validated
brackets can correctly contain 21 warmups and 50 measured rounds. `--last` is
applied to formal rounds, and a shorter selection uses the corresponding tail
of the raw samples. Missing brackets, changed run cadence or inconsistent Event
windows cause an explicit association failure. Older reports without the sample
file remain supported but cannot produce paired boundary analysis.

New runs align raw timestamps automatically using `dev_start.log*`, `host_start.log*`,
`info.json*` and `start_info*` in the original capture's device directory. Keep
these files locally and pass the capture root to `--msprof-dir`; no extra
benchmark flag or hand-written `--clock-map` is needed. Conversion follows
[msprof InfoConfReader](https://github.com/Ascend/msprof/blob/master/analysis/common_func/info_conf_reader.py):
device raw nanoseconds minus the device `clock_monotonic_raw` anchor, plus the
synchronized Host monotonic-raw anchor and export local-time offset. The HWTS
counter and its frequency are not used for v3 timestamps. `cntvct_diff` is accounted for using
the Host CPU frequency. Decimal arithmetic preserves sub-microsecond differences
when export timestamps are large. Multiple device directories require an
unambiguous device identity in the selected trace tasks.
Files ending in `.done` are completion markers and are excluded from metadata
discovery and match counts; they do not replace the corresponding log or JSON.

Converted boundaries must contain user entry/return within each associated task;
their durations must reconcile with the original raw-clock spans. The 1 us boundary tolerance covers trace
quantization; it is not a measured hardware synchronization error. No raw clock
anchor, absolute timestamp or source path is emitted in the bounded report.

Version-1 and version-2 samples remain readable. Version-1 containment bounds do **not** calibrate
the clock: the coupled ranges share an unknown offset, and their midpoint is
never reported as a measurement.

If an independent clock synchronization record is available, pass
`--clock-map clock-map.json`. Its schema is:

```text
format:       kvaf-aicpu-clock-map-v1
device_id:    same logical device ID as the benchmark
source_clock: aicpu_steady_ns
target_clock: msprof_trace_us
provenance:   msprof_clock_sync or independent_timestamp_calibration
anchors:      exactly two objects containing device_ns, trace_us, uncertainty_us
```

The anchors must come from an independent calibration, bracket the device
timestamps being analyzed and belong to this capture. Do not construct them by
placing user entry at task start or return at task end: that assumes the answer.
Unknown CANN clock-sync formats are rejected rather than guessed.
The extractor interpolates these anchors and checks task containment. It then
emits `stats_us.task_begin_to_first_entry`, `last_return_to_task_end`, their sum,
and calibration uncertainty. This manual mapping is for old steady-clock-only
samples; current version-3 samples use the automatic monotonic-raw path. A manual
mapping for v3 must name `aicpu_monotonic_raw_ns` as its source clock.

AICPU CSV rows with reused Task IDs can now be selected by ID plus a unique
exported timestamp contained in the corresponding selected task. Rows without
enough identity/time evidence remain unmatched. `timestamp_matched_rows` shows
how many rows used this disambiguation. These durations help investigate
dispatch but do not replace user entry/return timestamp alignment.

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
