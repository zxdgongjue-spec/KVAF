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
