# Random 656-byte concurrency diagnosis

This branch inherits V9-compatible KVAF and default THP from feat/v9-thp-memory.
It changes no runtime, device kernel, queue ordering or shared branch.

From the KVAF repository, using its existing build-npu:

```bash
KVAF_CPU_SERVICE_ADDR=127.0.0.1:29876 DEVICE_ID=0 \
python3 tests/diagnostics/random656_sweep.py \
  --build build-npu --output random656-results
cat random656-results/summary.json
```

Use the same CPU service, CPU affinity, memory node and verified THP setting
across all cases. This script does not configure the server or claim THP is
actually backed. `--build` must contain bin/kvaf_execute_batch_{get,put}_kernels.o
and python/kvaf. No rebuild is needed when these are already available with
AclGraph and kernel profile support, as on the V9-compatible THP branch.

Default: 16 subprocesses in seeded shuffled order, each 3 warmups + 20 measured
replays, plus correctness preflight and one separate instrumented launch.
Six configurations hold 7740 requests fixed while sweeping 8/16/30 AIV and
1/2 Jettys per AIV. Two configurations use 16384/32768 requests at 30 AIV and
2 Jettys. No undocumented queue-depth or WQE-ordering knobs are used.

Each case populates 1024 blocks x 128 tokens x 656 bytes = 86 MB (decimal),
selects unique random source tokens without replacement, and writes random
unique destination slots. Every replay changes source and destination plans.
Every output byte is verified outside timing. Same seed/repeat gives identical
input plans for a fixed request count across concurrency configurations.
Use `--blocks 8192` for a separate larger 688 MB random-address experiment;
source size is not silently increased with request count.

Reported GB/s uses useful bytes / device-event Graph replay time, INCLUDING
completion confirmation, EXCLUDING initialization, BatchPut, input copies and
validation. It is not directly identical to kernel-only profiler bandwidth.
An additional, separately instrumented eager launch saves per-core setup,
compaction, doorbell, post, completion and finish ticks. Summary includes max
per-core durations in ticks without assuming a hardware clock frequency.
Instrumented durations must not replace the uninstrumented event samples.

If larger requests improve throughput but extra queues do not, the fixed-size
case may be limited by startup/tail rather than queue count. If more queues at
7740 requests improve throughput, increasing usable concurrency is actionable.
Neither establishes a 40 GB/s production guarantee; the winner still needs V9
integration validation with its shared compute/communication workload.

A failed child stops the sweep and preserves its log; no failed result is
reported as zero bandwidth. Do not rerun into an old output directory.
To summarize completed cases after interruption:

```bash
python3 tests/diagnostics/random656_sweep.py \
  --output random656-results --summarize-only
```

Local validation: Python compilation, dry-run plan and three unit tests. Hardware
execution is pending. Full logs and per-case data remain local; send summary.json
first, and only the failing case log if a case fails.
