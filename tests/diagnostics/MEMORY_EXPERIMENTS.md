# Random 656B production-shaped memory experiments

Branch: feat/random-fetch-memory-diagnostics. No TopK-dependent allocation or
extra payload bytes. Existing CPU default stays THP. New `base` page mode uses
MADV_NOHUGEPAGE and first-touch, while THP uses existing MADV_HUGEPAGE/first-touch.
This is THP, not explicit hugetlb. Advice success does not prove huge pages:
inspect service smaps/numa_maps during the registered region's lifetime.
Do not change machine-wide THP or huge-page settings.

Rebuild the CPU service with your existing CPU build configuration after checkout:
`cmake --build build-cpu -j8` (replace build-cpu by your configured directory).
The NPU build must match this branch's ak-batchget-test ABI. No kernel changes.

Use an UNUSED service port, e.g. 29877. Never terminate the existing production
service. The wrapper starts and stops only its own PID. Choose valid CPU and
memory nodes using `numactl --hardware`; keep CPU_NODE unchanged for all runs.
Keep the same --ub-device/--eid-index across cases if configured in your service.

Terminal A, on the CPU service host, e.g. CPU node0 and memory node0:
```
bash tests/diagnostics/memory_service.sh \
  build-cpu/bin/kvaf_cpu_service 0 0 base memory-node0-base \
  --listen 127.0.0.1 --port 29877
```
Terminal B on the NPU host (change service IP if remote):
```
KVAF_CPU_SERVICE_ADDR=127.0.0.1:29877 DEVICE_ID=0 \
 bash tests/diagnostics/run_memory_client.sh build-npu results-node0-base 1000
```
Stop terminal A with Ctrl-C after client completes. Repeat with memory node1
(if present), still CPU node0 and base mode; then repeat node0 with thp. Use new
output names. Run baseline again with another output name to detect drift.
Single-NUMA machines cannot perform the node comparison; report unsupported.
All comparisons need node free memory for the largest registered pool (~2 GiB
plus allocator rounding and other service memory).

Each client runs strides656/704/768/1024 at fixed656B transferred per request,
5000 random logical requests, seed20260903, table16384 blocks, logical128 tokens
per block, 2 AIV / 2 Jetty. Storage has padding; output stays tightly packed.
Runtime block capacity is enlarged to legally contain padded offsets; extra
logical token slots are never requested. Existing payload preflight stays on.
Source benchmark is adapted in memory; original test files are untouched.
All variants have the same random logical request sequence, not identical
physical addresses. Padding changes page footprint as well as alignment, so a
win cannot by itself identify the exact hardware mechanism.

Compare stride656 across nodes (NUMA), stride656 across base/thp (pages), and
all strides within the same node/page case (layout). Use kernel_wall_us and
transfer_window_us, not host bandwidth. Clock1000 is an explicit assumption.
Preserve raw profiles. The source already enabled THP by default: enabling it
again is not an optimization. Check actual region placement/AnonHugePages;
whole-process summaries alone cannot establish transport-registration page size.

Local checks cover source transformation, bounds, unchanged payload length and
syntax. CPU service build and NPU execution require the target development
libraries/hardware; no bandwidth improvement is claimed before those runs.
