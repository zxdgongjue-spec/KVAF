# V9-compatible THP CPU service

This branch is based on feat/v4-fused-batchget 7c54bd5. Device headers,
NPU kernels, Python bindings and network protocol remain unchanged.
Only CPU allocation mode and standalone diagnostic scripts are added.
THP is enabled by default (2 MiB alignment, MADV_HUGEPAGE, first-touch before
registration). The page-mode CLI override is hidden from normal help and
retained only for diagnostic base/thp comparisons. THP advice does not
guarantee huge pages; verify smaps.

Configure/build CPU service:
```
cmake -S . -B build-cpu -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON
cmake --build build-cpu --target kvaf_cpu_service -j8
build-cpu/runtime/kvaf_cpu_service --listen 127.0.0.1 --port 29877
```
For diagnostic snapshots or base comparison, use `memory_service.sh` as before.
Preserve any required
URMA include/library CMake options from the working local CPU configuration.
Existing V9-compatible NPU artifacts can be reused; do not reuse artifacts
built from ak-batchget-test. Rebuild with your original V9 NPU build command
if that directory was overwritten. The V9 --kvaf-root points to THIS checkout.

Run benchmark_thp.py in vLLM-DSA as before. No second KVAF checkout is required.
All runtime measurements still require NPU hardware; local syntax checks are
not proof of DMA correctness or large-page allocation success.
