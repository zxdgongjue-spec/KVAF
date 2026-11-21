#!/usr/bin/env bash
# No CANN, torch, pybind11 or CMake required. Executes production kernel code
# through the deterministic CPU WQE/CQ model, plus Runtime metadata tests.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${KVAF_RANGE_HOST_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/kvaf-range-host.XXXXXX")}"
mkdir -p "$build"
cxx="${CXX:-c++}"
echo "RANGE_GET_HOST_BUILD=$build CXX=$cxx"
"$cxx" -std=c++17 -O2 -Wall -Wextra \
  -I"$root/tests/aiv_model" \
  -I"$root/access_operator/demo/aiv_batch_get/include" \
  "$root/tests/batch_get_kernel_model_test.cc" -o "$build/kernel_model"
"$cxx" -std=c++17 -O2 -Wall -Wextra \
  -I"$root/common/include" -I"$root/meta_manager/include" \
  -I"$root/access_operator/include" -I"$root/runtime/include" \
  "$root/tests/batch_get_ranges_runtime_test.cc" \
  "$root/runtime/src/runtime.cc" "$root/meta_manager/src/in_memory_meta_manager.cc" \
  -o "$build/ranges_runtime"
"$cxx" -std=c++17 -O2 -Wall -Wextra \
  -I"$root/tests/aiv_host_model" -I"$root/common/include" \
  -I"$root/meta_manager/include" -I"$root/access_operator/include" \
  -I"$root/access_operator/demo/aiv_batch_get/include" \
  "$root/tests/batch_get_ranges_operator_test.cc" "$root/access_operator/src/access_operator.cc" \
  -o "$build/ranges_operator"
"$build/kernel_model"
"$build/ranges_runtime"
"$build/ranges_operator"
echo "RANGE_GET_HOST_PASS"
