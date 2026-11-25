#!/usr/bin/env bash
# No CANN/URMA runtime required. Device DMA visibility still needs hardware.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
readonly KVAF_SGL_BUILD="${1:-$(mktemp -d "${TMPDIR:-/tmp}/kvaf-sgl-host.XXXXXX")}"
readonly KVAF_SGL_CXX="${CXX:-c++}"
readonly KVAF_SGL_CLANGXX="${CLANGXX:-clang++}"
mkdir -p "$KVAF_SGL_BUILD"
includes=(-Icommon/include -Iaccess_operator/include -Iaccess_operator/algorithms -Iruntime/include)
# Preserve GM as a distinct Clang address space. The executable model below
# erases that qualifier and cannot catch local/GM struct assignment errors.
if command -v "$KVAF_SGL_CLANGXX" >/dev/null 2>&1; then
  "$KVAF_SGL_CLANGXX" -std=c++17 -fsyntax-only -x c++ \
    -DKVAF_CPU_SGL_AIV_MODEL=1 -DKVAF_CPU_SGL_AIV_ADDRESS_SPACE_CHECK=1 \
    -Itests/diagnostics/aiv_sgl_model "${includes[@]}" \
    access_operator/algorithms/get/cpu_sgl_write/urma_aiv/kernel/cpu_sgl_write_aiv.asc
  printf '%s\n' 'AIV CPU SGL GM address-space check passed'
else
  printf '%s\n' 'SKIP: AIV GM address-space check requires clang++ (or CLANGXX)'
fi
"$KVAF_SGL_CXX" -std=c++17 -pthread -fsanitize=address,undefined -g "${includes[@]}" \
  -Itests/diagnostics/aiv_sgl_model tests/diagnostics/cpu_sgl_write_aiv.cc -o "$KVAF_SGL_BUILD/aiv"
"$KVAF_SGL_BUILD/aiv"
"$KVAF_SGL_CXX" -std=c++17 -pthread -fsanitize=address,undefined -g "${includes[@]}" \
  tests/diagnostics/cpu_sgl_write.cc \
  access_operator/algorithms/get/scatter/urma_aicpu/kernel/cpu_push_scatter.cc \
  runtime/execution/execution_config.cc runtime/peer/execution_protocol.cc -o "$KVAF_SGL_BUILD/aicpu"
"$KVAF_SGL_BUILD/aicpu"
# Both feature combinations must compile; these fake ACL declarations prove
# Host C++ syntax only, not compatibility with the deployed CANN version.
for features in '-DKVAF_ENABLE_AIV_BATCH_GET=1' '-DKVAF_ENABLE_AICPU_CPU_PUSH=1'; do
  "$KVAF_SGL_CXX" -std=c++17 -fsyntax-only "$features" "${includes[@]}" \
    -Itests/diagnostics/aiv_sgl_model -Iaccess_operator/providers/common \
    access_operator/algorithms/get/scatter/urma_aicpu/host/cpu_push_host.cc
done
"$KVAF_SGL_CXX" -std=c++17 -fsyntax-only "${includes[@]}" \
  -Itests/diagnostics/aiv_sgl_model -Iaccess_operator/providers/common \
  -Iaccess_operator/algorithms/get/cpu_sgl_write/urma_aiv/include \
  -Iaccess_operator/algorithms/get/direct/urma_aiv/include \
  access_operator/algorithms/get/cpu_sgl_write/urma_aiv/host/access_operator.cc \
  access_operator/algorithms/get/cpu_sgl_write/urma_aiv/host/loader.cpp
mkdir -p "$KVAF_SGL_BUILD/generated"
cp runtime/cmake/npu_kernel_paths.h.in "$KVAF_SGL_BUILD/generated/npu_kernel_paths.h"
"$KVAF_SGL_CXX" -std=c++17 -fsyntax-only -DKVAF_ENABLE_AIV_BATCH_GET=1 "${includes[@]}" \
  -Itests/diagnostics/aiv_sgl_model -Imeta_manager/include \
  -Iaccess_operator/algorithms/get/direct/urma_aiv/include -I"$KVAF_SGL_BUILD/generated" \
  runtime/src/npu_runtime.cc
python3 -m unittest discover -s tests/diagnostics -p 'test_*.py'
if [[ -n "${KVAF_TEST_URMA_INCLUDE_DIR:-}" ]]; then
  "$KVAF_SGL_CXX" -std=c++17 -O1 -g -pthread -fsanitize=address,undefined "${includes[@]}" \
    -Iruntime -I"$KVAF_TEST_URMA_INCLUDE_DIR" \
    tests/diagnostics/cpu_sgl_transport.cc runtime/cpu_service/push/cpu_push_get_transport.cc \
    -o "$KVAF_SGL_BUILD/transport"
  "$KVAF_SGL_BUILD/transport"
else
  printf '%s\n' 'SKIP: fake URMA transport requires KVAF_TEST_URMA_INCLUDE_DIR'
fi
