#!/usr/bin/env bash
# No CANN/URMA runtime required. Device DMA visibility still needs hardware.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
readonly KVAF_SGL_BUILD="${1:-$(mktemp -d "${TMPDIR:-/tmp}/kvaf-sgl-host.XXXXXX")}"
readonly KVAF_SGL_CXX="${CXX:-c++}"
mkdir -p "$KVAF_SGL_BUILD"
includes=(-Icommon/include -Iaccess_operator/include -Iaccess_operator/algorithms -Iruntime/include)
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
