#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
demo_dir=$(cd "${script_dir}/.." && pwd)
build_dir=${BUILD_DIR:-"${demo_dir}/build"}
toolkit=${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}
: "${JOBS:=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"

test -x "${toolkit}/tools/ccec_compiler/bin/ccec"
test -x "${toolkit}/tools/ccec_compiler/bin/ld.lld"
echo "[PREFLIGHT] toolkit=${toolkit} build=${build_dir}"
cmake -S "${demo_dir}" -B "${build_dir}" -DASCEND_CANN_PACKAGE_PATH="${toolkit}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel "${JOBS}"

kernel="${build_dir}/bin/kvaf_execute_batch_get_kernels.o"
test_binary="${build_dir}/bin/kvaf_execute_batch_get_test"
test -s "${kernel}"
test -x "${test_binary}"
file "${kernel}"
echo "BUILD_PASS kernel=${kernel} test=${test_binary}"
