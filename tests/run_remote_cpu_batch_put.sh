#!/usr/bin/env bash
# Run the variable-length single-Tensor vs Tensor-list BatchPut benchmark.
set -euo pipefail

usage() {
  echo "Usage: $0 /absolute/path/to/kvaf/build-npu" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

readonly build_dir="$1"
readonly script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly root_dir="$(cd "${script_dir}/.." && pwd)"
readonly get_kernel="${build_dir}/bin/kvaf_execute_batch_get_kernels.o"
readonly put_kernel="${build_dir}/bin/kvaf_execute_batch_put_kernels.o"
readonly test_program="${root_dir}/tests/python/test_npu_batch_put.py"

if [[ ! -s "${get_kernel}" ]]; then
  echo "ERROR: BatchGet kernel not found: ${get_kernel}" >&2
  exit 1
fi
if [[ ! -s "${put_kernel}" ]]; then
  echo "ERROR: BatchPut kernel not found: ${put_kernel}" >&2
  exit 1
fi
if [[ ! -d "${build_dir}/python/kvaf" ]]; then
  echo "ERROR: Python binding directory not found: ${build_dir}/python/kvaf" >&2
  exit 1
fi
if [[ ! -f "${test_program}" ]]; then
  echo "ERROR: test program not found: ${test_program}" >&2
  exit 1
fi

: "${KVAF_CPU_SERVICE_ADDR:=127.0.0.1:28999}"
: "${DEVICE_ID:=0}"
: "${KVAF_REMOTE_BATCH_PUT_TIMEOUT:=600}"
if ! [[ "${KVAF_REMOTE_BATCH_PUT_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: KVAF_REMOTE_BATCH_PUT_TIMEOUT must be a positive integer" >&2
  exit 2
fi

export KVAF_CPU_SERVICE_ADDR
export DEVICE_ID
export KVAF_RUN_NPU_TESTS=1
export KVAF_BATCH_GET_KERNEL="${get_kernel}"
export KVAF_BATCH_PUT_KERNEL="${put_kernel}"
export PYTHONPATH="${build_dir}/python${PYTHONPATH:+:${PYTHONPATH}}"

echo "[BATCH_PUT_TEST] build_dir=${build_dir}"
echo "[BATCH_PUT_TEST] cpu_service=${KVAF_CPU_SERVICE_ADDR} device_id=${DEVICE_ID}"
echo "[BATCH_PUT_TEST] get_kernel=${get_kernel}"
echo "[BATCH_PUT_TEST] put_kernel=${put_kernel}"
echo "[BATCH_PUT_TEST] timeout_seconds=${KVAF_REMOTE_BATCH_PUT_TIMEOUT}"

exec timeout "${KVAF_REMOTE_BATCH_PUT_TIMEOUT}s" \
  python3 "${test_program}"
