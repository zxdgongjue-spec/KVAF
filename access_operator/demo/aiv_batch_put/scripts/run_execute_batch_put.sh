#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
demo_dir=$(cd "${script_dir}/.." && pwd)
build_dir=${BUILD_DIR:-"${demo_dir}/build"}
result_dir=${RESULT_DIR:-"${demo_dir}/results/execute_batch_put_$(date +%Y%m%d_%H%M%S)"}
timeout_seconds=${TIMEOUT_SECONDS:-180}
toolkit=${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}

: "${DEVICE_ID:=0}"
: "${ROWS:=4096}"
: "${MIN_BYTES:=1024}"
: "${MAX_BYTES:=1024}"
: "${DESTINATION_STRIDE:=1024}"
: "${AIV_CORES:=1}"
: "${JETTYS_PER_CORE:=2}"
: "${ITERATIONS:=5}"

mkdir -p "${result_dir}"
log_file="${result_dir}/run.log"
feedback_file="${result_dir}/feedback.txt"
kernel=${KERNEL_OBJECT:-"${build_dir}/bin/kvaf_execute_batch_put_kernels.o"}
test_binary="${build_dir}/bin/kvaf_execute_batch_put_test"

if [[ ! -x "${test_binary}" || ! -f "${kernel}" ||
      -n "$(find "${demo_dir}" \( -path "${demo_dir}/build*" -o -path "${demo_dir}/results" \) -prune -o \
            -type f -newer "${test_binary}" -print -quit)" ]]; then
  echo "[EXECUTE_BATCH_PUT] build artifact missing_or_stale; running scripts/build.sh" | tee "${log_file}"
  BUILD_DIR="${build_dir}" bash "${script_dir}/build.sh" 2>&1 | tee -a "${log_file}"
fi

driver_lib=${ASCEND_DRIVER_LIB_PATH:-/usr/local/Ascend/driver/lib64}
cann_lib="${toolkit}/lib64"
export LD_LIBRARY_PATH="${build_dir}/bin:${cann_lib}:${driver_lib}:${LD_LIBRARY_PATH:-}"
echo "[EXECUTE_BATCH_PUT] mode=eager rows=${ROWS} min_bytes=${MIN_BYTES} max_bytes=${MAX_BYTES} destination_stride=${DESTINATION_STRIDE} aiv_cores=${AIV_CORES} jettys_per_core=${JETTYS_PER_CORE} iterations=${ITERATIONS}" | tee "${log_file}"

set +e
timeout "${timeout_seconds}" "${test_binary}" --kernel "${kernel}" \
  --device "${DEVICE_ID}" --rows "${ROWS}" --min-bytes "${MIN_BYTES}" \
  --max-bytes "${MAX_BYTES}" --destination-stride "${DESTINATION_STRIDE}" \
  --aiv-cores "${AIV_CORES}" --jettys-per-core "${JETTYS_PER_CORE}" \
  --iterations "${ITERATIONS}" 2>&1 | tee -a "${log_file}"
status=${PIPESTATUS[0]}
set -e

{
  printf 'result=%s\n' "$([[ ${status} -eq 0 ]] && printf pass || printf fail)"
  printf 'exit_status=%s\n' "${status}"
  printf 'run_log=%s\n' "${log_file}"
  grep -a -E 'PHASE_|EAGER_ITERATION_PASS|EXECUTE_BATCH_PUT_VALIDATION|CORE_' "${log_file}" || true
} > "${feedback_file}"
cat "${feedback_file}"
if [[ ${status} -eq 0 ]]; then
  echo "RUN_PASS feedback=${feedback_file} log=${log_file}"
else
  echo "RUN_FAIL status=${status} feedback=${feedback_file} log=${log_file}" >&2
fi
exit "${status}"
