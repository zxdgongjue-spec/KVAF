#!/usr/bin/env bash
set -euo pipefail

# This runner deliberately has a small surface: it exercises only the new
# device-side ExecuteBatchGet ABI, not the historic descriptor benchmark.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
demo_dir=$(cd "${script_dir}/.." && pwd)
build_dir=${BUILD_DIR:-"${demo_dir}/build"}
result_dir=${RESULT_DIR:-"${demo_dir}/results/execute_batch_get_$(date +%Y%m%d_%H%M%S)"}
timeout_seconds=${TIMEOUT_SECONDS:-180}
toolkit=${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}

: "${DEVICE_ID:=0}"
: "${TABLE_CAPACITY:=4096}"
: "${REQUESTS:=4096}"
: "${OUTPUT_ROWS:=${REQUESTS}}"
: "${TOKEN_BYTES:?TOKEN_BYTES must configure the KV bytes per token}"
: "${TOKENS_PER_BLOCK:?TOKENS_PER_BLOCK must configure the tokens per block}"
BLOCK_BYTES=${TOKEN_BYTES}
: "${SOURCE_STRIDE:=$((TOKEN_BYTES * TOKENS_PER_BLOCK))}"
: "${AIV_CORES:=4}"
: "${JETTYS_PER_CORE:=2}"
: "${WARMUP:=10}"
: "${ITERATIONS:=50}"
: "${MASK_EVERY:=0}"
: "${ACTIVE_EVERY:=0}"
: "${ACTIVE_PERCENT:=5}"
: "${GRAPH_REPLAYS:=0}"
: "${GRAPH_WARMUP:=5}"
: "${GRAPH_ITERATIONS:=20}"
: "${DYNAMIC_OUTPUT_SMOKE:=0}"

mkdir -p "${result_dir}"
log_file="${result_dir}/run.log"
feedback_file="${result_dir}/feedback.txt"
plog_dir="${result_dir}/plog"
run_started="${result_dir}/.run_started"
kernel="${build_dir}/bin/kvaf_execute_batch_get_kernels.o"
test_binary="${build_dir}/bin/kvaf_execute_batch_get_test"
touch "${run_started}"

if [[ ! -x "${test_binary}" || ! -f "${kernel}" ||
      -n "$(find "${demo_dir}" \( -path "${demo_dir}/build*" -o -path "${demo_dir}/results" \) -prune -o \
            -type f -newer "${test_binary}" -print -quit)" ]]; then
  echo "[EXECUTE_BATCH_GET] build artifact missing_or_stale; running scripts/build.sh" | tee "${log_file}"
  BUILD_DIR="${build_dir}" bash "${script_dir}/build.sh" 2>&1 | tee -a "${log_file}"
fi
if [[ ! -x "${test_binary}" || ! -f "${kernel}" ]]; then
  echo "ERROR: ExecuteBatchGet build artifacts are missing after build." | tee -a "${log_file}" >&2
  exit 2
fi

driver_lib=${ASCEND_DRIVER_LIB_PATH:-/usr/local/Ascend/driver/lib64}
cann_lib="${toolkit}/lib64"
if [[ ! -r "${cann_lib}/libunified_dlog.so" ]]; then
  echo "RUNTIME_LIB_PRECHECK_FAIL missing=${cann_lib}/libunified_dlog.so" | tee -a "${log_file}" >&2
  exit 2
fi
export ASCEND_CANN_PACKAGE_PATH="${toolkit}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${toolkit}}"
export LD_LIBRARY_PATH="${build_dir}/bin:${cann_lib}:${driver_lib}:${LD_LIBRARY_PATH:-}"
if ldd "${test_binary}" 2>/dev/null | grep -q 'not found'; then
  echo "RUNTIME_LIB_PRECHECK_FAIL executable_dependency_missing" | tee -a "${log_file}" >&2
  ldd "${test_binary}" 2>&1 | tee -a "${log_file}" >&2 || true
  exit 2
fi
if dumper=$(command -v asc_dumper 2>/dev/null) && ldd "${dumper}" 2>/dev/null | grep -q 'not found'; then
  echo "RUNTIME_LIB_PRECHECK_FAIL asc_dumper_dependency_missing binary=${dumper}" | tee -a "${log_file}" >&2
  ldd "${dumper}" 2>&1 | tee -a "${log_file}" >&2 || true
  exit 2
fi
echo "[EXECUTE_BATCH_GET] contract=npu_execute_address_table_to_udma_read device=${DEVICE_ID} table=${TABLE_CAPACITY} requests=${REQUESTS} output_rows=${OUTPUT_ROWS} token_bytes=${TOKEN_BYTES} tokens_per_block=${TOKENS_PER_BLOCK} block_bytes=${SOURCE_STRIDE} stride=${SOURCE_STRIDE} aiv_cores=${AIV_CORES} jettys_per_core=${JETTYS_PER_CORE} mask_every=${MASK_EVERY} active_every=${ACTIVE_EVERY} active_percent=${ACTIVE_PERCENT} dynamic_output_smoke=${DYNAMIC_OUTPUT_SMOKE} graph_replays=${GRAPH_REPLAYS} graph_warmup=${GRAPH_WARMUP} graph_iterations=${GRAPH_ITERATIONS} timeout=${timeout_seconds} toolkit_lib=${cann_lib}" | tee "${log_file}"

set +e
timeout "${timeout_seconds}" "${test_binary}" --kernel "${kernel}" \
  --device "${DEVICE_ID}" --table-capacity "${TABLE_CAPACITY}" --requests "${REQUESTS}" \
  --output-rows "${OUTPUT_ROWS}" --block-bytes "${BLOCK_BYTES}" --source-stride "${SOURCE_STRIDE}" \
  --aiv-cores "${AIV_CORES}" --jettys-per-core "${JETTYS_PER_CORE}" \
  --warmup "${WARMUP}" --iterations "${ITERATIONS}" \
  --mask-every "${MASK_EVERY}" --active-every "${ACTIVE_EVERY}" --active-percent "${ACTIVE_PERCENT}" \
  --graph-replays "${GRAPH_REPLAYS}" \
  --graph-warmup "${GRAPH_WARMUP}" --graph-iterations "${GRAPH_ITERATIONS}" \
  --dynamic-output-smoke "${DYNAMIC_OUTPUT_SMOKE}" 2>&1 | tee -a "${log_file}"
status=${PIPESTATUS[0]}
set -e

# Keep all run-local evidence together.  Some A5 deployments redirect PLOG,
# so the runner probes the documented overrides as well as common defaults.
mkdir -p "${plog_dir}"
for candidate in "${ASCEND_PROCESS_LOG_PATH:-}" "${ASCEND_SLOG_PATH:-}" "${ASCEND_PLOG_PATH:-}" \
                 "${HOME}/ascend/log/debug/plog" /var/log/npu/slog; do
  if [[ -n "${candidate}" && -d "${candidate}" && -r "${candidate}" ]]; then
    find "${candidate}" -type f -newer "${run_started}" -print0 2>/dev/null |
      xargs -0 -r -I{} cp -p {} "${plog_dir}" 2>/dev/null || true
  fi
done

{
  printf 'result=%s\n' "$([[ ${status} -eq 0 ]] && printf pass || printf fail)"
  printf 'exit_status=%s\n' "${status}"
  printf 'run_log=%s\n' "${log_file}"
  printf 'kernel=%s\n' "${kernel}"
  printf 'plog_directory=%s\n' "${plog_dir}"
  printf '\n[markers]\n'
  grep -a -E 'PHASE_|DIRECT_(TRANSPORT|LOCAL_BUFFER|AIV_ENTITY_TEMPLATE)|EAGER_RANDOM_INPUTS|EXECUTE_BATCH_GET|GRAPH_(CAPTURE|REPLAY|PERF)|PERF_RESULT|KVAF_EXECUTE' "${log_file}" | tail -n 120 || true
  printf '\n[plog]\n'
  find "${plog_dir}" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort || true
} > "${feedback_file}"
cat "${feedback_file}"
if [[ ${status} -eq 0 ]]; then
  echo "RUN_PASS feedback=${feedback_file} log=${log_file}"
else
  echo "RUN_FAIL status=${status} feedback=${feedback_file} log=${log_file}" >&2
fi
exit "${status}"
