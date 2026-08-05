#!/usr/bin/env bash
set -euo pipefail

# Capture the ExecuteBatchGet AIV launch once, mutate only fixed device-tensor
# contents between model runs, and validate every replay.  Resource creation,
# Host-DDR registration and HCCP setup stay deliberately outside capture.
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
demo_dir=$(cd "${script_dir}/.." && pwd)
result_dir=${RESULT_DIR:-"${demo_dir}/results/execute_batch_get_graph_$(date +%Y%m%d_%H%M%S)"}
build_dir=${BUILD_DIR:-"${demo_dir}/build_execute_batch_get_graph"}
: "${GRAPH_REPLAYS:=4}"
: "${GRAPH_WARMUP:=5}"
: "${GRAPH_ITERATIONS:=20}"

if [[ ! "${GRAPH_REPLAYS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: GRAPH_REPLAYS must be a positive integer." >&2
  exit 2
fi
if [[ ! "${GRAPH_WARMUP}" =~ ^[0-9]+$ || ! "${GRAPH_ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: GRAPH_WARMUP must be non-negative and GRAPH_ITERATIONS must be positive." >&2
  exit 2
fi

mkdir -p "${result_dir}"
build_log="${result_dir}/build.log"

# Graph support lives in the test executable.  Never reuse the ordinary
# ExecuteBatchGet build here: after a git pull it can be a valid but older
# binary that has no capture branch, which would otherwise run eager mode and
# make the graph acceptance check ambiguous.
echo "[EXECUTE_BATCH_GET_GRAPH] build=forced_isolated build_dir=${build_dir} contract=fixed_shape_device_tensors_capture_once_replay=${GRAPH_REPLAYS} graph_warmup=${GRAPH_WARMUP} graph_iterations=${GRAPH_ITERATIONS}" | tee "${build_log}"
BUILD_DIR="${build_dir}" bash "${script_dir}/build.sh" 2>&1 | tee -a "${build_log}"

set +e
RESULT_DIR="${result_dir}" BUILD_DIR="${build_dir}" GRAPH_REPLAYS="${GRAPH_REPLAYS}" \
  GRAPH_WARMUP="${GRAPH_WARMUP}" GRAPH_ITERATIONS="${GRAPH_ITERATIONS}" \
  bash "${script_dir}/run_execute_batch_get.sh"
status=$?
set -e

log_file="${result_dir}/run.log"
feedback_file="${result_dir}/feedback.txt"
if [[ ${status} -eq 0 ]] &&
   grep -aq '^GRAPH_CAPTURE_DONE model_ready=1$' "${log_file}" &&
   grep -aq "^GRAPH_REPLAY_PASS replays=${GRAPH_REPLAYS} " "${log_file}" &&
   grep -aq '^GRAPH_PERF_RESULT metric=graph_input_update_to_stream_completion ' "${log_file}"; then
  {
    printf '\n[graph_acceptance]\n'
    printf 'capture=passed\n'
    printf 'replays=%s\n' "${GRAPH_REPLAYS}"
    printf 'perf_warmup=%s\n' "${GRAPH_WARMUP}"
    printf 'perf_samples=%s\n' "${GRAPH_ITERATIONS}"
    grep -aE '^GRAPH_(CAPTURE|REPLAY|PERF)' "${log_file}" || true
  } >> "${feedback_file}"
  cat "${feedback_file}"
  echo "GRAPH_RUN_PASS feedback=${feedback_file}"
  exit 0
fi

if [[ ${status} -eq 0 ]]; then
  status=1
fi
echo "GRAPH_RUN_FAIL status=${status} feedback=${feedback_file}" >&2
exit "${status}"
