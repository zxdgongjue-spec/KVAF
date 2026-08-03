#!/usr/bin/env bash

# Server-B-host entry for one NPU2 zero-WQE hardware stage with bounded runtime
# plus matching kernel UDMA evidence. It never signals a KVAF worker, resets an
# NPU, or deletes a pre-existing active/quarantine file. Its delegated
# build/package step has no global wall timeout and may replace generated
# artifacts; the runner may clear only its own proven terminal active state.

set -Eeuo pipefail
umask 077

if (( $# != 0 )); then
  echo "Usage: $0" >&2
  exit 2
fi
if (( EUID != 0 )); then
  echo "ERROR: run this trace entry as root on the Server B host" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
kvaf_container=vllm-ascend-cs-kvaf
kvaf_state_root=/data/.kvaf-state/KVAF
kvaf_trace_instance_dir=
kvaf_function_graph=

kvaf_finalize_tracefs_instance()
{
  local cleanup_status=0

  [[ -n "${kvaf_trace_instance_dir}" ]] || return 0

  if ! printf '0\n' >"${kvaf_trace_instance_dir}/tracing_on"; then
    echo "ERROR: could not disable the KVAF tracefs instance" >&2
    cleanup_status=1
  fi
  if [[ -n "${kvaf_function_graph}" ]]; then
    if ! cat "${kvaf_trace_instance_dir}/trace" \
        >"${kvaf_function_graph}"; then
      echo "ERROR: could not archive the KVAF function graph" >&2
      cleanup_status=1
    fi
  fi
  if ! printf 'nop\n' >"${kvaf_trace_instance_dir}/current_tracer"; then
    echo "ERROR: could not restore the KVAF instance tracer to nop" >&2
    cleanup_status=1
  fi
  if [[ -w "${kvaf_trace_instance_dir}/set_graph_function" ]] &&
     ! : >"${kvaf_trace_instance_dir}/set_graph_function"; then
    echo "ERROR: could not clear the KVAF graph-function filter" >&2
    cleanup_status=1
  fi
  if rmdir -- "${kvaf_trace_instance_dir}"; then
    kvaf_trace_instance_dir=
  else
    echo "ERROR: could not remove the KVAF tracefs instance" >&2
    cleanup_status=1
  fi
  return "${cleanup_status}"
}

kvaf_trace_exit_cleanup()
{
  local exit_status=$?

  trap - EXIT HUP INT TERM
  if [[ -n "${kvaf_trace_instance_dir}" ]] &&
     ! kvaf_finalize_tracefs_instance; then
    if (( exit_status == 0 )); then
      exit_status=75
    fi
  fi
  exit "${exit_status}"
}

trap kvaf_trace_exit_cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is unavailable; this entry runs on Server B host" >&2
  exit 2
fi
if [[ "$(docker inspect -f '{{.State.Status}}' "${kvaf_container}" \
        2>/dev/null || true)" != running ]]; then
  echo "ERROR: ${kvaf_container} is not running" >&2
  exit 2
fi
if [[ -e "${kvaf_state_root}/direct-urma.active" ||
      -L "${kvaf_state_root}/direct-urma.active" ||
      -e "${kvaf_state_root}/direct-urma.quarantine" ||
      -L "${kvaf_state_root}/direct-urma.quarantine" ]]; then
  echo "ERROR: active/quarantine state exists; do not trace another run" >&2
  exit 2
fi

kvaf_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD) || {
  echo "ERROR: cannot identify the KVAF checkout" >&2
  exit 2
}
if [[ -n "$(git -C "${ROOT_DIR}" status --porcelain)" ]]; then
  echo "ERROR: KVAF checkout is not clean; trace identity would be ambiguous" >&2
  git -C "${ROOT_DIR}" status --short >&2
  exit 2
fi
kvaf_boot_id=$(tr -d '\n' </proc/sys/kernel/random/boot_id)
if [[ ! "${kvaf_boot_id}" =~ ^[0-9a-f-]{36}$ ||
      ! "${kvaf_commit}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: invalid boot-id or Git commit identity" >&2
  exit 2
fi

# Resolve and configure a private tracefs instance before consuming the
# boot-id/commit one-shot directory.  The first available root is selected in
# call-stack order: uburma parses the public command and calls ubcore, which
# in turn calls the UDMA driver.  function_graph then records every traceable
# child below that root in one zero-WQE attempt.
kvaf_tracefs=
for kvaf_tracefs_candidate in \
  /sys/kernel/tracing \
  /sys/kernel/debug/tracing
do
  if [[ -d "${kvaf_tracefs_candidate}/instances" &&
        -r "${kvaf_tracefs_candidate}/available_filter_functions" &&
        -w "${kvaf_tracefs_candidate}/instances" ]]; then
    kvaf_tracefs=${kvaf_tracefs_candidate}
    break
  fi
done
if [[ -z "${kvaf_tracefs}" ]]; then
  echo "ERROR: writable tracefs instances and available_filter_functions are required" >&2
  echo "No hardware operation was attempted and no one-shot trace directory was consumed." >&2
  exit 2
fi

kvaf_trace_root_symbol=
for kvaf_trace_root_candidate in \
  uburma_cmd_create_jfr \
  ubcore_create_jfr \
  udma_create_jfr
do
  kvaf_trace_root_symbol=$(
    awk -v wanted="${kvaf_trace_root_candidate}" '
      $1 == wanted || index($1, wanted ".") == 1 {
        print $1
        exit
      }
    ' "${kvaf_tracefs}/available_filter_functions"
  )
  if [[ -n "${kvaf_trace_root_symbol}" ]]; then
    break
  fi
done
if [[ -z "${kvaf_trace_root_symbol}" ]]; then
  echo "ERROR: no traceable CREATE_JFR root symbol is loaded" >&2
  echo "CHECKED=uburma_cmd_create_jfr,ubcore_create_jfr,udma_create_jfr" >&2
  echo "No hardware operation was attempted and no one-shot trace directory was consumed." >&2
  exit 2
fi

kvaf_trace_instance_name=kvaf_jfr_${BASHPID}_${RANDOM}
kvaf_trace_instance_candidate=\
"${kvaf_tracefs}/instances/${kvaf_trace_instance_name}"
if ! mkdir -- "${kvaf_trace_instance_candidate}"; then
  echo "ERROR: could not create an independent tracefs instance" >&2
  echo "No hardware operation was attempted and no one-shot trace directory was consumed." >&2
  exit 2
fi
kvaf_trace_instance_dir=${kvaf_trace_instance_candidate}

for kvaf_trace_control in \
  tracing_on \
  current_tracer \
  set_graph_function \
  max_graph_depth \
  trace
do
  if [[ ! -r "${kvaf_trace_instance_dir}/${kvaf_trace_control}" ||
        ! -w "${kvaf_trace_instance_dir}/${kvaf_trace_control}" ]]; then
    echo "ERROR: tracefs instance control is not readable/writable: ${kvaf_trace_control}" >&2
    echo "No hardware operation was attempted and no one-shot trace directory was consumed." >&2
    exit 2
  fi
done
printf '0\n' >"${kvaf_trace_instance_dir}/tracing_on"
printf 'function_graph\n' >"${kvaf_trace_instance_dir}/current_tracer"
printf '%s\n' "${kvaf_trace_root_symbol}" \
  >"${kvaf_trace_instance_dir}/set_graph_function"
printf '16\n' >"${kvaf_trace_instance_dir}/max_graph_depth"
if [[ -r "${kvaf_trace_instance_dir}/trace_options" &&
      -w "${kvaf_trace_instance_dir}/trace_options" ]]; then
  for kvaf_graph_option in \
    funcgraph-proc \
    funcgraph-duration \
    funcgraph-abstime \
    funcgraph-retval
  do
    if grep -qw -- "${kvaf_graph_option}" \
         "${kvaf_trace_instance_dir}/trace_options" ||
       grep -qw -- "no${kvaf_graph_option}" \
         "${kvaf_trace_instance_dir}/trace_options"; then
      if ! printf '%s\n' "${kvaf_graph_option}" \
          >"${kvaf_trace_instance_dir}/trace_options"; then
        echo "WARNING: optional trace option unavailable: ${kvaf_graph_option}" >&2
      fi
    fi
  done
fi
if [[ "$(tr -d '[:space:]' \
        <"${kvaf_trace_instance_dir}/current_tracer")" != function_graph ]] ||
   ! awk -v wanted="${kvaf_trace_root_symbol}" \
       '$1 == wanted { found = 1 } END { exit !found }' \
       "${kvaf_trace_instance_dir}/set_graph_function"; then
  echo "ERROR: function_graph root installation was not verified" >&2
  echo "No hardware operation was attempted and no one-shot trace directory was consumed." >&2
  exit 2
fi
# Prove that this private instance can toggle before the one-shot identity is
# consumed.  Its tracer is filtered and no KVAF hardware process is running.
printf '1\n' >"${kvaf_trace_instance_dir}/tracing_on"
printf '0\n' >"${kvaf_trace_instance_dir}/tracing_on"
: >"${kvaf_trace_instance_dir}/trace"

kvaf_trace_root=${kvaf_state_root}/diagnostics/jfr-create
kvaf_trace_dir=${kvaf_trace_root}/${kvaf_boot_id}/${kvaf_commit}
mkdir -p "$(dirname "${kvaf_trace_dir}")"
if ! mkdir "${kvaf_trace_dir}" 2>/dev/null; then
  echo "ERROR: this boot-id/commit trace was already selected" >&2
  echo "TRACE_DIR=${kvaf_trace_dir}" >&2
  exit 2
fi
chmod 0700 "${kvaf_trace_dir}"

kvaf_console=${kvaf_trace_dir}/console.log
kvaf_kernel=${kvaf_trace_dir}/kernel.log
kvaf_kernel_filtered=${kvaf_trace_dir}/kernel-udma-filtered.log
kvaf_function_graph=${kvaf_trace_dir}/function-graph.log
kvaf_metadata=${kvaf_trace_dir}/metadata.txt
kvaf_start_wall=$(date --iso-8601=seconds)
kvaf_journal_cursor=
if command -v journalctl >/dev/null 2>&1; then
  if kvaf_journal_cursor=$(
    journalctl -k -n 0 --show-cursor --no-pager 2>/dev/null |
      sed -n 's/^-- cursor: //p' |
      tail -n 1
  ); then
    :
  else
    kvaf_journal_cursor=
  fi
fi

{
  echo "schema=kvaf-a5-jfr-kernel-trace-v1"
  echo "boot_id=${kvaf_boot_id}"
  echo "kvaf_commit=${kvaf_commit}"
  echo "started_at=${kvaf_start_wall}"
  echo "container=${kvaf_container}"
  echo "logical_device_id=2"
  echo "physical_device_id=2"
  echo "journal_cursor=${kvaf_journal_cursor}"
  echo "tracefs=${kvaf_tracefs}"
  echo "function_graph_root=${kvaf_trace_root_symbol}"
  echo "function_graph_instance=${kvaf_trace_instance_name}"
} >"${kvaf_metadata}"

echo "EXECUTION_ENV=SERVER_B_HOST"
echo "KVAF_JFR_TRACE_DIR=${kvaf_trace_dir}"
echo "KVAF_JFR_TRACE_STAGE=single-zero-wqe"
echo "KVAF_JFR_FUNCTION_GRAPH_ROOT=${kvaf_trace_root_symbol}"
printf '1\n' >"${kvaf_trace_instance_dir}/tracing_on"
set +e
docker exec -u 0 \
  -e USER=root \
  -e KVAF_NPU2_H2D_TIMEOUT_SECONDS=20 \
  -e KVAF_NPU2_RUN_TIMEOUT_SECONDS=90 \
  -e KVAF_NPU2_SERVER_READY_TIMEOUT_SECONDS=15 \
  -e KVAF_NPU2_VALIDATION_LIMIT=zero-wqe \
  -w "${ROOT_DIR}" \
  "${kvaf_container}" \
  bash scripts/prepare_and_run_a5_direct_urma_npu2_clean.sh \
  2>&1 | tee "${kvaf_console}"
kvaf_pipeline_status=("${PIPESTATUS[@]}")
kvaf_run_rc=${kvaf_pipeline_status[0]}
kvaf_tee_rc=${kvaf_pipeline_status[1]}
set -e
kvaf_function_graph_rc=0
kvaf_finalize_tracefs_instance || kvaf_function_graph_rc=$?
kvaf_evidence_rc=${kvaf_run_rc}
if (( kvaf_tee_rc != 0 )); then
  echo "ERROR: console evidence write failed: tee_rc=${kvaf_tee_rc}" >&2
  kvaf_evidence_rc=74
fi
if (( kvaf_function_graph_rc != 0 )); then
  echo "ERROR: function-graph evidence finalization failed" >&2
  kvaf_evidence_rc=75
fi

kvaf_kernel_mode=unavailable
if [[ -n "${kvaf_journal_cursor}" ]] &&
   journalctl -k --after-cursor "${kvaf_journal_cursor}" \
     --no-pager -o short-monotonic >"${kvaf_kernel}" 2>/dev/null; then
  kvaf_kernel_mode=journal-cursor
elif command -v journalctl >/dev/null 2>&1 &&
     journalctl -k --since "${kvaf_start_wall}" \
       --no-pager -o short-monotonic >"${kvaf_kernel}" 2>/dev/null; then
  kvaf_kernel_mode=journal-since
elif command -v dmesg >/dev/null 2>&1 &&
     dmesg --time-format=iso --since "${kvaf_start_wall}" \
       >"${kvaf_kernel}" 2>/dev/null; then
  kvaf_kernel_mode=dmesg-since
elif command -v dmesg >/dev/null 2>&1 &&
     dmesg --time-format=iso 2>/dev/null |
       tail -n 20000 >"${kvaf_kernel}"; then
  kvaf_kernel_mode=dmesg-ring-unscoped-fallback
else
  : >"${kvaf_kernel}"
fi

grep -Ei \
  '(^|[^[:alnum:]_])(udma|uburma|ubcore|urma|ummu|jfr|jfc|jetty|aicpu)([^[:alnum:]_]|$)|failed to find|in_len|pin.*(rq|idx|db|sleep)' \
  "${kvaf_kernel}" >"${kvaf_kernel_filtered}" || true
kvaf_function_graph_lines=0
if [[ -r "${kvaf_function_graph}" ]]; then
  kvaf_function_graph_lines=$(wc -l <"${kvaf_function_graph}")
fi
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "run_rc=${kvaf_run_rc}"
  echo "tee_rc=${kvaf_tee_rc}"
  echo "evidence_rc=${kvaf_evidence_rc}"
  echo "function_graph_rc=${kvaf_function_graph_rc}"
  echo "function_graph_lines=${kvaf_function_graph_lines}"
  echo "kernel_capture=${kvaf_kernel_mode}"
  echo "kernel_lines=$(wc -l <"${kvaf_kernel}")"
  echo "kernel_filtered_lines=$(wc -l <"${kvaf_kernel_filtered}")"
} >>"${kvaf_metadata}"

kvaf_result_dir=$(
  awk '
    /^RESULT_DIR=/ { value = substr($0, length("RESULT_DIR=") + 1) }
    END { print value }
  ' "${kvaf_console}"
)
if [[ -n "${kvaf_result_dir}" &&
      "${kvaf_result_dir}" == "${kvaf_state_root}/results/"* &&
      -d "${kvaf_result_dir}" && ! -L "${kvaf_result_dir}" ]]; then
  install -m 0600 "${kvaf_metadata}" \
    "${kvaf_result_dir}/jfr-kernel-trace.metadata"
  install -m 0600 "${kvaf_kernel_filtered}" \
    "${kvaf_result_dir}/jfr-kernel-trace.log"
  if [[ -r "${kvaf_function_graph}" ]]; then
    install -m 0600 "${kvaf_function_graph}" \
      "${kvaf_result_dir}/jfr-function-graph.log"
  fi
  echo "KVAF_JFR_TRACE_RESULT_DIR=${kvaf_result_dir}"
fi

echo "KVAF_JFR_TRACE_RC=${kvaf_run_rc}"
echo "KVAF_JFR_TRACE_EVIDENCE_RC=${kvaf_evidence_rc}"
echo "KVAF_JFR_KERNEL_CAPTURE=${kvaf_kernel_mode}"
echo "KVAF_JFR_KERNEL_LOG=${kvaf_kernel_filtered}"
echo "KVAF_JFR_FUNCTION_GRAPH_LOG=${kvaf_function_graph}"
echo "KVAF_JFR_FUNCTION_GRAPH_TAIL_BEGIN"
if [[ -r "${kvaf_function_graph}" ]]; then
  tail -n 240 "${kvaf_function_graph}" || true
else
  echo "FUNCTION_GRAPH_UNAVAILABLE"
fi
echo "KVAF_JFR_FUNCTION_GRAPH_TAIL_END"
echo "KVAF_JFR_KERNEL_TAIL_BEGIN"
tail -n 160 "${kvaf_kernel_filtered}" || true
echo "KVAF_JFR_KERNEL_TAIL_END"
exit "${kvaf_evidence_rc}"
