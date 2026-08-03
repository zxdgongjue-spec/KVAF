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
} >"${kvaf_metadata}"

echo "EXECUTION_ENV=SERVER_B_HOST"
echo "KVAF_JFR_TRACE_DIR=${kvaf_trace_dir}"
echo "KVAF_JFR_TRACE_STAGE=single-zero-wqe"
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
kvaf_evidence_rc=${kvaf_run_rc}
if (( kvaf_tee_rc != 0 )); then
  echo "ERROR: console evidence write failed: tee_rc=${kvaf_tee_rc}" >&2
  kvaf_evidence_rc=74
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
{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "run_rc=${kvaf_run_rc}"
  echo "tee_rc=${kvaf_tee_rc}"
  echo "evidence_rc=${kvaf_evidence_rc}"
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
  echo "KVAF_JFR_TRACE_RESULT_DIR=${kvaf_result_dir}"
fi

echo "KVAF_JFR_TRACE_RC=${kvaf_run_rc}"
echo "KVAF_JFR_TRACE_EVIDENCE_RC=${kvaf_evidence_rc}"
echo "KVAF_JFR_KERNEL_CAPTURE=${kvaf_kernel_mode}"
echo "KVAF_JFR_KERNEL_LOG=${kvaf_kernel_filtered}"
echo "KVAF_JFR_KERNEL_TAIL_BEGIN"
tail -n 160 "${kvaf_kernel_filtered}" || true
echo "KVAF_JFR_KERNEL_TAIL_END"
exit "${kvaf_evidence_rc}"
