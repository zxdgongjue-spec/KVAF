#!/usr/bin/env bash

# Detached transaction owner for one resource-bearing public-URMA AICPU
# probe.  The parent supplies already-validated identity through environment
# variables and passes the worker argv unchanged.  FD 8 and FD 9 must be the
# inherited active/deploy lock descriptions.

set -Eeuo pipefail
umask 077

: "${KVAF_RUNTIME_SUPERVISOR_DURABLE_HELPER:?}"
: "${KVAF_RUNTIME_SUPERVISOR_STATE_HELPER:?}"
: "${KVAF_RUNTIME_SUPERVISOR_ACTIVE_LOCK:?}"
: "${KVAF_RUNTIME_SUPERVISOR_DEPLOY_LOCK:?}"
: "${KVAF_RUNTIME_SUPERVISOR_QUARANTINE:?}"
: "${KVAF_RUNTIME_SUPERVISOR_ACTIVE:?}"
: "${KVAF_RUNTIME_SUPERVISOR_READY:?}"
: "${KVAF_RUNTIME_SUPERVISOR_TERMINAL:?}"
: "${KVAF_RUNTIME_SUPERVISOR_LOG:?}"
: "${KVAF_RUNTIME_SUPERVISOR_RUN_ID:?}"
: "${KVAF_RUNTIME_SUPERVISOR_MODE:?}"
: "${KVAF_RUNTIME_SUPERVISOR_BOOT_ID:?}"
: "${KVAF_RUNTIME_SUPERVISOR_LOGICAL_ID:?}"
: "${KVAF_RUNTIME_SUPERVISOR_PHYSICAL_ID:?}"
: "${KVAF_RUNTIME_SUPERVISOR_WORKER_PATH:?}"
: "${KVAF_RUNTIME_SUPERVISOR_WORKER_SHA256:?}"
: "${KVAF_RUNTIME_SUPERVISOR_PROVIDER_SONAME:?}"
: "${KVAF_RUNTIME_SUPERVISOR_PROVIDER_BUILD_ID:?}"
: "${KVAF_RUNTIME_SUPERVISOR_PROVIDER_ARTIFACT_SHA256:?}"
: "${KVAF_RUNTIME_SUPERVISOR_CALLER_DOMAIN_CONTRACT:?}"

source "${KVAF_RUNTIME_SUPERVISOR_STATE_HELPER}"

kvaf_supervisor_publish()
{
  local kvaf_target=$1
  local kvaf_source=$2

  chmod 0600 "${kvaf_source}" || return 1
  python3 "${KVAF_RUNTIME_SUPERVISOR_DURABLE_HELPER}" publish \
    "${kvaf_source}" "${kvaf_target}"
}

kvaf_supervisor_publish_terminal()
{
  local kvaf_terminal_rc=$1
  local kvaf_terminal_reason=$2
  local kvaf_log_sha kvaf_terminal_tmp

  python3 "${KVAF_RUNTIME_SUPERVISOR_DURABLE_HELPER}" sync \
    "${KVAF_RUNTIME_SUPERVISOR_LOG}" || return 1
  kvaf_log_sha=$(sha256sum -- "${KVAF_RUNTIME_SUPERVISOR_LOG}" |
    awk '{print $1}') || return 1
  [[ ${kvaf_log_sha} =~ ^[0-9a-f]{64}$ ]] || return 1
  kvaf_terminal_tmp="${KVAF_RUNTIME_SUPERVISOR_TERMINAL}.tmp.$$"
  printf '%s\n' \
    'schema=kvaf-a5-runtime-urma-probe-terminal-v1' \
    "run_id=${KVAF_RUNTIME_SUPERVISOR_RUN_ID}" \
    "mode=${KVAF_RUNTIME_SUPERVISOR_MODE}" \
    "boot_id=${KVAF_RUNTIME_SUPERVISOR_BOOT_ID}" \
    "logical_device_id=${KVAF_RUNTIME_SUPERVISOR_LOGICAL_ID}" \
    "physical_device_id=${KVAF_RUNTIME_SUPERVISOR_PHYSICAL_ID}" \
    "supervisor_pid=$$" \
    "supervisor_start_ticks=${kvaf_supervisor_start_ticks}" \
    "worker_path=${KVAF_RUNTIME_SUPERVISOR_WORKER_PATH}" \
    "worker_sha256=${KVAF_RUNTIME_SUPERVISOR_WORKER_SHA256}" \
    "provider_soname=${KVAF_RUNTIME_SUPERVISOR_PROVIDER_SONAME}" \
    "provider_build_id_sha256=${KVAF_RUNTIME_SUPERVISOR_PROVIDER_BUILD_ID}" \
    "expected_packaged_provider_artifact_sha256=${KVAF_RUNTIME_SUPERVISOR_PROVIDER_ARTIFACT_SHA256}" \
    "log=${KVAF_RUNTIME_SUPERVISOR_LOG}" \
    "log_sha256=${kvaf_log_sha}" \
    "rc=${kvaf_terminal_rc}" \
    "reason=${kvaf_terminal_reason}" \
    'terminal_contract=runtime-probe-exact-terminal-v2' \
    'lock_contract=inherited-supervisor-worker-v4-held-flock-pinned-worker-fd' \
    "caller_domain_contract=${KVAF_RUNTIME_SUPERVISOR_CALLER_DOMAIN_CONTRACT}" \
    >"${kvaf_terminal_tmp}" || return 1
  kvaf_supervisor_publish \
    "${KVAF_RUNTIME_SUPERVISOR_TERMINAL}" "${kvaf_terminal_tmp}"
}

kvaf_supervisor_prelaunch_safe_exit()
{
  local kvaf_prelaunch_stage=$1

  printf '%s\n' \
    "KVAF_A5_RUNTIME_URMA_PRELAUNCH_SAFE mode=${KVAF_RUNTIME_SUPERVISOR_MODE}" \
    "KVAF_A5_RUNTIME_URMA_PRELAUNCH_STAGE=${kvaf_prelaunch_stage}" \
    >>"${KVAF_RUNTIME_SUPERVISOR_LOG}" || exit 255
  kvaf_supervisor_publish_terminal \
    1 supervisor_prelaunch_failed_log_synced || exit 255
  exit 1
}

# Validate inherited lock descriptions before publishing state or launching
# anything that can enter ACL/AICPU/URMA.  A path string in the environment is
# not proof that this detached process actually retained the lock.
kvaf_fd8_target=$(readlink -f /proc/$$/fd/8 2>/dev/null || true)
kvaf_fd9_target=$(readlink -f /proc/$$/fd/9 2>/dev/null || true)
kvaf_expected_fd8=$(realpath -e \
  "${KVAF_RUNTIME_SUPERVISOR_ACTIVE_LOCK}")
kvaf_expected_fd9=$(realpath -e \
  "${KVAF_RUNTIME_SUPERVISOR_DEPLOY_LOCK}")
[[ ${kvaf_fd8_target} == "${kvaf_expected_fd8}" &&
   ${kvaf_fd9_target} == "${kvaf_expected_fd9}" ]] || exit 255
# readlink proves only which inode is open.  Acquire (or reassert, for the
# inherited open-file description) both advisory locks before publishing any
# durable ownership claim.  An unlocked descriptor pointing at the right path
# is not a lock contract.
flock -n 8 || exit 255
flock -n 9 || exit 255

[[ ${KVAF_RUNTIME_SUPERVISOR_CALLER_DOMAIN_CONTRACT} == \
   undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1 ]] || exit 255
[[ ${KVAF_RUNTIME_SUPERVISOR_MODE} == discover ||
   ${KVAF_RUNTIME_SUPERVISOR_MODE} == init-device ||
   ${KVAF_RUNTIME_SUPERVISOR_MODE} == create-delete ]] || exit 255
[[ $(realpath -e "${KVAF_RUNTIME_SUPERVISOR_WORKER_PATH}") == \
   "${KVAF_RUNTIME_SUPERVISOR_WORKER_PATH}" ]] || exit 255
# Open the verified worker before any durable publication and execute the same
# inode through this descriptor.  A concurrent git pull/relink may replace the
# checkout pathname without taking KVAF's deploy lock; hashing a pathname and
# later executing that pathname would otherwise be a real TOCTOU.  FD 7 stays
# open in the supervisor and its child until the worker exec resolves it.
exec 7<"${KVAF_RUNTIME_SUPERVISOR_WORKER_PATH}" || exit 255
kvaf_worker_path_identity=$(LC_ALL=C stat -Lc '%d:%i' -- \
  "${KVAF_RUNTIME_SUPERVISOR_WORKER_PATH}") || exit 255
kvaf_worker_fd_identity=$(LC_ALL=C stat -Lc '%d:%i' -- \
  /proc/$$/fd/7) || exit 255
[[ ${kvaf_worker_path_identity} == "${kvaf_worker_fd_identity}" ]] || exit 255
[[ -x /proc/$$/fd/7 ]] || exit 255
[[ $(sha256sum -- /proc/$$/fd/7 |
      awk '{print $1}') == "${KVAF_RUNTIME_SUPERVISOR_WORKER_SHA256}" ]] ||
  exit 255
kvaf_worker_exec_path=/proc/self/fd/7
[[ ${KVAF_RUNTIME_SUPERVISOR_PROVIDER_BUILD_ID} =~ ^[0-9a-f]{64}$ &&
   ${KVAF_RUNTIME_SUPERVISOR_PROVIDER_ARTIFACT_SHA256} =~ ^[0-9a-f]{64}$ &&
   ${KVAF_RUNTIME_SUPERVISOR_PROVIDER_SONAME} == \
     "libkvaf_rt_${KVAF_RUNTIME_SUPERVISOR_PROVIDER_BUILD_ID:0:16}.so" &&
   ${#KVAF_RUNTIME_SUPERVISOR_PROVIDER_SONAME} -lt 32 ]] || exit 255
kvaf_provider_soname_argv_count=0
kvaf_provider_build_id_argv_count=0
kvaf_previous_argument=
for kvaf_argument in "$@"; do
  if [[ ${kvaf_previous_argument} == --provider-soname ]]; then
    [[ ${kvaf_argument} == "${KVAF_RUNTIME_SUPERVISOR_PROVIDER_SONAME}" ]] ||
      exit 255
    ((kvaf_provider_soname_argv_count += 1))
  elif [[ ${kvaf_previous_argument} == --provider-build-id ]]; then
    [[ ${kvaf_argument} == "${KVAF_RUNTIME_SUPERVISOR_PROVIDER_BUILD_ID}" ]] ||
      exit 255
    ((kvaf_provider_build_id_argv_count += 1))
  fi
  kvaf_previous_argument=${kvaf_argument}
done
[[ ${kvaf_provider_soname_argv_count} -eq 1 &&
   ${kvaf_provider_build_id_argv_count} -eq 1 ]] || exit 255

kvaf_supervisor_identity=$(
  kvaf_runtime_probe_read_proc_state_and_start_ticks "$$"
) || exit 255
read -r kvaf_supervisor_state kvaf_supervisor_start_ticks \
  <<<"${kvaf_supervisor_identity}" || exit 255
[[ ${kvaf_supervisor_state} != Z ]] || exit 255
kvaf_supervisor_exe=$(readlink -f /proc/$$/exe) || exit 255

[[ ! -e ${KVAF_RUNTIME_SUPERVISOR_LOG} &&
   ! -L ${KVAF_RUNTIME_SUPERVISOR_LOG} &&
   ! -e ${KVAF_RUNTIME_SUPERVISOR_READY} &&
   ! -L ${KVAF_RUNTIME_SUPERVISOR_READY} &&
   ! -e ${KVAF_RUNTIME_SUPERVISOR_TERMINAL} &&
   ! -L ${KVAF_RUNTIME_SUPERVISOR_TERMINAL} ]] || exit 255
: >"${KVAF_RUNTIME_SUPERVISOR_LOG}" || exit 255
chmod 0600 "${KVAF_RUNTIME_SUPERVISOR_LOG}" || exit 255

# Publication order is part of the safety protocol: quarantine -> active ->
# ready -> worker.  The worker must never run in a gap with no durable owner
# identity.
kvaf_quarantine_tmp="${KVAF_RUNTIME_SUPERVISOR_QUARANTINE}.tmp.$$"
printf '%s\n' \
  'schema=kvaf-a5-runtime-urma-probe-quarantine-v1' \
  "run_id=${KVAF_RUNTIME_SUPERVISOR_RUN_ID}" \
  "mode=${KVAF_RUNTIME_SUPERVISOR_MODE}" \
  "boot_id=${KVAF_RUNTIME_SUPERVISOR_BOOT_ID}" \
  "logical_device_id=${KVAF_RUNTIME_SUPERVISOR_LOGICAL_ID}" \
  "physical_device_id=${KVAF_RUNTIME_SUPERVISOR_PHYSICAL_ID}" \
  "supervisor_pid=$$" \
  "supervisor_start_ticks=${kvaf_supervisor_start_ticks}" \
  "supervisor_observed_exe=${kvaf_supervisor_exe}" \
  "worker_path=${KVAF_RUNTIME_SUPERVISOR_WORKER_PATH}" \
  "worker_sha256=${KVAF_RUNTIME_SUPERVISOR_WORKER_SHA256}" \
  "provider_soname=${KVAF_RUNTIME_SUPERVISOR_PROVIDER_SONAME}" \
  "provider_build_id_sha256=${KVAF_RUNTIME_SUPERVISOR_PROVIDER_BUILD_ID}" \
  "expected_packaged_provider_artifact_sha256=${KVAF_RUNTIME_SUPERVISOR_PROVIDER_ARTIFACT_SHA256}" \
  "log=${KVAF_RUNTIME_SUPERVISOR_LOG}" \
  "ready_file=${KVAF_RUNTIME_SUPERVISOR_READY}" \
  "terminal_file=${KVAF_RUNTIME_SUPERVISOR_TERMINAL}" \
  'terminal_contract=runtime-probe-exact-terminal-v2' \
  'lock_contract=inherited-supervisor-worker-v4-held-flock-pinned-worker-fd' \
  "caller_domain_contract=${KVAF_RUNTIME_SUPERVISOR_CALLER_DOMAIN_CONTRACT}" \
  'reason=runtime_public_probe_supervisor_prelaunch' \
  'cleanup_certainty=unknown' \
  >"${kvaf_quarantine_tmp}" || exit 255
kvaf_supervisor_publish \
  "${KVAF_RUNTIME_SUPERVISOR_QUARANTINE}" "${kvaf_quarantine_tmp}" ||
  exit 255

kvaf_active_tmp="${KVAF_RUNTIME_SUPERVISOR_ACTIVE}.tmp.$$"
printf '%s\n' \
  'schema=kvaf-a5-runtime-urma-probe-active-v2' \
  "run_id=${KVAF_RUNTIME_SUPERVISOR_RUN_ID}" \
  "mode=${KVAF_RUNTIME_SUPERVISOR_MODE}" \
  "boot_id=${KVAF_RUNTIME_SUPERVISOR_BOOT_ID}" \
  "logical_device_id=${KVAF_RUNTIME_SUPERVISOR_LOGICAL_ID}" \
  "physical_device_id=${KVAF_RUNTIME_SUPERVISOR_PHYSICAL_ID}" \
  'phase=owner_supervisor_ready' \
  "supervisor_pid=$$" \
  "supervisor_start_ticks=${kvaf_supervisor_start_ticks}" \
  "supervisor_observed_exe=${kvaf_supervisor_exe}" \
  "worker_path=${KVAF_RUNTIME_SUPERVISOR_WORKER_PATH}" \
  "worker_sha256=${KVAF_RUNTIME_SUPERVISOR_WORKER_SHA256}" \
  "provider_soname=${KVAF_RUNTIME_SUPERVISOR_PROVIDER_SONAME}" \
  "provider_build_id_sha256=${KVAF_RUNTIME_SUPERVISOR_PROVIDER_BUILD_ID}" \
  "expected_packaged_provider_artifact_sha256=${KVAF_RUNTIME_SUPERVISOR_PROVIDER_ARTIFACT_SHA256}" \
  "log=${KVAF_RUNTIME_SUPERVISOR_LOG}" \
  "ready_file=${KVAF_RUNTIME_SUPERVISOR_READY}" \
  "terminal_file=${KVAF_RUNTIME_SUPERVISOR_TERMINAL}" \
  'reason=supervisor_owns_locks_and_worker_lifetime' \
  'terminal_contract=runtime-probe-exact-terminal-v2' \
  'lock_contract=inherited-supervisor-worker-v4-held-flock-pinned-worker-fd' \
  "caller_domain_contract=${KVAF_RUNTIME_SUPERVISOR_CALLER_DOMAIN_CONTRACT}" \
  >"${kvaf_active_tmp}" || kvaf_supervisor_prelaunch_safe_exit active-write
kvaf_supervisor_publish \
  "${KVAF_RUNTIME_SUPERVISOR_ACTIVE}" "${kvaf_active_tmp}" ||
  kvaf_supervisor_prelaunch_safe_exit active-publish

kvaf_ready_tmp="${KVAF_RUNTIME_SUPERVISOR_READY}.tmp.$$"
printf '%s\n' \
  'schema=kvaf-a5-runtime-urma-probe-ready-v1' \
  "run_id=${KVAF_RUNTIME_SUPERVISOR_RUN_ID}" \
  "mode=${KVAF_RUNTIME_SUPERVISOR_MODE}" \
  "boot_id=${KVAF_RUNTIME_SUPERVISOR_BOOT_ID}" \
  "logical_device_id=${KVAF_RUNTIME_SUPERVISOR_LOGICAL_ID}" \
  "physical_device_id=${KVAF_RUNTIME_SUPERVISOR_PHYSICAL_ID}" \
  "supervisor_pid=$$" \
  "supervisor_start_ticks=${kvaf_supervisor_start_ticks}" \
  "worker_path=${KVAF_RUNTIME_SUPERVISOR_WORKER_PATH}" \
  "worker_sha256=${KVAF_RUNTIME_SUPERVISOR_WORKER_SHA256}" \
  "provider_soname=${KVAF_RUNTIME_SUPERVISOR_PROVIDER_SONAME}" \
  "provider_build_id_sha256=${KVAF_RUNTIME_SUPERVISOR_PROVIDER_BUILD_ID}" \
  "expected_packaged_provider_artifact_sha256=${KVAF_RUNTIME_SUPERVISOR_PROVIDER_ARTIFACT_SHA256}" \
  "log=${KVAF_RUNTIME_SUPERVISOR_LOG}" \
  "terminal_file=${KVAF_RUNTIME_SUPERVISOR_TERMINAL}" \
  'reason=supervisor_ready_to_own_worker_lifetime' \
  'terminal_contract=runtime-probe-exact-terminal-v2' \
  'lock_contract=inherited-supervisor-worker-v4-held-flock-pinned-worker-fd' \
  "caller_domain_contract=${KVAF_RUNTIME_SUPERVISOR_CALLER_DOMAIN_CONTRACT}" \
  >"${kvaf_ready_tmp}" || kvaf_supervisor_prelaunch_safe_exit ready-write
kvaf_supervisor_publish \
  "${KVAF_RUNTIME_SUPERVISOR_READY}" "${kvaf_ready_tmp}" ||
  kvaf_supervisor_prelaunch_safe_exit ready-publish

set +e
"${kvaf_worker_exec_path}" "$@" \
  >"${KVAF_RUNTIME_SUPERVISOR_LOG}" 2>&1
kvaf_worker_rc=$?
set -e
kvaf_supervisor_publish_terminal \
  "${kvaf_worker_rc}" worker_exited_log_synced || exit 255
exit "${kvaf_worker_rc}"
