#!/usr/bin/env bash

# Guarded three-stage AICPU probe for the Device OS system liburma provider.
# Even `discover` executes the official liburma DSO constructor during dlopen,
# so every mode is supervised and quarantined until exact cleanup evidence is
# durably sealed.

set -Eeuo pipefail
umask 077
unset BASH_ENV ENV
unset -f exit builtin 2>/dev/null || true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
source "${SCRIPT_DIR}/a5_direct_urma_legacy_guard.sh"
source "${SCRIPT_DIR}/a5_direct_urma_manifest_v3.sh"
source "${SCRIPT_DIR}/a5_direct_urma_runtime_preflight.sh"
source "${SCRIPT_DIR}/a5_runtime_probe_state.sh"
kvaf_configure_direct_urma_state_paths "${ROOT_DIR}" || exit 2

KVAF_A5_DIRECT_BUILD_DIR=${KVAF_A5_DIRECT_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma"}
KVAF_A5_DIRECT_BINARY=${KVAF_A5_DIRECT_BINARY:-"${KVAF_A5_DIRECT_BUILD_DIR}/access_operator/a5/kvaf_a5_aiv_direct_urma_h2d_smoke"}
KVAF_A5_RUNTIME_URMA_PROBE_BINARY=${KVAF_A5_RUNTIME_URMA_PROBE_BINARY:-"${KVAF_A5_DIRECT_BUILD_DIR}/access_operator/a5/kvaf_a5_aicpu_runtime_urma_probe"}
KVAF_A5_DIRECT_BUILD_MANIFEST=${KVAF_A5_DIRECT_BUILD_MANIFEST:-"${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke.manifest"}
KVAF_DIRECT_PROVIDER_BUILD_DIR=${KVAF_DIRECT_PROVIDER_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma-provider"}
KVAF_DIRECT_HOST_PROVIDER_LIBRARY=${KVAF_DIRECT_HOST_PROVIDER_LIBRARY:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/libkvaf_urma_provider_host.so"}
KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY=${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/libkvaf_urma_provider_device.so"}
KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST=${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/source.sha256"}
KVAF_DIRECT_PROVIDER_BUILD_MANIFEST=${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/build.manifest"}
KVAF_DIRECT_PROVIDER_SUCCESS_STAMP=${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/build.success"}
KVAF_ASCEND_HAL_LIBRARY=/usr/local/Ascend/driver/lib64/driver/libascend_hal.so
KVAF_NPU_IDLE_CHECKER=${KVAF_NPU_IDLE_CHECKER:-"${SCRIPT_DIR}/check_a5_npu_idle.py"}
KVAF_ENDPOINT_RESOLVER=${KVAF_ENDPOINT_RESOLVER:-"${SCRIPT_DIR}/resolve_a5_urma_endpoints.py"}
KVAF_DURABLE_STATE_HELPER=${KVAF_DURABLE_STATE_HELPER:-"${SCRIPT_DIR}/durable_state_file.py"}
KVAF_RUNTIME_PROBE_OBSERVATION_TIMEOUT_SECONDS=${KVAF_RUNTIME_PROBE_OBSERVATION_TIMEOUT_SECONDS:-120}
KVAF_RUNTIME_PROBE_READY_TIMEOUT_SECONDS=${KVAF_RUNTIME_PROBE_READY_TIMEOUT_SECONDS:-30}
kvaf_runtime_probe_recovered=0

die()
{
  echo "ERROR: AICPU runtime URMA probe preflight: $*" >&2
  exit 2
}

kvaf_owned_single_link_regular_file()
{
  local kvaf_state_file=$1
  local kvaf_state_metadata

  [[ -f ${kvaf_state_file} && ! -L ${kvaf_state_file} ]] || return 1
  kvaf_state_metadata=$(LC_ALL=C stat -c '%u:%h:%F' "${kvaf_state_file}") ||
    return 1
  [[ ${kvaf_state_metadata} == "${EUID}:1:regular file" ]]
}

kvaf_publish_runtime_probe_blocked()
{
  local kvaf_blocked_run_id=$1
  local kvaf_blocked_mode=$2
  local kvaf_blocked_boot_id=$3
  local kvaf_blocked_logical_id=$4
  local kvaf_blocked_physical_id=$5
  local kvaf_blocked_worker_path=$6
  local kvaf_blocked_worker_sha=$7
  local kvaf_blocked_log=$8
  local kvaf_blocked_terminal=$9
  shift 9
  local kvaf_blocked_provider_soname=$1
  local kvaf_blocked_provider_build_id=$2
  local kvaf_blocked_provider_artifact_sha=$3
  local kvaf_blocked_log_sha kvaf_blocked_tmp

  kvaf_blocked_log_sha=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_blocked_terminal}" log_sha256) || return 1
  [[ ${kvaf_blocked_log_sha} =~ ^[0-9a-f]{64}$ ]] || return 1
  kvaf_blocked_tmp="${KVAF_DIRECT_QUARANTINE_FILE}.tmp.$$"
  printf '%s\n' \
    'schema=kvaf-a5-runtime-urma-probe-blocked-v1' \
    "run_id=${kvaf_blocked_run_id}" \
    "mode=${kvaf_blocked_mode}" \
    "boot_id=${kvaf_blocked_boot_id}" \
    "logical_device_id=${kvaf_blocked_logical_id}" \
    "physical_device_id=${kvaf_blocked_physical_id}" \
    "worker_path=${kvaf_blocked_worker_path}" \
    "worker_sha256=${kvaf_blocked_worker_sha}" \
    "provider_soname=${kvaf_blocked_provider_soname}" \
    "provider_build_id_sha256=${kvaf_blocked_provider_build_id}" \
    "expected_packaged_provider_artifact_sha256=${kvaf_blocked_provider_artifact_sha}" \
    "log=${kvaf_blocked_log}" \
    "log_sha256=${kvaf_blocked_log_sha}" \
    "terminal_file=${kvaf_blocked_terminal}" \
    'terminal_contract=runtime-probe-exact-terminal-v2' \
    'caller_domain_contract=undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1' \
    'reason=aicpu_runtime_already_initialized_by_unattributed_owner' \
    'cleanup_certainty=this-invocation-clean-borrowed-runtime-blocked' \
    >"${kvaf_blocked_tmp}" || return 1
  chmod 0600 "${kvaf_blocked_tmp}" || return 1
  python3 "${KVAF_DURABLE_STATE_HELPER}" publish \
    "${kvaf_blocked_tmp}" "${KVAF_DIRECT_QUARANTINE_FILE}"
}

# A borrowed-runtime terminal is intentionally retained for the rest of the
# current boot: this probe cannot identify or stop the process that initialized
# liburma first.  A different kernel boot ID is the only local evidence that
# all such process and kernel state was reset.  Recovery is still a standalone
# transaction; it never starts a new probe in the same invocation.
kvaf_try_recover_rebooted_runtime_probe_blocked()
{
  local kvaf_active=${KVAF_DIRECT_ACTIVE_SESSION_FILE}
  local kvaf_quarantine=${KVAF_DIRECT_QUARANTINE_FILE}
  local kvaf_schema kvaf_run_id kvaf_mode kvaf_boot_id kvaf_current_boot_id
  local kvaf_logical_id kvaf_physical_id kvaf_worker_path kvaf_worker_sha
  local kvaf_log kvaf_log_sha kvaf_actual_log_sha kvaf_terminal_file
  local kvaf_terminal_rc kvaf_terminal_reason kvaf_ready_file
  local kvaf_supervisor_pid kvaf_supervisor_start_ticks kvaf_expected_prefix
  local kvaf_provider_soname kvaf_provider_build_id
  local kvaf_provider_artifact_sha

  [[ -e ${kvaf_quarantine} || -L ${kvaf_quarantine} ]] || return 0
  [[ ! -e ${kvaf_active} && ! -L ${kvaf_active} ]] || return 1
  kvaf_owned_single_link_regular_file "${kvaf_quarantine}" || return 1
  kvaf_runtime_probe_file_has_exact_keys "${kvaf_quarantine}" \
    schema run_id mode boot_id logical_device_id physical_device_id \
    worker_path worker_sha256 provider_soname provider_build_id_sha256 \
    expected_packaged_provider_artifact_sha256 log log_sha256 terminal_file \
    terminal_contract caller_domain_contract reason cleanup_certainty ||
    return 1
  kvaf_schema=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" schema) || return 1
  [[ ${kvaf_schema} == kvaf-a5-runtime-urma-probe-blocked-v1 ]] || return 1
  kvaf_run_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" run_id) || return 1
  kvaf_mode=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" mode) || return 1
  kvaf_boot_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" boot_id) || return 1
  kvaf_logical_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" logical_device_id) || return 1
  kvaf_physical_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" physical_device_id) || return 1
  kvaf_worker_path=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" worker_path) || return 1
  kvaf_worker_sha=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" worker_sha256) || return 1
  kvaf_provider_soname=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" provider_soname) || return 1
  kvaf_provider_build_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" provider_build_id_sha256) || return 1
  kvaf_provider_artifact_sha=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" \
    expected_packaged_provider_artifact_sha256) || return 1
  kvaf_log=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" log) || return 1
  kvaf_log_sha=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" log_sha256) || return 1
  kvaf_terminal_file=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" terminal_file) || return 1

  [[ ${kvaf_mode} == discover || ${kvaf_mode} == init-device ||
     ${kvaf_mode} == create-delete ]] || return 1
  [[ ${kvaf_run_id} =~ ^runtime-urma-${kvaf_mode}-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9]+$ &&
     ${kvaf_boot_id} =~ ^[0-9a-f-]{36}$ &&
     ${kvaf_logical_id} =~ ^(0|[1-9][0-9]*)$ &&
     ${kvaf_physical_id} =~ ^(0|[1-9][0-9]*)$ &&
     ${kvaf_worker_path} == /* &&
     ${kvaf_worker_sha} =~ ^[0-9a-f]{64}$ &&
     ${kvaf_provider_build_id} =~ ^[0-9a-f]{64}$ &&
     ${kvaf_provider_artifact_sha} =~ ^[0-9a-f]{64}$ &&
     ${kvaf_provider_soname} == \
       "libkvaf_rt_${kvaf_provider_build_id:0:16}.so" &&
     ${kvaf_log_sha} =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ $(kvaf_runtime_probe_read_single_value \
         "${kvaf_quarantine}" terminal_contract) == \
       runtime-probe-exact-terminal-v2 &&
     $(kvaf_runtime_probe_read_single_value \
         "${kvaf_quarantine}" caller_domain_contract) == \
       undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1 &&
     $(kvaf_runtime_probe_read_single_value \
         "${kvaf_quarantine}" reason) == \
       aicpu_runtime_already_initialized_by_unattributed_owner &&
     $(kvaf_runtime_probe_read_single_value \
         "${kvaf_quarantine}" cleanup_certainty) == \
       this-invocation-clean-borrowed-runtime-blocked ]] || return 1

  kvaf_current_boot_id=$(< /proc/sys/kernel/random/boot_id)
  [[ ${kvaf_current_boot_id} =~ ^[0-9a-f-]{36}$ &&
     ${kvaf_boot_id} != "${kvaf_current_boot_id}" ]] || return 1
  kvaf_expected_prefix=${KVAF_DIRECT_STATE_DIR}/runtime-probes/${kvaf_run_id}
  kvaf_ready_file=${kvaf_expected_prefix}.ready
  [[ ${kvaf_log} == "${kvaf_expected_prefix}.log" &&
     ${kvaf_terminal_file} == "${kvaf_expected_prefix}.terminal" &&
     ! -e ${kvaf_ready_file} && ! -L ${kvaf_ready_file} ]] || return 1
  kvaf_owned_single_link_regular_file "${kvaf_log}" || return 1
  kvaf_owned_single_link_regular_file "${kvaf_terminal_file}" || return 1
  kvaf_actual_log_sha=$(sha256sum -- "${kvaf_log}" | awk '{print $1}') ||
    return 1
  [[ ${kvaf_actual_log_sha} == "${kvaf_log_sha}" ]] || return 1

  kvaf_supervisor_pid=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" supervisor_pid) || return 1
  kvaf_supervisor_start_ticks=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" supervisor_start_ticks) || return 1
  kvaf_runtime_probe_terminal_evidence_matches \
    "${kvaf_terminal_file}" "${kvaf_run_id}" "${kvaf_mode}" \
    "${kvaf_boot_id}" "${kvaf_logical_id}" "${kvaf_physical_id}" \
    "${kvaf_worker_path}" "${kvaf_worker_sha}" "${kvaf_log}" \
    "${kvaf_supervisor_pid}" "${kvaf_supervisor_start_ticks}" \
    "${kvaf_provider_soname}" "${kvaf_provider_build_id}" \
    "${kvaf_provider_artifact_sha}" || return 1
  kvaf_terminal_rc=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" rc) || return 1
  kvaf_terminal_reason=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" reason) || return 1
  [[ ${kvaf_terminal_rc} == 5 &&
     ${kvaf_terminal_reason} == worker_exited_log_synced ]] || return 1
  grep -Fxq KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK "${kvaf_log}" || return 1
  grep -Fxq KVAF_A5_RUNTIME_URMA_PROCESS_CONTAMINATED "${kvaf_log}" ||
    return 1

  python3 "${KVAF_DURABLE_STATE_HELPER}" remove "${kvaf_quarantine}" ||
    return 1
  kvaf_runtime_probe_recovered=3
  echo "KVAF_A5_RUNTIME_URMA_BLOCKED_REBOOT_RECOVERED run_id=${kvaf_run_id} mode=${kvaf_mode}"
  return 0
}

# A worker that deliberately holds after an unverifiable official-library
# result cannot publish a terminal record.  After a real OS reboot, however,
# its old boot ID proves that neither the Host supervisor nor its AICPU process
# can still own resources.  Validate every durable identity field that exists,
# then retire only the barriers.  The unsealed log remains diagnostic and is
# never promoted to a capability result.
kvaf_try_recover_rebooted_runtime_probe_hold()
{
  local kvaf_active=${KVAF_DIRECT_ACTIVE_SESSION_FILE}
  local kvaf_quarantine=${KVAF_DIRECT_QUARANTINE_FILE}
  local kvaf_schema kvaf_run_id kvaf_mode kvaf_boot_id kvaf_current_boot_id
  local kvaf_logical_id kvaf_physical_id kvaf_supervisor_pid
  local kvaf_supervisor_start_ticks kvaf_supervisor_observed_exe
  local kvaf_worker_path kvaf_worker_sha kvaf_provider_soname
  local kvaf_provider_build_id kvaf_provider_artifact_sha
  local kvaf_log kvaf_ready_file kvaf_terminal_file kvaf_expected_prefix

  [[ -e ${kvaf_quarantine} || -L ${kvaf_quarantine} ]] || return 0
  kvaf_owned_single_link_regular_file "${kvaf_quarantine}" || return 1
  kvaf_runtime_probe_file_has_exact_keys "${kvaf_quarantine}" \
    schema run_id mode boot_id logical_device_id physical_device_id \
    supervisor_pid supervisor_start_ticks supervisor_observed_exe worker_path \
    worker_sha256 provider_soname provider_build_id_sha256 \
    expected_packaged_provider_artifact_sha256 log ready_file terminal_file terminal_contract \
    lock_contract caller_domain_contract reason cleanup_certainty || return 1
  kvaf_schema=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" schema) || return 1
  [[ ${kvaf_schema} == kvaf-a5-runtime-urma-probe-quarantine-v1 ]] ||
    return 1
  kvaf_run_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" run_id) || return 1
  kvaf_mode=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" mode) || return 1
  kvaf_boot_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" boot_id) || return 1
  kvaf_logical_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" logical_device_id) || return 1
  kvaf_physical_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" physical_device_id) || return 1
  kvaf_supervisor_pid=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" supervisor_pid) || return 1
  kvaf_supervisor_start_ticks=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" supervisor_start_ticks) || return 1
  kvaf_supervisor_observed_exe=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" supervisor_observed_exe) || return 1
  kvaf_worker_path=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" worker_path) || return 1
  kvaf_worker_sha=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" worker_sha256) || return 1
  kvaf_provider_soname=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" provider_soname) || return 1
  kvaf_provider_build_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" provider_build_id_sha256) || return 1
  kvaf_provider_artifact_sha=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" \
    expected_packaged_provider_artifact_sha256) || return 1
  kvaf_log=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" log) || return 1
  kvaf_ready_file=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" ready_file) || return 1
  kvaf_terminal_file=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" terminal_file) || return 1

  [[ ${kvaf_mode} == discover || ${kvaf_mode} == init-device ||
     ${kvaf_mode} == create-delete ]] || return 1
  [[ ${kvaf_run_id} =~ ^runtime-urma-${kvaf_mode}-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9]+$ &&
     ${kvaf_boot_id} =~ ^[0-9a-f-]{36}$ &&
     ${kvaf_logical_id} =~ ^(0|[1-9][0-9]*)$ &&
     ${kvaf_physical_id} =~ ^(0|[1-9][0-9]*)$ &&
     ${kvaf_supervisor_pid} =~ ^[1-9][0-9]*$ &&
     ${kvaf_supervisor_start_ticks} =~ ^[1-9][0-9]*$ &&
     ${kvaf_supervisor_observed_exe} == /* &&
     ${kvaf_worker_path} == /* &&
     ${kvaf_worker_sha} =~ ^[0-9a-f]{64}$ &&
     ${kvaf_provider_build_id} =~ ^[0-9a-f]{64}$ &&
     ${kvaf_provider_artifact_sha} =~ ^[0-9a-f]{64}$ &&
     ${kvaf_provider_soname} == \
       "libkvaf_rt_${kvaf_provider_build_id:0:16}.so" ]] || return 1
  [[ $(kvaf_runtime_probe_read_single_value \
         "${kvaf_quarantine}" terminal_contract) == \
       runtime-probe-exact-terminal-v2 &&
     $(kvaf_runtime_probe_read_single_value \
         "${kvaf_quarantine}" lock_contract) == \
       inherited-supervisor-worker-v4-held-flock-pinned-worker-fd &&
     $(kvaf_runtime_probe_read_single_value \
         "${kvaf_quarantine}" caller_domain_contract) == \
       undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1 &&
     $(kvaf_runtime_probe_read_single_value \
         "${kvaf_quarantine}" reason) == \
       runtime_public_probe_supervisor_prelaunch &&
     $(kvaf_runtime_probe_read_single_value \
         "${kvaf_quarantine}" cleanup_certainty) == unknown ]] || return 1
  kvaf_expected_prefix=${KVAF_DIRECT_STATE_DIR}/runtime-probes/${kvaf_run_id}
  [[ ${kvaf_log} == "${kvaf_expected_prefix}.log" &&
     ${kvaf_ready_file} == "${kvaf_expected_prefix}.ready" &&
     ${kvaf_terminal_file} == "${kvaf_expected_prefix}.terminal" &&
     ! -e ${kvaf_terminal_file} && ! -L ${kvaf_terminal_file} ]] || return 1
  kvaf_owned_single_link_regular_file "${kvaf_log}" || return 1
  kvaf_current_boot_id=$(< /proc/sys/kernel/random/boot_id)
  [[ ${kvaf_current_boot_id} =~ ^[0-9a-f-]{36}$ &&
     ${kvaf_boot_id} != "${kvaf_current_boot_id}" ]] || return 1

  if [[ -e ${kvaf_active} || -L ${kvaf_active} ]]; then
    kvaf_owned_single_link_regular_file "${kvaf_active}" || return 1
    kvaf_runtime_probe_file_has_exact_keys "${kvaf_active}" \
      schema run_id mode boot_id logical_device_id physical_device_id phase \
      supervisor_pid supervisor_start_ticks supervisor_observed_exe \
      worker_path worker_sha256 provider_soname provider_build_id_sha256 \
      expected_packaged_provider_artifact_sha256 log ready_file terminal_file reason \
      terminal_contract lock_contract caller_domain_contract || return 1
    [[ $(kvaf_runtime_probe_read_single_value "${kvaf_active}" schema) == \
         kvaf-a5-runtime-urma-probe-active-v2 &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" run_id) == \
         "${kvaf_run_id}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" mode) == \
         "${kvaf_mode}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" boot_id) == \
         "${kvaf_boot_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" logical_device_id) == "${kvaf_logical_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" physical_device_id) == "${kvaf_physical_id}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" phase) == \
         owner_supervisor_ready &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" supervisor_pid) == "${kvaf_supervisor_pid}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" supervisor_start_ticks) == \
         "${kvaf_supervisor_start_ticks}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" supervisor_observed_exe) == \
         "${kvaf_supervisor_observed_exe}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" worker_path) == \
         "${kvaf_worker_path}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" worker_sha256) == "${kvaf_worker_sha}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" provider_soname) == "${kvaf_provider_soname}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" provider_build_id_sha256) == \
         "${kvaf_provider_build_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" expected_packaged_provider_artifact_sha256) == \
         "${kvaf_provider_artifact_sha}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" log) == \
         "${kvaf_log}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" ready_file) == \
         "${kvaf_ready_file}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" terminal_file) == "${kvaf_terminal_file}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" reason) == \
         supervisor_owns_locks_and_worker_lifetime &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" terminal_contract) == \
         runtime-probe-exact-terminal-v2 &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" lock_contract) == inherited-supervisor-worker-v4-held-flock-pinned-worker-fd &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" caller_domain_contract) == \
         undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1 ]] || return 1
  elif [[ -e ${kvaf_ready_file} || -L ${kvaf_ready_file} ]]; then
    return 1
  fi

  if [[ -e ${kvaf_ready_file} || -L ${kvaf_ready_file} ]]; then
    kvaf_owned_single_link_regular_file "${kvaf_ready_file}" || return 1
    kvaf_runtime_probe_file_has_exact_keys "${kvaf_ready_file}" \
      schema run_id mode boot_id logical_device_id physical_device_id \
      supervisor_pid supervisor_start_ticks worker_path worker_sha256 \
      provider_soname provider_build_id_sha256 \
      expected_packaged_provider_artifact_sha256 log \
      terminal_file reason terminal_contract lock_contract \
      caller_domain_contract || return 1
    [[ $(kvaf_runtime_probe_read_single_value "${kvaf_ready_file}" schema) == \
         kvaf-a5-runtime-urma-probe-ready-v1 &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_ready_file}" run_id) == \
         "${kvaf_run_id}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_ready_file}" mode) == \
         "${kvaf_mode}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_ready_file}" boot_id) == \
         "${kvaf_boot_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" logical_device_id) == "${kvaf_logical_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" physical_device_id) == "${kvaf_physical_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" supervisor_pid) == "${kvaf_supervisor_pid}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" supervisor_start_ticks) == \
         "${kvaf_supervisor_start_ticks}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" worker_path) == "${kvaf_worker_path}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" worker_sha256) == "${kvaf_worker_sha}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" provider_soname) == "${kvaf_provider_soname}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" provider_build_id_sha256) == \
         "${kvaf_provider_build_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" expected_packaged_provider_artifact_sha256) == \
         "${kvaf_provider_artifact_sha}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_ready_file}" log) == \
         "${kvaf_log}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" terminal_file) == "${kvaf_terminal_file}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_ready_file}" reason) == \
         supervisor_ready_to_own_worker_lifetime &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" terminal_contract) == \
         runtime-probe-exact-terminal-v2 &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" lock_contract) == \
         inherited-supervisor-worker-v4-held-flock-pinned-worker-fd &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" caller_domain_contract) == \
         undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1 ]] || return 1
  fi

  python3 "${KVAF_DURABLE_STATE_HELPER}" remove "${kvaf_active}" || return 1
  python3 "${KVAF_DURABLE_STATE_HELPER}" remove "${kvaf_ready_file}" || return 1
  python3 "${KVAF_DURABLE_STATE_HELPER}" remove "${kvaf_quarantine}" ||
    return 1
  kvaf_runtime_probe_recovered=4
  echo "KVAF_A5_RUNTIME_URMA_HELD_REBOOT_RECOVERED run_id=${kvaf_run_id} mode=${kvaf_mode}"
  return 0
}

# Recover only from the previous run's self-contained, sealed evidence.  No
# current build binary, manifest, CANN installation or provider artifact is
# opened here: rebuilding the checkout must not make a safely completed old
# run unrecoverable.
kvaf_try_recover_completed_runtime_probe()
{
  local kvaf_active=${KVAF_DIRECT_ACTIVE_SESSION_FILE}
  local kvaf_quarantine=${KVAF_DIRECT_QUARANTINE_FILE}
  local kvaf_schema kvaf_run_id kvaf_mode kvaf_boot_id kvaf_current_boot_id
  local kvaf_logical_id kvaf_physical_id kvaf_worker_path kvaf_worker_sha
  local kvaf_log kvaf_ready_file kvaf_terminal_file
  local kvaf_supervisor_pid kvaf_supervisor_start_ticks
  local kvaf_supervisor_observed_exe
  local kvaf_actual_state kvaf_actual_start_ticks kvaf_proc_identity
  local kvaf_terminal_rc kvaf_terminal_reason kvaf_expected_prefix
  local kvaf_provider_soname kvaf_provider_build_id
  local kvaf_provider_artifact_sha

  [[ -e ${kvaf_quarantine} || -L ${kvaf_quarantine} ||
     -e ${kvaf_active} || -L ${kvaf_active} ]] || return 0
  kvaf_owned_single_link_regular_file "${kvaf_quarantine}" || return 1
  kvaf_runtime_probe_file_has_exact_keys "${kvaf_quarantine}" \
    schema run_id mode boot_id logical_device_id physical_device_id \
    supervisor_pid supervisor_start_ticks supervisor_observed_exe worker_path \
    worker_sha256 provider_soname provider_build_id_sha256 \
    expected_packaged_provider_artifact_sha256 log ready_file terminal_file terminal_contract lock_contract \
    caller_domain_contract reason cleanup_certainty || return 1
  kvaf_schema=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" schema) || return 1
  [[ ${kvaf_schema} == kvaf-a5-runtime-urma-probe-quarantine-v1 ]] ||
    return 1
  kvaf_run_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" run_id) || return 1
  kvaf_mode=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" mode) || return 1
  [[ ${kvaf_mode} == discover || ${kvaf_mode} == init-device ||
     ${kvaf_mode} == create-delete ]] ||
    return 1
  [[ ${kvaf_run_id} =~ ^runtime-urma-${kvaf_mode}-[0-9]{8}T[0-9]{6}Z-[0-9]+-[0-9]+$ ]] ||
    return 1
  kvaf_boot_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" boot_id) || return 1
  kvaf_logical_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" logical_device_id) || return 1
  kvaf_physical_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" physical_device_id) || return 1
  kvaf_supervisor_pid=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" supervisor_pid) || return 1
  kvaf_supervisor_start_ticks=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" supervisor_start_ticks) || return 1
  kvaf_supervisor_observed_exe=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" supervisor_observed_exe) || return 1
  kvaf_worker_path=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" worker_path) || return 1
  kvaf_worker_sha=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" worker_sha256) || return 1
  kvaf_provider_soname=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" provider_soname) || return 1
  kvaf_provider_build_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" provider_build_id_sha256) || return 1
  kvaf_provider_artifact_sha=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" \
    expected_packaged_provider_artifact_sha256) || return 1
  kvaf_log=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" log) || return 1
  kvaf_ready_file=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" ready_file) || return 1
  kvaf_terminal_file=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_quarantine}" terminal_file) || return 1
  [[ ${kvaf_boot_id} =~ ^[0-9a-f-]{36}$ &&
     ${kvaf_logical_id} =~ ^(0|[1-9][0-9]*)$ &&
     ${kvaf_physical_id} =~ ^(0|[1-9][0-9]*)$ &&
     ${kvaf_supervisor_pid} =~ ^[1-9][0-9]*$ &&
     ${kvaf_supervisor_start_ticks} =~ ^[1-9][0-9]*$ &&
     ${kvaf_supervisor_observed_exe} == /* &&
     ${kvaf_worker_path} == /* &&
     ${kvaf_worker_sha} =~ ^[0-9a-f]{64}$ ]] || return 1
  [[ ${kvaf_provider_build_id} =~ ^[0-9a-f]{64}$ &&
     ${kvaf_provider_artifact_sha} =~ ^[0-9a-f]{64}$ &&
     ${kvaf_provider_soname} == \
       "libkvaf_rt_${kvaf_provider_build_id:0:16}.so" ]] || return 1
  [[ $(kvaf_runtime_probe_read_single_value \
         "${kvaf_quarantine}" terminal_contract) == \
       runtime-probe-exact-terminal-v2 &&
     $(kvaf_runtime_probe_read_single_value \
         "${kvaf_quarantine}" lock_contract) == \
       inherited-supervisor-worker-v4-held-flock-pinned-worker-fd &&
     $(kvaf_runtime_probe_read_single_value \
         "${kvaf_quarantine}" caller_domain_contract) == \
       undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1 &&
     $(kvaf_runtime_probe_read_single_value \
         "${kvaf_quarantine}" reason) == \
       runtime_public_probe_supervisor_prelaunch &&
     $(kvaf_runtime_probe_read_single_value \
         "${kvaf_quarantine}" cleanup_certainty) == unknown ]] || return 1
  kvaf_expected_prefix=${KVAF_DIRECT_STATE_DIR}/runtime-probes/${kvaf_run_id}
  [[ ${kvaf_log} == "${kvaf_expected_prefix}.log" &&
     ${kvaf_ready_file} == "${kvaf_expected_prefix}.ready" &&
     ${kvaf_terminal_file} == "${kvaf_expected_prefix}.terminal" ]] ||
    return 1

  kvaf_current_boot_id=$(< /proc/sys/kernel/random/boot_id)
  if [[ ${kvaf_boot_id} == "${kvaf_current_boot_id}" &&
       -r /proc/${kvaf_supervisor_pid}/stat ]]; then
    kvaf_proc_identity=$(kvaf_runtime_probe_read_proc_state_and_start_ticks \
      "${kvaf_supervisor_pid}") || return 1
    read -r kvaf_actual_state kvaf_actual_start_ticks \
      <<<"${kvaf_proc_identity}" || return 1
    if [[ ${kvaf_actual_start_ticks} == "${kvaf_supervisor_start_ticks}" &&
          ${kvaf_actual_state} != Z ]]; then
      echo "KVAF_A5_RUNTIME_URMA_RECOVERY_BLOCKED owner=live"
      return 1
    fi
  fi

  if [[ -e ${kvaf_active} || -L ${kvaf_active} ]]; then
    kvaf_owned_single_link_regular_file "${kvaf_active}" || return 1
    kvaf_runtime_probe_file_has_exact_keys "${kvaf_active}" \
      schema run_id mode boot_id logical_device_id physical_device_id phase \
      supervisor_pid supervisor_start_ticks supervisor_observed_exe \
      worker_path worker_sha256 provider_soname provider_build_id_sha256 \
      expected_packaged_provider_artifact_sha256 log ready_file terminal_file \
      reason terminal_contract lock_contract caller_domain_contract || return 1
    [[ $(kvaf_runtime_probe_read_single_value "${kvaf_active}" schema) == \
         kvaf-a5-runtime-urma-probe-active-v2 &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" run_id) == \
         "${kvaf_run_id}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" mode) == \
         "${kvaf_mode}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" boot_id) == \
         "${kvaf_boot_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" logical_device_id) == "${kvaf_logical_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" physical_device_id) == "${kvaf_physical_id}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" phase) == \
         owner_supervisor_ready &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" supervisor_pid) == "${kvaf_supervisor_pid}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" supervisor_start_ticks) == \
         "${kvaf_supervisor_start_ticks}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" supervisor_observed_exe) == \
         "${kvaf_supervisor_observed_exe}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" worker_path) == \
         "${kvaf_worker_path}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" worker_sha256) == "${kvaf_worker_sha}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" provider_soname) == "${kvaf_provider_soname}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" provider_build_id_sha256) == \
         "${kvaf_provider_build_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" expected_packaged_provider_artifact_sha256) == \
         "${kvaf_provider_artifact_sha}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" log) == \
         "${kvaf_log}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" ready_file) == \
         "${kvaf_ready_file}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" terminal_file) == "${kvaf_terminal_file}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_active}" reason) == \
         supervisor_owns_locks_and_worker_lifetime &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" terminal_contract) == \
         runtime-probe-exact-terminal-v2 &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" lock_contract) == \
         inherited-supervisor-worker-v4-held-flock-pinned-worker-fd &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_active}" caller_domain_contract) == \
         undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1 ]] || return 1
  fi

  kvaf_owned_single_link_regular_file "${kvaf_terminal_file}" || return 1
  kvaf_owned_single_link_regular_file "${kvaf_log}" || return 1
  kvaf_runtime_probe_terminal_evidence_matches \
    "${kvaf_terminal_file}" "${kvaf_run_id}" "${kvaf_mode}" \
    "${kvaf_boot_id}" "${kvaf_logical_id}" "${kvaf_physical_id}" \
    "${kvaf_worker_path}" "${kvaf_worker_sha}" "${kvaf_log}" \
    "${kvaf_supervisor_pid}" "${kvaf_supervisor_start_ticks}" \
    "${kvaf_provider_soname}" "${kvaf_provider_build_id}" \
    "${kvaf_provider_artifact_sha}" || return 1
  kvaf_terminal_rc=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" rc) || return 1
  kvaf_terminal_reason=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" reason) || return 1
  if [[ ${kvaf_terminal_rc} != 5 ]]; then
    kvaf_runtime_probe_log_is_safe_terminal \
      "${kvaf_terminal_rc}" "${kvaf_log}" "${kvaf_mode}" || return 1
  else
    grep -Fxq KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK "${kvaf_log}" ||
      return 1
    grep -Fxq KVAF_A5_RUNTIME_URMA_PROCESS_CONTAMINATED "${kvaf_log}" ||
      return 1
  fi

  if [[ ${kvaf_terminal_reason} == worker_exited_log_synced ]]; then
    kvaf_owned_single_link_regular_file "${kvaf_ready_file}" || return 1
    kvaf_runtime_probe_file_has_exact_keys "${kvaf_ready_file}" \
      schema run_id mode boot_id logical_device_id physical_device_id \
      supervisor_pid supervisor_start_ticks worker_path worker_sha256 log \
      provider_soname provider_build_id_sha256 \
      expected_packaged_provider_artifact_sha256 \
      terminal_file reason terminal_contract lock_contract \
      caller_domain_contract || return 1
    [[ $(kvaf_runtime_probe_read_single_value "${kvaf_ready_file}" schema) == \
         kvaf-a5-runtime-urma-probe-ready-v1 &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_ready_file}" run_id) == \
         "${kvaf_run_id}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_ready_file}" mode) == \
         "${kvaf_mode}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_ready_file}" boot_id) == \
         "${kvaf_boot_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" logical_device_id) == "${kvaf_logical_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" physical_device_id) == "${kvaf_physical_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" supervisor_pid) == "${kvaf_supervisor_pid}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" supervisor_start_ticks) == \
         "${kvaf_supervisor_start_ticks}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" worker_path) == "${kvaf_worker_path}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" worker_sha256) == "${kvaf_worker_sha}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" provider_soname) == "${kvaf_provider_soname}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" provider_build_id_sha256) == \
         "${kvaf_provider_build_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" expected_packaged_provider_artifact_sha256) == \
         "${kvaf_provider_artifact_sha}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_ready_file}" log) == \
         "${kvaf_log}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" terminal_file) == "${kvaf_terminal_file}" &&
       $(kvaf_runtime_probe_read_single_value "${kvaf_ready_file}" reason) == \
         supervisor_ready_to_own_worker_lifetime &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" terminal_contract) == \
         runtime-probe-exact-terminal-v2 &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" lock_contract) == \
         inherited-supervisor-worker-v4-held-flock-pinned-worker-fd &&
       $(kvaf_runtime_probe_read_single_value \
         "${kvaf_ready_file}" caller_domain_contract) == \
         undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1 ]] || return 1
  elif [[ -e ${kvaf_ready_file} || -L ${kvaf_ready_file} ]]; then
    # A prelaunch terminal and a ready record are mutually exclusive.
    return 1
  fi

  if [[ ${kvaf_terminal_rc} == 5 ]]; then
    python3 "${KVAF_DURABLE_STATE_HELPER}" remove "${kvaf_active}" ||
      return 1
    python3 "${KVAF_DURABLE_STATE_HELPER}" remove "${kvaf_ready_file}" ||
      return 1
    kvaf_publish_runtime_probe_blocked \
      "${kvaf_run_id}" "${kvaf_mode}" "${kvaf_boot_id}" \
      "${kvaf_logical_id}" "${kvaf_physical_id}" \
      "${kvaf_worker_path}" "${kvaf_worker_sha}" \
      "${kvaf_log}" "${kvaf_terminal_file}" \
      "${kvaf_provider_soname}" "${kvaf_provider_build_id}" \
      "${kvaf_provider_artifact_sha}" || return 1
    kvaf_runtime_probe_recovered=2
    echo "KVAF_A5_RUNTIME_URMA_BLOCKED_TERMINAL_RECOVERED run_id=${kvaf_run_id} mode=${kvaf_mode}"
    return 0
  fi

  python3 "${KVAF_DURABLE_STATE_HELPER}" remove "${kvaf_active}" ||
    return 1
  python3 "${KVAF_DURABLE_STATE_HELPER}" remove "${kvaf_ready_file}" ||
    return 1
  python3 "${KVAF_DURABLE_STATE_HELPER}" remove "${kvaf_quarantine}" ||
    return 1
  kvaf_runtime_probe_recovered=1
  echo "KVAF_A5_RUNTIME_URMA_SAFE_TERMINAL_RECOVERED run_id=${kvaf_run_id} mode=${kvaf_mode}"
  return 0
}

mode=
mode_count=0
logical_device_id=
logical_device_count=0
urma_device=
urma_device_count=0
eid_index=
eid_index_count=0
queue_depth=
queue_depth_count=0
provider_soname=
provider_soname_count=0
declare -a kvaf_probe_arguments=("$@")
for ((kvaf_index = 0;
      kvaf_index < ${#kvaf_probe_arguments[@]};
      kvaf_index += 1)); do
  argument=${kvaf_probe_arguments[kvaf_index]}
  case ${argument} in
    --mode|--logical-device-id|--urma-device|--eid-index|--queue-depth|--provider-soname)
      ((kvaf_index + 1 < ${#kvaf_probe_arguments[@]})) ||
        die "${argument} requires one value"
      value=${kvaf_probe_arguments[kvaf_index + 1]}
      ((kvaf_index += 1))
      case ${argument} in
        --mode) mode=${value}; ((mode_count += 1)) ;;
        --logical-device-id)
          logical_device_id=${value}; ((logical_device_count += 1)) ;;
        --urma-device) urma_device=${value}; ((urma_device_count += 1)) ;;
        --eid-index) eid_index=${value}; ((eid_index_count += 1)) ;;
        --queue-depth)
          queue_depth=${value}; ((queue_depth_count += 1)) ;;
        --provider-soname)
          provider_soname=${value}; ((provider_soname_count += 1)) ;;
      esac
      ;;
    *) die "unknown or positional argument: ${argument}" ;;
  esac
done
[[ ${mode_count} -eq 1 &&
   ( ${mode} == discover || ${mode} == init-device ||
     ${mode} == create-delete ) ]] ||
  die "pass exactly one --mode discover|init-device|create-delete"
[[ ${logical_device_count} -eq 1 ]] ||
  die "pass exactly one logical device"
[[ ${logical_device_id} =~ ^(0|[1-9][0-9]*)$ &&
   ${#logical_device_id} -le 10 &&
   ${logical_device_id} -le 2147483647 ]] ||
  die "logical device must be an unsigned integer"
[[ ${queue_depth_count} -le 1 && ${provider_soname_count} -le 1 ]] ||
  die "queue depth and provider soname may be passed at most once"
if (( queue_depth_count == 1 )); then
  [[ ${queue_depth} =~ ^[1-9][0-9]*$ && ${#queue_depth} -le 4 &&
     ${queue_depth} -le 4096 &&
     $((queue_depth & (queue_depth - 1))) -eq 0 ]] ||
    die "queue depth must be a power of two in [1,4096]"
fi
if (( provider_soname_count == 1 )); then
  [[ ${provider_soname} =~ ^[A-Za-z0-9._-]+$ &&
     ${#provider_soname} -lt 32 && ${provider_soname} != . &&
     ${provider_soname} != .. ]] ||
    die "provider soname must be a basename shorter than 32 bytes"
fi
if [[ ${mode} != discover ]]; then
  [[ ${urma_device_count} -eq 1 && ${eid_index_count} -eq 1 ]] ||
    die "resource-bearing mode requires exactly one URMA device and EID index"
else
  [[ ${urma_device_count} -eq ${eid_index_count} &&
     ${urma_device_count} -le 1 ]] ||
    die "discover accepts neither endpoint option or one complete optional pair"
fi
if (( urma_device_count == 1 )); then
  [[ ${urma_device} =~ ^[A-Za-z0-9._-]+$ &&
     ${#urma_device} -lt 32 && ${urma_device} != . &&
     ${urma_device} != .. &&
     ${eid_index} =~ ^(0|[1-9][0-9]*)$ &&
     ${#eid_index} -le 10 && ${eid_index} -le 4294967295 ]] ||
    die "URMA device must be a short basename and EID index unsigned"
fi

[[ -e /.dockerenv && ${EUID} -eq 0 ]] ||
  die "run as container uid 0 inside vllm-ascend-cs-kvaf"
for command in awk cat chmod dirname flock grep ldd mkdir nm \
  python3 readelf readlink realpath sha256sum stat tar tee timeout; do
  command -v "${command}" >/dev/null 2>&1 || die "${command} is required"
done
for command in npu-smi setsid sleep tail; do
  command -v "${command}" >/dev/null 2>&1 || die "${command} is required"
done

kvaf_check_legacy_direct_urma_state "${ROOT_DIR}" ||
  die "legacy direct URMA state blocks the probe"
mkdir -p "${KVAF_GLOBAL_STATE_DIR}" "${KVAF_DIRECT_STATE_DIR}" \
  "${KVAF_DIRECT_STATE_DIR}/runtime-probes" \
  "$(dirname "${KVAF_DIRECT_DEPLOY_RUN_LOCK}")" \
  "$(dirname "${KVAF_DIRECT_ACTIVE_SESSION_LOCK}")"
chmod 0700 "${KVAF_GLOBAL_STATE_DIR}" "${KVAF_DIRECT_STATE_DIR}" \
  "${KVAF_DIRECT_STATE_DIR}/runtime-probes"
[[ -f ${KVAF_DURABLE_STATE_HELPER} && ! -L ${KVAF_DURABLE_STATE_HELPER} ]] ||
  die "durable state helper is missing or unsafe"
for path in "${KVAF_DIRECT_DEPLOY_RUN_LOCK}" \
  "${KVAF_DIRECT_ACTIVE_SESSION_LOCK}" \
  "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" \
  "${KVAF_DIRECT_QUARANTINE_FILE}"; do
  [[ ! -L ${path} ]] || die "persistent state path is a symlink: ${path}"
done

exec 9>"${KVAF_DIRECT_DEPLOY_RUN_LOCK}"
flock -w 30 9 || die "another deploy/run owns the direct URMA lock"
exec 8>"${KVAF_DIRECT_ACTIVE_SESSION_LOCK}"
flock -n 8 || die "a live direct URMA worker blocks the probe"

# This recovery-only transaction deliberately precedes every current build,
# CANN, package and provider verification.  It trusts only the previous
# supervisor's exact state identity (and a sealed terminal when one exists)
# and never opens today's worker.
kvaf_try_recover_rebooted_runtime_probe_blocked || true
if (( kvaf_runtime_probe_recovered == 3 )); then
  echo "KVAF_A5_RUNTIME_URMA_REBOOT_RECOVERY_ONLY_OK"
  echo "The boot-ID change retired the blocked runtime state; rerun to start a probe."
  exit 6
fi
kvaf_try_recover_rebooted_runtime_probe_hold || true
if (( kvaf_runtime_probe_recovered == 4 )); then
  echo "KVAF_A5_RUNTIME_URMA_HELD_REBOOT_RECOVERY_ONLY_OK"
  echo "The old-boot held owner was reset; its unsealed log was not accepted as capability evidence."
  exit 6
fi
kvaf_try_recover_completed_runtime_probe || true
if (( kvaf_runtime_probe_recovered == 2 )); then
  echo "KVAF_A5_RUNTIME_URMA_BLOCKED_RECOVERY_ONLY"
  echo "The probe invocation tore down safely, but another process owns liburma runtime state."
  exit 5
fi
if (( kvaf_runtime_probe_recovered == 1 )); then
  echo "KVAF_A5_RUNTIME_URMA_RECOVERY_ONLY_OK"
  echo "Rerun the same command to start a new probe; no hardware operation was attempted."
  exit 6
fi
[[ ! -e ${KVAF_DIRECT_QUARANTINE_FILE} ]] ||
  die "an unsafe or unverifiable direct URMA session is quarantined"
[[ ! -e ${KVAF_DIRECT_ACTIVE_SESSION_FILE} ]] ||
  die "an incomplete direct URMA session blocks the probe"

# The current supervisor deliberately retains every official-DSO invocation
# as cleanup-unproven.  It therefore cannot seal an observation that a later
# boot can consume as a prerequisite for the next mode.  Running discover,
# init-device or create-delete now would consume a reboot window without
# advancing the machine-verifiable capability chain.  Disable all three until
# an observation ledger is implemented; recovery-only handling above remains
# available.  No environment override is accepted.
echo "KVAF_A5_RUNTIME_URMA_DYNAMIC_PROBE_DISABLED" >&2
echo "CAPABILITY_OBSERVATION_LEDGER=not-implemented" >&2
echo "DOCUMENTED_OFFICIAL_AICPU_CALL_CONTRACT=not-found-in-audited-sources" >&2
echo "LOCAL_RUNTIME_COMPATIBILITY=unproven" >&2
echo "OFFICIAL_FUNCTION_SEMANTICS=unproven" >&2
echo "No ACL/AICPU/URMA operation was attempted." >&2
builtin exit 2

# Retain the dynamic implementation for review without leaving an execution
# path: even a shadowed `exit` can only enter this constant-false branch.  The
# compiled probe binary is independently inert.
if [[ 0 == 1 ]]; then

: "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"
KVAF_DIRECT_AICPU_ARCHIVE=${KVAF_DIRECT_AICPU_ARCHIVE:-"${ASCEND_HOME_PATH}/opp/vendors/cust/op_impl/aicpu/kernel/cann-kvaf-direct-urma-compat.tar.gz"}
KVAF_DIRECT_AICPU_REGISTRATION_FILE=${KVAF_DIRECT_AICPU_REGISTRATION_FILE:-"${ASCEND_HOME_PATH}/conf/ascend_package_load.ini"}

kvaf_verify_direct_provider_build_v3 ||
  die "provider build verification failed"
kvaf_verify_direct_runtime_artifacts_v3 ||
  die "runtime/package/registration verification failed"

runtime_probe_provider_filename=${KVAF_VERIFIED_RUNTIME_PROBE_PROVIDER_FILENAME:-}
runtime_probe_provider_build_id=${KVAF_VERIFIED_RUNTIME_PROBE_PROVIDER_BUILD_ID:-}
runtime_probe_provider_sha=${KVAF_VERIFIED_RUNTIME_PROBE_PROVIDER_SHA:-}
[[ ${runtime_probe_provider_filename} =~ ^libkvaf_rt_[0-9a-f]{16}\.so$ &&
   ${runtime_probe_provider_build_id} =~ ^[0-9a-f]{64}$ &&
   ${runtime_probe_provider_filename} == \
     "libkvaf_rt_${runtime_probe_provider_build_id:0:16}.so" &&
   ${runtime_probe_provider_sha} =~ ^[0-9a-f]{64}$ ]] ||
  die "verified versioned runtime provider identity is malformed"
if (( provider_soname_count == 1 )); then
  [[ ${provider_soname} == "${runtime_probe_provider_filename}" ]] ||
    die "provider soname override differs from the verified build identity"
else
  kvaf_probe_arguments+=(
    --provider-soname "${runtime_probe_provider_filename}")
fi
kvaf_probe_arguments+=(
  --provider-build-id "${runtime_probe_provider_build_id}")

manifest_probe=$(kvaf_runtime_preflight_manifest_value \
  runtime_urma_probe_path "${KVAF_A5_DIRECT_BUILD_MANIFEST}") ||
  die "direct build manifest lacks runtime_urma_probe_path"
manifest_probe_sha=$(kvaf_runtime_preflight_manifest_value \
  runtime_urma_probe_sha256 "${KVAF_A5_DIRECT_BUILD_MANIFEST}") ||
  die "direct build manifest lacks runtime_urma_probe_sha256"
[[ ! -L ${KVAF_A5_RUNTIME_URMA_PROBE_BINARY} &&
   -x ${KVAF_A5_RUNTIME_URMA_PROBE_BINARY} ]] ||
  die "runtime URMA probe binary is missing or unsafe"
[[ $(realpath -e "${KVAF_A5_RUNTIME_URMA_PROBE_BINARY}") == \
   $(realpath -e "${manifest_probe}") ]] ||
  die "runtime URMA probe path differs from its build manifest"
actual_probe_sha=$(sha256sum "${KVAF_A5_RUNTIME_URMA_PROBE_BINARY}" |
  awk '{print $1}')
[[ ${actual_probe_sha} == "${manifest_probe_sha}" ]] ||
  die "runtime URMA probe changed after its successful build"

probe_dynamic=$(readelf -d "${KVAF_A5_RUNTIME_URMA_PROBE_BINARY}")
grep -Fq libhcomm <<<"${probe_dynamic}" &&
  die "runtime URMA probe unexpectedly links HCOMM"
probe_ldd=$(ldd -r "${KVAF_A5_RUNTIME_URMA_PROBE_BINARY}" 2>&1) || {
  echo "${probe_ldd}" >&2
  die "runtime URMA probe linkage is unresolved"
}
grep -Eq 'not found|undefined symbol' <<<"${probe_ldd}" && {
  echo "${probe_ldd}" >&2
  die "runtime URMA probe linkage is unresolved"
}

[[ ${KVAF_RUNTIME_PROBE_OBSERVATION_TIMEOUT_SECONDS} =~ ^[1-9][0-9]*$ &&
   ${KVAF_RUNTIME_PROBE_OBSERVATION_TIMEOUT_SECONDS} -le 3600 ]] ||
  die "KVAF_RUNTIME_PROBE_OBSERVATION_TIMEOUT_SECONDS must be in [1,3600]"
[[ ${KVAF_RUNTIME_PROBE_READY_TIMEOUT_SECONDS} =~ ^[1-9][0-9]*$ &&
   ${KVAF_RUNTIME_PROBE_READY_TIMEOUT_SECONDS} -le 120 ]] ||
  die "KVAF_RUNTIME_PROBE_READY_TIMEOUT_SECONDS must be in [1,120]"
: "${KVAF_PHYSICAL_DEVICE_ID:?set KVAF_PHYSICAL_DEVICE_ID}"
[[ ${KVAF_PHYSICAL_DEVICE_ID} =~ ^(0|[1-9][0-9]*)$ ]] ||
  die "KVAF_PHYSICAL_DEVICE_ID must be an unsigned integer"
[[ ${logical_device_id} == "${KVAF_PHYSICAL_DEVICE_ID}" ]] ||
  die "all-device container requires logical and physical device IDs to match"
[[ -f ${KVAF_NPU_IDLE_CHECKER} && ! -L ${KVAF_NPU_IDLE_CHECKER} ]] ||
  die "NPU idle checker is missing or unsafe"
set +e
kvaf_npu_smi_output=$(LC_ALL=C timeout --signal=TERM --kill-after=2s \
  30s npu-smi info 2>&1)
kvaf_npu_smi_rc=$?
set -e
(( kvaf_npu_smi_rc == 0 )) || {
  echo "${kvaf_npu_smi_output}" >&2
  die "npu-smi could not prove the selected NPU idle"
}
set +e
kvaf_idle_output=$(python3 "${KVAF_NPU_IDLE_CHECKER}" \
  --physical-device-id "${KVAF_PHYSICAL_DEVICE_ID}" \
  <<<"${kvaf_npu_smi_output}" 2>&1)
kvaf_idle_rc=$?
set -e
(( kvaf_idle_rc == 0 )) &&
  grep -Fxq KVAF_NPU_IDLE_STATUS=idle <<<"${kvaf_idle_output}" || {
    echo "${kvaf_idle_output}" >&2
    die "selected physical NPU is not strictly proven idle"
  }
echo "KVAF_A5_NPU_IDLE_PROVEN physical_device_id=${KVAF_PHYSICAL_DEVICE_ID}"

# dlopen itself executes the official liburma constructor.  The official
# public API documents a Host Linux caller domain, not AICPU/Device OS, so all
# three modes require the same explicit risk decision before quarantine and
# before any ACL/AICPU launch.
[[ ${KVAF_ACCEPT_UNDOCUMENTED_AICPU_URMA:-} == \
   I_ACCEPT_UNDOCUMENTED_AICPU_URMA_ABI_SOURCE_CALLER_AND_CLEANUP_RISK ]] ||
  die "set KVAF_ACCEPT_UNDOCUMENTED_AICPU_URMA=I_ACCEPT_UNDOCUMENTED_AICPU_URMA_ABI_SOURCE_CALLER_AND_CLEANUP_RISK for every mode"
echo "KVAF_A5_RUNTIME_URMA_ABI_SOURCE_CALLER_AND_CLEANUP_RISK_ACCEPTED"

if [[ ${mode} != discover ]]; then
  : "${KVAF_HOST_EID:?set KVAF_HOST_EID for exact endpoint affinity}"
  : "${KVAF_DEVICE_EID:?set KVAF_DEVICE_EID for exact endpoint affinity}"
  [[ -f ${KVAF_ENDPOINT_RESOLVER} && ! -L ${KVAF_ENDPOINT_RESOLVER} ]] ||
    die "URMA endpoint resolver is missing or unsafe"
  endpoint_output=$(python3 "${KVAF_ENDPOINT_RESOLVER}" \
    --physical-device-id "${KVAF_PHYSICAL_DEVICE_ID}" \
    --host-eid "${KVAF_HOST_EID}" \
    --device-eid "${KVAF_DEVICE_EID}" \
    --shell) || die "exact Host/Device URMA endpoint resolution failed"
  resolved_device=
  resolved_eid_index=
  while IFS='=' read -r key value; do
    case ${key} in
      KVAF_DEVICE_URMA_DEVICE) resolved_device=${value} ;;
      KVAF_DEVICE_EID_INDEX) resolved_eid_index=${value} ;;
      KVAF_HOST_URMA_DEVICE|KVAF_HOST_EID_INDEX) ;;
      *) die "unexpected endpoint resolver field: ${key}" ;;
    esac
  done <<<"${endpoint_output}"
  [[ ${resolved_device} == "${urma_device}" &&
     ${resolved_eid_index} == "${eid_index}" ]] ||
    die "requested Device URMA endpoint does not match NPU/EID inventory"
  echo "KVAF_A5_RUNTIME_URMA_ENDPOINT_AFFINITY_PROVEN device=${urma_device}/eid${eid_index}"
fi

run_id="runtime-urma-${mode}-$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM}"
probe_log="${KVAF_DIRECT_STATE_DIR}/runtime-probes/${run_id}.log"
probe_ready_file="${KVAF_DIRECT_STATE_DIR}/runtime-probes/${run_id}.ready"
probe_terminal_file="${KVAF_DIRECT_STATE_DIR}/runtime-probes/${run_id}.terminal"
[[ ! -e ${probe_log} && ! -L ${probe_log} &&
   ! -e ${probe_ready_file} && ! -L ${probe_ready_file} &&
   ! -e ${probe_terminal_file} && ! -L ${probe_terminal_file} ]] ||
  die "runtime probe result path already exists"

# Every mode runs under a new session and a detached supervisor.  The
# supervisor, not this observational parent, validates inherited lock FDs and
# durably publishes quarantine -> active -> ready before it starts the worker.
kvaf_supervisor_script=${SCRIPT_DIR}/a5_runtime_probe_supervisor.sh
[[ -f ${kvaf_supervisor_script} && ! -L ${kvaf_supervisor_script} &&
   -x ${kvaf_supervisor_script} ]] ||
  die "runtime probe supervisor is missing, unsafe or not executable"
kvaf_worker_path=$(realpath -e "${KVAF_A5_RUNTIME_URMA_PROBE_BINARY}")
kvaf_boot_id=$(< /proc/sys/kernel/random/boot_id)
export KVAF_RUNTIME_SUPERVISOR_DURABLE_HELPER=${KVAF_DURABLE_STATE_HELPER}
export KVAF_RUNTIME_SUPERVISOR_STATE_HELPER=${SCRIPT_DIR}/a5_runtime_probe_state.sh
export KVAF_RUNTIME_SUPERVISOR_ACTIVE_LOCK=${KVAF_DIRECT_ACTIVE_SESSION_LOCK}
export KVAF_RUNTIME_SUPERVISOR_DEPLOY_LOCK=${KVAF_DIRECT_DEPLOY_RUN_LOCK}
export KVAF_RUNTIME_SUPERVISOR_QUARANTINE=${KVAF_DIRECT_QUARANTINE_FILE}
export KVAF_RUNTIME_SUPERVISOR_ACTIVE=${KVAF_DIRECT_ACTIVE_SESSION_FILE}
export KVAF_RUNTIME_SUPERVISOR_READY=${probe_ready_file}
export KVAF_RUNTIME_SUPERVISOR_TERMINAL=${probe_terminal_file}
export KVAF_RUNTIME_SUPERVISOR_LOG=${probe_log}
export KVAF_RUNTIME_SUPERVISOR_RUN_ID=${run_id}
export KVAF_RUNTIME_SUPERVISOR_MODE=${mode}
export KVAF_RUNTIME_SUPERVISOR_BOOT_ID=${kvaf_boot_id}
export KVAF_RUNTIME_SUPERVISOR_LOGICAL_ID=${logical_device_id}
export KVAF_RUNTIME_SUPERVISOR_PHYSICAL_ID=${KVAF_PHYSICAL_DEVICE_ID}
export KVAF_RUNTIME_SUPERVISOR_WORKER_PATH=${kvaf_worker_path}
export KVAF_RUNTIME_SUPERVISOR_WORKER_SHA256=${actual_probe_sha}
export KVAF_RUNTIME_SUPERVISOR_PROVIDER_SONAME=${runtime_probe_provider_filename}
export KVAF_RUNTIME_SUPERVISOR_PROVIDER_BUILD_ID=${runtime_probe_provider_build_id}
export KVAF_RUNTIME_SUPERVISOR_PROVIDER_ARTIFACT_SHA256=${runtime_probe_provider_sha}
export KVAF_RUNTIME_SUPERVISOR_CALLER_DOMAIN_CONTRACT=undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1
setsid "${kvaf_supervisor_script}" "${kvaf_probe_arguments[@]}" 8>&8 9>&9 &
kvaf_supervisor_pid=$!

# Parent/supervisor ready handshake.  Seeing a process ID is insufficient: the
# ready record must bind that PID/starttime, mode, boot, worker path+digest and
# result paths before the parent reports that a resource owner was launched.
kvaf_ready_seen=0
kvaf_ready_deadline=$((SECONDS + KVAF_RUNTIME_PROBE_READY_TIMEOUT_SECONDS))
while (( kvaf_ready_seen == 0 )); do
  if [[ -e ${probe_terminal_file} ]]; then
    break
  fi
  if [[ -e ${probe_ready_file} ]]; then
    kvaf_owned_single_link_regular_file "${KVAF_DIRECT_QUARANTINE_FILE}" &&
      kvaf_owned_single_link_regular_file \
        "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" || {
      echo "ERROR: ready appeared without durable quarantine/active state" >&2
      exit 4
    }
    kvaf_owned_single_link_regular_file "${probe_ready_file}" || {
      echo "ERROR: detached supervisor published unsafe ready evidence" >&2
      exit 4
    }
    kvaf_runtime_probe_file_has_exact_keys "${probe_ready_file}" \
      schema run_id mode boot_id logical_device_id physical_device_id \
      supervisor_pid supervisor_start_ticks worker_path worker_sha256 log \
      provider_soname provider_build_id_sha256 \
      expected_packaged_provider_artifact_sha256 \
      terminal_file reason terminal_contract lock_contract \
      caller_domain_contract || {
        echo "ERROR: detached supervisor published malformed ready evidence" >&2
        exit 4
      }
    [[ $(kvaf_runtime_probe_read_single_value "${probe_ready_file}" schema) == \
         kvaf-a5-runtime-urma-probe-ready-v1 &&
       $(kvaf_runtime_probe_read_single_value "${probe_ready_file}" run_id) == \
         "${run_id}" &&
       $(kvaf_runtime_probe_read_single_value "${probe_ready_file}" mode) == \
         "${mode}" &&
       $(kvaf_runtime_probe_read_single_value "${probe_ready_file}" boot_id) == \
         "${kvaf_boot_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${probe_ready_file}" logical_device_id) == "${logical_device_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${probe_ready_file}" physical_device_id) == \
         "${KVAF_PHYSICAL_DEVICE_ID}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${probe_ready_file}" supervisor_pid) == "${kvaf_supervisor_pid}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${probe_ready_file}" worker_path) == "${kvaf_worker_path}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${probe_ready_file}" worker_sha256) == "${actual_probe_sha}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${probe_ready_file}" provider_soname) == \
         "${runtime_probe_provider_filename}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${probe_ready_file}" provider_build_id_sha256) == \
         "${runtime_probe_provider_build_id}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${probe_ready_file}" expected_packaged_provider_artifact_sha256) == \
         "${runtime_probe_provider_sha}" &&
       $(kvaf_runtime_probe_read_single_value "${probe_ready_file}" log) == \
         "${probe_log}" &&
       $(kvaf_runtime_probe_read_single_value \
         "${probe_ready_file}" terminal_file) == "${probe_terminal_file}" &&
       $(kvaf_runtime_probe_read_single_value "${probe_ready_file}" reason) == \
         supervisor_ready_to_own_worker_lifetime &&
       $(kvaf_runtime_probe_read_single_value \
         "${probe_ready_file}" terminal_contract) == \
         runtime-probe-exact-terminal-v2 &&
       $(kvaf_runtime_probe_read_single_value \
         "${probe_ready_file}" lock_contract) == \
         inherited-supervisor-worker-v4-held-flock-pinned-worker-fd &&
       $(kvaf_runtime_probe_read_single_value \
         "${probe_ready_file}" caller_domain_contract) == \
         undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1 ]] || {
      echo "ERROR: detached supervisor ready identity does not match launch" >&2
      exit 4
    }
    kvaf_supervisor_start_ticks=$(kvaf_runtime_probe_read_single_value \
      "${probe_ready_file}" supervisor_start_ticks) || exit 4
    [[ ${kvaf_supervisor_start_ticks} =~ ^[1-9][0-9]*$ ]] || exit 4
    kvaf_supervisor_identity=$(
      kvaf_runtime_probe_read_proc_state_and_start_ticks \
        "${kvaf_supervisor_pid}"
    ) || {
      [[ -e ${probe_terminal_file} ]] || exit 4
    }
    if [[ -n ${kvaf_supervisor_identity:-} ]]; then
      read -r kvaf_supervisor_state kvaf_observed_start_ticks \
        <<<"${kvaf_supervisor_identity}" || exit 4
      [[ ${kvaf_observed_start_ticks} == "${kvaf_supervisor_start_ticks}" &&
         ${kvaf_supervisor_state} != Z ]] || {
        [[ -e ${probe_terminal_file} ]] || exit 4
      }
    fi
    kvaf_ready_seen=1
    echo "KVAF_A5_RUNTIME_URMA_SUPERVISOR_READY run_id=${run_id} mode=${mode}"
    break
  fi
  if (( SECONDS >= kvaf_ready_deadline )); then
    echo "ERROR: detached supervisor ready handshake timed out" >&2
    echo "SUPERVISOR_PID=${kvaf_supervisor_pid}" >&2
    exit 4
  fi
  if ! kill -0 "${kvaf_supervisor_pid}" 2>/dev/null; then
    sleep 1
    [[ -e ${probe_terminal_file} ]] || {
      echo "ERROR: detached supervisor exited before ready/terminal publication" >&2
      exit 4
    }
    break
  fi
  sleep 1
done

kvaf_deadline=$((SECONDS + KVAF_RUNTIME_PROBE_OBSERVATION_TIMEOUT_SECONDS))
while [[ ! -e ${probe_terminal_file} ]]; do
  if (( SECONDS >= kvaf_deadline )); then
    echo "KVAF_A5_RUNTIME_URMA_OBSERVATION_TIMEOUT"
    echo "SUPERVISOR_PID=${kvaf_supervisor_pid}"
    echo "ACTIVE_SESSION=${KVAF_DIRECT_ACTIVE_SESSION_FILE}"
    echo "QUARANTINE=${KVAF_DIRECT_QUARANTINE_FILE}"
    echo "LOG=${probe_log}"
    [[ ! -r ${probe_log} ]] || tail -n 120 "${probe_log}"
    exit 4
  fi
  if ! kill -0 "${kvaf_supervisor_pid}" 2>/dev/null; then
    # Give the supervisor's final log fsync + terminal publication one bounded
    # grace interval.  Absence remains unknown and fail-closed.
    sleep 1
    [[ -e ${probe_terminal_file} ]] || {
      echo "ERROR: detached supervisor exited without durable terminal" >&2
      exit 4
    }
    break
  fi
  sleep 1
done

kvaf_owned_single_link_regular_file "${probe_terminal_file}" || {
  echo "ERROR: detached supervisor terminal is unsafe" >&2
  exit 4
}
if (( kvaf_ready_seen == 0 )); then
  kvaf_supervisor_start_ticks=$(kvaf_runtime_probe_read_single_value \
    "${probe_terminal_file}" supervisor_start_ticks) || exit 4
fi
kvaf_runtime_probe_terminal_evidence_matches \
  "${probe_terminal_file}" "${run_id}" "${mode}" "${kvaf_boot_id}" \
  "${logical_device_id}" "${KVAF_PHYSICAL_DEVICE_ID}" \
  "${kvaf_worker_path}" "${actual_probe_sha}" "${probe_log}" \
  "${kvaf_supervisor_pid}" "${kvaf_supervisor_start_ticks}" \
  "${runtime_probe_provider_filename}" \
  "${runtime_probe_provider_build_id}" \
  "${runtime_probe_provider_sha}" || {
    echo "ERROR: detached supervisor terminal evidence does not match launch/log" >&2
    exit 4
  }
probe_rc=$(kvaf_runtime_probe_read_single_value \
  "${probe_terminal_file}" rc) || exit 4

# Terminal is published immediately before supervisor exit.  Do not block
# indefinitely in wait(2) if that final exit is unexpectedly wedged.
for kvaf_exit_poll in 1 2 3 4 5; do
  kvaf_supervisor_identity=$(
    kvaf_runtime_probe_read_proc_state_and_start_ticks \
      "${kvaf_supervisor_pid}" 2>/dev/null
  ) || break
  read -r kvaf_supervisor_state kvaf_observed_start_ticks \
    <<<"${kvaf_supervisor_identity}" || break
  if [[ ${kvaf_observed_start_ticks} != "${kvaf_supervisor_start_ticks}" ||
        ${kvaf_supervisor_state} == Z ]]; then
    break
  fi
  sleep 1
done
kvaf_supervisor_still_live=0
kvaf_supervisor_identity=$(
  kvaf_runtime_probe_read_proc_state_and_start_ticks \
    "${kvaf_supervisor_pid}" 2>/dev/null
) || true
if [[ -n ${kvaf_supervisor_identity} ]]; then
  read -r kvaf_supervisor_state kvaf_observed_start_ticks \
    <<<"${kvaf_supervisor_identity}" || exit 4
  if [[ ${kvaf_observed_start_ticks} == "${kvaf_supervisor_start_ticks}" &&
        ${kvaf_supervisor_state} != Z ]]; then
    kvaf_supervisor_still_live=1
  fi
fi
if (( kvaf_supervisor_still_live == 1 )); then
  echo "ERROR: supervisor remained live after sealed terminal publication" >&2
  exit 4
fi
set +e
wait "${kvaf_supervisor_pid}" 2>/dev/null
set -e
cat "${probe_log}"

kvaf_safe_terminal=0
if kvaf_runtime_probe_log_is_safe_terminal \
    "${probe_rc}" "${probe_log}" "${mode}"; then
  kvaf_safe_terminal=1
fi

if (( kvaf_safe_terminal == 1 )); then
  python3 "${KVAF_DURABLE_STATE_HELPER}" remove \
    "${KVAF_DIRECT_ACTIVE_SESSION_FILE}"
  python3 "${KVAF_DURABLE_STATE_HELPER}" remove "${probe_ready_file}"
  python3 "${KVAF_DURABLE_STATE_HELPER}" remove \
    "${KVAF_DIRECT_QUARANTINE_FILE}"
elif (( probe_rc == 5 )) &&
     grep -Fxq KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK "${probe_log}" &&
     grep -Fxq KVAF_A5_RUNTIME_URMA_PROCESS_CONTAMINATED "${probe_log}"; then
  python3 "${KVAF_DURABLE_STATE_HELPER}" remove \
    "${KVAF_DIRECT_ACTIVE_SESSION_FILE}"
  python3 "${KVAF_DURABLE_STATE_HELPER}" remove "${probe_ready_file}"
  kvaf_publish_runtime_probe_blocked \
    "${run_id}" "${mode}" "${kvaf_boot_id}" \
    "${logical_device_id}" "${KVAF_PHYSICAL_DEVICE_ID}" \
    "${kvaf_worker_path}" "${actual_probe_sha}" \
    "${probe_log}" "${probe_terminal_file}" \
    "${runtime_probe_provider_filename}" \
    "${runtime_probe_provider_build_id}" \
    "${runtime_probe_provider_sha}" || exit 4
else
  echo "KVAF_A5_RUNTIME_URMA_TERMINAL_UNPROVEN" >&2
  echo "ACTIVE_SESSION=${KVAF_DIRECT_ACTIVE_SESSION_FILE}" >&2
  echo "QUARANTINE=${KVAF_DIRECT_QUARANTINE_FILE}" >&2
fi
echo "KVAF_RUNTIME_URMA_PROBE_RC=${probe_rc}"
echo "KVAF_RUNTIME_URMA_PROBE_LOG=${probe_log}"
echo "KVAF_RUNTIME_URMA_PROBE_TERMINAL=${probe_terminal_file}"
if (( kvaf_safe_terminal != 1 )); then
  echo "KVAF_RUNTIME_URMA_PROBE_QUARANTINED=${KVAF_DIRECT_QUARANTINE_FILE}" >&2
fi
exit "${probe_rc}"
fi
/bin/sh -c 'exit 2'
