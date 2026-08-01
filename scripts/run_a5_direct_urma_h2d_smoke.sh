#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

kvaf_affinity_run_mode=${1:-production}
if (( $# > 1 )) ||
   [[ "${kvaf_affinity_run_mode}" != production &&
      "${kvaf_affinity_run_mode}" != \
        candidate-epoch-owner-validation-v1 ]]; then
  echo "ERROR: invalid direct URMA runner mode" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
source "${SCRIPT_DIR}/a5_direct_urma_legacy_guard.sh"
source "${SCRIPT_DIR}/a5_aicpu_affinity_guard.sh"
source "${SCRIPT_DIR}/a5_direct_urma_manifest_v3.sh"
source "${SCRIPT_DIR}/a5_direct_urma_runtime_preflight.sh"
kvaf_configure_direct_urma_state_paths "${ROOT_DIR}" || exit 2

: "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"
KVAF_A5_DIRECT_BUILD_DIR=${KVAF_A5_DIRECT_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma"}
KVAF_A5_DIRECT_BINARY=${KVAF_A5_DIRECT_BINARY:-"${KVAF_A5_DIRECT_BUILD_DIR}/access_operator/a5/kvaf_a5_aiv_direct_urma_h2d_smoke"}
KVAF_A5_DIRECT_BUILD_MANIFEST=${KVAF_A5_DIRECT_BUILD_MANIFEST:-"${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke.manifest"}
KVAF_DIRECT_PROVIDER_BUILD_DIR=${KVAF_DIRECT_PROVIDER_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma-provider"}
KVAF_DIRECT_HOST_PROVIDER_LIBRARY=${KVAF_DIRECT_HOST_PROVIDER_LIBRARY:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/libkvaf_urma_provider_host.so"}
KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY=${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/libkvaf_urma_provider_device.so"}
KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST=${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/source.sha256"}
KVAF_DIRECT_PROVIDER_BUILD_MANIFEST=${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/build.manifest"}
KVAF_DIRECT_PROVIDER_SUCCESS_STAMP=${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/build.success"}
KVAF_DIRECT_PROVIDER_SONAME=${KVAF_DIRECT_PROVIDER_SONAME:-libkvaf_urma_provider_device.so}
KVAF_DIRECT_AICPU_ARCHIVE=${KVAF_DIRECT_AICPU_ARCHIVE:-"${ASCEND_HOME_PATH}/opp/vendors/cust/op_impl/aicpu/kernel/cann-kvaf-direct-urma-compat.tar.gz"}
KVAF_DIRECT_AICPU_REGISTRATION_FILE=${KVAF_DIRECT_AICPU_REGISTRATION_FILE:-"${ASCEND_HOME_PATH}/conf/ascend_package_load.ini"}
KVAF_DURABLE_STATE_HELPER="${SCRIPT_DIR}/durable_state_file.py"
KVAF_HOST_EID=${KVAF_HOST_EID:-}
KVAF_DEVICE_EID=${KVAF_DEVICE_EID:-}
KVAF_PAIR_CHANNEL=${KVAF_PAIR_CHANNEL:-}
KVAF_HOST_URMA_DEVICE=${KVAF_HOST_URMA_DEVICE:-}
KVAF_DEVICE_URMA_DEVICE=${KVAF_DEVICE_URMA_DEVICE:-}
KVAF_HOST_EID_INDEX=${KVAF_HOST_EID_INDEX:-}
KVAF_DEVICE_EID_INDEX=${KVAF_DEVICE_EID_INDEX:-}
KVAF_EID_PROC_ROOT=${KVAF_EID_PROC_ROOT:-/proc/asdrv_ub}
KVAF_EID_LOCK_TIMEOUT_SECONDS=${KVAF_EID_LOCK_TIMEOUT_SECONDS:-10}
KVAF_UBCORE_SYSFS_ROOT=${KVAF_UBCORE_SYSFS_ROOT:-/sys/class/ubcore}
KVAF_HCCN_TOOL=${KVAF_HCCN_TOOL:-}
KVAF_HCCN_TIMEOUT_SECONDS=${KVAF_HCCN_TIMEOUT_SECONDS:-30}
KVAF_LOGICAL_DEVICE_ID=${KVAF_LOGICAL_DEVICE_ID:-0}
KVAF_PHYSICAL_DEVICE_ID=${KVAF_PHYSICAL_DEVICE_ID:-}
KVAF_DIRECT_CONTROL_PORT=${KVAF_DIRECT_CONTROL_PORT:-21115}
KVAF_DIRECT_CONTROL_SERVER_ADDRESS=${KVAF_DIRECT_CONTROL_SERVER_ADDRESS:-}
KVAF_DIRECT_CONTROL_BIND_ADDRESS=${KVAF_DIRECT_CONTROL_BIND_ADDRESS:-}
kvaf_legacy_h2d_bytes_is_set=0
kvaf_transfer_count_is_set=0
kvaf_bytes_per_transfer_is_set=0
kvaf_batch_iterations_is_set=0
kvaf_descriptor_capacity_is_set=0
kvaf_run_timeout_is_set=0
[[ -n "${KVAF_H2D_BYTES+x}" ]] && kvaf_legacy_h2d_bytes_is_set=1
[[ -n "${KVAF_H2D_TRANSFER_COUNT+x}" ]] && kvaf_transfer_count_is_set=1
[[ -n "${KVAF_H2D_BYTES_PER_TRANSFER+x}" ]] &&
  kvaf_bytes_per_transfer_is_set=1
[[ -n "${KVAF_H2D_BATCH_ITERATIONS+x}" ]] &&
  kvaf_batch_iterations_is_set=1
[[ -n "${KVAF_H2D_DESCRIPTOR_CAPACITY+x}" ]] &&
  kvaf_descriptor_capacity_is_set=1
[[ -n "${KVAF_H2D_RUN_TIMEOUT_SECONDS+x}" ]] &&
  kvaf_run_timeout_is_set=1
if (( kvaf_legacy_h2d_bytes_is_set != 0 &&
      (kvaf_transfer_count_is_set != 0 ||
       kvaf_bytes_per_transfer_is_set != 0 ||
       kvaf_batch_iterations_is_set != 0 ||
       kvaf_descriptor_capacity_is_set != 0) )); then
  echo "ERROR: legacy KVAF_H2D_BYTES cannot be combined with the new H2D workload variables" >&2
  exit 2
fi
if (( kvaf_legacy_h2d_bytes_is_set != 0 )); then
  KVAF_H2D_TRANSFER_COUNT=1
  KVAF_H2D_BYTES_PER_TRANSFER=${KVAF_H2D_BYTES}
  KVAF_H2D_BATCH_ITERATIONS=1
  KVAF_H2D_DESCRIPTOR_CAPACITY=1
  kvaf_h2d_workload_mode=legacy-total-bytes
else
  KVAF_H2D_TRANSFER_COUNT=${KVAF_H2D_TRANSFER_COUNT:-4096}
  KVAF_H2D_BYTES_PER_TRANSFER=${KVAF_H2D_BYTES_PER_TRANSFER:-1024}
  KVAF_H2D_BATCH_ITERATIONS=${KVAF_H2D_BATCH_ITERATIONS:-1}
  KVAF_H2D_DESCRIPTOR_CAPACITY=${KVAF_H2D_DESCRIPTOR_CAPACITY:-4096}
  kvaf_h2d_workload_mode=explicit-transfer-shape
fi
KVAF_H2D_TIMEOUT_SECONDS=${KVAF_H2D_TIMEOUT_SECONDS:-120}
if [[ -n "${KVAF_DIRECT_SERVER_START_DELAY_SECONDS+x}" ]]; then
  echo "ERROR: KVAF_DIRECT_SERVER_START_DELAY_SECONDS was removed; fixed sleeps consume the Host accept deadline" >&2
  echo "Set KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS instead; Device starts immediately after PID-bound LISTEN proof." >&2
  exit 2
fi
KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS=${KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS:-30}
KVAF_HOST_CONTROL_TIMEOUT_SECONDS=${KVAF_HOST_CONTROL_TIMEOUT_SECONDS:-}
KVAF_A5_RUN_TAG=${KVAF_A5_RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}
KVAF_A5_RESULT_DIR=${KVAF_A5_RESULT_DIR:-"${KVAF_DIRECT_RESULT_ROOT}/a5_direct_urma_h2d_${KVAF_A5_RUN_TAG}"}
KVAF_DIRECT_ALLOW_BUSY_NPU=${KVAF_DIRECT_ALLOW_BUSY_NPU:-0}
KVAF_NPU_SMI_TIMEOUT_SECONDS=${KVAF_NPU_SMI_TIMEOUT_SECONDS:-15}
KVAF_ASCEND_HAL_LIBRARY=/usr/local/Ascend/driver/lib64/driver/libascend_hal.so
KVAF_NPU_IDLE_CHECKER="${SCRIPT_DIR}/check_a5_npu_idle.py"

require_uint()
{
  local name=$1
  local value=$2
  if ! [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "ERROR: ${name} must be canonical unsigned decimal, got ${value}" >&2
    exit 2
  fi
}

decimal_at_most()
{
  local value=$1
  local maximum=$2
  local LC_ALL=C

  if (( ${#value} < ${#maximum} )); then
    return 0
  fi
  if (( ${#value} > ${#maximum} )); then
    return 1
  fi
  [[ "${value}" == "${maximum}" || "${value}" < "${maximum}" ]]
}

require_uint_max()
{
  local name=$1
  local value=$2
  local maximum=$3

  require_uint "${name}" "${value}"
  decimal_at_most "${value}" "${maximum}" || {
    echo "ERROR: ${name} exceeds ${maximum}, got ${value}" >&2
    exit 2
  }
}

normalize_numeric_ip()
{
  python3 -c '
import ipaddress
import sys
try:
    print(ipaddress.ip_address(sys.argv[1]).compressed)
except ValueError:
    raise SystemExit(1)
' "$1"
}

resolve_ipourma_global_ipv6_for_eid()
{
  local eid=$1
  local address_output

  command -v ip >/dev/null 2>&1 || return 1
  address_output=$(ip -6 -o addr show scope global 2>/dev/null) || return 1
  python3 -c '
import ipaddress
import sys

try:
    expected = ipaddress.IPv6Address(int(sys.argv[1], 16))
except ValueError:
    raise SystemExit(1)

matches = []
for line in sys.stdin:
    fields = line.split()
    if len(fields) < 4 or fields[2] != "inet6":
        continue
    interface = fields[1].split("@", 1)[0]
    if not interface.startswith("ipourma"):
        continue
    if "tentative" in fields or "dadfailed" in fields:
        continue
    try:
        address = ipaddress.ip_interface(fields[3]).ip
    except ValueError:
        continue
    if address == expected:
        matches.append((address.compressed, interface))

if len(matches) != 1:
    raise SystemExit(1)
print(matches[0][0])
print(matches[0][1])
' "${eid}" <<<"${address_output}"
}

manifest_value()
{
  local key=$1
  local file=$2
  awk -F= -v key="${key}" '
    $1 == key {
      value = substr($0, length(key) + 2)
      count++
    }
    END {
      if (count != 1 || value == "") {
        exit 1
      }
      print value
    }
  ' "${file}"
}

valid_fatal_marker()
{
  local file=$1
  local schema run_id pid reason

  [[ ! -L "${file}" && -f "${file}" && -r "${file}" ]] || return 1
  schema=$(manifest_value schema "${file}") || return 1
  run_id=$(manifest_value run_id "${file}") || return 1
  pid=$(manifest_value pid "${file}") || return 1
  reason=$(manifest_value reason "${file}") || return 1
  [[ "${schema}" == kvaf-a5-direct-fatal-v1 &&
     "${run_id}" == "${kvaf_run_id}" &&
     "${pid}" =~ ^[1-9][0-9]*$ &&
     -n "${reason}" ]]
}

any_valid_fatal_marker()
{
  valid_fatal_marker "${exchange_dir}/fatal.host" ||
    valid_fatal_marker "${exchange_dir}/fatal.device"
}

durably_publish_state()
{
  python3 "${KVAF_DURABLE_STATE_HELPER}" publish "$1" "$2"
}

durably_remove_state()
{
  python3 "${KVAF_DURABLE_STATE_HELPER}" remove "$1"
}

recover_stale_active_session()
{
  local schema run_id phase result_dir recorded_manifest
  local recorded_run_id teardown_assurance host_pid device_pid
  local logical_device_id
  local host_start_ticks device_start_ticks
  local host_observed_exe device_observed_exe worker_expected_exe
  local outcome classifier_rc host_armed device_armed
  local results_root resolved_result pair_required recorded_classifier
  local recorded_classifier_sha actual_classifier_sha
  local quarantine_temporary quarantine_reason

  [[ ! -L "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" &&
     -f "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" &&
     -r "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ]] || return 1
  schema=$(manifest_value schema "${KVAF_DIRECT_ACTIVE_SESSION_FILE}") ||
    return 1
  run_id=$(manifest_value run_id "${KVAF_DIRECT_ACTIVE_SESSION_FILE}") ||
    return 1
  phase=$(manifest_value phase "${KVAF_DIRECT_ACTIVE_SESSION_FILE}") ||
    return 1
  result_dir=$(manifest_value result_dir \
    "${KVAF_DIRECT_ACTIVE_SESSION_FILE}") || return 1
  host_pid=$(manifest_value host_pid \
    "${KVAF_DIRECT_ACTIVE_SESSION_FILE}") || return 1
  device_pid=$(manifest_value device_pid \
    "${KVAF_DIRECT_ACTIVE_SESSION_FILE}") || return 1
  logical_device_id=$(manifest_value logical_device_id \
    "${KVAF_DIRECT_ACTIVE_SESSION_FILE}") || return 1
  host_start_ticks=$(manifest_value host_start_ticks \
    "${KVAF_DIRECT_ACTIVE_SESSION_FILE}") || return 1
  device_start_ticks=$(manifest_value device_start_ticks \
    "${KVAF_DIRECT_ACTIVE_SESSION_FILE}") || return 1
  host_observed_exe=$(manifest_value host_observed_exe \
    "${KVAF_DIRECT_ACTIVE_SESSION_FILE}") || return 1
  device_observed_exe=$(manifest_value device_observed_exe \
    "${KVAF_DIRECT_ACTIVE_SESSION_FILE}") || return 1
  worker_expected_exe=$(manifest_value worker_expected_exe \
    "${KVAF_DIRECT_ACTIVE_SESSION_FILE}") || return 1
  [[ "${schema}" == kvaf-a5-direct-active-v2 &&
     "${run_id}" =~ ^[0-9A-Fa-f]{32}$ &&
     "${host_pid}" =~ ^(0|[1-9][0-9]*)$ &&
     "${device_pid}" =~ ^(0|[1-9][0-9]*)$ &&
     "${logical_device_id}" =~ ^(0|[1-9][0-9]*)$ &&
     "${host_start_ticks}" =~ ^(0|[1-9][0-9]*)$ &&
     "${device_start_ticks}" =~ ^(0|[1-9][0-9]*)$ &&
     -n "${host_observed_exe}" && -n "${device_observed_exe}" &&
     "${worker_expected_exe}" == /* ]] || return 1

  case "${phase}" in
    before_host_launch)
      pair_required=0
      [[ "${host_pid}" == 0 && "${device_pid}" == 0 ]] || return 1
      ;;
    host_launch_intent)
      pair_required=0
      [[ "${host_pid}" == 0 && "${device_pid}" == 0 ]] || return 1
      ;;
    host_started|host_safe_terminal_process_stuck|host_terminal_unproven_process_stuck)
      pair_required=0
      [[ "${host_pid}" != 0 && "${device_pid}" == 0 ]] || return 1
      ;;
    device_launch_intent)
      pair_required=1
      [[ "${host_pid}" != 0 && "${device_pid}" == 0 ]] || return 1
      ;;
    device_started|safe_terminal_process_stuck|pair_terminal_unproven_process_stuck)
      pair_required=1
      [[ "${host_pid}" != 0 && "${device_pid}" != 0 ]] || return 1
      ;;
    *)
      return 1
      ;;
  esac

  # This phase is written before any fork intent. Once host_launch_intent is
  # durable, no recovery path may infer "nothing launched" from PID zero:
  # SIGKILL can occur between fork and the subsequent PID-state update.
  if [[ "${phase}" == before_host_launch &&
        "${host_pid}" == 0 && "${device_pid}" == 0 ]]; then
    durably_remove_state "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" || return 1
    echo "KVAF_A5_DIRECT_STALE_ACTIVE_RECOVERED phase=before_host_launch" >&2
    return 0
  fi

  results_root=$(realpath -e "${KVAF_DIRECT_RESULT_ROOT}") || return 1
  resolved_result=$(realpath -e "${result_dir}") || return 1
  [[ "${resolved_result}" == "${results_root}/"* ]] || return 1
  recorded_manifest="${resolved_result}/manifest.txt"
  [[ ! -L "${recorded_manifest}" && -f "${recorded_manifest}" ]] ||
    return 1
  recorded_run_id=$(manifest_value run_id "${recorded_manifest}") ||
    return 1
  teardown_assurance=$(manifest_value teardown_assurance \
    "${recorded_manifest}") || return 1
  [[ "${teardown_assurance}" == verified ||
     "${teardown_assurance}" == experimental-unverified ]] || return 1
  recorded_classifier_sha=$(manifest_value outcome_classifier_sha256 \
    "${recorded_manifest}") || return 1
  [[ "${recorded_run_id}" == "${run_id}" ]] || return 1
  recorded_classifier="${resolved_result}/classify_a5_direct_urma_outcome.py"
  [[ ! -L "${recorded_classifier}" && -f "${recorded_classifier}" &&
     -r "${recorded_classifier}" ]] || return 1
  actual_classifier_sha=$(sha256sum "${recorded_classifier}" |
    awk '{print $1}') || return 1
  [[ "${actual_classifier_sha}" == "${recorded_classifier_sha}" ]] ||
    return 1

  if (( pair_required == 0 )); then
    if outcome=$(python3 \
        "${recorded_classifier}" \
        --run-id "${run_id}" \
        --host-final "${resolved_result}/exchange/host.final" \
        --single-role host 2>/dev/null); then
      classifier_rc=0
    else
      classifier_rc=$?
    fi
  else
    if outcome=$(python3 \
        "${recorded_classifier}" \
        --run-id "${run_id}" \
        --host-final "${resolved_result}/exchange/host.final" \
        --device-final "${resolved_result}/exchange/device.final" \
        2>/dev/null); then
      classifier_rc=0
    else
      classifier_rc=$?
    fi
  fi
  quarantine_reason=
  if (( classifier_rc == 4 )) && [[ "${outcome}" == FAIL_UNSAFE ]]; then
    quarantine_reason=stale_explicit_fail_unsafe
  elif { (( classifier_rc != 0 )) || [[ "${outcome}" != PASS ]]; } &&
       { (( classifier_rc != 3 )) || [[ "${outcome}" != FAIL_SAFE ]]; }; then
    # Missing, malformed, stale, or incomplete evidence is UNPROVEN. Preserve
    # the active barrier so a late durable marker can be recovered later.
    return 1
  fi
  if [[ -z "${quarantine_reason}" &&
        "${teardown_assurance}" != verified ]]; then
    host_armed=$(manifest_value resource_armed \
      "${resolved_result}/exchange/host.final") || return 1
    device_armed=0
    if (( pair_required != 0 )); then
      device_armed=$(manifest_value resource_armed \
        "${resolved_result}/exchange/device.final") || return 1
    fi
    if [[ "${host_armed}" != 0 || "${device_armed}" != 0 ]]; then
      quarantine_reason="stale_unverified_udma_armed_terminal_${outcome}"
    fi
  fi
  if [[ -n "${quarantine_reason}" ]]; then
    quarantine_temporary="${KVAF_DIRECT_QUARANTINE_FILE}.tmp.$$"
    if ! {
      printf 'schema=kvaf-a5-direct-quarantine-v1\n'
      printf 'reason=%s\n' "${quarantine_reason}"
      printf 'run_id=%s\n' "${run_id}"
      printf 'logical_device_id=%s\n' "${logical_device_id}"
      printf 'host_pid=%s\n' "${host_pid}"
      printf 'device_pid=%s\n' "${device_pid}"
      printf 'host_start_ticks=%s\n' "${host_start_ticks}"
      printf 'device_start_ticks=%s\n' "${device_start_ticks}"
      printf 'host_observed_exe=%s\n' "${host_observed_exe}"
      printf 'device_observed_exe=%s\n' "${device_observed_exe}"
      printf 'worker_expected_exe=%s\n' "${worker_expected_exe}"
      printf 'result_dir=%s\n' "${resolved_result}"
      printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } >"${quarantine_temporary}"; then
      rm -f "${quarantine_temporary}"
      return 1
    fi
    if ! chmod 0600 "${quarantine_temporary}" ||
       ! durably_publish_state \
         "${quarantine_temporary}" "${KVAF_DIRECT_QUARANTINE_FILE}"; then
      rm -f "${quarantine_temporary}"
      return 1
    fi
    if ! durably_remove_state "${KVAF_DIRECT_ACTIVE_SESSION_FILE}"; then
      echo "WARNING: quarantine is durable, but stale active state remains" >&2
    fi
    echo "KVAF_A5_DIRECT_STALE_ACTIVE_QUARANTINED outcome=${outcome}" >&2
    echo "QUARANTINE_FILE=${KVAF_DIRECT_QUARANTINE_FILE}" >&2
    return 4
  fi
  durably_remove_state "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" || return 1
  echo "KVAF_A5_DIRECT_STALE_ACTIVE_RECOVERED outcome=${outcome}" >&2
  return 0
}

acquire_persistent_session_guards()
{
  local recovery_rc
  for command in awk flock python3 realpath sha256sum; do
    command -v "${command}" >/dev/null 2>&1 || {
      echo "ERROR: ${command} is required for persistent session guards" >&2
      exit 2
    }
  done
  if [[ -L "${KVAF_DURABLE_STATE_HELPER}" ||
        ! -f "${KVAF_DURABLE_STATE_HELPER}" ||
        ! -r "${KVAF_DURABLE_STATE_HELPER}" ]]; then
    echo "ERROR: durable state helper is missing or unsafe" >&2
    exit 2
  fi
  if kvaf_check_legacy_direct_urma_state "${ROOT_DIR}"; then
    :
  else
    exit $?
  fi
  if [[ -L "${KVAF_GLOBAL_STATE_DIR}" ||
        ( -e "${KVAF_GLOBAL_STATE_DIR}" &&
          ! -d "${KVAF_GLOBAL_STATE_DIR}" ) ]]; then
    echo "ERROR: global KVAF state path must be a real directory" >&2
    exit 2
  fi
  mkdir -p "${KVAF_GLOBAL_STATE_DIR}"
  chmod 0700 "${KVAF_GLOBAL_STATE_DIR}"
  if [[ -L "${KVAF_DIRECT_STATE_DIR}" ||
        ( -e "${KVAF_DIRECT_STATE_DIR}" &&
          ! -d "${KVAF_DIRECT_STATE_DIR}" ) ]]; then
    echo "ERROR: direct URMA state path must be a real directory" >&2
    exit 2
  fi
  mkdir -p "${KVAF_DIRECT_STATE_DIR}"
  chmod 0700 "${KVAF_DIRECT_STATE_DIR}"
  kvaf_global_state_root=$(realpath -e "${KVAF_GLOBAL_STATE_DIR}") || exit 2
  kvaf_state_root=$(realpath -e "${KVAF_DIRECT_STATE_DIR}") || exit 2
  if [[ "${kvaf_state_root}" != "${kvaf_global_state_root}/KVAF" ]]; then
    echo "ERROR: direct URMA state directory escaped its fixed global root" >&2
    exit 2
  fi
  if [[ -L "${KVAF_DIRECT_RESULT_ROOT}" ||
        ( -e "${KVAF_DIRECT_RESULT_ROOT}" &&
          ! -d "${KVAF_DIRECT_RESULT_ROOT}" ) ]]; then
    echo "ERROR: direct URMA result root must be a real directory" >&2
    exit 2
  fi
  mkdir -p "${KVAF_DIRECT_RESULT_ROOT}"
  chmod 0700 "${KVAF_DIRECT_RESULT_ROOT}"
  kvaf_results_root=$(realpath -e "${KVAF_DIRECT_RESULT_ROOT}") || exit 2
  if [[ "${kvaf_results_root}" != "${kvaf_state_root}/"* ]]; then
    echo "ERROR: direct URMA result root must remain below the persistent state directory" >&2
    echo "STATE_ROOT=${kvaf_state_root}" >&2
    echo "RESULT_ROOT=${kvaf_results_root}" >&2
    exit 2
  fi
  KVAF_DIRECT_RESULT_ROOT=${kvaf_results_root}
  mkdir -p \
    "$(dirname "${KVAF_DIRECT_DEPLOY_RUN_LOCK}")" \
    "$(dirname "${KVAF_DIRECT_ACTIVE_SESSION_LOCK}")" \
    "$(dirname "${KVAF_DIRECT_ACTIVE_SESSION_FILE}")" \
    "$(dirname "${KVAF_DIRECT_QUARANTINE_FILE}")"
  for guarded_path in \
    "${KVAF_DIRECT_DEPLOY_RUN_LOCK}" \
    "${KVAF_DIRECT_ACTIVE_SESSION_LOCK}" \
    "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" \
    "${KVAF_DIRECT_QUARANTINE_FILE}" \
    "${KVAF_EID_LOCK_FILE}"
  do
    if [[ -L "${guarded_path}" ]]; then
      echo "ERROR: persistent state path must not be a symlink: ${guarded_path}" >&2
      exit 2
    fi
  done
  exec 9>"${KVAF_DIRECT_DEPLOY_RUN_LOCK}"
  if ! flock -w "${KVAF_EID_LOCK_TIMEOUT_SECONDS}" 9; then
    echo "ERROR: another direct URMA deploy/run owns ${KVAF_DIRECT_DEPLOY_RUN_LOCK}" >&2
    exit 5
  fi

  if [[ -e "${KVAF_DIRECT_QUARANTINE_FILE}" ||
        -L "${KVAF_DIRECT_QUARANTINE_FILE}" ]]; then
    if [[ -L "${KVAF_DIRECT_QUARANTINE_FILE}" ||
          ! -f "${KVAF_DIRECT_QUARANTINE_FILE}" ||
          ! -r "${KVAF_DIRECT_QUARANTINE_FILE}" ]]; then
      echo "ERROR: quarantine path exists but is not a readable regular file" >&2
      echo "PATH=${KVAF_DIRECT_QUARANTINE_FILE}" >&2
      exit 4
    fi
    echo "ERROR: an unsafe direct URMA session is quarantined" >&2
    cat "${KVAF_DIRECT_QUARANTINE_FILE}" >&2
    echo "Resolve the recorded session before any topology/EID/NPU access." >&2
    exit 4
  fi

  if [[ -L "${KVAF_DIRECT_ACTIVE_SESSION_LOCK}" ]]; then
    echo "ERROR: active-session lock path must not be a symlink" >&2
    exit 2
  fi
  exec 8>"${KVAF_DIRECT_ACTIVE_SESSION_LOCK}"
  if ! flock -n 8; then
    echo "ERROR: a previous direct URMA worker still owns the active-session lock" >&2
    echo "LOCK=${KVAF_DIRECT_ACTIVE_SESSION_LOCK}" >&2
    if [[ -f "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" &&
          ! -L "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ]]; then
      echo "ACTIVE_SESSION:" >&2
      cat "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" >&2
    fi
    echo "No topology/EID/NPU operation was attempted." >&2
    exit 5
  fi

  if [[ -e "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ||
        -L "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ]]; then
    if recover_stale_active_session; then
      :
    else
      recovery_rc=$?
      if (( recovery_rc == 4 )) &&
         [[ -f "${KVAF_DIRECT_QUARANTINE_FILE}" &&
            ! -L "${KVAF_DIRECT_QUARANTINE_FILE}" ]]; then
        echo "ERROR: stale active session was promoted to quarantine" >&2
        cat "${KVAF_DIRECT_QUARANTINE_FILE}" >&2
        exit 4
      fi
      if [[ -f "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" &&
            ! -L "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ]]; then
        echo "ERROR: a previous direct URMA run lacks a proven terminal cleanup" >&2
        cat "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" >&2
      else
        echo "ERROR: active-session state is malformed or unsafe" >&2
      fi
      echo "STATE=${KVAF_DIRECT_ACTIVE_SESSION_FILE}" >&2
      echo "No topology/EID/NPU operation was attempted." >&2
      exit 5
    fi
  fi
}

require_uint_max KVAF_LOGICAL_DEVICE_ID \
  "${KVAF_LOGICAL_DEVICE_ID}" 4294967295
require_uint_max KVAF_DIRECT_CONTROL_PORT \
  "${KVAF_DIRECT_CONTROL_PORT}" 65535
require_uint_max KVAF_H2D_TRANSFER_COUNT \
  "${KVAF_H2D_TRANSFER_COUNT}" 67108864
require_uint_max KVAF_H2D_BYTES_PER_TRANSFER \
  "${KVAF_H2D_BYTES_PER_TRANSFER}" 67108864
require_uint_max KVAF_H2D_BATCH_ITERATIONS \
  "${KVAF_H2D_BATCH_ITERATIONS}" 67108864
require_uint_max KVAF_H2D_DESCRIPTOR_CAPACITY \
  "${KVAF_H2D_DESCRIPTOR_CAPACITY}" 4096
require_uint_max KVAF_H2D_TIMEOUT_SECONDS \
  "${KVAF_H2D_TIMEOUT_SECONDS}" 3600
if (( kvaf_run_timeout_is_set != 0 )); then
  require_uint_max KVAF_H2D_RUN_TIMEOUT_SECONDS \
    "${KVAF_H2D_RUN_TIMEOUT_SECONDS}" 604800
else
  # Preserve the historical worker-supervision deadline. This is separate
  # from --timeout-seconds below, which remains the provider/control timeout.
  KVAF_H2D_RUN_TIMEOUT_SECONDS=$((
    KVAF_H2D_TIMEOUT_SECONDS * 3 + 10
  ))
fi
require_uint_max KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS \
  "${KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS}" 3600
if [[ -z "${KVAF_HOST_CONTROL_TIMEOUT_SECONDS}" ]]; then
  KVAF_HOST_CONTROL_TIMEOUT_SECONDS=$((
    KVAF_H2D_TIMEOUT_SECONDS +
    KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS + 5
  ))
  if (( KVAF_HOST_CONTROL_TIMEOUT_SECONDS > 3600 )); then
    KVAF_HOST_CONTROL_TIMEOUT_SECONDS=3600
  fi
fi
if (( kvaf_run_timeout_is_set == 0 &&
      KVAF_H2D_RUN_TIMEOUT_SECONDS <
        KVAF_HOST_CONTROL_TIMEOUT_SECONDS + 10 )); then
  # A short Device timeout must not make the historical 3*T+10 default
  # expire before the independently longer Host accept window.  Explicit
  # user values still fail closed below when they cannot supervise Host.
  KVAF_H2D_RUN_TIMEOUT_SECONDS=$((
    KVAF_HOST_CONTROL_TIMEOUT_SECONDS + 10
  ))
fi
require_uint KVAF_EID_LOCK_TIMEOUT_SECONDS \
  "${KVAF_EID_LOCK_TIMEOUT_SECONDS}"
require_uint_max KVAF_HOST_CONTROL_TIMEOUT_SECONDS \
  "${KVAF_HOST_CONTROL_TIMEOUT_SECONDS}" 3600
require_uint_max KVAF_NPU_SMI_TIMEOUT_SECONDS \
  "${KVAF_NPU_SMI_TIMEOUT_SECONDS}" 120
require_uint_max KVAF_HCCN_TIMEOUT_SECONDS \
  "${KVAF_HCCN_TIMEOUT_SECONDS}" 120
if [[ "${KVAF_DIRECT_ALLOW_BUSY_NPU}" != 0 &&
      "${KVAF_DIRECT_ALLOW_BUSY_NPU}" != 1 ]]; then
  echo "ERROR: KVAF_DIRECT_ALLOW_BUSY_NPU must be 0 or 1" >&2
  exit 2
fi
if [[ -n "${KVAF_PHYSICAL_DEVICE_ID}" ]]; then
  require_uint_max KVAF_PHYSICAL_DEVICE_ID \
    "${KVAF_PHYSICAL_DEVICE_ID}" 4294967295
fi
if [[ -n "${KVAF_PAIR_CHANNEL}" ]]; then
  require_uint_max KVAF_PAIR_CHANNEL \
    "${KVAF_PAIR_CHANNEL}" 4294967295
fi
if (( KVAF_DIRECT_CONTROL_PORT < 1024 ||
      KVAF_DIRECT_CONTROL_PORT > 65535 )); then
  echo "ERROR: KVAF_DIRECT_CONTROL_PORT must be in [1024, 65535]" >&2
  exit 2
fi
if (( KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS == 0 ||
      KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS >
        KVAF_H2D_RUN_TIMEOUT_SECONDS )); then
  echo "ERROR: KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS must be in [1, KVAF_H2D_RUN_TIMEOUT_SECONDS]" >&2
  exit 2
fi
if (( KVAF_HOST_CONTROL_TIMEOUT_SECONDS < KVAF_H2D_TIMEOUT_SECONDS ||
      KVAF_HOST_CONTROL_TIMEOUT_SECONDS >
        KVAF_H2D_RUN_TIMEOUT_SECONDS )); then
  echo "ERROR: KVAF_HOST_CONTROL_TIMEOUT_SECONDS must be in [KVAF_H2D_TIMEOUT_SECONDS, KVAF_H2D_RUN_TIMEOUT_SECONDS]" >&2
  exit 2
fi
KVAF_DIRECT_CONTROL_SERVER_ADDRESS_SOURCE=explicit
KVAF_DIRECT_CONTROL_SERVER_INTERFACE=none
if [[ "${KVAF_DIRECT_CONTROL_SERVER_ADDRESS}" == auto ]]; then
  KVAF_DIRECT_CONTROL_SERVER_ADDRESS_SOURCE=auto-ipourma-eid
else
  if ! KVAF_DIRECT_CONTROL_SERVER_ADDRESS=$(
    normalize_numeric_ip "${KVAF_DIRECT_CONTROL_SERVER_ADDRESS}"
  ); then
    echo "ERROR: KVAF_DIRECT_CONTROL_SERVER_ADDRESS must be auto or a numeric IPv4/IPv6 address" >&2
    echo "URMA EIDs are not TCP management addresses." >&2
    exit 2
  fi
fi
if [[ -n "${KVAF_DIRECT_CONTROL_BIND_ADDRESS}" ]]; then
  if ! KVAF_DIRECT_CONTROL_BIND_ADDRESS=$(
    normalize_numeric_ip "${KVAF_DIRECT_CONTROL_BIND_ADDRESS}"
  ); then
    echo "ERROR: KVAF_DIRECT_CONTROL_BIND_ADDRESS must be empty or a numeric IPv4/IPv6 address" >&2
    exit 2
  fi
  if [[ "${KVAF_DIRECT_CONTROL_SERVER_ADDRESS_SOURCE}" == explicit ]] &&
     { { [[ "${KVAF_DIRECT_CONTROL_SERVER_ADDRESS}" == *:* ]] &&
         [[ "${KVAF_DIRECT_CONTROL_BIND_ADDRESS}" != *:* ]]; } ||
       { [[ "${KVAF_DIRECT_CONTROL_SERVER_ADDRESS}" != *:* ]] &&
         [[ "${KVAF_DIRECT_CONTROL_BIND_ADDRESS}" == *:* ]]; }; }; then
    echo "ERROR: control server and client bind addresses must use the same IP family" >&2
    exit 2
  fi
fi
if (( KVAF_H2D_BYTES_PER_TRANSFER == 0 ||
      KVAF_H2D_BATCH_ITERATIONS == 0 ||
      KVAF_H2D_DESCRIPTOR_CAPACITY == 0 ||
      KVAF_H2D_TIMEOUT_SECONDS == 0 ||
      KVAF_H2D_TIMEOUT_SECONDS > 3600 )); then
  echo "ERROR: invalid H2D workload shape or timeout" >&2
  exit 2
fi
if (( KVAF_H2D_RUN_TIMEOUT_SECONDS == 0 )); then
  echo "ERROR: KVAF_H2D_RUN_TIMEOUT_SECONDS must be in [1, 604800]" >&2
  exit 2
fi
if (( KVAF_H2D_TRANSFER_COUNT == 0 )) &&
   [[ "${kvaf_affinity_run_mode}" != \
      candidate-epoch-owner-validation-v1 ]]; then
  echo "ERROR: zero-WQE lifecycle is available only through the staged epoch-owner validation wrapper" >&2
  exit 2
fi
if (( KVAF_H2D_TRANSFER_COUNT > KVAF_H2D_DESCRIPTOR_CAPACITY )); then
  echo "ERROR: KVAF_H2D_TRANSFER_COUNT exceeds KVAF_H2D_DESCRIPTOR_CAPACITY" >&2
  exit 2
fi
if (( KVAF_H2D_DESCRIPTOR_CAPACITY >
      67108864 / KVAF_H2D_BYTES_PER_TRANSFER )); then
  echo "ERROR: the descriptor pool exceeds the 64 MiB registration limit" >&2
  exit 2
fi
KVAF_H2D_PAYLOAD_CAPACITY_BYTES=$((
  KVAF_H2D_DESCRIPTOR_CAPACITY * KVAF_H2D_BYTES_PER_TRANSFER
))
# Mirror smoke RegistrationBytes(): reserve the payload canary, round one
# half to the provider's 4 KiB page contract, then register two halves.
KVAF_H2D_PROVIDER_SEGMENT_PAYLOAD_BYTES=$((
  KVAF_H2D_PAYLOAD_CAPACITY_BYTES + 64
))
kvaf_h2d_provider_half_bytes=${KVAF_H2D_PROVIDER_SEGMENT_PAYLOAD_BYTES}
if (( kvaf_h2d_provider_half_bytes < 4096 )); then
  kvaf_h2d_provider_half_bytes=4096
fi
kvaf_h2d_provider_half_remainder=$((
  kvaf_h2d_provider_half_bytes % 4096
))
if (( kvaf_h2d_provider_half_remainder != 0 )); then
  kvaf_h2d_provider_half_bytes=$((
    kvaf_h2d_provider_half_bytes +
    4096 - kvaf_h2d_provider_half_remainder
  ))
fi
KVAF_H2D_PROVIDER_LOCAL_ALLOCATION_BYTES=$((
  kvaf_h2d_provider_half_bytes * 2
))
KVAF_H2D_BATCH_BYTES=$((
  KVAF_H2D_TRANSFER_COUNT * KVAF_H2D_BYTES_PER_TRANSFER
))
if (( KVAF_H2D_TRANSFER_COUNT != 0 &&
      KVAF_H2D_BATCH_ITERATIONS >
        4294967295 / KVAF_H2D_TRANSFER_COUNT )); then
  echo "ERROR: total transfer count exceeds the 32-bit epoch queue-counter limit" >&2
  exit 2
fi
KVAF_H2D_TOTAL_BYTES=$((
  KVAF_H2D_BATCH_BYTES * KVAF_H2D_BATCH_ITERATIONS
))
if (( KVAF_NPU_SMI_TIMEOUT_SECONDS == 0 )); then
  echo "ERROR: KVAF_NPU_SMI_TIMEOUT_SECONDS must be in [1, 120]" >&2
  exit 2
fi
if (( KVAF_HCCN_TIMEOUT_SECONDS == 0 )); then
  echo "ERROR: KVAF_HCCN_TIMEOUT_SECONDS must be in [1, 120]" >&2
  exit 2
fi
if [[ ! "${KVAF_A5_RUN_TAG}" =~ ^[A-Za-z0-9._-]{1,96}$ ]]; then
  echo "ERROR: KVAF_A5_RUN_TAG must use 1-96 safe filename characters" >&2
  exit 2
fi
if [[ "${KVAF_DIRECT_PROVIDER_SONAME}" == */* ||
      ${#KVAF_DIRECT_PROVIDER_SONAME} -ge 32 ]]; then
  echo "ERROR: provider soname must be a basename shorter than 32 bytes" >&2
  exit 2
fi
if [[ "${KVAF_DIRECT_PROVIDER_SONAME}" != \
      "libkvaf_urma_provider_device.so" ]]; then
  echo "ERROR: provider soname must match the AICPU package entry" >&2
  exit 2
fi

# Stale-session recovery deliberately trusts only nonce-bound evidence under
# the persistent state root. Keeping it outside the Git worktree prevents a
# clean/reclone from deleting the only proof that an active run became safe.
command -v realpath >/dev/null 2>&1 || {
  echo "ERROR: realpath is required" >&2
  exit 2
}

# This guard intentionally precedes EID selector and NPU access. A prior
# unsafe run must be reported without touching ACL, npu-smi, or the
# host-global /proc/asdrv_ub/dev_id selector again.
acquire_persistent_session_guards

# Recovery must not depend on the current checkout or provider build: its
# nonce-bound classifier and final markers were durably copied into the result
# bundle.  After recovery, however, reject a v2/partial/mismatched provider
# before EID discovery, endpoint discovery, /proc, npu-smi, ACL, DDR/HBM or
# URMA.  This also re-runs all three ABI probes and binds the build to the
# current kernel and its UDMA/UMMU UAPI files.
if ! kvaf_verify_direct_provider_build_v3; then
  echo "ERROR: rebuild the paired official UMDK/libummu provider before running hardware" >&2
  exit 2
fi
host_provider_sha=${KVAF_VERIFIED_HOST_PROVIDER_SHA}
device_provider_sha=${KVAF_VERIFIED_DEVICE_PROVIDER_SHA}
source_manifest_sha=${KVAF_VERIFIED_SOURCE_MANIFEST_SHA}
umdk_source_manifest_sha=${KVAF_VERIFIED_UMDK_SOURCE_MANIFEST_SHA}
libummu_source_manifest_sha=${KVAF_VERIFIED_LIBUMMU_SOURCE_MANIFEST_SHA}
build_manifest_sha=${KVAF_VERIFIED_BUILD_MANIFEST_SHA}
kvaf_teardown_assurance=verified
udma_teardown_contract=checked-retryable-v1

# Recovery above must remain available even while new hardware runs are
# blocked.  For a new run, however, this ownership proof must precede every
# EID selector, topology query, npu-smi call, ACL context, AICPU launch and
# URMA/UMMU allocation.  The public KFC ABI does not currently provide such a
# proof, so the source-owned guard fails closed without an operator override.
kvaf_require_aicpu_affinity_contract \
  "${kvaf_affinity_run_mode}" || exit $?

# The deploy lock acquired during recovery closes the build/package race.
# Validate the complete direct-smoke and installed AICPU artifact chain before
# any EID selector, hccn/sysfs query, npu-smi call, ACL context or URMA access.
if ! kvaf_verify_direct_runtime_artifacts_v3; then
  echo "ERROR: rebuild and repackage the direct URMA runtime artifacts" >&2
  exit 2
fi
direct_binary_sha=${KVAF_VERIFIED_DIRECT_BINARY_SHA}
direct_build_manifest_sha=${KVAF_VERIFIED_DIRECT_BUILD_MANIFEST_SHA}
direct_source_manifest=${KVAF_VERIFIED_DIRECT_SOURCE_MANIFEST}
direct_source_manifest_sha=${KVAF_VERIFIED_DIRECT_SOURCE_MANIFEST_SHA}
direct_runtime_header=${KVAF_VERIFIED_DIRECT_RUNTIME_HEADER}

kvaf_result_candidate=$(realpath -m "${KVAF_A5_RESULT_DIR}")
kvaf_result_parent=${kvaf_result_candidate%/*}
kvaf_result_name=${kvaf_result_candidate##*/}
if [[ "${kvaf_result_parent}" != "${kvaf_results_root}" ||
      ! "${kvaf_result_name}" =~ \
        ^a5_direct_urma_h2d_[A-Za-z0-9._-]{1,96}$ ]]; then
  echo "ERROR: KVAF_A5_RESULT_DIR must be one safe direct child of ${kvaf_results_root}" >&2
  exit 2
fi
KVAF_A5_RESULT_DIR=${kvaf_result_candidate}

if { [[ -n "${KVAF_HOST_EID}" ]] && [[ -z "${KVAF_DEVICE_EID}" ]]; } ||
   { [[ -z "${KVAF_HOST_EID}" ]] && [[ -n "${KVAF_DEVICE_EID}" ]]; }; then
  echo "ERROR: set both EIDs, or leave both empty for driver discovery" >&2
  exit 2
fi

if [[ -z "${KVAF_PHYSICAL_DEVICE_ID}" ]]; then
  echo "ERROR: set KVAF_PHYSICAL_DEVICE_ID for the resource-free NPU idle preflight" >&2
  echo "Automatic ACL topology discovery would select/create a device context before" >&2
  echo "npu-smi proves that the card is idle, so the direct runner fails closed." >&2
  echo "No Host/Device DDR, HBM, URMA, SQ, or CQ resource was created." >&2
  exit 2
fi
require_uint_max KVAF_PHYSICAL_DEVICE_ID \
  "${KVAF_PHYSICAL_DEVICE_ID}" 4294967295
if [[ "${KVAF_LOGICAL_DEVICE_ID}" != \
      "${KVAF_PHYSICAL_DEVICE_ID}" ]]; then
  echo "ERROR: direct runner cannot prove a remapped logical/physical NPU pair without selecting the device" >&2
  echo "This all-device container requires KVAF_LOGICAL_DEVICE_ID == KVAF_PHYSICAL_DEVICE_ID." >&2
  echo "No Host/Device DDR, HBM, URMA, SQ, or CQ resource was created." >&2
  exit 2
fi

if [[ -z "${KVAF_HOST_EID}" ]]; then
  resolver_args=(
    --physical-device-id "${KVAF_PHYSICAL_DEVICE_ID}"
    --proc-root "${KVAF_EID_PROC_ROOT}"
    --lock-file "${KVAF_EID_LOCK_FILE}"
    --legacy-lock-file "${KVAF_EID_LEGACY_LOCK_FILE}"
    --lock-timeout-seconds "${KVAF_EID_LOCK_TIMEOUT_SECONDS}"
    --shell
  )
  if [[ -n "${KVAF_PAIR_CHANNEL}" ]]; then
    require_uint_max KVAF_PAIR_CHANNEL \
      "${KVAF_PAIR_CHANNEL}" 4294967295
    resolver_args+=(--pair-channel "${KVAF_PAIR_CHANNEL}")
  fi
  eid_output=$(
    python3 "${SCRIPT_DIR}/resolve_a5_urma_eids.py" \
      "${resolver_args[@]}"
  )
  while IFS='=' read -r key value; do
    case "${key}" in
      KVAF_HOST_EID) KVAF_HOST_EID=${value} ;;
      KVAF_DEVICE_EID) KVAF_DEVICE_EID=${value} ;;
      KVAF_PAIR_CHANNEL) KVAF_PAIR_CHANNEL=${value} ;;
      *)
        echo "ERROR: unknown EID resolver output: ${key}" >&2
        exit 2
        ;;
    esac
  done <<<"${eid_output}"
  require_uint_max KVAF_PAIR_CHANNEL \
    "${KVAF_PAIR_CHANNEL}" 4294967295
  echo "KVAF_A5_EID_RESOLVED source=driver_pair_info channel=${KVAF_PAIR_CHANNEL}"
fi

normalize_eid()
{
  local value=${1//:/}
  value=${value//-/}
  value=$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')
  if ! [[ "${value}" =~ ^[0-9a-f]{32}$ ]] ||
     [[ "${value}" == 00000000000000000000000000000000 ]]; then
    return 1
  fi
  printf '%s\n' "${value}"
}

if ! host_eid=$(normalize_eid "${KVAF_HOST_EID}"); then
  echo "ERROR: KVAF_HOST_EID is not a non-zero 16-byte EID" >&2
  exit 2
fi
if ! device_eid=$(normalize_eid "${KVAF_DEVICE_EID}"); then
  echo "ERROR: KVAF_DEVICE_EID is not a non-zero 16-byte EID" >&2
  exit 2
fi
if [[ "${host_eid}" == "${device_eid}" ]]; then
  echo "ERROR: Host and Device EIDs must be different" >&2
  exit 2
fi

if [[ "${KVAF_DIRECT_CONTROL_SERVER_ADDRESS_SOURCE}" == \
      auto-ipourma-eid ]]; then
  if ! control_address_output=$(
    resolve_ipourma_global_ipv6_for_eid "${host_eid}"
  ); then
    echo "ERROR: Host EID has no unique exact global IPv6 assignment on ipourma*" >&2
    echo "HOST_EID=${host_eid}" >&2
    echo "Set KVAF_DIRECT_CONTROL_SERVER_ADDRESS to an explicit proven address if needed." >&2
    echo "No Host/Device DDR, HBM, URMA, SQ, or CQ resource was created." >&2
    exit 2
  fi
  mapfile -t control_address_fields <<<"${control_address_output}"
  if (( ${#control_address_fields[@]} != 2 )) ||
     [[ -z "${control_address_fields[0]}" ||
        ! "${control_address_fields[1]}" =~ ^ipourma[A-Za-z0-9_.:-]*$ ]]; then
    echo "ERROR: malformed ipourma control-address resolution" >&2
    exit 2
  fi
  KVAF_DIRECT_CONTROL_SERVER_ADDRESS=${control_address_fields[0]}
  KVAF_DIRECT_CONTROL_SERVER_INTERFACE=${control_address_fields[1]}
  if ! KVAF_DIRECT_CONTROL_SERVER_ADDRESS=$(
    normalize_numeric_ip "${KVAF_DIRECT_CONTROL_SERVER_ADDRESS}"
  ); then
    echo "ERROR: resolved ipourma control address is not a numeric IP" >&2
    exit 2
  fi
  if [[ "${KVAF_DIRECT_CONTROL_SERVER_ADDRESS}" != *:* ]]; then
    echo "ERROR: auto control address must resolve to Host-EID IPv6" >&2
    exit 2
  fi
  if [[ -n "${KVAF_DIRECT_CONTROL_BIND_ADDRESS}" &&
        "${KVAF_DIRECT_CONTROL_BIND_ADDRESS}" != *:* ]]; then
    echo "ERROR: control server and client bind addresses must use the same IP family" >&2
    exit 2
  fi
  echo "KVAF_A5_CONTROL_ADDRESS_RESOLVED source=${KVAF_DIRECT_CONTROL_SERVER_ADDRESS_SOURCE} interface=${KVAF_DIRECT_CONTROL_SERVER_INTERFACE} address=${KVAF_DIRECT_CONTROL_SERVER_ADDRESS}"
fi

endpoint_resolver_args=(
  --physical-device-id "${KVAF_PHYSICAL_DEVICE_ID}"
  --host-eid "${host_eid}"
  --device-eid "${device_eid}"
  --ubcore-root "${KVAF_UBCORE_SYSFS_ROOT}"
  --hccn-timeout-seconds "${KVAF_HCCN_TIMEOUT_SECONDS}"
  --shell
)
if [[ -n "${KVAF_HCCN_TOOL}" ]]; then
  endpoint_resolver_args+=(--hccn-tool "${KVAF_HCCN_TOOL}")
fi
if ! endpoint_output=$(
  python3 "${SCRIPT_DIR}/resolve_a5_urma_endpoints.py" \
    "${endpoint_resolver_args[@]}"
); then
  echo "ERROR: automatic A5 UDMA endpoint discovery failed" >&2
  echo "No Host/Device DDR, HBM, URMA, SQ, or CQ resource was created." >&2
  exit 2
fi

resolved_host_urma_device=
resolved_device_urma_device=
resolved_host_eid_index=
resolved_device_eid_index=
seen_host_urma_device=0
seen_device_urma_device=0
seen_host_eid_index=0
seen_device_eid_index=0
while IFS='=' read -r key value; do
  case "${key}" in
    KVAF_HOST_URMA_DEVICE)
      (( seen_host_urma_device == 0 )) || {
        echo "ERROR: endpoint resolver returned duplicate KVAF_HOST_URMA_DEVICE" >&2
        exit 2
      }
      resolved_host_urma_device=${value}
      seen_host_urma_device=1
      ;;
    KVAF_DEVICE_URMA_DEVICE)
      (( seen_device_urma_device == 0 )) || {
        echo "ERROR: endpoint resolver returned duplicate KVAF_DEVICE_URMA_DEVICE" >&2
        exit 2
      }
      resolved_device_urma_device=${value}
      seen_device_urma_device=1
      ;;
    KVAF_HOST_EID_INDEX)
      (( seen_host_eid_index == 0 )) || {
        echo "ERROR: endpoint resolver returned duplicate KVAF_HOST_EID_INDEX" >&2
        exit 2
      }
      resolved_host_eid_index=${value}
      seen_host_eid_index=1
      ;;
    KVAF_DEVICE_EID_INDEX)
      (( seen_device_eid_index == 0 )) || {
        echo "ERROR: endpoint resolver returned duplicate KVAF_DEVICE_EID_INDEX" >&2
        exit 2
      }
      resolved_device_eid_index=${value}
      seen_device_eid_index=1
      ;;
    *)
      echo "ERROR: unknown UDMA endpoint resolver output: ${key}" >&2
      exit 2
      ;;
  esac
done <<<"${endpoint_output}"
if (( seen_host_urma_device == 0 ||
      seen_device_urma_device == 0 ||
      seen_host_eid_index == 0 ||
      seen_device_eid_index == 0 )); then
  echo "ERROR: UDMA endpoint resolver output is incomplete" >&2
  exit 2
fi

for endpoint_variable in \
  KVAF_HOST_URMA_DEVICE \
  KVAF_DEVICE_URMA_DEVICE \
  KVAF_HOST_EID_INDEX \
  KVAF_DEVICE_EID_INDEX
do
  case "${endpoint_variable}" in
    KVAF_HOST_URMA_DEVICE) resolved_value=${resolved_host_urma_device} ;;
    KVAF_DEVICE_URMA_DEVICE) resolved_value=${resolved_device_urma_device} ;;
    KVAF_HOST_EID_INDEX) resolved_value=${resolved_host_eid_index} ;;
    KVAF_DEVICE_EID_INDEX) resolved_value=${resolved_device_eid_index} ;;
  esac
  if [[ -n "${!endpoint_variable}" &&
        "${!endpoint_variable}" != "${resolved_value}" ]]; then
    echo "ERROR: ${endpoint_variable}=${!endpoint_variable} disagrees with the exact EID match ${resolved_value}" >&2
    echo "Unset the four KVAF_*URMA_DEVICE/KVAF_*EID_INDEX overrides and retry." >&2
    exit 2
  fi
  printf -v "${endpoint_variable}" '%s' "${resolved_value}"
done

require_uint_max KVAF_HOST_EID_INDEX \
  "${KVAF_HOST_EID_INDEX}" 4294967295
require_uint_max KVAF_DEVICE_EID_INDEX \
  "${KVAF_DEVICE_EID_INDEX}" 4294967295
if [[ "${KVAF_HOST_URMA_DEVICE}" == */* ||
      "${KVAF_DEVICE_URMA_DEVICE}" == */* ||
      ${#KVAF_HOST_URMA_DEVICE} -ge 32 ||
      ${#KVAF_DEVICE_URMA_DEVICE} -ge 32 ||
      ! "${KVAF_HOST_URMA_DEVICE}" =~ ^[A-Za-z0-9._:-]+$ ||
      ! "${KVAF_DEVICE_URMA_DEVICE}" =~ ^[A-Za-z0-9._:-]+$ ]]; then
  echo "ERROR: resolved UDMA names must be safe basenames shorter than 32 bytes" >&2
  exit 2
fi
echo "KVAF_A5_URMA_ENDPOINTS_RESOLVED source=host_sysfs+hccn_tool host=${KVAF_HOST_URMA_DEVICE}/eid${KVAF_HOST_EID_INDEX} device=${KVAF_DEVICE_URMA_DEVICE}/eid${KVAF_DEVICE_EID_INDEX}"

# The ABI-pinned accessor exports the normal userspace qbuf geometry and
# deliberately rejects reserved-SQ contexts.  The device-address and queue
# ownership semantics of reserved-SQ mode have not been validated on A5, so
# fail closed before any private queue object is consumed.
if [[ ! -r /sys/module/udma/parameters/sq_reserved ]]; then
  echo "ERROR: cannot prove UDMA reserved-SQ mode is disabled" >&2
  echo "Missing readable /sys/module/udma/parameters/sq_reserved." >&2
  exit 2
fi
kvaf_sq_reserved=$(
  tr '[:upper:]' '[:lower:]' \
    </sys/module/udma/parameters/sq_reserved |
    tr -d '[:space:]'
)
case "${kvaf_sq_reserved}" in
  0|n|no|false)
    ;;
  *)
    echo "ERROR: direct AIV SQ is unsupported while udma sq_reserved=${kvaf_sq_reserved:-unknown}" >&2
    echo "The current perftest sidecar does not export the actual reserved ring size." >&2
    exit 2
    ;;
esac

printf -v kvaf_control_port_hex '%04X' \
  "${KVAF_DIRECT_CONTROL_PORT}"
if awk -v expected_port="${kvaf_control_port_hex}" '
    $4 == "0A" && $2 ~ ":" expected_port "$" { found = 1 }
    END { exit found ? 0 : 1 }
  ' /proc/net/tcp /proc/net/tcp6 2>/dev/null; then
  echo "ERROR: control port ${KVAF_DIRECT_CONTROL_PORT} is already listening" >&2
  exit 2
fi

# Take the occupancy snapshot as late as possible, after all build/package
# checks but before creating the result bundle or publishing launch intent.
# Absence of a process row is not proof of idleness: the parser requires the
# explicit idle record emitted by the npu-smi process table.
if [[ -L "${KVAF_NPU_IDLE_CHECKER}" ||
      ! -f "${KVAF_NPU_IDLE_CHECKER}" ||
      ! -r "${KVAF_NPU_IDLE_CHECKER}" ]]; then
  echo "ERROR: missing regular NPU idle checker: ${KVAF_NPU_IDLE_CHECKER}" >&2
  exit 2
fi
kvaf_npu_smi_binary=$(command -v npu-smi 2>/dev/null || true)
kvaf_npu_smi_rc=127
kvaf_npu_smi_output=
kvaf_npu_idle_check_output=
kvaf_npu_idle_check_rc=2
kvaf_npu_idle_status=unproven
kvaf_npu_idle_override=0
if [[ -n "${kvaf_npu_smi_binary}" &&
      -x "${kvaf_npu_smi_binary}" ]]; then
  set +e
  kvaf_npu_smi_output=$(
    LC_ALL=C timeout --signal=TERM --kill-after=2s \
      "${KVAF_NPU_SMI_TIMEOUT_SECONDS}s" \
      "${kvaf_npu_smi_binary}" info 2>&1
  )
  kvaf_npu_smi_rc=$?
  set -e
  if (( kvaf_npu_smi_rc == 0 )); then
    set +e
    kvaf_npu_idle_check_output=$(
      python3 "${KVAF_NPU_IDLE_CHECKER}" \
        --physical-device-id "${KVAF_PHYSICAL_DEVICE_ID}" \
        <<<"${kvaf_npu_smi_output}" 2>&1
    )
    kvaf_npu_idle_check_rc=$?
    set -e
    case "${kvaf_npu_idle_check_rc}" in
      0)
        if grep -Fxq KVAF_NPU_IDLE_STATUS=idle \
            <<<"${kvaf_npu_idle_check_output}"; then
          kvaf_npu_idle_status=idle
        fi
        ;;
      3)
        if grep -Fxq KVAF_NPU_IDLE_STATUS=busy \
            <<<"${kvaf_npu_idle_check_output}"; then
          kvaf_npu_idle_status=busy
        fi
        ;;
    esac
  else
    kvaf_npu_idle_check_output="npu-smi info failed with rc=${kvaf_npu_smi_rc}"
    if [[ -n "${kvaf_npu_smi_output}" ]]; then
      kvaf_npu_idle_check_output+=$'\n'"${kvaf_npu_smi_output}"
    fi
  fi
else
  kvaf_npu_idle_check_output="npu-smi is unavailable or not executable"
fi

if [[ "${kvaf_npu_idle_status}" != idle ]]; then
  if [[ "${KVAF_DIRECT_ALLOW_BUSY_NPU}" != 1 ]]; then
    echo "ERROR: target physical NPU ${KVAF_PHYSICAL_DEVICE_ID} is ${kvaf_npu_idle_status}" >&2
    printf '%s\n' "${kvaf_npu_idle_check_output}" >&2
    echo "No Host/Device DDR, HBM, URMA, SQ, or CQ resource was created." >&2
    echo "To accept the collision/reset risk explicitly, set KVAF_DIRECT_ALLOW_BUSY_NPU=1." >&2
    exit 2
  fi
  kvaf_npu_idle_override=1
  echo "WARNING: proceeding with target physical NPU ${KVAF_PHYSICAL_DEVICE_ID} status=${kvaf_npu_idle_status}" >&2
  printf '%s\n' "${kvaf_npu_idle_check_output}" >&2
  echo "WARNING: KVAF_DIRECT_ALLOW_BUSY_NPU=1 accepts interference and reset risk" >&2
else
  echo "KVAF_A5_NPU_IDLE_PROVEN physical_device_id=${KVAF_PHYSICAL_DEVICE_ID}"
fi

if [[ -e "${KVAF_A5_RESULT_DIR}" ||
      -L "${KVAF_A5_RESULT_DIR}" ]]; then
  echo "ERROR: refusing an existing result directory: ${KVAF_A5_RESULT_DIR}" >&2
  exit 2
fi
mkdir -p "$(dirname "${KVAF_A5_RESULT_DIR}")"
if ! mkdir "${KVAF_A5_RESULT_DIR}" ||
   [[ -L "${KVAF_A5_RESULT_DIR}" ]] ||
   [[ ! -d "${KVAF_A5_RESULT_DIR}" ]]; then
  echo "ERROR: could not atomically create a fresh result directory" >&2
  exit 2
fi
if ! mkdir "${KVAF_A5_RESULT_DIR}/exchange" ||
   [[ -L "${KVAF_A5_RESULT_DIR}/exchange" ]]; then
  echo "ERROR: could not create a fresh exchange directory" >&2
  exit 2
fi
kvaf_npu_smi_snapshot="${KVAF_A5_RESULT_DIR}/npu-smi-info.txt"
kvaf_npu_idle_check_record="${KVAF_A5_RESULT_DIR}/npu-idle-check.txt"
kvaf_npu_idle_checker_record="${KVAF_A5_RESULT_DIR}/check_a5_npu_idle.py"
printf '%s\n' "${kvaf_npu_smi_output}" >"${kvaf_npu_smi_snapshot}"
printf '%s\n' "${kvaf_npu_idle_check_output}" \
  >"${kvaf_npu_idle_check_record}"
cp "${KVAF_NPU_IDLE_CHECKER}" "${kvaf_npu_idle_checker_record}"
chmod 0400 \
  "${kvaf_npu_smi_snapshot}" \
  "${kvaf_npu_idle_check_record}" \
  "${kvaf_npu_idle_checker_record}"
if [[ ! -r /proc/sys/kernel/random/uuid ]]; then
  echo "ERROR: cannot create a per-run control-plane nonce" >&2
  exit 2
fi
kvaf_run_id=$(tr -d '-' </proc/sys/kernel/random/uuid)
if ! [[ "${kvaf_run_id}" =~ ^[0-9a-fA-F]{32}$ ]]; then
  echo "ERROR: kernel returned an invalid run nonce" >&2
  exit 2
fi
host_log="${KVAF_A5_RESULT_DIR}/host.log"
device_log="${KVAF_A5_RESULT_DIR}/device.log"
exchange_dir="${KVAF_A5_RESULT_DIR}/exchange"
host_final="${exchange_dir}/host.final"
device_final="${exchange_dir}/device.final"
outcome_classifier_source="${SCRIPT_DIR}/classify_a5_direct_urma_outcome.py"
outcome_classifier="${KVAF_A5_RESULT_DIR}/classify_a5_direct_urma_outcome.py"
manifest="${KVAF_A5_RESULT_DIR}/manifest.txt"
if [[ -L "${outcome_classifier_source}" ||
      ! -f "${outcome_classifier_source}" ||
      ! -r "${outcome_classifier_source}" ]]; then
  echo "ERROR: missing regular direct URMA outcome classifier: ${outcome_classifier_source}" >&2
  exit 2
fi
cp "${outcome_classifier_source}" "${outcome_classifier}"
chmod 0500 "${outcome_classifier}"
outcome_classifier_sha=$(sha256sum "${outcome_classifier}" |
  awk '{print $1}')
if [[ "${outcome_classifier_sha}" != \
      "$(sha256sum "${outcome_classifier_source}" | awk '{print $1}')" ]]; then
  echo "ERROR: result-bundled outcome classifier differs from its source" >&2
  exit 2
fi

kvaf_commit=$(
  git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null ||
    printf 'unknown'
)
archive_sha=$(
  sha256sum "${KVAF_DIRECT_AICPU_ARCHIVE}" |
    awk '{print $1}'
)
{
  printf 'kvaf_commit=%s\n' "${kvaf_commit}"
  printf 'aicpu_affinity_contract=%s\n' \
    "${KVAF_AICPU_AFFINITY_CONTRACT}"
  printf 'aicpu_affinity_run_mode=%s\n' "${kvaf_affinity_run_mode}"
  printf 'run_id=%s\n' "${kvaf_run_id}"
  printf 'ascend_home_path=%s\n' "${ASCEND_HOME_PATH}"
  printf 'logical_device_id=%s\n' "${KVAF_LOGICAL_DEVICE_ID}"
  printf 'physical_device_id=%s\n' "${KVAF_PHYSICAL_DEVICE_ID}"
  printf 'npu_idle_preflight_status=%s\n' "${kvaf_npu_idle_status}"
  printf 'npu_idle_preflight_override=%s\n' "${kvaf_npu_idle_override}"
  printf 'npu_smi_exit_code=%s\n' "${kvaf_npu_smi_rc}"
  printf 'npu_smi_timeout_seconds=%s\n' "${KVAF_NPU_SMI_TIMEOUT_SECONDS}"
  printf 'npu_smi_snapshot_sha256=%s\n' "$(
    sha256sum "${kvaf_npu_smi_snapshot}" | awk '{print $1}'
  )"
  printf 'npu_idle_check_sha256=%s\n' "$(
    sha256sum "${kvaf_npu_idle_check_record}" | awk '{print $1}'
  )"
  printf 'npu_idle_checker_sha256=%s\n' "$(
    sha256sum "${kvaf_npu_idle_checker_record}" | awk '{print $1}'
  )"
  printf 'pair_channel=%s\n' "${KVAF_PAIR_CHANNEL}"
  printf 'host_eid=%s\n' "${host_eid}"
  printf 'device_eid=%s\n' "${device_eid}"
  printf 'host_urma_device=%s\n' "${KVAF_HOST_URMA_DEVICE}"
  printf 'host_eid_index=%s\n' "${KVAF_HOST_EID_INDEX}"
  printf 'device_urma_device=%s\n' "${KVAF_DEVICE_URMA_DEVICE}"
  printf 'device_eid_index=%s\n' "${KVAF_DEVICE_EID_INDEX}"
  printf 'control_port=%s\n' "${KVAF_DIRECT_CONTROL_PORT}"
  printf 'control_server_address_source=%s\n' \
    "${KVAF_DIRECT_CONTROL_SERVER_ADDRESS_SOURCE}"
  printf 'control_server_interface=%s\n' \
    "${KVAF_DIRECT_CONTROL_SERVER_INTERFACE}"
  printf 'control_server_address=%s\n' \
    "${KVAF_DIRECT_CONTROL_SERVER_ADDRESS}"
  printf 'control_bind_address=%s\n' \
    "${KVAF_DIRECT_CONTROL_BIND_ADDRESS:-none}"
  printf 'h2d_workload_mode=%s\n' "${kvaf_h2d_workload_mode}"
  printf 'h2d_transfer_count=%s\n' "${KVAF_H2D_TRANSFER_COUNT}"
  printf 'h2d_bytes_per_transfer=%s\n' \
    "${KVAF_H2D_BYTES_PER_TRANSFER}"
  printf 'h2d_batch_iterations=%s\n' \
    "${KVAF_H2D_BATCH_ITERATIONS}"
  printf 'h2d_descriptor_capacity=%s\n' \
    "${KVAF_H2D_DESCRIPTOR_CAPACITY}"
  printf 'h2d_payload_capacity_bytes=%s\n' \
    "${KVAF_H2D_PAYLOAD_CAPACITY_BYTES}"
  printf 'h2d_provider_segment_payload_bytes=%s\n' \
    "${KVAF_H2D_PROVIDER_SEGMENT_PAYLOAD_BYTES}"
  # Both endpoints pass this exact local_allocation_bytes value to their
  # provider. It is not a claim about additional provider-internal memory.
  printf 'h2d_provider_local_allocation_bytes_per_endpoint=%s\n' \
    "${KVAF_H2D_PROVIDER_LOCAL_ALLOCATION_BYTES}"
  printf 'h2d_total_bytes_per_batch=%s\n' "${KVAF_H2D_BATCH_BYTES}"
  printf 'h2d_total_bytes=%s\n' "${KVAF_H2D_TOTAL_BYTES}"
  printf 'h2d_control_timeout_seconds=%s\n' \
    "${KVAF_H2D_TIMEOUT_SECONDS}"
  printf 'host_control_timeout_seconds=%s\n' \
    "${KVAF_HOST_CONTROL_TIMEOUT_SECONDS}"
  printf 'host_listener_ready_timeout_seconds=%s\n' \
    "${KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS}"
  printf 'h2d_run_timeout_seconds=%s\n' \
    "${KVAF_H2D_RUN_TIMEOUT_SECONDS}"
  printf 'binary_sha256=%s\n' "${direct_binary_sha}"
  printf 'direct_build_manifest_sha256=%s\n' \
    "${direct_build_manifest_sha}"
  printf 'direct_source_manifest_sha256=%s\n' \
    "${direct_source_manifest_sha}"
  printf 'runtime_header_sha256=%s\n' "$(
    sha256sum "${direct_runtime_header}" | awk '{print $1}'
  )"
  printf 'ascend_hal_library_path=%s\n' "${KVAF_ASCEND_HAL_LIBRARY}"
  printf 'ascend_hal_library_sha256=%s\n' "$(
    sha256sum "${KVAF_ASCEND_HAL_LIBRARY}" | awk '{print $1}'
  )"
  printf 'host_provider_sha256=%s\n' "${host_provider_sha}"
  printf 'device_provider_sha256=%s\n' "${device_provider_sha}"
  printf 'provider_build_manifest_sha256=%s\n' \
    "${build_manifest_sha}"
  printf 'provider_source_manifest_sha256=%s\n' \
    "${source_manifest_sha}"
  printf 'provider_umdk_source_manifest_sha256=%s\n' \
    "${umdk_source_manifest_sha}"
  printf 'provider_libummu_source_manifest_sha256=%s\n' \
    "${libummu_source_manifest_sha}"
  printf 'aicpu_archive_sha256=%s\n' "${archive_sha}"
  printf 'udma_sq_reserved=%s\n' "${kvaf_sq_reserved}"
  printf 'udma_teardown_contract=%s\n' "${udma_teardown_contract}"
  printf 'teardown_assurance=%s\n' "${kvaf_teardown_assurance}"
  printf 'outcome_classifier_sha256=%s\n' \
    "${outcome_classifier_sha}"
} >"${manifest}"
cp "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" \
  "${KVAF_A5_RESULT_DIR}/provider-build.manifest"
cp "${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST}" \
  "${KVAF_A5_RESULT_DIR}/provider-source.sha256"
cp "${KVAF_VERIFIED_UMDK_SOURCE_MANIFEST}" \
  "${KVAF_A5_RESULT_DIR}/provider-umdk-source.sha256"
cp "${KVAF_VERIFIED_LIBUMMU_SOURCE_MANIFEST}" \
  "${KVAF_A5_RESULT_DIR}/provider-libummu-source.sha256"
cp "${KVAF_A5_DIRECT_BUILD_MANIFEST}" \
  "${KVAF_A5_RESULT_DIR}/direct-smoke-build.manifest"
cp "${direct_source_manifest}" \
  "${KVAF_A5_RESULT_DIR}/direct-smoke-source.sha256"

# The active launch-intent state must never outlive the evidence needed to
# classify it. Flush the immutable recovery bundle and directory entries
# before publishing the first active record.
python3 "${KVAF_DURABLE_STATE_HELPER}" sync \
  "${outcome_classifier}" \
  "${manifest}" \
  "${kvaf_npu_smi_snapshot}" \
  "${kvaf_npu_idle_check_record}" \
  "${kvaf_npu_idle_checker_record}" \
  "${KVAF_A5_RESULT_DIR}/provider-build.manifest" \
  "${KVAF_A5_RESULT_DIR}/provider-source.sha256" \
  "${KVAF_A5_RESULT_DIR}/provider-umdk-source.sha256" \
  "${KVAF_A5_RESULT_DIR}/provider-libummu-source.sha256" \
  "${KVAF_A5_RESULT_DIR}/direct-smoke-build.manifest" \
  "${KVAF_A5_RESULT_DIR}/direct-smoke-source.sha256" \
  "${exchange_dir}" \
  "${KVAF_A5_RESULT_DIR}" \
  "${KVAF_DIRECT_RESULT_ROOT}"

common_args=(
  --host-eid "${host_eid}"
  --device-eid "${device_eid}"
  --host-urma-device "${KVAF_HOST_URMA_DEVICE}"
  --device-urma-device "${KVAF_DEVICE_URMA_DEVICE}"
  --host-eid-index "${KVAF_HOST_EID_INDEX}"
  --device-eid-index "${KVAF_DEVICE_EID_INDEX}"
  --provider-library "${KVAF_DIRECT_HOST_PROVIDER_LIBRARY}"
  --provider-soname "${KVAF_DIRECT_PROVIDER_SONAME}"
  --exchange-dir "${exchange_dir}"
  --run-id "${kvaf_run_id}"
  --logical-device-id "${KVAF_LOGICAL_DEVICE_ID}"
  --control-port "${KVAF_DIRECT_CONTROL_PORT}"
  --control-server-address "${KVAF_DIRECT_CONTROL_SERVER_ADDRESS}"
  --transfer-count "${KVAF_H2D_TRANSFER_COUNT}"
  --bytes-per-transfer "${KVAF_H2D_BYTES_PER_TRANSFER}"
  --batch-iterations "${KVAF_H2D_BATCH_ITERATIONS}"
  --descriptor-capacity "${KVAF_H2D_DESCRIPTOR_CAPACITY}"
)
if [[ -n "${KVAF_DIRECT_CONTROL_BIND_ADDRESS}" ]]; then
  common_args+=(
    --control-bind-address "${KVAF_DIRECT_CONTROL_BIND_ADDRESS}"
  )
fi
host_args=(
  "${common_args[@]}"
  --timeout-seconds "${KVAF_HOST_CONTROL_TIMEOUT_SECONDS}"
)
device_args=(
  "${common_args[@]}"
  --timeout-seconds "${KVAF_H2D_TIMEOUT_SECONDS}"
)

host_pid=
device_pid=
run_completed=0
active_session_written=0
active_phase=before_host_launch
worker_expected_exe=$(realpath -e "${KVAF_A5_DIRECT_BINARY}")

process_start_ticks()
{
  local pid=${1:-0}
  local stat_line stat_tail ticks
  local -a stat_fields
  if [[ "${pid}" =~ ^[1-9][0-9]*$ && -r "/proc/${pid}/stat" ]]; then
    stat_line=$(<"/proc/${pid}/stat")
    # comm is parenthesized and may itself contain spaces. Strip through the
    # final ") " before indexing field 22 (field 20 in the remaining tail).
    stat_tail=${stat_line##*) }
    read -r -a stat_fields <<<"${stat_tail}"
    ticks=${stat_fields[19]:-}
    if [[ "${ticks}" =~ ^[1-9][0-9]*$ ]]; then
      printf '%s\n' "${ticks}"
      return 0
    fi
  fi
  printf '0\n'
}

process_executable()
{
  local pid=${1:-0}
  local executable
  if [[ "${pid}" =~ ^[1-9][0-9]*$ ]]; then
    executable=$(readlink -e "/proc/${pid}/exe" 2>/dev/null || true)
    if [[ -n "${executable}" ]]; then
      printf '%s\n' "${executable}"
      return 0
    fi
  fi
  printf 'none\n'
}

process_listens_on_control_port()
{
  local pid=$1
  local expected_start_ticks=$2
  local expected_executable=$3
  local port=$4
  local server_address=$5
  local current_start_ticks current_executable descriptor target inode
  local port_hex expected_local
  local -A process_socket_inodes=()
  local -a socket_tables=()

  current_start_ticks=$(process_start_ticks "${pid}")
  current_executable=$(process_executable "${pid}")
  [[ "${current_start_ticks}" == "${expected_start_ticks}" ]] || return 2
  # nohup may still be between fork and exec on the first observation.  The
  # stable start tick prevents PID reuse; accept readiness only after exec
  # resolves to the manifest-pinned worker binary.
  [[ "${current_executable}" == "${expected_executable}" ]] || return 1

  for descriptor in "/proc/${pid}/fd/"*; do
    target=$(readlink "${descriptor}" 2>/dev/null || true)
    if [[ "${target}" =~ ^socket:\[([1-9][0-9]*)\]$ ]]; then
      process_socket_inodes["${BASH_REMATCH[1]}"]=1
    fi
  done
  printf -v port_hex '%04X' "${port}"
  if [[ "${server_address}" == *:* ]]; then
    expected_local="00000000000000000000000000000000:${port_hex}"
    socket_tables=("/proc/${pid}/net/tcp6")
  else
    expected_local="00000000:${port_hex}"
    socket_tables=("/proc/${pid}/net/tcp")
  fi
  while read -r inode; do
    if [[ -n "${process_socket_inodes[${inode}]+x}" ]]; then
      return 0
    fi
  done < <(
    awk -v expected_local="${expected_local}" '
      $4 == "0A" && $2 == expected_local { print $10 }
    ' "${socket_tables[@]}" 2>/dev/null
  )
  return 1
}

wait_for_host_control_listener()
{
  local pid=$1
  local expected_start_ticks=$2
  local expected_executable=$3
  local port=$4
  local server_address=$5
  local timeout_seconds=$6
  local deadline=$((SECONDS + timeout_seconds))
  local listener_status

  while :; do
    listener_status=0
    process_listens_on_control_port \
      "${pid}" "${expected_start_ticks}" \
      "${expected_executable}" "${port}" \
      "${server_address}" || listener_status=$?
    if (( listener_status == 0 )); then
      return 0
    fi
    if (( listener_status == 2 )) ||
       valid_fatal_marker "${exchange_dir}/fatal.host" ||
       [[ -f "${host_final}" || -L "${host_final}" ]]; then
      return 3
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 1
    fi
    if (( SECONDS >= deadline )); then
      return 2
    fi
    sleep 0.05 || true
  done
}

write_active_session()
{
  local temporary="${KVAF_DIRECT_ACTIVE_SESSION_FILE}.tmp.$$"
  if ! {
    printf 'schema=kvaf-a5-direct-active-v2\n'
    printf 'run_id=%s\n' "${kvaf_run_id}"
    printf 'phase=%s\n' "${active_phase}"
    printf 'logical_device_id=%s\n' "${KVAF_LOGICAL_DEVICE_ID}"
    printf 'host_pid=%s\n' "${host_pid:-0}"
    printf 'device_pid=%s\n' "${device_pid:-0}"
    printf 'host_start_ticks=%s\n' \
      "$(process_start_ticks "${host_pid:-0}")"
    printf 'device_start_ticks=%s\n' \
      "$(process_start_ticks "${device_pid:-0}")"
    printf 'host_observed_exe=%s\n' \
      "$(process_executable "${host_pid:-0}")"
    printf 'device_observed_exe=%s\n' \
      "$(process_executable "${device_pid:-0}")"
    printf 'worker_expected_exe=%s\n' "${worker_expected_exe}"
    printf 'result_dir=%s\n' "${KVAF_A5_RESULT_DIR}"
    printf 'updated_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${temporary}"; then
    rm -f "${temporary}"
    echo "ERROR: could not write active-session temporary state" >&2
    return 1
  fi
  if ! chmod 0600 "${temporary}" ||
     ! durably_publish_state \
       "${temporary}" "${KVAF_DIRECT_ACTIVE_SESSION_FILE}"; then
    rm -f "${temporary}"
    echo "ERROR: could not publish active-session state" >&2
    return 1
  fi
  active_session_written=1
}

clear_active_session()
{
  local recorded_run_id
  if (( active_session_written == 0 )); then
    return 0
  fi
  if [[ -L "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ||
        ! -f "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ]]; then
    echo "ERROR: active-session proof path changed during this run" >&2
    return 1
  fi
  recorded_run_id=$(manifest_value run_id \
    "${KVAF_DIRECT_ACTIVE_SESSION_FILE}") || return 1
  if [[ "${recorded_run_id}" != "${kvaf_run_id}" ]]; then
    echo "ERROR: refusing to clear another run's active-session state" >&2
    return 1
  fi
  if ! durably_remove_state "${KVAF_DIRECT_ACTIVE_SESSION_FILE}"; then
    echo "ERROR: could not remove this run's active-session state" >&2
    return 1
  fi
  active_session_written=0
}

record_quarantine()
{
  local reason=$1
  local temporary="${KVAF_DIRECT_QUARANTINE_FILE}.tmp.$$"
  if ! {
    printf 'schema=kvaf-a5-direct-quarantine-v1\n'
    printf 'reason=%s\n' "${reason}"
    printf 'run_id=%s\n' "${kvaf_run_id}"
    printf 'logical_device_id=%s\n' "${KVAF_LOGICAL_DEVICE_ID}"
    printf 'host_pid=%s\n' "${host_pid:-0}"
    printf 'device_pid=%s\n' "${device_pid:-0}"
    printf 'host_start_ticks=%s\n' \
      "$(process_start_ticks "${host_pid:-0}")"
    printf 'device_start_ticks=%s\n' \
      "$(process_start_ticks "${device_pid:-0}")"
    printf 'host_observed_exe=%s\n' \
      "$(process_executable "${host_pid:-0}")"
    printf 'device_observed_exe=%s\n' \
      "$(process_executable "${device_pid:-0}")"
    printf 'worker_expected_exe=%s\n' "${worker_expected_exe}"
    printf 'result_dir=%s\n' "${KVAF_A5_RESULT_DIR}"
    printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${temporary}"; then
    rm -f "${temporary}"
    echo "ERROR: could not write quarantine temporary state" >&2
    return 1
  fi
  if ! chmod 0600 "${temporary}" ||
     ! durably_publish_state \
       "${temporary}" "${KVAF_DIRECT_QUARANTINE_FILE}"; then
    rm -f "${temporary}"
    echo "ERROR: could not publish quarantine state" >&2
    return 1
  fi
  if ! clear_active_session; then
    echo "WARNING: quarantine is durable, but stale active-session state remains" >&2
  fi
  echo "KVAF_A5_DIRECT_URMA_QUARANTINED reason=${reason}" >&2
  echo "QUARANTINE_FILE=${KVAF_DIRECT_QUARANTINE_FILE}" >&2
  echo "Do not kill the recorded processes until the selected NPU is reset or hardware-isolated." >&2
}

retain_unproven_active_session()
{
  local reason=$1
  local phase=$2
  active_phase=${phase}
  if ! write_active_session; then
    echo "KVAF_A5_DIRECT_PENDING_STATE_UPDATE_FAILED reason=${reason}" >&2
    echo "The previous atomic active state remains the recovery barrier." >&2
  fi
  echo "KVAF_A5_DIRECT_PENDING_UNPROVEN reason=${reason}" >&2
  echo "ACTIVE_SESSION=${KVAF_DIRECT_ACTIVE_SESSION_FILE}" >&2
  echo "Do not kill the recorded workers. A later run will auto-recover only after nonce-bound safe final markers exist." >&2
}

terminal_cleanup_proven()
{
  local outcome classifier_rc host_armed device_armed pair_required
  pair_required=0
  case "${active_phase}" in
    device_launch_intent|device_started|safe_terminal_process_stuck|pair_terminal_unproven_process_stuck)
      pair_required=1
      ;;
  esac
  if (( pair_required != 0 )); then
    if outcome=$(python3 "${outcome_classifier}" \
        --run-id "${kvaf_run_id}" \
        --host-final "${host_final}" \
        --device-final "${device_final}" 2>/dev/null); then
      classifier_rc=0
    else
      classifier_rc=$?
    fi
  else
    if outcome=$(python3 "${outcome_classifier}" \
        --run-id "${kvaf_run_id}" \
        --host-final "${host_final}" \
        --single-role host 2>/dev/null); then
      classifier_rc=0
    else
      classifier_rc=$?
    fi
  fi

  if { (( classifier_rc != 0 )) || [[ "${outcome}" != PASS ]]; } &&
     { (( classifier_rc != 3 )) || [[ "${outcome}" != FAIL_SAFE ]]; }; then
    return 1
  fi
  if [[ "${kvaf_teardown_assurance}" == verified ]]; then
    return 0
  fi
  host_armed=$(manifest_value resource_armed "${host_final}") || return 1
  device_armed=0
  if (( pair_required != 0 )); then
    device_armed=$(manifest_value resource_armed "${device_final}") ||
      return 1
  fi
  [[ "${host_armed}" == 0 && "${device_armed}" == 0 ]]
}

terminal_explicit_unsafe_proven()
{
  local outcome classifier_rc pair_required
  pair_required=0
  case "${active_phase}" in
    device_launch_intent|device_started|safe_terminal_process_stuck|pair_terminal_unproven_process_stuck)
      pair_required=1
      ;;
  esac
  if (( pair_required != 0 )); then
    if outcome=$(python3 "${outcome_classifier}" \
        --run-id "${kvaf_run_id}" \
        --host-final "${host_final}" \
        --device-final "${device_final}" 2>/dev/null); then
      classifier_rc=0
    else
      classifier_rc=$?
    fi
  else
    if outcome=$(python3 "${outcome_classifier}" \
        --run-id "${kvaf_run_id}" \
        --host-final "${host_final}" \
        --single-role host 2>/dev/null); then
      classifier_rc=0
    else
      classifier_rc=$?
    fi
  fi
  (( classifier_rc == 4 )) && [[ "${outcome}" == FAIL_UNSAFE ]]
}

cleanup_on_exit()
{
  local rc=$?
  local worker_live=0
  if (( run_completed == 0 )); then
    # An ordinary EXIT with no captured child PID can only occur before the
    # first fork: there is no fallible command between nohup and `$!`. SIGKILL
    # does not run this trap and is covered by the durable launch-intent state.
    # A prelaunch filesystem/configuration error is never a DMA quarantine.
    if [[ -z "${host_pid}" && -z "${device_pid}" ]]; then
      set +e
      if clear_active_session; then
        echo "KVAF_A5_DIRECT_FAIL_SAFE_PRELAUNCH rc=${rc}" >&2
      else
        echo "KVAF_A5_DIRECT_SAFE_STATE_MAINTENANCE_FAILED action=prelaunch_clear rc=${rc}" >&2
        echo "No worker was forked; do not reset the NPU." >&2
      fi
      set -e
      return
    fi
    if (( active_session_written != 0 )) ||
       [[ -n "${host_pid}" || -n "${device_pid}" ]]; then
      set +e
      if terminal_cleanup_proven; then
        if [[ -n "${host_pid}" ]] && kill -0 "${host_pid}" 2>/dev/null; then
          worker_live=1
        fi
        if [[ -n "${device_pid}" ]] && kill -0 "${device_pid}" 2>/dev/null; then
          worker_live=1
        fi
        if (( worker_live != 0 )); then
          case "${active_phase}" in
            device_launch_intent|device_started|safe_terminal_process_stuck|pair_terminal_unproven_process_stuck)
              active_phase=safe_terminal_process_stuck
              ;;
            *)
              active_phase=host_safe_terminal_process_stuck
              ;;
          esac
          if write_active_session; then
            echo "KVAF_A5_DIRECT_RUNNER_EXIT_WITH_SAFE_WORKER rc=${rc}" >&2
            echo "ACTIVE_SESSION=${KVAF_DIRECT_ACTIVE_SESSION_FILE}" >&2
          else
            echo "KVAF_A5_DIRECT_SAFE_STATE_MAINTENANCE_FAILED action=write rc=${rc}" >&2
            echo "Transport teardown is already proven safe; do not reset the NPU." >&2
          fi
        elif clear_active_session; then
          echo "KVAF_A5_DIRECT_RUNNER_EXIT_AFTER_SAFE_TERMINAL_PROOF rc=${rc}" >&2
        else
          echo "KVAF_A5_DIRECT_SAFE_STATE_MAINTENANCE_FAILED action=clear rc=${rc}" >&2
          echo "Transport teardown is already proven safe; retry stale-state recovery." >&2
        fi
      else
        if any_valid_fatal_marker || terminal_explicit_unsafe_proven; then
          record_quarantine "runner_exit_with_explicit_unsafe_rc_${rc}"
        elif [[ -n "${device_pid}" ]]; then
          retain_unproven_active_session \
            "runner_exit_rc_${rc}" \
            pair_terminal_unproven_process_stuck
        else
          retain_unproven_active_session \
            "runner_exit_rc_${rc}" \
            host_terminal_unproven_process_stuck
        fi
      fi
      set -e
    fi
  fi
}
trap cleanup_on_exit EXIT

runner_signal_requested=0
defer_runner_signal()
{
  runner_signal_requested=1
  echo "Runner interruption requested; waiting for bounded checked teardown." >&2
}
trap defer_runner_signal HUP INT TERM

echo "PHASE=direct_urma_host_ddr_to_aiv_hbm"
echo "RESULT_DIR=${KVAF_A5_RESULT_DIR}"
echo "LOGICAL_DEVICE_ID=${KVAF_LOGICAL_DEVICE_ID}"
echo "HOST_URMA=${KVAF_HOST_URMA_DEVICE}/eid${KVAF_HOST_EID_INDEX}"
echo "DEVICE_URMA=${KVAF_DEVICE_URMA_DEVICE}/eid${KVAF_DEVICE_EID_INDEX}"
echo "CONTROL_PORT=${KVAF_DIRECT_CONTROL_PORT}"
echo "CONTROL_SERVER_ADDRESS_SOURCE=${KVAF_DIRECT_CONTROL_SERVER_ADDRESS_SOURCE}"
echo "CONTROL_SERVER_INTERFACE=${KVAF_DIRECT_CONTROL_SERVER_INTERFACE}"
echo "CONTROL_SERVER_ADDRESS=${KVAF_DIRECT_CONTROL_SERVER_ADDRESS}"
echo "CONTROL_BIND_ADDRESS=${KVAF_DIRECT_CONTROL_BIND_ADDRESS:-none}"
echo "RUN_ID=${kvaf_run_id}"
echo "H2D_CONTROL_TIMEOUT_SECONDS=${KVAF_H2D_TIMEOUT_SECONDS}"
echo "HOST_CONTROL_TIMEOUT_SECONDS=${KVAF_HOST_CONTROL_TIMEOUT_SECONDS}"
echo "H2D_RUN_TIMEOUT_SECONDS=${KVAF_H2D_RUN_TIMEOUT_SECONDS}"
echo "HOST_LISTENER_READY_TIMEOUT_SECONDS=${KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS}"
echo "TEARDOWN_ASSURANCE=${kvaf_teardown_assurance}"
echo "MANIFEST=${manifest}"

if (( runner_signal_requested != 0 )); then
  exit 130
fi
write_active_session
active_phase=host_launch_intent
write_active_session
nohup "${KVAF_A5_DIRECT_BINARY}" --role host "${host_args[@]}" \
  9>&- >"${host_log}" 2>&1 &
host_pid=$!
active_phase=host_started
write_active_session
host_expected_start_ticks=$(process_start_ticks "${host_pid}")
if [[ "${host_expected_start_ticks}" == 0 ]]; then
  echo "ERROR: could not pin Host start ticks before listener readiness" >&2
  exit 5
fi

host_listener_wait_rc=0
wait_for_host_control_listener \
  "${host_pid}" "${host_expected_start_ticks}" \
  "${worker_expected_exe}" "${KVAF_DIRECT_CONTROL_PORT}" \
  "${KVAF_DIRECT_CONTROL_SERVER_ADDRESS}" \
  "${KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS}" ||
  host_listener_wait_rc=$?
if (( host_listener_wait_rc == 0 )); then
  echo "KVAF_A5_HOST_CONTROL_LISTENER_READY pid=${host_pid} port=${KVAF_DIRECT_CONTROL_PORT}"
elif (( host_listener_wait_rc == 2 )); then
  echo "ERROR: this run's Host PID did not own the control LISTEN socket before the readiness deadline" >&2
  retain_unproven_active_session \
    "host_listener_readiness_timeout" \
    host_terminal_unproven_process_stuck
  run_completed=1
  trap - EXIT
  exit 5
elif (( host_listener_wait_rc == 3 )); then
  echo "ERROR: Host identity changed or terminal evidence appeared before listener readiness" >&2
  exit 5
fi
if ! kill -0 "${host_pid}" 2>/dev/null; then
  set +e
  wait "${host_pid}"
  host_rc=$?
  early_outcome=$(
    python3 "${outcome_classifier}" \
      --run-id "${kvaf_run_id}" \
      --host-final "${host_final}" \
      --single-role host
  )
  early_classifier_rc=$?
  set -e
  if (( early_classifier_rc == 3 )) &&
     [[ "${early_outcome}" == FAIL_SAFE ]]; then
    if [[ "${kvaf_teardown_assurance}" != verified &&
          "$(manifest_value resource_armed "${host_final}")" == 1 ]]; then
      record_quarantine "unverified_udma_armed_host_failure"
      run_completed=1
      trap - EXIT
      echo "KVAF_A5_DIRECT_URMA_H2D_FAIL_UNSAFE unverified teardown" >&2
      exit 4
    fi
    clear_active_session
    host_pid=
    run_completed=1
    trap - EXIT
    sed -n '1,240p' "${host_log}" >&2
    echo "KVAF_A5_DIRECT_URMA_H2D_FAIL_SAFE host_rc=${host_rc}" >&2
    echo "RESULT_DIR=${KVAF_A5_RESULT_DIR}" >&2
    exit 3
  fi
  if { (( early_classifier_rc == 4 )) &&
       [[ "${early_outcome}" == FAIL_UNSAFE ]]; } ||
     valid_fatal_marker "${exchange_dir}/fatal.host"; then
    record_quarantine \
      "host_exited_with_explicit_fatal_${early_outcome:-FAIL_UNSAFE}"
    early_exit_rc=4
  else
    retain_unproven_active_session \
      "host_exited_before_device_start_${early_outcome:-UNKNOWN}" \
      host_terminal_unproven_process_stuck
    early_exit_rc=5
  fi
  run_completed=1
  trap - EXIT
  sed -n '1,240p' "${host_log}" >&2
  echo "ERROR: Host direct provider exited before Device startup" >&2
  exit "${early_exit_rc}"
fi

# A quarantined Host intentionally remains alive to retain DDR ownership.
# Therefore kill -0 alone is not proof that it is waiting for a client. Check
# its nonce-bound terminal evidence before exposing the Device/AICPU side.
set +e
pre_device_host_outcome=$(
  python3 "${outcome_classifier}" \
    --run-id "${kvaf_run_id}" \
    --host-final "${host_final}" \
    --single-role host 2>/dev/null
)
pre_device_host_classifier_rc=$?
set -e
if { (( pre_device_host_classifier_rc == 4 )) &&
     [[ "${pre_device_host_outcome}" == FAIL_UNSAFE ]]; } ||
   valid_fatal_marker "${exchange_dir}/fatal.host"; then
  record_quarantine "host_unsafe_before_device_launch"
  run_completed=1
  trap - EXIT
  exit 4
fi
if (( pre_device_host_classifier_rc == 3 )) &&
   [[ "${pre_device_host_outcome}" == FAIL_SAFE ]]; then
  if [[ "${kvaf_teardown_assurance}" != verified &&
        "$(manifest_value resource_armed "${host_final}")" == 1 ]]; then
    record_quarantine "unverified_udma_armed_host_terminal_before_device"
    pre_device_safe_exit_rc=4
  else
    active_phase=host_safe_terminal_process_stuck
    write_active_session
    pre_device_safe_exit_rc=3
  fi
  run_completed=1
  trap - EXIT
  echo "Device was not launched because Host already published a safe terminal marker." >&2
  exit "${pre_device_safe_exit_rc}"
fi
if [[ -e "${host_final}" || -L "${host_final}" ]]; then
  retain_unproven_active_session \
    host_terminal_marker_unproven_before_device \
    host_terminal_unproven_process_stuck
  run_completed=1
  trap - EXIT
  echo "Device was not launched because Host terminal evidence is malformed or incomplete." >&2
  exit 5
fi

active_phase=device_launch_intent
write_active_session
nohup "${KVAF_A5_DIRECT_BINARY}" --role device "${device_args[@]}" \
  9>&- >"${device_log}" 2>&1 &
device_pid=$!
active_phase=device_started
write_active_session

deadline=$(( $(date +%s) + KVAF_H2D_RUN_TIMEOUT_SECONDS ))
while kill -0 "${host_pid}" 2>/dev/null ||
      kill -0 "${device_pid}" 2>/dev/null
do
  if any_valid_fatal_marker || terminal_explicit_unsafe_proven; then
    record_quarantine "unsafe_provider_teardown"
    run_completed=1
    trap - EXIT
    exit 4
  fi
  if (( runner_signal_requested != 0 )); then
    retain_unproven_active_session \
      runner_interrupted \
      pair_terminal_unproven_process_stuck
    run_completed=1
    trap - EXIT
    exit 130
  fi
  if (( $(date +%s) >= deadline )); then
    if terminal_cleanup_proven; then
      # The transport is safe, but at least one worker has not exited. Keep
      # the durable state and PID evidence while the inherited fd 8 lock
      # prevents a second session. A later invocation removes this state only
      # after the worker lock is free and the same nonce-bound final markers
      # still classify PASS/FAIL_SAFE.
      active_phase=safe_terminal_process_stuck
      write_active_session
      run_completed=1
      trap - EXIT
      echo "KVAF_A5_DIRECT_URMA_H2D_FAIL_SAFE terminal proof arrived before process exit" >&2
      echo "ACTIVE_SESSION=${KVAF_DIRECT_ACTIVE_SESSION_FILE}" >&2
      exit 3
    fi
    retain_unproven_active_session \
      bounded_wait_expired \
      pair_terminal_unproven_process_stuck
    run_completed=1
    trap - EXIT
    exit 5
  fi
  sleep 1 || true
done

set +e
wait "${host_pid}"
host_rc=$?
wait "${device_pid}"
device_rc=$?
set -e

set +e
pair_outcome=$(
  python3 "${outcome_classifier}" \
    --run-id "${kvaf_run_id}" \
    --host-final "${host_final}" \
    --device-final "${device_final}"
)
pair_classifier_rc=$?
set -e

if (( pair_classifier_rc != 0 &&
      pair_classifier_rc != 3 &&
      pair_classifier_rc != 4 )); then
  retain_unproven_active_session \
    "terminal_classifier_error_rc_${pair_classifier_rc}" \
    pair_terminal_unproven_process_stuck
  run_completed=1
  trap - EXIT
  exit 5
fi
if { (( pair_classifier_rc == 0 )) &&
     [[ "${pair_outcome}" != PASS ]]; } ||
   { (( pair_classifier_rc == 3 )) &&
     [[ "${pair_outcome}" != FAIL_SAFE ]]; } ||
   { (( pair_classifier_rc == 4 )) &&
     [[ "${pair_outcome}" != FAIL_UNSAFE ]]; }; then
  retain_unproven_active_session \
    terminal_classifier_output_mismatch \
    pair_terminal_unproven_process_stuck
  run_completed=1
  trap - EXIT
  exit 5
fi

if [[ "${kvaf_teardown_assurance}" != verified ]] &&
   (( pair_classifier_rc == 0 || pair_classifier_rc == 3 )) &&
   { [[ "$(manifest_value resource_armed "${host_final}")" == 1 ]] ||
     [[ "$(manifest_value resource_armed "${device_final}")" == 1 ]]; }; then
  record_quarantine "unverified_udma_armed_terminal_${pair_outcome}"
  run_completed=1
  trap - EXIT
  echo "KVAF_A5_DIRECT_DATA_PATH_RESULT=${pair_outcome}" >&2
  echo "KVAF_A5_DIRECT_URMA_H2D_FAIL_UNSAFE unverified teardown" >&2
  echo "System UDMA cannot prove lower ungrant/free; active state was retained as quarantine." >&2
  exit 4
fi

if (( pair_classifier_rc == 4 )); then
  # The classifier has parsed nonce-bound endpoint proofs successfully.  A
  # fatal.* file is only a low-latency wake-up edge and is not required for
  # the authoritative FAIL_UNSAFE decision.
  record_quarantine \
    "terminal_explicit_fail_unsafe_host_${host_rc}_device_${device_rc}"
  unsafe_exit_rc=4
  run_completed=1
  trap - EXIT
  echo "HOST_LOG=${host_log}" >&2
  sed -n '1,260p' "${host_log}" >&2
  echo "DEVICE_LOG=${device_log}" >&2
  sed -n '1,320p' "${device_log}" >&2
  exit "${unsafe_exit_rc}"
fi

if (( pair_classifier_rc == 3 )); then
  clear_active_session
  host_pid=
  device_pid=
  run_completed=1
  trap - EXIT
  echo "KVAF_A5_DIRECT_URMA_H2D_FAIL_SAFE host_rc=${host_rc} device_rc=${device_rc}" >&2
  echo "HOST_LOG=${host_log}" >&2
  sed -n '1,260p' "${host_log}" >&2
  echo "DEVICE_LOG=${device_log}" >&2
  sed -n '1,320p' "${device_log}" >&2
  echo "RESULT_DIR=${KVAF_A5_RESULT_DIR}" >&2
  exit 3
fi

required_host_markers=(
  "KVAF_A5_DIRECT_DDR_REGISTERED"
  "KVAF_A5_DIRECT_URMA_READY role=host"
  "KVAF_A5_DIRECT_TEARDOWN_OK role=host"
)
required_device_markers=(
  "KVAF_A5_DIRECT_URMA_READY role=device"
  "KVAF_A5_DIRECT_H2D_SMOKE_OK"
  "KVAF_A5_DIRECT_TEARDOWN_OK role=device"
)
if (( KVAF_H2D_TRANSFER_COUNT == 0 )); then
  required_device_markers+=("KVAF_A5_DIRECT_ZERO_WQE_OK")
else
  required_device_markers+=("KVAF_A5_DIRECT_URMA_CHANNEL")
fi
evidence_ok=1
for marker in "${required_host_markers[@]}"; do
  if ! grep -Fq "${marker}" "${host_log}"; then
    echo "ERROR: missing Host evidence: ${marker}" >&2
    evidence_ok=0
  fi
done
for marker in "${required_device_markers[@]}"; do
  if ! grep -Fq "${marker}" "${device_log}"; then
    echo "ERROR: missing Device evidence: ${marker}" >&2
    evidence_ok=0
  fi
done

if (( host_rc != 0 || device_rc != 0 || evidence_ok == 0 )); then
  # Both endpoint markers independently prove teardown safe. A process/log
  # inconsistency is therefore a functional acceptance failure, not a reason
  # to retain DDR/HBM or reset the NPU.
  clear_active_session
  host_pid=
  device_pid=
  run_completed=1
  trap - EXIT
  echo "KVAF_A5_DIRECT_URMA_H2D_FAIL_SAFE host_rc=${host_rc} device_rc=${device_rc}" >&2
  echo "HOST_LOG=${host_log}" >&2
  sed -n '1,260p' "${host_log}" >&2
  echo "DEVICE_LOG=${device_log}" >&2
  sed -n '1,320p' "${device_log}" >&2
  echo "RESULT_DIR=${KVAF_A5_RESULT_DIR}" >&2
  exit 3
fi

clear_active_session
host_pid=
device_pid=
run_completed=1
trap - EXIT
grep -E 'KVAF_A5_(DIRECT|EID)' "${host_log}" "${device_log}"
echo "KVAF_A5_AIV_SIMT_DIRECT_URMA_H2D_ACCEPTANCE=PASS"
echo "RESULT_DIR=${KVAF_A5_RESULT_DIR}"
