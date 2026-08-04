#!/usr/bin/env bash

# Establish and destroy a public-HCOMM Host-CPU / Device-AIV channel without
# submitting a WQE.  Both workers have their own bounded waits.  This
# supervisor intentionally never sends a signal to a resource-owning worker:
# if a worker cannot prove teardown, it remains alive holding its backing
# memory rather than freeing it beneath a possibly-live DMA reference.

set -Eeuo pipefail
umask 077
unset BASH_ENV ENV
unset -f exit builtin 2>/dev/null || true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)

die()
{
  echo "ERROR: $*" >&2
  builtin exit 2
}

require_uint()
{
  local name=$1
  local value=$2
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be an unsigned integer, got ${value}"
}

marker_has_stage()
{
  local path=$1
  local run_id=$2
  local stage=$3
  local -a lines=()
  [[ -r "${path}" ]] || return 1
  mapfile -t lines < "${path}" || return 1
  (( ${#lines[@]} == 3 || ${#lines[@]} == 4 )) || return 1
  [[ "${lines[0]}" == 'schema=kvaf-hccp-aiv-channel-probe-v1' &&
     "${lines[1]}" == "run_id=${run_id}" &&
     "${lines[2]}" == "stage=${stage}" ]] || return 1
  (( ${#lines[@]} == 3 )) || [[ "${lines[3]}" == detail=* ]]
}

worker_running()
{
  local pid=$1
  local state
  state=$(ps -p "${pid}" -o stat= 2>/dev/null | tr -d '[:space:]')
  [[ -n "${state}" && "${state}" != Z* ]]
}

marker_has_safe_terminal()
{
  local path=$1
  local run_id=$2
  marker_has_stage "${path}" "${run_id}" PASS ||
    marker_has_stage "${path}" "${run_id}" FAIL_SAFE
}

process_start_ticks()
{
  local pid=$1
  awk '{ print $22 }' "/proc/${pid}/stat" 2>/dev/null || true
}

legacy_active_field()
{
  local path=$1
  local key=$2
  awk -F= -v key="${key}" '$1 == key { print $2; exit }' \
    "${path}" 2>/dev/null || true
}

# Old HCOMM headers cannot carry a run-id channel name. Never let an unsafe
# owner on NPU2 block an independent NPU7 pair, but never allow a second
# legacy channel-probe worker on the selected physical NPU either.
legacy_workers_for_current_npu()
{
  local -a candidate_pids=()
  local candidate_pid
  local cmdline

  mapfile -t candidate_pids < <(
    pgrep -f '[k]vaf_a5_hccp_aiv_channel_probe' 2>/dev/null || true
  )
  for candidate_pid in "${candidate_pids[@]}"; do
    [[ "${candidate_pid}" =~ ^[0-9]+$ &&
       -r "/proc/${candidate_pid}/cmdline" ]] || continue
    cmdline=$(tr '\0' ' ' < "/proc/${candidate_pid}/cmdline" 2>/dev/null || true)
    [[ " ${cmdline} " == *" --physical-device-id ${physical_device_id} "* ]] ||
      continue
    printf '%s %s\n' "${candidate_pid}" "${cmdline}"
  done
}

: "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"

if [[ "${KVAF_HCCP_AIV_CHANNEL_PROBE_ACK:-}" != \
      "I_ACKNOWLEDGE_HCCP_AIV_CHANNEL_ZERO_WQE_PROBE" ]]; then
  die "set KVAF_HCCP_AIV_CHANNEL_PROBE_ACK=I_ACKNOWLEDGE_HCCP_AIV_CHANNEL_ZERO_WQE_PROBE"
fi

logical_device_id=${KVAF_LOGICAL_DEVICE_ID:-}
pair_channel=${KVAF_PAIR_CHANNEL:-0}
bytes=${KVAF_HCCP_AIV_CHANNEL_PROBE_BYTES:-4096}
worker_timeout_seconds=${KVAF_HCCP_AIV_CHANNEL_TIMEOUT_SECONDS:-20}
run_timeout_seconds=${KVAF_HCCP_AIV_CHANNEL_RUN_TIMEOUT_SECONDS:-90}
teardown_retries=${KVAF_HCCP_AIV_CHANNEL_TEARDOWN_RETRIES:-3}
global_state_dir=${KVAF_GLOBAL_STATE_DIR:-/data/.kvaf-state}
result_root=${KVAF_HCCP_AIV_CHANNEL_RESULT_ROOT:-"${global_state_dir}/KVAF/results"}
binary=${KVAF_HCCP_AIV_CHANNEL_PROBE_BINARY:-"${KVAF_A5_BUILD_DIR:-${ROOT_DIR}/build-a5-aiv-urma}/access_operator/a5/kvaf_a5_hccp_aiv_channel_probe"}
idle_checker=${KVAF_NPU_IDLE_CHECKER:-"${SCRIPT_DIR}/check_a5_npu_idle.py"}
legacy_anonymous_requested=${KVAF_HCCP_AIV_CHANNEL_LEGACY_ANONYMOUS:-0}
legacy_anonymous_ack=${KVAF_HCCP_AIV_CHANNEL_LEGACY_ANONYMOUS_ACK:-}

[[ -n "${logical_device_id}" ]] || die "set KVAF_LOGICAL_DEVICE_ID explicitly"
require_uint KVAF_LOGICAL_DEVICE_ID "${logical_device_id}"
require_uint KVAF_PAIR_CHANNEL "${pair_channel}"
require_uint KVAF_HCCP_AIV_CHANNEL_PROBE_BYTES "${bytes}"
require_uint KVAF_HCCP_AIV_CHANNEL_TIMEOUT_SECONDS "${worker_timeout_seconds}"
require_uint KVAF_HCCP_AIV_CHANNEL_RUN_TIMEOUT_SECONDS "${run_timeout_seconds}"
require_uint KVAF_HCCP_AIV_CHANNEL_TEARDOWN_RETRIES "${teardown_retries}"
(( bytes > 0 && bytes <= 1073741824 )) || die "probe bytes must be in 1..1073741824"
(( worker_timeout_seconds >= 5 && worker_timeout_seconds <= 300 )) ||
  die "worker timeout must be in 5..300 seconds"
(( run_timeout_seconds >= worker_timeout_seconds * 2 + 20 &&
   run_timeout_seconds <= 900 )) ||
  die "run timeout must be at least 2*worker timeout+20 and at most 900 seconds"
(( teardown_retries >= 1 && teardown_retries <= 100 )) ||
  die "teardown retries must be in 1..100"
[[ -x "${binary}" ]] || die "missing channel probe ${binary}; run scripts/build_a5_aiv_urma.sh first"
[[ -f "${idle_checker}" && ! -L "${idle_checker}" ]] ||
  die "missing regular NPU idle checker ${idle_checker}"
[[ "${legacy_anonymous_requested}" == 0 ||
   "${legacy_anonymous_requested}" == 1 ]] ||
  die "KVAF_HCCP_AIV_CHANNEL_LEGACY_ANONYMOUS must be 0 or 1"

# This is a compile/header capability query only.  It does not initialize ACL
# or create an HCOMM resource.  Keep the normal path fail-closed: decode and
# production use require v3's run-id-bound channelName.
if ! channel_capability_output=$("${binary}" --print-capabilities 2>&1); then
  die "could not query HCOMM channel-name capability: ${channel_capability_output}"
fi
case "${channel_capability_output}" in
  KVAF_HCCP_CHANNEL_NAME_SUPPORT=1)
    channel_name_support=1
    ;;
  KVAF_HCCP_CHANNEL_NAME_SUPPORT=0)
    channel_name_support=0
    ;;
  *)
    die "invalid HCOMM channel-name capability output: ${channel_capability_output}"
    ;;
esac

channel_name_mode=named-required
channel_identity=named-run-id
if [[ "${channel_name_support}" -eq 0 ]]; then
  if [[ "${legacy_anonymous_requested}" != 1 ]]; then
    die "installed HCOMM header has no ChannelDesc.channelName; default named-required mode performed no hardware operation. Upgrade the paired CANN/HCOMM stack, or explicitly opt into the zero-WQE legacy compatibility probe."
  fi
  [[ "${legacy_anonymous_ack}" == \
     "I_ACCEPT_LEGACY_ANONYMOUS_CHANNEL_CROSS_PROCESS_PAIRING_RISK" ]] ||
    die "set KVAF_HCCP_AIV_CHANNEL_LEGACY_ANONYMOUS_ACK=I_ACCEPT_LEGACY_ANONYMOUS_CHANNEL_CROSS_PROCESS_PAIRING_RISK for the old-header zero-WQE compatibility probe"
  channel_name_mode=legacy-anonymous-risk-ack
  channel_identity=anonymous-old-header-exclusive
elif [[ "${legacy_anonymous_requested}" == 1 ]]; then
  die "legacy anonymous mode is only valid when the installed public HCOMM header lacks ChannelDesc.channelName"
fi

mkdir -p "${global_state_dir}/KVAF" "${result_root}"
chmod 0700 "${global_state_dir}/KVAF" "${result_root}"
lock_file=

# The legacy ABI has no run-id namespace.  Its lock and persistent owner
# state are selected only after topology and the exact EID pair are resolved.
# They intentionally remain independent from KVAF_GLOBAL_STATE_DIR so two
# containers cannot evade a retained-owner guard by choosing different result
# roots.
resource_guard_fd=9
legacy_state_dir=/data/.kvaf-state/KVAF
legacy_unscoped_active_file=${legacy_state_dir}/hccp-aiv-legacy-anonymous.active
legacy_active_file=
legacy_active_owned=0
legacy_boot_id=
legacy_scope=

# The topology helper selects then releases its own short-lived ACL context.
# Parse only its known KEY=UINT32 output rather than sourcing arbitrary text.
if ! topology_output=$(python3 "${SCRIPT_DIR}/query_a5_topology.py" \
    --logical-device-id "${logical_device_id}" --shell); then
  die "A5 topology query failed"
fi
topology_value()
{
  local key=$1
  local value
  value=$(printf '%s\n' "${topology_output}" |
    awk -F= -v key="${key}" '$1 == key { print $2 }')
  [[ "${value}" =~ ^[0-9]+$ ]] || die "topology did not return ${key}"
  printf '%s\n' "${value}"
}
physical_device_id=$(topology_value KVAF_PHYSICAL_DEVICE_ID)
super_device_id=$(topology_value KVAF_SUPER_DEVICE_ID)
server_index=$(topology_value KVAF_SERVER_INDEX)
super_pod_index=$(topology_value KVAF_SUPER_POD_INDEX)

# A worker retains inherited lock descriptors after an unsafe owner exits its
# supervisor.  Scope the supervisor lock to the physical NPU, so an NPU2
# retained worker cannot falsely block an independently selected NPU7.  The
# old-header fallback must use the fixed host-shared state root: permitting
# KVAF_GLOBAL_STATE_DIR to select this lock would let another container evade
# the same-NPU exclusion simply by choosing a different result directory.
lock_dir="${global_state_dir}/KVAF"
if [[ "${channel_name_mode}" == legacy-anonymous-risk-ack ]]; then
  mkdir -p "${legacy_state_dir}"
  chmod 0700 "${legacy_state_dir}"
  lock_dir="${legacy_state_dir}"
fi
lock_file="${lock_dir}/hccp-aiv-channel.npu${physical_device_id}.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
  die "another HCCP AIV channel probe is active on this NPU: ${lock_file}"
fi

run_id=${KVAF_HCCP_AIV_CHANNEL_RUN_ID:-"aivch_${physical_device_id}_$(date +%Y%m%d_%H%M%S)_$$"}
[[ "${run_id}" =~ ^[A-Za-z0-9_-]{1,128}$ ]] ||
  die "KVAF_HCCP_AIV_CHANNEL_RUN_ID must contain only letters, digits, _ or -"
run_tag=${KVAF_A5_RUN_TAG:-"hccp_aiv_channel_npu${physical_device_id}_$(date +%Y%m%d_%H%M%S)_$$"}
result_dir=${KVAF_HCCP_AIV_CHANNEL_RESULT_DIR:-"${result_root}/${run_tag}"}
[[ ! -e "${result_dir}" && ! -L "${result_dir}" ]] ||
  die "result directory already exists: ${result_dir}"
mkdir -m 0700 "${result_dir}"
exchange_dir="${result_dir}/exchange"
mkdir -m 0700 "${exchange_dir}"

echo "PHASE=hccp_aiv_channel_zero_wqe"
echo "RESULT_DIR=${result_dir}"
echo "RUN_ID=${run_id}"
echo "LOGICAL_DEVICE_ID=${logical_device_id}"
echo "PHYSICAL_DEVICE_ID=${physical_device_id}"
echo "HCCP_CONTROL_PLANE=installed-libhcomm-resource-api"
echo "HOT_PATH=none-this-is-channel-snapshot-only"
echo "WQE_SUBMISSIONS=0"
echo "CHANNEL_NAME_SUPPORT=${channel_name_support}"
echo "CHANNEL_NAME_MODE=${channel_name_mode}"
echo "CHANNEL_IDENTITY=${channel_identity}"
if [[ "${channel_name_mode}" == legacy-anonymous-risk-ack ]]; then
  echo "CONTRACT=best-effort-exclusive-legacy-anonymous-v1"
fi

if ! preflight_output=$(bash "${SCRIPT_DIR}/preflight_a5_hccp_hal_capabilities.sh" 2>&1); then
  printf '%s\n' "${preflight_output}" > "${result_dir}/static-preflight.log"
  die "HCCP static preflight failed; see ${result_dir}/static-preflight.log"
fi
printf '%s\n' "${preflight_output}" > "${result_dir}/static-preflight.log"
grep -Fxq 'KVAF_HCCP_HCOMM_RESOURCE_SURFACE=complete-symbols-only' \
  "${result_dir}/static-preflight.log" ||
  die "installed public HCOMM resource API is incomplete; see ${result_dir}/static-preflight.log"
echo "KVAF_A5_HCCP_STATIC_CHANNEL_PRECHECK_OK"

if ! inventory=$(LC_ALL=C timeout 20s npu-smi info 2>&1); then
  printf '%s\n' "${inventory}" > "${result_dir}/npu-smi-info.txt"
  die "npu-smi info failed; cannot prove selected NPU is idle"
fi
printf '%s\n' "${inventory}" > "${result_dir}/npu-smi-info.txt"
if ! idle_output=$(printf '%s\n' "${inventory}" |
     python3 "${idle_checker}" --physical-device-id "${physical_device_id}"); then
  printf '%s\n' "${idle_output}" > "${result_dir}/npu-idle.txt"
  die "NPU idle parser rejected the inventory; see ${result_dir}/npu-idle.txt"
fi
printf '%s\n' "${idle_output}" > "${result_dir}/npu-idle.txt"
grep -Fxq 'KVAF_NPU_IDLE_STATUS=idle' "${result_dir}/npu-idle.txt" ||
  die "selected NPU is not proven idle; see ${result_dir}/npu-idle.txt"
echo "KVAF_A5_NPU_IDLE_PROVEN physical_device_id=${physical_device_id}"

if ! endpoint_output=$(python3 "${SCRIPT_DIR}/resolve_a5_urma_eids.py" \
    --physical-device-id "${physical_device_id}" \
    --pair-channel "${pair_channel}" --shell); then
  die "A5 Host/Device EID resolution failed"
fi
printf '%s\n' "${endpoint_output}" > "${result_dir}/eid-resolution.env"
host_eid=$(awk -F= '$1 == "KVAF_HOST_EID" { print $2 }' "${result_dir}/eid-resolution.env")
device_eid=$(awk -F= '$1 == "KVAF_DEVICE_EID" { print $2 }' "${result_dir}/eid-resolution.env")
[[ "${host_eid}" =~ ^[0-9A-Fa-f]{32}$ &&
   "${device_eid}" =~ ^[0-9A-Fa-f]{32}$ &&
   "${host_eid}" != "${device_eid}" ]] ||
  die "EID resolver returned an invalid or identical endpoint pair"
echo "KVAF_A5_HCCP_CHANNEL_EIDS_RESOLVED source=driver-pair-info"

# Scope the old-header fallback by the hardware identity HCOMM uses for
# matching.  The same-EID scope remains fail-closed; a retained owner on a
# different physical NPU is reported and left intact, not deleted or reused.
if [[ "${channel_name_mode}" == legacy-anonymous-risk-ack ]]; then
  command -v pgrep >/dev/null 2>&1 ||
    die "legacy anonymous compatibility probe requires pgrep"
  mkdir -p "${legacy_state_dir}"
  chmod 0700 "${legacy_state_dir}"
  legacy_scope="npu${physical_device_id}_${host_eid}_${device_eid}"
  [[ "${legacy_scope}" =~ ^npu[0-9]+_[0-9A-Fa-f]{32}_[0-9A-Fa-f]{32}$ ]] ||
    die "invalid legacy anonymous scope"
  legacy_active_file=${legacy_state_dir}/hccp-aiv-legacy-anonymous.${legacy_scope}.active
  legacy_lock_file=${legacy_state_dir}/hccp-aiv-anonymous-channel.${legacy_scope}.lock
  exec 10>"${legacy_lock_file}"
  if ! flock -n 10; then
    die "legacy anonymous mode refused: same NPU/EID-pair lock is held: ${legacy_lock_file}"
  fi
  resource_guard_fd=10

  if legacy_workers=$(legacy_workers_for_current_npu); [[ -n "${legacy_workers}" ]]; then
    die "legacy anonymous mode refused: channel-probe worker(s) remain on this NPU: ${legacy_workers}"
  fi

  # Releases before this change used one unscoped active file.  Preserve it
  # as an owner guard for its recorded NPU while allowing a different NPU/EID
  # pair to proceed.  Never remove the old file during this migration.
  if [[ -e "${legacy_unscoped_active_file}" ||
        -L "${legacy_unscoped_active_file}" ]]; then
    legacy_unscoped_device=$(legacy_active_field "${legacy_unscoped_active_file}" \
      physical_device_id)
    legacy_unscoped_boot_id=$(legacy_active_field "${legacy_unscoped_active_file}" \
      boot_id)
    [[ "${legacy_unscoped_device}" =~ ^[0-9]+$ &&
       "${legacy_unscoped_boot_id}" =~ ^[0-9a-f-]{36}$ ]] ||
      die "legacy anonymous active state is malformed: ${legacy_unscoped_active_file}"
    if [[ "${legacy_unscoped_device}" == "${physical_device_id}" ]]; then
      legacy_boot_id=$(cat /proc/sys/kernel/random/boot_id)
      [[ "${legacy_boot_id}" =~ ^[0-9a-f-]{36}$ ]] ||
        die "could not read a valid current boot ID"
      if [[ "${legacy_unscoped_boot_id}" == "${legacy_boot_id}" ]]; then
        die "legacy anonymous active state remains on this NPU: ${legacy_unscoped_active_file}; do not retry until its owner is proven safe or the host reboots"
      fi
      legacy_reboot_archive=${legacy_unscoped_active_file}.previous-boot.$(date +%Y%m%d_%H%M%S).$$
      mv -- "${legacy_unscoped_active_file}" "${legacy_reboot_archive}"
      echo "KVAF_A5_HCCP_LEGACY_ANONYMOUS_PREVIOUS_BOOT_STATE_ARCHIVED=${legacy_reboot_archive}"
    else
      echo "KVAF_A5_HCCP_LEGACY_ANONYMOUS_OTHER_NPU_RETAINED physical_device_id=${legacy_unscoped_device}"
    fi
  fi
fi

# A stale legacy owner is never silently cleared on the same boot.  A normal
# PASS or FAIL_SAFE run archives this file only after both exact teardown
# markers exist; timeout, interrupt, or unsafe retention leaves it in place
# and blocks the next old-header attempt until reboot/operator isolation.
if [[ "${channel_name_mode}" == legacy-anonymous-risk-ack ]]; then
  if [[ -z "${legacy_boot_id}" ]]; then
    legacy_boot_id=$(cat /proc/sys/kernel/random/boot_id)
  fi
  [[ "${legacy_boot_id}" =~ ^[0-9a-f-]{36}$ ]] ||
    die "could not read a valid current boot ID"
  if [[ -e "${legacy_active_file}" || -L "${legacy_active_file}" ]]; then
    legacy_previous_boot_id=$(awk -F= '$1 == "boot_id" { print $2 }' \
      "${legacy_active_file}" 2>/dev/null || true)
    [[ "${legacy_previous_boot_id}" =~ ^[0-9a-f-]{36}$ ]] ||
      die "legacy anonymous active state is malformed: ${legacy_active_file}"
    [[ "${legacy_previous_boot_id}" != "${legacy_boot_id}" ]] ||
      die "legacy anonymous active state remains from this boot: ${legacy_active_file}; do not retry until its owner is proven safe or the host reboots"
    legacy_reboot_archive="${legacy_active_file}.previous-boot.$(date +%Y%m%d_%H%M%S).$$"
    mv -- "${legacy_active_file}" "${legacy_reboot_archive}"
    echo "KVAF_A5_HCCP_LEGACY_ANONYMOUS_PREVIOUS_BOOT_STATE_ARCHIVED=${legacy_reboot_archive}"
  fi
  if ! (set -o noclobber; printf '%s\n' \
      'schema=kvaf-hccp-aiv-legacy-anonymous-active-v1' \
      "boot_id=${legacy_boot_id}" \
      "run_id=${run_id}" \
      "physical_device_id=${physical_device_id}" \
      "host_eid=${host_eid}" \
      "device_eid=${device_eid}" \
      'state=launching' > "${legacy_active_file}"); then
    die "could not claim legacy anonymous active state: ${legacy_active_file}"
  fi
  legacy_active_owned=1
fi

common_args=(
  --run-id "${run_id}"
  --exchange-dir "${exchange_dir}"
  --host-eid "${host_eid}"
  --device-eid "${device_eid}"
  --logical-device-id "${logical_device_id}"
  --physical-device-id "${physical_device_id}"
  --super-device-id "${super_device_id}"
  --server-index "${server_index}"
  --super-pod-index "${super_pod_index}"
  --bytes "${bytes}"
  --timeout-seconds "${worker_timeout_seconds}"
  --teardown-retries "${teardown_retries}"
  --channel-name-mode "${channel_name_mode}"
  --resource-guard-fd "${resource_guard_fd}"
)

"${binary}" --role host "${common_args[@]}" >"${result_dir}/host.log" 2>&1 &
host_pid=$!
"${binary}" --role device "${common_args[@]}" >"${result_dir}/device.log" 2>&1 &
device_pid=$!
printf 'run_id=%s\nhost_pid=%s\ndevice_pid=%s\n' \
  "${run_id}" "${host_pid}" "${device_pid}" > "${result_dir}/workers.env"
if [[ "${legacy_active_owned}" -eq 1 ]]; then
  printf 'host_pid=%s\nhost_start_ticks=%s\ndevice_pid=%s\ndevice_start_ticks=%s\nstate=workers-launched\n' \
    "${host_pid}" "$(process_start_ticks "${host_pid}")" \
    "${device_pid}" "$(process_start_ticks "${device_pid}")" \
    >> "${legacy_active_file}"
fi

interrupted=0
on_interrupt()
{
  interrupted=1
  echo "KVAF_A5_HCCP_CHANNEL_SUPERVISOR_INTERRUPTED workers_retained host_pid=${host_pid} device_pid=${device_pid}" >&2
}
trap on_interrupt INT TERM

deadline=$(( $(date +%s) + run_timeout_seconds ))
while :; do
  if [[ "${interrupted}" -ne 0 ]]; then
    exit 1
  fi
  if [[ -e "${exchange_dir}/unsafe.host" || -e "${exchange_dir}/unsafe.device" ]]; then
    echo "KVAF_A5_HCCP_CHANNEL_TEARDOWN_UNPROVEN workers_retained=1" >&2
    exit 1
  fi
  if [[ -r "${exchange_dir}/host.final" && -r "${exchange_dir}/device.final" ]]; then
    break
  fi
  # A worker that has already written teardown-ok holds no HCOMM/MR/HBM/DDR
  # resource.  It can safely be interrupted during the later ACL-only
  # context cleanup after its signal guard is restored.  Report that distinct
  # safe-but-incomplete state immediately rather than waiting until the
  # supervisor deadline on a zombie child.
  if ! worker_running "${host_pid}" && ! worker_running "${device_pid}"; then
    if marker_has_stage "${exchange_dir}/host.teardown_ok" "${run_id}" teardown-ok &&
       marker_has_stage "${exchange_dir}/device.teardown_ok" "${run_id}" teardown-ok; then
      echo "KVAF_A5_HCCP_CHANNEL_RUNTIME_CLEANUP_INCOMPLETE resources_already_closed=1" >&2
    else
      echo "KVAF_A5_HCCP_CHANNEL_WORKERS_EXITED_BEFORE_FINAL resources_not_proven=1" >&2
    fi
    exit 1
  fi
  if (( $(date +%s) >= deadline )); then
    echo "KVAF_A5_HCCP_CHANNEL_SUPERVISOR_TIMEOUT workers_retained=1 host_running=$(worker_running "${host_pid}" && echo yes || echo no) device_running=$(worker_running "${device_pid}" && echo yes || echo no)" >&2
    echo "No signal was sent to either worker." >&2
    exit 1
  fi
  sleep 1
done

set +e
wait "${host_pid}"
host_rc=$?
wait "${device_pid}"
device_rc=$?
set -e

archive_legacy_active_if_terminal_safe()
{
  [[ "${legacy_active_owned}" -eq 1 ]] || return 0
  [[ ! -e "${exchange_dir}/unsafe.host" &&
     ! -e "${exchange_dir}/unsafe.device" ]] || return 0
  marker_has_safe_terminal "${exchange_dir}/host.final" "${run_id}" &&
    marker_has_safe_terminal "${exchange_dir}/device.final" "${run_id}" &&
    marker_has_stage "${exchange_dir}/host.teardown_ok" "${run_id}" teardown-ok &&
    marker_has_stage "${exchange_dir}/device.teardown_ok" "${run_id}" teardown-ok ||
    return 0
  legacy_complete_file="${legacy_active_file}.completed.${run_id}"
  mv -- "${legacy_active_file}" "${legacy_complete_file}"
  legacy_active_owned=0
  echo "KVAF_A5_HCCP_LEGACY_ANONYMOUS_TERMINAL_SAFE=${legacy_complete_file}"
}

archive_legacy_active_if_terminal_safe

if [[ "${host_rc}" -eq 0 && "${device_rc}" -eq 0 ]] &&
   marker_has_stage "${exchange_dir}/host.final" "${run_id}" PASS &&
   marker_has_stage "${exchange_dir}/device.final" "${run_id}" PASS &&
   marker_has_stage "${exchange_dir}/host.teardown_ok" "${run_id}" teardown-ok &&
   marker_has_stage "${exchange_dir}/device.teardown_ok" "${run_id}" teardown-ok &&
   grep -Fxq 'KVAF_A5_HCCP_AIV_CHANNEL_ZERO_WQE_OK role=host' "${result_dir}/host.log" &&
   grep -Fxq 'KVAF_A5_HCCP_AIV_CHANNEL_ZERO_WQE_OK role=device' "${result_dir}/device.log"; then
  echo "KVAF_A5_HCCP_AIV_CHANNEL_ZERO_WQE_PROBE_OK"
  echo "RESULT_DIR=${result_dir}"
  exit 0
fi

echo "KVAF_A5_HCCP_AIV_CHANNEL_ZERO_WQE_PROBE_FAILED host_rc=${host_rc} device_rc=${device_rc}" >&2
echo "RESULT_DIR=${result_dir}" >&2
exit 1
