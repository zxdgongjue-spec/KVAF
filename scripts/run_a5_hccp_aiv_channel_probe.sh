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

mkdir -p "${global_state_dir}/KVAF" "${result_root}"
chmod 0700 "${global_state_dir}/KVAF" "${result_root}"
lock_file="${global_state_dir}/KVAF/hccp-aiv-channel.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
  die "another HCCP AIV channel probe is active: ${lock_file}"
fi

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
)

"${binary}" --role host "${common_args[@]}" >"${result_dir}/host.log" 2>&1 &
host_pid=$!
"${binary}" --role device "${common_args[@]}" >"${result_dir}/device.log" 2>&1 &
device_pid=$!
printf 'run_id=%s\nhost_pid=%s\ndevice_pid=%s\n' \
  "${run_id}" "${host_pid}" "${device_pid}" > "${result_dir}/workers.env"

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
