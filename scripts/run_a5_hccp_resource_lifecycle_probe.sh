#!/usr/bin/env bash

# Run one bounded HCCP-backed resource lifecycle stage.  This is intentionally
# not the H2D runner: it creates no channel/Jetty/SQ/CQ and submits no WQE.
# It proves that the installed public HCOMM resource ABI can ask the matching
# HCCP control plane to register and then destroy a single local MR safely.

set -Ee
set -o pipefail
umask 077
unset BASH_ENV ENV
unset -f exit builtin 2>/dev/null || true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)

die()
{
  echo "ERROR: $*" >&2
  exit 2
}

require_uint()
{
  local name=$1
  local value=$2
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be an unsigned integer, got ${value}"
}

: "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"

phase=${1:-${KVAF_HCCP_RESOURCE_PROBE_PHASE:-host-ddr}}
case "${phase}" in
  host-ddr|device-hbm) ;;
  *) die "phase must be host-ddr or device-hbm" ;;
esac

if [[ "${KVAF_HCCP_RESOURCE_PROBE_ACK:-}" != \
      "I_ACKNOWLEDGE_HCCP_RESOURCE_LIFECYCLE_PROBE" ]]; then
  die "set KVAF_HCCP_RESOURCE_PROBE_ACK=I_ACKNOWLEDGE_HCCP_RESOURCE_LIFECYCLE_PROBE"
fi

logical_device_id=${KVAF_LOGICAL_DEVICE_ID:-}
physical_device_id=${KVAF_PHYSICAL_DEVICE_ID:-}
bytes=${KVAF_HCCP_RESOURCE_PROBE_BYTES:-4096}
teardown_retries=${KVAF_HCCP_RESOURCE_TEARDOWN_RETRIES:-3}
pair_channel=${KVAF_PAIR_CHANNEL:-}
global_state_dir=${KVAF_GLOBAL_STATE_DIR:-/data/.kvaf-state}
result_root=${KVAF_HCCP_RESOURCE_RESULT_ROOT:-"${global_state_dir}/KVAF/results"}
run_tag=${KVAF_A5_RUN_TAG:-"hccp_resource_${phase}_$(date +%Y%m%d_%H%M%S)_$$"}
result_dir=${KVAF_HCCP_RESOURCE_RESULT_DIR:-"${result_root}/${run_tag}"}
binary=${KVAF_HCCP_RESOURCE_PROBE_BINARY:-"${KVAF_A5_BUILD_DIR:-${ROOT_DIR}/build-a5-aiv-urma}/access_operator/a5/kvaf_a5_hccp_resource_lifecycle_probe"}
idle_checker=${KVAF_NPU_IDLE_CHECKER:-"${SCRIPT_DIR}/check_a5_npu_idle.py"}

[[ -n "${logical_device_id}" ]] || die "set KVAF_LOGICAL_DEVICE_ID explicitly; this probe does not reset/query topology"
[[ -n "${physical_device_id}" ]] || die "set KVAF_PHYSICAL_DEVICE_ID explicitly; this probe does not reset/query topology"
require_uint KVAF_LOGICAL_DEVICE_ID "${logical_device_id}"
require_uint KVAF_PHYSICAL_DEVICE_ID "${physical_device_id}"
require_uint KVAF_HCCP_RESOURCE_PROBE_BYTES "${bytes}"
require_uint KVAF_HCCP_RESOURCE_TEARDOWN_RETRIES "${teardown_retries}"
(( bytes > 0 )) || die "KVAF_HCCP_RESOURCE_PROBE_BYTES must be positive"
(( teardown_retries > 0 && teardown_retries <= 100 )) ||
  die "KVAF_HCCP_RESOURCE_TEARDOWN_RETRIES must be in 1..100"
if [[ -n "${pair_channel}" ]]; then
  require_uint KVAF_PAIR_CHANNEL "${pair_channel}"
fi

if [[ "${phase}" == device-hbm ]]; then
  super_device_id=${KVAF_SUPER_DEVICE_ID:-}
  server_index=${KVAF_SERVER_INDEX:-}
  super_pod_index=${KVAF_SUPER_POD_INDEX:-}
  [[ -n "${super_device_id}" && -n "${server_index}" && -n "${super_pod_index}" ]] ||
    die "device-hbm requires KVAF_SUPER_DEVICE_ID, KVAF_SERVER_INDEX, and KVAF_SUPER_POD_INDEX"
  require_uint KVAF_SUPER_DEVICE_ID "${super_device_id}"
  require_uint KVAF_SERVER_INDEX "${server_index}"
  require_uint KVAF_SUPER_POD_INDEX "${super_pod_index}"
else
  super_device_id=0
  server_index=0
  super_pod_index=0
fi

[[ -x "${binary}" ]] || die "missing probe binary ${binary}; run scripts/build_a5_aiv_urma.sh first"
[[ -f "${idle_checker}" && ! -L "${idle_checker}" ]] ||
  die "missing regular NPU idle checker ${idle_checker}"

mkdir -p "${global_state_dir}/KVAF" "${result_root}"
# `/data/.kvaf-state` may already be a shared mount root.  Do not change its
# mode; only KVAF's own child directories are private to this probe.
chmod 0700 "${global_state_dir}/KVAF" "${result_root}"
[[ ! -e "${result_dir}" && ! -L "${result_dir}" ]] ||
  die "result directory already exists: ${result_dir}"
mkdir -m 0700 "${result_dir}"

lock_file="${global_state_dir}/KVAF/hccp-resource-lifecycle.lock"
exec 9>"${lock_file}"
if ! flock -n 9; then
  die "another HCCP resource lifecycle probe is active: ${lock_file}"
fi

echo "PHASE=${phase}"
echo "RESULT_DIR=${result_dir}"
echo "LOGICAL_DEVICE_ID=${logical_device_id}"
echo "PHYSICAL_DEVICE_ID=${physical_device_id}"
echo "HCCP_CONTROL_PLANE=installed-libhcomm-resource-api"
echo "HOT_PATH=none-this-is-resource-lifecycle-only"
echo "WQE_SUBMISSIONS=0"

if ! preflight_output=$(bash "${SCRIPT_DIR}/preflight_a5_hccp_hal_capabilities.sh" 2>&1); then
  printf '%s\n' "${preflight_output}" > "${result_dir}/static-preflight.log"
  die "HCCP static preflight failed; see ${result_dir}/static-preflight.log"
fi
printf '%s\n' "${preflight_output}" > "${result_dir}/static-preflight.log"
grep -Fxq 'KVAF_HCCP_HCOMM_RESOURCE_SURFACE=complete-symbols-only' \
  "${result_dir}/static-preflight.log" ||
  die "installed public HCOMM resource API is incomplete; see ${result_dir}/static-preflight.log"
grep -Eq '^KVAF_HCCP_HCOMM_LIBRARY=/.+libhcomm\.so$' \
  "${result_dir}/static-preflight.log" ||
  die "preflight did not select an absolute installed libhcomm.so"
echo "KVAF_A5_HCCP_STATIC_RESOURCE_PRECHECK_OK"

if ! inventory=$(LC_ALL=C timeout 20s npu-smi info 2>&1); then
  printf '%s\n' "${inventory}" > "${result_dir}/npu-smi-info.txt"
  die "npu-smi info failed; cannot prove the selected NPU is idle"
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

if [[ -n "${KVAF_HCCP_RESOURCE_PROBE_EID:-}" ]]; then
  eid=${KVAF_HCCP_RESOURCE_PROBE_EID}
  eid_source=caller
else
  resolver_args=(
    --physical-device-id "${physical_device_id}"
    --shell
  )
  if [[ -n "${pair_channel}" ]]; then
    resolver_args+=(--pair-channel "${pair_channel}")
  fi
  if ! eid_output=$(python3 "${SCRIPT_DIR}/resolve_a5_urma_eids.py" "${resolver_args[@]}"); then
    die "A5 EID resolution failed before the resource probe"
  fi
  printf '%s\n' "${eid_output}" > "${result_dir}/eid-resolution.env"
  host_eid=$(awk -F= '$1 == "KVAF_HOST_EID" { print $2 }' "${result_dir}/eid-resolution.env")
  device_eid=$(awk -F= '$1 == "KVAF_DEVICE_EID" { print $2 }' "${result_dir}/eid-resolution.env")
  if [[ "${phase}" == host-ddr ]]; then
    eid=${host_eid}
  else
    eid=${device_eid}
  fi
  eid_source=driver-pair-info
fi
[[ "${eid}" =~ ^[0-9A-Fa-f:=-]{32,47}$ ]] || die "resolved EID has an invalid format"
echo "KVAF_A5_HCCP_RESOURCE_EID_RESOLVED source=${eid_source} phase=${phase}"

probe_args=(
  --phase "${phase}"
  --eid "${eid}"
  --logical-device-id "${logical_device_id}"
  --physical-device-id "${physical_device_id}"
  --super-device-id "${super_device_id}"
  --server-index "${server_index}"
  --super-pod-index "${super_pod_index}"
  --bytes "${bytes}"
  --teardown-retries "${teardown_retries}"
)

set +e
"${binary}" "${probe_args[@]}" 2>&1 | tee "${result_dir}/probe.log"
probe_rc=${PIPESTATUS[0]}
set -e
echo "KVAF_A5_HCCP_RESOURCE_PROBE_RC=${probe_rc}"
if [[ "${probe_rc}" -eq 0 ]] &&
   grep -Fxq "KVAF_A5_HCCP_RESOURCE_PROBE_OK phase=${phase}" \
     "${result_dir}/probe.log"; then
  echo "KVAF_A5_HCCP_RESOURCE_LIFECYCLE_OK phase=${phase}"
  exit 0
fi
if grep -Fq 'KVAF_A5_HCCP_RESOURCE_TEARDOWN_UNPROVEN' \
   "${result_dir}/probe.log"; then
  echo "KVAF_A5_HCCP_RESOURCE_LIFECYCLE_UNPROVEN phase=${phase}" >&2
  echo "Do not advance to channel or WQE stages; retain ${result_dir}." >&2
fi
exit "${probe_rc:-1}"
