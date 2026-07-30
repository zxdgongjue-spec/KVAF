#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

KVAF_A5_BUILD_DIR=${KVAF_A5_BUILD_DIR:-"${ROOT_DIR}/build-a5-aiv-urma"}
KVAF_A5_BINARY=${KVAF_A5_BINARY:-"${KVAF_A5_BUILD_DIR}/access_operator/a5/kvaf_a5_aiv_urma_h2d_smoke"}
KVAF_HOST_EID=${KVAF_HOST_EID:-}
KVAF_DEVICE_EID=${KVAF_DEVICE_EID:-}
KVAF_PAIR_CHANNEL=${KVAF_PAIR_CHANNEL:-}
KVAF_EID_PROC_ROOT=${KVAF_EID_PROC_ROOT:-/proc/asdrv_ub}
KVAF_EID_LOCK_FILE=${KVAF_EID_LOCK_FILE:-/run/lock/kvaf_asdrv_ub_pair.lock}
KVAF_EID_LOCK_TIMEOUT_SECONDS=${KVAF_EID_LOCK_TIMEOUT_SECONDS:-10}
KVAF_LOGICAL_DEVICE_ID=${KVAF_LOGICAL_DEVICE_ID:-0}
KVAF_PHYSICAL_DEVICE_ID=${KVAF_PHYSICAL_DEVICE_ID:-}
KVAF_SUPER_DEVICE_ID=${KVAF_SUPER_DEVICE_ID:-}
KVAF_SERVER_INDEX=${KVAF_SERVER_INDEX:-}
KVAF_SUPER_POD_INDEX=${KVAF_SUPER_POD_INDEX:-}
KVAF_H2D_BYTES=${KVAF_H2D_BYTES:-4096}
KVAF_H2D_TIMEOUT_SECONDS=${KVAF_H2D_TIMEOUT_SECONDS:-120}
KVAF_A5_RUN_TAG=${KVAF_A5_RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}
KVAF_A5_RESULT_DIR=${KVAF_A5_RESULT_DIR:-"${ROOT_DIR}/results/a5_aiv_urma_h2d_${KVAF_A5_RUN_TAG}"}

require_uint()
{
  local name=$1
  local value=$2
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ${name} must be an unsigned integer, got ${value}" >&2
    exit 2
  fi
}

if [[ -z "${KVAF_PHYSICAL_DEVICE_ID}" ||
      -z "${KVAF_SUPER_DEVICE_ID}" ||
      -z "${KVAF_SERVER_INDEX}" ||
      -z "${KVAF_SUPER_POD_INDEX}" ]]; then
  topology_output=$(
    python3 "${SCRIPT_DIR}/query_a5_topology.py" \
      --logical-device-id "${KVAF_LOGICAL_DEVICE_ID}" \
      --shell
  )
  while IFS='=' read -r key value; do
    case "${key}" in
      KVAF_PHYSICAL_DEVICE_ID)
        KVAF_PHYSICAL_DEVICE_ID=${KVAF_PHYSICAL_DEVICE_ID:-${value}}
        ;;
      KVAF_SUPER_DEVICE_ID)
        KVAF_SUPER_DEVICE_ID=${KVAF_SUPER_DEVICE_ID:-${value}}
        ;;
      KVAF_SERVER_INDEX)
        KVAF_SERVER_INDEX=${KVAF_SERVER_INDEX:-${value}}
        ;;
      KVAF_SUPER_POD_INDEX)
        KVAF_SUPER_POD_INDEX=${KVAF_SUPER_POD_INDEX:-${value}}
        ;;
    esac
  done <<<"${topology_output}"
fi

for name in \
  KVAF_LOGICAL_DEVICE_ID \
  KVAF_PHYSICAL_DEVICE_ID \
  KVAF_SUPER_DEVICE_ID \
  KVAF_SERVER_INDEX \
  KVAF_SUPER_POD_INDEX \
  KVAF_H2D_BYTES \
  KVAF_H2D_TIMEOUT_SECONDS
do
  require_uint "${name}" "${!name}"
done
if (( KVAF_H2D_BYTES == 0 || KVAF_H2D_TIMEOUT_SECONDS == 0 )); then
  echo "ERROR: KVAF_H2D_BYTES and KVAF_H2D_TIMEOUT_SECONDS must be positive" >&2
  exit 2
fi
if (( KVAF_H2D_TIMEOUT_SECONDS > 4294967 )); then
  echo "ERROR: KVAF_H2D_TIMEOUT_SECONDS is too large for the C++ timeout" >&2
  exit 2
fi
if (( KVAF_H2D_BYTES > 4294967295 )); then
  echo "ERROR: KVAF_H2D_BYTES must fit the A5 URMA 32-bit SGE length" >&2
  exit 2
fi
if [[ -n "${KVAF_PAIR_CHANNEL}" ]]; then
  require_uint KVAF_PAIR_CHANNEL "${KVAF_PAIR_CHANNEL}"
fi

if { [[ -n "${KVAF_HOST_EID}" ]] && [[ -z "${KVAF_DEVICE_EID}" ]]; } ||
   { [[ -z "${KVAF_HOST_EID}" ]] && [[ -n "${KVAF_DEVICE_EID}" ]]; }; then
  echo "ERROR: set both KVAF_HOST_EID and KVAF_DEVICE_EID, or leave both empty for driver discovery" >&2
  exit 2
fi

if [[ -z "${KVAF_HOST_EID}" && -z "${KVAF_DEVICE_EID}" ]]; then
  resolver_args=(
    --physical-device-id "${KVAF_PHYSICAL_DEVICE_ID}"
    --proc-root "${KVAF_EID_PROC_ROOT}"
    --lock-file "${KVAF_EID_LOCK_FILE}"
    --lock-timeout-seconds "${KVAF_EID_LOCK_TIMEOUT_SECONDS}"
    --shell
  )
  if [[ -n "${KVAF_PAIR_CHANNEL}" ]]; then
    resolver_args+=(--pair-channel "${KVAF_PAIR_CHANNEL}")
  fi
  if ! eid_output=$(
    python3 "${SCRIPT_DIR}/resolve_a5_urma_eids.py" "${resolver_args[@]}"
  ); then
    echo "ERROR: automatic A5 Host/Device EID discovery failed" >&2
    exit 2
  fi

  seen_host_eid=0
  seen_device_eid=0
  seen_pair_channel=0
  while IFS='=' read -r key value; do
    case "${key}" in
      KVAF_HOST_EID)
        if (( seen_host_eid != 0 )); then
          echo "ERROR: EID resolver returned duplicate KVAF_HOST_EID" >&2
          exit 2
        fi
        KVAF_HOST_EID=${value}
        seen_host_eid=1
        ;;
      KVAF_DEVICE_EID)
        if (( seen_device_eid != 0 )); then
          echo "ERROR: EID resolver returned duplicate KVAF_DEVICE_EID" >&2
          exit 2
        fi
        KVAF_DEVICE_EID=${value}
        seen_device_eid=1
        ;;
      KVAF_PAIR_CHANNEL)
        if (( seen_pair_channel != 0 )); then
          echo "ERROR: EID resolver returned duplicate KVAF_PAIR_CHANNEL" >&2
          exit 2
        fi
        KVAF_PAIR_CHANNEL=${value}
        seen_pair_channel=1
        ;;
      *)
        echo "ERROR: EID resolver returned an unknown record: ${key}" >&2
        exit 2
        ;;
    esac
  done <<<"${eid_output}"
  if (( seen_host_eid == 0 ||
        seen_device_eid == 0 ||
        seen_pair_channel == 0 )); then
    echo "ERROR: EID resolver output is incomplete" >&2
    exit 2
  fi
  require_uint KVAF_PAIR_CHANNEL "${KVAF_PAIR_CHANNEL}"
  echo "KVAF_A5_EID_RESOLVED source=driver_pair_info physical_device=${KVAF_PHYSICAL_DEVICE_ID} channel=${KVAF_PAIR_CHANNEL}"
fi

normalize_eid()
{
  local value=${1//:/}
  value=${value//-/}
  value=$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')
  if ! [[ "${value}" =~ ^[0-9a-f]{32}$ ]] ||
     [[ "${value}" == "00000000000000000000000000000000" ]]; then
    return 1
  fi
  printf '%s\n' "${value}"
}

if ! host_eid=$(normalize_eid "${KVAF_HOST_EID}"); then
  echo "ERROR: KVAF_HOST_EID must be a non-zero 16-byte EID (32 hex digits)" >&2
  exit 2
fi
if ! device_eid=$(normalize_eid "${KVAF_DEVICE_EID}"); then
  echo "ERROR: KVAF_DEVICE_EID must be a non-zero 16-byte EID (32 hex digits)" >&2
  exit 2
fi
if [[ "${host_eid}" == "${device_eid}" ]]; then
  echo "ERROR: Host and Device EIDs must be the distinct driver pair" >&2
  exit 2
fi
if [[ ! -x "${KVAF_A5_BINARY}" ]]; then
  echo "ERROR: smoke binary is missing: ${KVAF_A5_BINARY}" >&2
  echo "Run scripts/build_a5_aiv_urma.sh first." >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "ERROR: GNU timeout is required for bounded hardware validation" >&2
  exit 2
fi
if [[ -e "${KVAF_A5_RESULT_DIR}" || -L "${KVAF_A5_RESULT_DIR}" ]]; then
  echo "ERROR: result directory already exists; refusing stale evidence: ${KVAF_A5_RESULT_DIR}" >&2
  exit 2
fi

mkdir -p "${KVAF_A5_RESULT_DIR}/exchange"
host_log="${KVAF_A5_RESULT_DIR}/host.log"
device_log="${KVAF_A5_RESULT_DIR}/device.log"
exchange_dir="${KVAF_A5_RESULT_DIR}/exchange"

common_args=(
  --host-eid "${host_eid}"
  --device-eid "${device_eid}"
  --exchange-dir "${exchange_dir}"
  --logical-device-id "${KVAF_LOGICAL_DEVICE_ID}"
  --physical-device-id "${KVAF_PHYSICAL_DEVICE_ID}"
  --super-device-id "${KVAF_SUPER_DEVICE_ID}"
  --server-index "${KVAF_SERVER_INDEX}"
  --super-pod-index "${KVAF_SUPER_POD_INDEX}"
  --bytes "${KVAF_H2D_BYTES}"
  --timeout-seconds "${KVAF_H2D_TIMEOUT_SECONDS}"
)
total_timeout_seconds=$((KVAF_H2D_TIMEOUT_SECONDS * 3 + 10))

host_pid=
cleanup()
{
  if [[ -n "${host_pid}" ]] && kill -0 "${host_pid}" 2>/dev/null; then
    kill "${host_pid}" 2>/dev/null || true
    wait "${host_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "PHASE=host_device_h2d"
echo "RESULT_DIR=${KVAF_A5_RESULT_DIR}"
echo "LOGICAL_DEVICE_ID=${KVAF_LOGICAL_DEVICE_ID}"
echo "PHYSICAL_DEVICE_ID=${KVAF_PHYSICAL_DEVICE_ID}"
echo "BYTES=${KVAF_H2D_BYTES}"

timeout --signal=TERM --kill-after=5s \
  "${total_timeout_seconds}s" \
  "${KVAF_A5_BINARY}" --role host "${common_args[@]}" \
  >"${host_log}" 2>&1 &
host_pid=$!

set +e
timeout --signal=TERM --kill-after=5s \
  "${total_timeout_seconds}s" \
  "${KVAF_A5_BINARY}" --role device "${common_args[@]}" \
  >"${device_log}" 2>&1
device_rc=$?
if (( device_rc != 0 )) && kill -0 "${host_pid}" 2>/dev/null; then
  kill "${host_pid}" 2>/dev/null || true
fi
wait "${host_pid}"
host_rc=$?
set -e
host_pid=

required_host_markers=(
  "KVAF_A5_DDR_REGISTERED"
  "KVAF_A5_CHANNEL_READY role=host engine=CPU"
  "KVAF_A5_TEARDOWN_OK role=host"
)
required_device_markers=(
  "KVAF_A5_DDR_IMPORTED"
  "KVAF_A5_CHANNEL_READY role=device engine=AIV"
  "KVAF_A5_DIRECT_URMA_READY"
  "KVAF_A5_H2D_SMOKE_OK"
  "KVAF_A5_TEARDOWN_OK role=device"
)

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
  echo "KVAF_A5_H2D_SMOKE_FAILED host_rc=${host_rc} device_rc=${device_rc}" >&2
  echo "HOST_LOG=${host_log}" >&2
  sed -n '1,240p' "${host_log}" >&2
  echo "DEVICE_LOG=${device_log}" >&2
  sed -n '1,240p' "${device_log}" >&2
  exit 1
fi

grep -E 'KVAF_A5_(DDR|CHANNEL|DIRECT|H2D|TEARDOWN)' \
  "${host_log}" "${device_log}"
echo "KVAF_A5_AIV_SIMT_URMA_H2D_ACCEPTANCE=PASS"
echo "RESULT_DIR=${KVAF_A5_RESULT_DIR}"
