#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

: "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"
: "${KVAF_HOST_URMA_DEVICE:?set the Host-side UDMA device name}"
: "${KVAF_DEVICE_URMA_DEVICE:?set the AICPU-side UDMA device name}"
: "${KVAF_HOST_EID_INDEX:?set the Host-side UDMA EID index}"
: "${KVAF_DEVICE_EID_INDEX:?set the AICPU-side UDMA EID index}"

KVAF_A5_DIRECT_BUILD_DIR=${KVAF_A5_DIRECT_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma"}
KVAF_A5_DIRECT_BINARY=${KVAF_A5_DIRECT_BINARY:-"${KVAF_A5_DIRECT_BUILD_DIR}/access_operator/a5/kvaf_a5_aiv_direct_urma_h2d_smoke"}
KVAF_DIRECT_PROVIDER_BUILD_DIR=${KVAF_DIRECT_PROVIDER_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma-provider"}
KVAF_DIRECT_HOST_PROVIDER_LIBRARY=${KVAF_DIRECT_HOST_PROVIDER_LIBRARY:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/libkvaf_urma_provider_host.so"}
KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY=${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/libkvaf_urma_provider_device.so"}
KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST=${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/source.sha256"}
KVAF_DIRECT_PROVIDER_BUILD_MANIFEST=${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/build.manifest"}
KVAF_DIRECT_PROVIDER_SUCCESS_STAMP=${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/build.success"}
KVAF_DIRECT_PROVIDER_SONAME=${KVAF_DIRECT_PROVIDER_SONAME:-libkvaf_urma_provider_device.so}
KVAF_DIRECT_AICPU_ARCHIVE=${KVAF_DIRECT_AICPU_ARCHIVE:-"${ASCEND_HOME_PATH}/compat/liburma_perftest.tar.gz"}
KVAF_HOST_EID=${KVAF_HOST_EID:-}
KVAF_DEVICE_EID=${KVAF_DEVICE_EID:-}
KVAF_PAIR_CHANNEL=${KVAF_PAIR_CHANNEL:-}
KVAF_EID_PROC_ROOT=${KVAF_EID_PROC_ROOT:-/proc/asdrv_ub}
KVAF_EID_LOCK_FILE=${KVAF_EID_LOCK_FILE:-/run/lock/kvaf_asdrv_ub_pair.lock}
KVAF_EID_LOCK_TIMEOUT_SECONDS=${KVAF_EID_LOCK_TIMEOUT_SECONDS:-10}
KVAF_LOGICAL_DEVICE_ID=${KVAF_LOGICAL_DEVICE_ID:-0}
KVAF_PHYSICAL_DEVICE_ID=${KVAF_PHYSICAL_DEVICE_ID:-}
KVAF_DIRECT_CONTROL_PORT=${KVAF_DIRECT_CONTROL_PORT:-21115}
KVAF_H2D_BYTES=${KVAF_H2D_BYTES:-4096}
KVAF_H2D_TIMEOUT_SECONDS=${KVAF_H2D_TIMEOUT_SECONDS:-120}
KVAF_DIRECT_SERVER_START_DELAY_SECONDS=${KVAF_DIRECT_SERVER_START_DELAY_SECONDS:-1}
KVAF_A5_RUN_TAG=${KVAF_A5_RUN_TAG:-$(date +%Y%m%d_%H%M%S)_$$}
KVAF_A5_RESULT_DIR=${KVAF_A5_RESULT_DIR:-"${ROOT_DIR}/results/a5_direct_urma_h2d_${KVAF_A5_RUN_TAG}"}
KVAF_DIRECT_DEPLOY_RUN_LOCK=${KVAF_DIRECT_DEPLOY_RUN_LOCK:-/run/lock/kvaf-a5-direct-urma.lock}
KVAF_DIRECT_QUARANTINE_FILE=${KVAF_DIRECT_QUARANTINE_FILE:-/run/lock/kvaf-a5-direct-urma.quarantine}

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

verify_runtime_manifest()
{
  local prefix path expected actual
  for prefix in urma_library urma_common_library udma_provider_library; do
    path=$(manifest_value "${prefix}_path" \
      "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") ||
      return 1
    expected=$(manifest_value "${prefix}_sha256" \
      "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") ||
      return 1
    [[ -r "${path}" ]] || return 1
    actual=$(sha256sum "${path}" | awk '{print $1}')
    [[ "${actual}" == "${expected}" ]] || return 1
  done
}

verify_provider_role_manifest()
{
  local key=$1
  local expected=$2
  local actual

  actual=$(manifest_value "${key}" \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  [[ "${actual}" == "${expected}" ]]
}

ldd_resolved_path()
{
  local soname=$1
  awk -v soname="${soname}" '
    ($1 == soname || index($1, soname ".") == 1) && $2 == "=>" {
      path = $3
      count++
    }
    END {
      if (count != 1 || path == "") {
        exit 1
      }
      print path
    }
  '
}

verify_provider_runtime_resolution()
{
  local label=$1
  local provider=$2
  local ldd_output resolved_urma resolved_common
  local actual_urma actual_common expected_urma expected_common
  local expected_urma_sha expected_common_sha
  local actual_urma_sha actual_common_sha

  if ! ldd_output=$(ldd -r "${provider}" 2>&1); then
    echo "${ldd_output}" >&2
    echo "ERROR: ldd could not load the ${label} provider" >&2
    return 1
  fi
  if grep -Eq 'not found|undefined symbol' <<<"${ldd_output}"; then
    echo "${ldd_output}" >&2
    echo "ERROR: ${label} provider has unresolved runtime dependencies" >&2
    return 1
  fi
  resolved_urma=$(ldd_resolved_path liburma.so \
    <<<"${ldd_output}") || {
    echo "ERROR: ldd did not resolve exactly one ${label} liburma" >&2
    return 1
  }
  resolved_common=$(ldd_resolved_path liburma_common.so \
    <<<"${ldd_output}") || {
    echo "ERROR: ldd did not resolve exactly one ${label} liburma_common" >&2
    return 1
  }
  actual_urma=$(realpath -e "${resolved_urma}") || return 1
  actual_common=$(realpath -e "${resolved_common}") || return 1
  expected_urma=$(manifest_value urma_library_path \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  expected_common=$(manifest_value urma_common_library_path \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  expected_urma_sha=$(manifest_value urma_library_sha256 \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  expected_common_sha=$(manifest_value urma_common_library_sha256 \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  actual_urma_sha=$(sha256sum "${actual_urma}" | awk '{print $1}') ||
    return 1
  actual_common_sha=$(sha256sum "${actual_common}" | awk '{print $1}') ||
    return 1
  if [[ "${actual_urma}" != "${expected_urma}" ||
        "${actual_common}" != "${expected_common}" ||
        "${actual_urma_sha}" != "${expected_urma_sha}" ||
        "${actual_common_sha}" != "${expected_common_sha}" ]]; then
    echo "ERROR: ${label} provider runtime resolution differs from the build manifest" >&2
    echo "RESOLVED_LIBURMA=${actual_urma}" >&2
    echo "RESOLVED_LIBURMA_COMMON=${actual_common}" >&2
    return 1
  fi
}

require_uint_max KVAF_LOGICAL_DEVICE_ID \
  "${KVAF_LOGICAL_DEVICE_ID}" 4294967295
require_uint_max KVAF_HOST_EID_INDEX \
  "${KVAF_HOST_EID_INDEX}" 4294967295
require_uint_max KVAF_DEVICE_EID_INDEX \
  "${KVAF_DEVICE_EID_INDEX}" 4294967295
require_uint_max KVAF_DIRECT_CONTROL_PORT \
  "${KVAF_DIRECT_CONTROL_PORT}" 65535
require_uint_max KVAF_H2D_BYTES "${KVAF_H2D_BYTES}" 4294967231
require_uint_max KVAF_H2D_TIMEOUT_SECONDS \
  "${KVAF_H2D_TIMEOUT_SECONDS}" 4294967
require_uint KVAF_EID_LOCK_TIMEOUT_SECONDS \
  "${KVAF_EID_LOCK_TIMEOUT_SECONDS}"
require_uint KVAF_DIRECT_SERVER_START_DELAY_SECONDS \
  "${KVAF_DIRECT_SERVER_START_DELAY_SECONDS}"
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
if (( KVAF_H2D_BYTES == 0 ||
      KVAF_H2D_BYTES > 4294967231 ||
      KVAF_H2D_TIMEOUT_SECONDS == 0 ||
      KVAF_H2D_TIMEOUT_SECONDS > 4294967 )); then
  echo "ERROR: invalid H2D byte count or timeout" >&2
  exit 2
fi
if [[ "${KVAF_HOST_URMA_DEVICE}" == */* ||
      "${KVAF_DEVICE_URMA_DEVICE}" == */* ||
      ${#KVAF_HOST_URMA_DEVICE} -ge 32 ||
      ${#KVAF_DEVICE_URMA_DEVICE} -ge 32 ]]; then
  echo "ERROR: UDMA device names must be basenames shorter than 32 bytes" >&2
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

if { [[ -n "${KVAF_HOST_EID}" ]] && [[ -z "${KVAF_DEVICE_EID}" ]]; } ||
   { [[ -z "${KVAF_HOST_EID}" ]] && [[ -n "${KVAF_DEVICE_EID}" ]]; }; then
  echo "ERROR: set both EIDs, or leave both empty for driver discovery" >&2
  exit 2
fi

if [[ -z "${KVAF_HOST_EID}" ]]; then
  if [[ -z "${KVAF_PHYSICAL_DEVICE_ID}" ]]; then
    topology_output=$(
      python3 "${SCRIPT_DIR}/query_a5_topology.py" \
        --logical-device-id "${KVAF_LOGICAL_DEVICE_ID}" \
        --shell
    )
    while IFS='=' read -r key value; do
      if [[ "${key}" == KVAF_PHYSICAL_DEVICE_ID ]]; then
        KVAF_PHYSICAL_DEVICE_ID=${value}
      fi
    done <<<"${topology_output}"
  fi
  require_uint_max KVAF_PHYSICAL_DEVICE_ID \
    "${KVAF_PHYSICAL_DEVICE_ID}" 4294967295

  resolver_args=(
    --physical-device-id "${KVAF_PHYSICAL_DEVICE_ID}"
    --proc-root "${KVAF_EID_PROC_ROOT}"
    --lock-file "${KVAF_EID_LOCK_FILE}"
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

if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock is required to serialize the singleton direct provider" >&2
  exit 2
fi
mkdir -p "$(dirname "${KVAF_DIRECT_DEPLOY_RUN_LOCK}")"
exec 9>"${KVAF_DIRECT_DEPLOY_RUN_LOCK}"
if ! flock -w "${KVAF_EID_LOCK_TIMEOUT_SECONDS}" 9; then
  echo "ERROR: another direct URMA deploy/run owns ${KVAF_DIRECT_DEPLOY_RUN_LOCK}" >&2
  exit 2
fi

if [[ -L "${KVAF_DIRECT_QUARANTINE_FILE}" ]]; then
  echo "ERROR: quarantine path must not be a symlink: ${KVAF_DIRECT_QUARANTINE_FILE}" >&2
  exit 2
fi
if [[ -r "${KVAF_DIRECT_QUARANTINE_FILE}" ]]; then
  quarantine_live=0
  while IFS='=' read -r quarantine_key quarantine_value; do
    case "${quarantine_key}" in
      host_pid|device_pid)
        if [[ "${quarantine_value}" =~ ^[1-9][0-9]*$ ]] &&
           kill -0 "${quarantine_value}" 2>/dev/null; then
          quarantine_live=1
        fi
        ;;
    esac
  done <"${KVAF_DIRECT_QUARANTINE_FILE}"
  if (( quarantine_live != 0 )); then
    echo "ERROR: an unsafe direct URMA session is quarantined" >&2
    cat "${KVAF_DIRECT_QUARANTINE_FILE}" >&2
    echo "Reset or hardware-isolate the selected NPU before terminating those PIDs." >&2
    exit 4
  fi
  echo "ERROR: a previous unsafe direct URMA session remains quarantined" >&2
  cat "${KVAF_DIRECT_QUARANTINE_FILE}" >&2
  echo "Dead PIDs do not prove that DMA has stopped. Reset or hardware-isolate" >&2
  echo "the selected NPU first, then remove ${KVAF_DIRECT_QUARANTINE_FILE}." >&2
  exit 4
fi

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

for command in awk ldd nm readelf realpath sha256sum tar; do
  command -v "${command}" >/dev/null 2>&1 || {
    echo "ERROR: ${command} is required" >&2
    exit 2
  }
done

if [[ ! -x "${KVAF_A5_DIRECT_BINARY}" ||
      ! -r "${KVAF_DIRECT_HOST_PROVIDER_LIBRARY}" ||
      ! -r "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}" ||
      ! -r "${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST}" ||
      ! -r "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" ||
      ! -r "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}" ]]; then
  echo "ERROR: build the direct smoke and source-compiled provider first" >&2
  exit 2
fi
host_provider_sha=$(
  sha256sum "${KVAF_DIRECT_HOST_PROVIDER_LIBRARY}" | awk '{print $1}'
)
device_provider_sha=$(
  sha256sum "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}" | awk '{print $1}'
)
source_manifest_sha=$(
  sha256sum "${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST}" |
    awk '{print $1}'
)
build_manifest_sha=$(
  sha256sum "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" |
    awk '{print $1}'
)
if [[ "$(manifest_value host_provider_sha256 \
        "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}")" != \
        "${host_provider_sha}" ||
      "$(manifest_value device_provider_sha256 \
        "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}")" != \
        "${device_provider_sha}" ||
      "$(manifest_value build_manifest_sha256 \
        "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}")" != \
        "${build_manifest_sha}" ||
      "$(manifest_value host_provider_sha256 \
        "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}")" != \
        "${host_provider_sha}" ||
      "$(manifest_value device_provider_sha256 \
        "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}")" != \
      "${device_provider_sha}" ||
      "${host_provider_sha}" == "${device_provider_sha}" ||
      "$(manifest_value source_manifest_sha256 \
        "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}")" != \
        "${source_manifest_sha}" ]]; then
  echo "ERROR: provider atomic build provenance is inconsistent" >&2
  exit 2
fi
for role_binding in \
  'host_provider_filename=libkvaf_urma_provider_host.so' \
  'host_provider_execution_domain=host-process' \
  'host_private_abi=1' \
  'host_private_abi_name=perftest-aiv-v1-host' \
  'device_provider_filename=libkvaf_urma_provider_device.so' \
  'device_provider_execution_domain=aicpu' \
  'device_private_abi=2' \
  'device_private_abi_name=perftest-aiv-v1-device' \
  'reference_perftest_abi=perftest-aiv-v1'
do
  role_key=${role_binding%%=*}
  role_value=${role_binding#*=}
  if ! verify_provider_role_manifest "${role_key}" "${role_value}"; then
    echo "ERROR: provider role manifest is inconsistent: ${role_binding}" >&2
    exit 2
  fi
done
if ! sha256sum -c "${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST}" \
    >/dev/null; then
  echo "ERROR: direct provider source changed after its atomic build" >&2
  exit 2
fi
if ! verify_runtime_manifest; then
  echo "ERROR: pinned liburma/liburma_common/liburma-udma changed" >&2
  exit 2
fi
if [[ ! -r "${KVAF_DIRECT_AICPU_ARCHIVE}" ]]; then
  echo "ERROR: AICPU provider archive is missing: ${KVAF_DIRECT_AICPU_ARCHIVE}" >&2
  exit 2
fi
expected_archive="${ASCEND_HOME_PATH}/compat/liburma_perftest.tar.gz"
if [[ "$(realpath -m "${KVAF_DIRECT_AICPU_ARCHIVE}")" != \
      "$(realpath -m "${expected_archive}")" ]]; then
  echo "ERROR: the runtime AICPU archive must be ${expected_archive}" >&2
  echo "CANN does not load an arbitrary same-content archive from another directory." >&2
  exit 2
fi
archive_entries=$(tar -tzf "${KVAF_DIRECT_AICPU_ARCHIVE}")
for entry in \
  aicpu_kernels_device/libkvaf_urma_provider_device.so \
  aicpu_kernels_device/kvaf-direct-urma-provider.manifest \
  aicpu_kernels_device/kvaf-direct-urma-source.sha256
do
  if ! grep -Fxq "${entry}" <<<"${archive_entries}"; then
    echo "ERROR: AICPU archive does not contain ${entry}" >&2
    exit 2
  fi
done
if grep -Ev \
    '^aicpu_kernels_device/?$|^aicpu_kernels_device/(libkvaf_urma_provider_device\.so|kvaf-direct-urma-provider\.manifest|kvaf-direct-urma-source\.sha256)$' \
    <<<"${archive_entries}" |
    grep -q .; then
  echo "ERROR: AICPU archive contains a stale or unexpected entry" >&2
  exit 2
fi
archive_provider_sha=$(
  tar -xOzf "${KVAF_DIRECT_AICPU_ARCHIVE}" \
    aicpu_kernels_device/libkvaf_urma_provider_device.so |
    sha256sum |
    awk '{print $1}'
)
archive_build_manifest_sha=$(
  tar -xOzf "${KVAF_DIRECT_AICPU_ARCHIVE}" \
    aicpu_kernels_device/kvaf-direct-urma-provider.manifest |
    sha256sum |
    awk '{print $1}'
)
archive_source_manifest_sha=$(
  tar -xOzf "${KVAF_DIRECT_AICPU_ARCHIVE}" \
    aicpu_kernels_device/kvaf-direct-urma-source.sha256 |
    sha256sum |
    awk '{print $1}'
)
if [[ "${archive_provider_sha}" != "${device_provider_sha}" ||
      "${archive_build_manifest_sha}" != "${build_manifest_sha}" ||
      "${archive_source_manifest_sha}" != "${source_manifest_sha}" ]]; then
  echo "ERROR: AICPU provider or provenance does not match the build" >&2
  echo "Re-run build_a5_direct_urma_provider.sh and package_a5_direct_urma_aicpu.sh." >&2
  exit 2
fi
if command -v ss >/dev/null 2>&1; then
  listening_addresses=$(
    ss -H -ltn |
      awk '{print $4}'
  )
  if grep -Eq ":${KVAF_DIRECT_CONTROL_PORT}$" \
      <<<"${listening_addresses}"; then
    echo "ERROR: control port ${KVAF_DIRECT_CONTROL_PORT} is already listening" >&2
    exit 2
  fi
fi

binary_dynamic=$(readelf -d "${KVAF_A5_DIRECT_BINARY}")
for provider in \
  "${KVAF_DIRECT_HOST_PROVIDER_LIBRARY}" \
  "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}"
do
  provider_dynamic=$(readelf -d "${provider}")
  if grep -Eq 'lib(hcomm|urma_perftest)' <<<"${provider_dynamic}" ||
     grep -Eq '\((RPATH|RUNPATH)\)' <<<"${provider_dynamic}"; then
    echo "ERROR: provider links HCOMM/prebuilt perftest or has RPATH" >&2
    exit 2
  fi
  provider_symbols=$(nm -D --defined-only "${provider}")
  for symbol in \
    kvaf_urma_provider_open_v2 \
    kvaf_urma_provider_close_v2 \
    kvaf_urma_provider_private_abi_v2 \
    aicpu_kvaf_urma_open_v2 \
    aicpu_kvaf_urma_close_v2
  do
    grep -Eq \
      "[[:space:]]${symbol}(@@KVAF_URMA_PROVIDER_2)?$" \
      <<<"${provider_symbols}" || {
      echo "ERROR: provider does not export ${symbol}: ${provider}" >&2
      exit 2
    }
  done
done
verify_provider_runtime_resolution \
  host "${KVAF_DIRECT_HOST_PROVIDER_LIBRARY}" || exit 2
verify_provider_runtime_resolution \
  device "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}" || exit 2
if grep -Fq libhcomm <<<"${binary_dynamic}"; then
  echo "ERROR: direct smoke unexpectedly links HCOMM" >&2
  exit 2
fi

if [[ -e "${KVAF_A5_RESULT_DIR}" ||
      -L "${KVAF_A5_RESULT_DIR}" ]]; then
  echo "ERROR: refusing an existing result directory: ${KVAF_A5_RESULT_DIR}" >&2
  exit 2
fi
mkdir -p "${KVAF_A5_RESULT_DIR}/exchange"
host_log="${KVAF_A5_RESULT_DIR}/host.log"
device_log="${KVAF_A5_RESULT_DIR}/device.log"
exchange_dir="${KVAF_A5_RESULT_DIR}/exchange"
manifest="${KVAF_A5_RESULT_DIR}/manifest.txt"

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
  printf 'ascend_home_path=%s\n' "${ASCEND_HOME_PATH}"
  printf 'logical_device_id=%s\n' "${KVAF_LOGICAL_DEVICE_ID}"
  printf 'physical_device_id=%s\n' "${KVAF_PHYSICAL_DEVICE_ID}"
  printf 'pair_channel=%s\n' "${KVAF_PAIR_CHANNEL}"
  printf 'host_eid=%s\n' "${host_eid}"
  printf 'device_eid=%s\n' "${device_eid}"
  printf 'host_urma_device=%s\n' "${KVAF_HOST_URMA_DEVICE}"
  printf 'host_eid_index=%s\n' "${KVAF_HOST_EID_INDEX}"
  printf 'device_urma_device=%s\n' "${KVAF_DEVICE_URMA_DEVICE}"
  printf 'device_eid_index=%s\n' "${KVAF_DEVICE_EID_INDEX}"
  printf 'control_port=%s\n' "${KVAF_DIRECT_CONTROL_PORT}"
  printf 'h2d_bytes=%s\n' "${KVAF_H2D_BYTES}"
  printf 'binary_sha256=%s\n' "$(
    sha256sum "${KVAF_A5_DIRECT_BINARY}" | awk '{print $1}'
  )"
  printf 'host_provider_sha256=%s\n' "${host_provider_sha}"
  printf 'device_provider_sha256=%s\n' "${device_provider_sha}"
  printf 'provider_build_manifest_sha256=%s\n' \
    "${build_manifest_sha}"
  printf 'provider_source_manifest_sha256=%s\n' \
    "${source_manifest_sha}"
  printf 'aicpu_archive_sha256=%s\n' "${archive_sha}"
  printf 'udma_sq_reserved=%s\n' "${kvaf_sq_reserved}"
} >"${manifest}"
cp "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" \
  "${KVAF_A5_RESULT_DIR}/provider-build.manifest"
cp "${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST}" \
  "${KVAF_A5_RESULT_DIR}/provider-source.sha256"

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
  --logical-device-id "${KVAF_LOGICAL_DEVICE_ID}"
  --control-port "${KVAF_DIRECT_CONTROL_PORT}"
  --bytes "${KVAF_H2D_BYTES}"
  --timeout-seconds "${KVAF_H2D_TIMEOUT_SECONDS}"
)
total_timeout_seconds=$((KVAF_H2D_TIMEOUT_SECONDS * 3 + 10))

host_pid=
device_pid=
run_completed=0

record_quarantine()
{
  local reason=$1
  local temporary="${KVAF_DIRECT_QUARANTINE_FILE}.tmp.$$"
  {
    printf 'reason=%s\n' "${reason}"
    printf 'logical_device_id=%s\n' "${KVAF_LOGICAL_DEVICE_ID}"
    printf 'host_pid=%s\n' "${host_pid:-0}"
    printf 'device_pid=%s\n' "${device_pid:-0}"
    printf 'result_dir=%s\n' "${KVAF_A5_RESULT_DIR}"
    printf 'created_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } >"${temporary}"
  chmod 0600 "${temporary}"
  mv -f "${temporary}" "${KVAF_DIRECT_QUARANTINE_FILE}"
  echo "KVAF_A5_DIRECT_URMA_QUARANTINED reason=${reason}" >&2
  echo "QUARANTINE_FILE=${KVAF_DIRECT_QUARANTINE_FILE}" >&2
  echo "Do not kill the recorded processes until the selected NPU is reset or hardware-isolated." >&2
}

cleanup_on_exit()
{
  local rc=$?
  if (( run_completed == 0 )); then
    if [[ -n "${host_pid}" || -n "${device_pid}" ]]; then
      set +e
      record_quarantine "runner_exit_rc_${rc}"
      set -e
    fi
  fi
}
trap cleanup_on_exit EXIT

echo "PHASE=direct_urma_host_ddr_to_aiv_hbm"
echo "RESULT_DIR=${KVAF_A5_RESULT_DIR}"
echo "LOGICAL_DEVICE_ID=${KVAF_LOGICAL_DEVICE_ID}"
echo "HOST_URMA=${KVAF_HOST_URMA_DEVICE}/eid${KVAF_HOST_EID_INDEX}"
echo "DEVICE_URMA=${KVAF_DEVICE_URMA_DEVICE}/eid${KVAF_DEVICE_EID_INDEX}"
echo "CONTROL_PORT=${KVAF_DIRECT_CONTROL_PORT}"
echo "MANIFEST=${manifest}"

nohup "${KVAF_A5_DIRECT_BINARY}" --role host "${common_args[@]}" \
  >"${host_log}" 2>&1 &
host_pid=$!

sleep "${KVAF_DIRECT_SERVER_START_DELAY_SECONDS}"
if ! kill -0 "${host_pid}" 2>/dev/null; then
  wait "${host_pid}" || true
  record_quarantine "host_exited_before_device_start"
  run_completed=1
  trap - EXIT
  sed -n '1,240p' "${host_log}" >&2
  echo "ERROR: Host direct provider exited before Device startup" >&2
  exit 4
fi

nohup "${KVAF_A5_DIRECT_BINARY}" --role device "${common_args[@]}" \
  >"${device_log}" 2>&1 &
device_pid=$!

deadline=$(( $(date +%s) + total_timeout_seconds ))
while kill -0 "${host_pid}" 2>/dev/null ||
      kill -0 "${device_pid}" 2>/dev/null
do
  if [[ -r "${exchange_dir}/fatal.host" ||
        -r "${exchange_dir}/fatal.device" ]]; then
    record_quarantine "unsafe_provider_teardown"
    run_completed=1
    trap - EXIT
    exit 4
  fi
  if (( $(date +%s) >= deadline )); then
    record_quarantine "bounded_wait_expired"
    run_completed=1
    trap - EXIT
    exit 4
  fi
  sleep 1
done

set +e
wait "${host_pid}"
host_rc=$?
wait "${device_pid}"
device_rc=$?
set -e

required_host_markers=(
  "KVAF_A5_DIRECT_DDR_REGISTERED"
  "KVAF_A5_DIRECT_URMA_READY role=host"
  "KVAF_A5_DIRECT_TEARDOWN_OK role=host"
)
required_device_markers=(
  "KVAF_A5_DIRECT_URMA_READY role=device"
  "KVAF_A5_DIRECT_URMA_CHANNEL"
  "KVAF_A5_DIRECT_H2D_SMOKE_OK"
  "KVAF_A5_DIRECT_TEARDOWN_OK role=device"
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

if (( host_rc != 0 || device_rc != 0 ||
      evidence_ok == 0 )); then
  record_quarantine \
    "process_or_teardown_evidence_failed_host_${host_rc}_device_${device_rc}"
  run_completed=1
  trap - EXIT
  echo "KVAF_A5_DIRECT_URMA_H2D_FAILED host_rc=${host_rc} device_rc=${device_rc}" >&2
  echo "HOST_LOG=${host_log}" >&2
  sed -n '1,260p' "${host_log}" >&2
  echo "DEVICE_LOG=${device_log}" >&2
  sed -n '1,320p' "${device_log}" >&2
  exit 4
fi

host_pid=
device_pid=
run_completed=1
trap - EXIT
grep -E 'KVAF_A5_(DIRECT|EID)' "${host_log}" "${device_log}"
echo "KVAF_A5_AIV_SIMT_DIRECT_URMA_H2D_ACCEPTANCE=PASS"
echo "RESULT_DIR=${KVAF_A5_RESULT_DIR}"
