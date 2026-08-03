#!/usr/bin/env bash

# Read-only capability discovery for a possible HCCP-style A5 control plane.
# It inspects exported symbols and performs syntax-only compilation against the
# installed driver header.  It never loads libascend_hal, calls a HAL function,
# opens a device node, creates a resource, or changes host/container state.

set -Eeuo pipefail
umask 077

if (( $# != 0 )); then
  echo "Usage: $0" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
PROBE_SOURCE=${ROOT_DIR}/scripts/probes/a5_hccp_hal_capability_probe.c
KVAF_PUBLIC_INCLUDE_DIR=${ROOT_DIR}/access_operator/a5/include
KVAF_PRIVATE_ABI_HEADER=${KVAF_PUBLIC_INCLUDE_DIR}/kvaf/access_operator/a5/hccp_hal_private_abi.h

kvaf_fail()
{
  echo "ERROR: A5 HCCP HAL capability preflight: $*" >&2
  echo "No HAL function or hardware operation was executed." >&2
  exit 2
}

for kvaf_command in awk dirname realpath sed; do
  command -v "${kvaf_command}" >/dev/null 2>&1 ||
    kvaf_fail "${kvaf_command} is required"
done

KVAF_HCCP_HAL_CC=${KVAF_HCCP_HAL_CC:-cc}
KVAF_HCCP_HAL_NM=${KVAF_HCCP_HAL_NM:-nm}
command -v "${KVAF_HCCP_HAL_CC}" >/dev/null 2>&1 ||
  kvaf_fail "C compiler is unavailable: ${KVAF_HCCP_HAL_CC}"
command -v "${KVAF_HCCP_HAL_NM}" >/dev/null 2>&1 ||
  kvaf_fail "dynamic-symbol reader is unavailable: ${KVAF_HCCP_HAL_NM}"

[[ -f "${PROBE_SOURCE}" && -r "${PROBE_SOURCE}" &&
   ! -L "${PROBE_SOURCE}" ]] ||
  kvaf_fail "compile-only probe source is missing or unsafe: ${PROBE_SOURCE}"
[[ -f "${KVAF_PRIVATE_ABI_HEADER}" && -r "${KVAF_PRIVATE_ABI_HEADER}" &&
   ! -L "${KVAF_PRIVATE_ABI_HEADER}" ]] ||
  kvaf_fail "source-audited HCCP private ABI header is missing or unsafe"

KVAF_ASCEND_DRIVER_ROOT=${KVAF_ASCEND_DRIVER_ROOT:-/usr/local/Ascend/driver}
[[ "${KVAF_ASCEND_DRIVER_ROOT}" == /* ]] ||
  kvaf_fail "KVAF_ASCEND_DRIVER_ROOT must be absolute"
[[ -d "${KVAF_ASCEND_DRIVER_ROOT}" ]] ||
  kvaf_fail "Ascend driver root is missing: ${KVAF_ASCEND_DRIVER_ROOT}"

kvaf_driver_root_real=$(realpath "${KVAF_ASCEND_DRIVER_ROOT}") ||
  kvaf_fail "cannot resolve Ascend driver root"

if [[ -n "${KVAF_HCCP_HAL_HEADER:-}" ]]; then
  kvaf_hal_header=${KVAF_HCCP_HAL_HEADER}
else
  declare -a kvaf_hal_header_candidates=()
  for kvaf_candidate in \
    "${KVAF_ASCEND_DRIVER_ROOT}/include/driver/ascend_hal.h" \
    "${KVAF_ASCEND_DRIVER_ROOT}/include/ascend_hal.h"
  do
    if [[ -f "${kvaf_candidate}" && -r "${kvaf_candidate}" ]]; then
      kvaf_candidate_real=$(realpath "${kvaf_candidate}") ||
        kvaf_fail "cannot resolve HAL header candidate: ${kvaf_candidate}"
      if (( ${#kvaf_hal_header_candidates[@]} == 0 )) ||
         [[ ${kvaf_hal_header_candidates[0]} != "${kvaf_candidate_real}" ]]; then
        kvaf_hal_header_candidates+=("${kvaf_candidate_real}")
      fi
    fi
  done
  (( ${#kvaf_hal_header_candidates[@]} <= 1 )) ||
    kvaf_fail \
      "multiple distinct ascend_hal.h candidates exist; select KVAF_HCCP_HAL_HEADER explicitly"
  kvaf_hal_header=${kvaf_hal_header_candidates[0]:-}
fi
[[ -n "${kvaf_hal_header}" && "${kvaf_hal_header}" == /* ]] ||
  kvaf_fail \
    "installed ascend_hal.h was not found; set KVAF_HCCP_HAL_HEADER explicitly"
[[ -f "${kvaf_hal_header}" && -r "${kvaf_hal_header}" ]] ||
  kvaf_fail "HAL header is missing or unreadable: ${kvaf_hal_header}"
kvaf_hal_header_real=$(realpath "${kvaf_hal_header}") ||
  kvaf_fail "cannot resolve HAL header: ${kvaf_hal_header}"

if [[ -n "${KVAF_HCCP_HAL_LIBRARY:-}" ]]; then
  kvaf_hal_library=${KVAF_HCCP_HAL_LIBRARY}
else
  declare -a kvaf_hal_library_candidates=()
  for kvaf_candidate in \
    "${KVAF_ASCEND_DRIVER_ROOT}/lib64/driver/libascend_hal.so" \
    "${KVAF_ASCEND_DRIVER_ROOT}/lib64/libascend_hal.so"
  do
    if [[ -f "${kvaf_candidate}" && -r "${kvaf_candidate}" ]]; then
      kvaf_candidate_real=$(realpath "${kvaf_candidate}") ||
        kvaf_fail "cannot resolve HAL library candidate: ${kvaf_candidate}"
      if (( ${#kvaf_hal_library_candidates[@]} == 0 )) ||
         [[ ${kvaf_hal_library_candidates[0]} != "${kvaf_candidate_real}" ]]; then
        kvaf_hal_library_candidates+=("${kvaf_candidate_real}")
      fi
    fi
  done
  (( ${#kvaf_hal_library_candidates[@]} <= 1 )) ||
    kvaf_fail \
      "multiple distinct libascend_hal.so candidates exist; select KVAF_HCCP_HAL_LIBRARY explicitly"
  kvaf_hal_library=${kvaf_hal_library_candidates[0]:-}
fi
[[ -n "${kvaf_hal_library}" && "${kvaf_hal_library}" == /* ]] ||
  kvaf_fail \
    "installed libascend_hal.so was not found; set KVAF_HCCP_HAL_LIBRARY explicitly"
[[ -f "${kvaf_hal_library}" && -r "${kvaf_hal_library}" ]] ||
  kvaf_fail "HAL library is missing or unreadable: ${kvaf_hal_library}"
kvaf_hal_library_real=$(realpath "${kvaf_hal_library}") ||
  kvaf_fail "cannot resolve HAL library: ${kvaf_hal_library}"

case "${kvaf_hal_header_real}" in
  "${kvaf_driver_root_real}"/*) ;;
  *) kvaf_fail "HAL header resolves outside KVAF_ASCEND_DRIVER_ROOT" ;;
esac
case "${kvaf_hal_library_real}" in
  "${kvaf_driver_root_real}"/*) ;;
  *) kvaf_fail "HAL library resolves outside KVAF_ASCEND_DRIVER_ROOT" ;;
esac

declare -a kvaf_include_args=()
kvaf_add_include_dir()
{
  local kvaf_dir=$1
  [[ -n "${kvaf_dir}" ]] || return 0
  [[ "${kvaf_dir}" == /* && -d "${kvaf_dir}" ]] ||
    kvaf_fail "HAL include directory is missing or not absolute: ${kvaf_dir}"
  kvaf_dir=$(realpath "${kvaf_dir}") ||
    kvaf_fail "cannot resolve HAL include directory: ${kvaf_dir}"
  case "${kvaf_dir}" in
    "${kvaf_driver_root_real}"|"${kvaf_driver_root_real}"/*) ;;
    *) kvaf_fail \
         "HAL include directory resolves outside KVAF_ASCEND_DRIVER_ROOT: ${kvaf_dir}" ;;
  esac
  kvaf_include_args+=("-I${kvaf_dir}")
}

kvaf_add_include_dir "${KVAF_ASCEND_DRIVER_ROOT}/include"
kvaf_add_include_dir "$(dirname "${kvaf_hal_header_real}")"
kvaf_include_args+=("-I${KVAF_PUBLIC_INCLUDE_DIR}")
if [[ -n "${KVAF_HCCP_HAL_INCLUDE_DIRS:-}" ]]; then
  IFS=: read -r -a kvaf_extra_include_dirs \
    <<<"${KVAF_HCCP_HAL_INCLUDE_DIRS}"
  for kvaf_include_dir in "${kvaf_extra_include_dirs[@]}"; do
    kvaf_add_include_dir "${kvaf_include_dir}"
  done
fi

echo "EXECUTION_ENV=CALLER_SELECTED_READ_ONLY_PREFLIGHT"
echo "KVAF_HCCP_HAL_HEADER=${kvaf_hal_header_real}"
echo "KVAF_HCCP_HAL_LIBRARY=${kvaf_hal_library_real}"
echo "KVAF_HCCP_HAL_PAIRING=co-located-under-selected-root-not-version-proof"
echo "KVAF_HCCP_HAL_OPERATION_MODE=compile-and-symbol-inspection-only"
echo "KVAF_HCCP_HAL_RUNTIME_LOAD_CONTRACT=absolute-path-required-not-yet-implemented"
echo "KVAF_HCCP_PRIVATE_PAYLOAD_PUBLIC_API=no"
echo "KVAF_HCCP_PRIVATE_PAYLOAD_SOURCE=HCOMM@2591d7ff74c18fb9829dd5c0c157f97558405169"
echo "KVAF_HCCP_PRIVATE_PAYLOAD_SOURCE_SHA256=9753b9de6dbd8106677672959ffb909708dcd36aab071baa6b67c0a548c5792d"

if kvaf_baseline_output=$(
  env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
    -u OBJC_INCLUDE_PATH \
    "${KVAF_HCCP_HAL_CC}" \
    -std=gnu11 -fsyntax-only \
    "${kvaf_include_args[@]}" \
    -include "${kvaf_hal_header_real}" \
    -DKVAF_HAL_PROBE_BASELINE=1 \
    "${PROBE_SOURCE}" 2>&1
); then
  echo "KVAF_HCCP_HAL_HEADER_BASELINE=usable"
else
  echo "KVAF_HCCP_HAL_HEADER_BASELINE=unusable"
  echo "KVAF_HCCP_HAL_HEADER_BASELINE_DIAGNOSTIC_BEGIN" >&2
  sed -n '1,12p' <<<"${kvaf_baseline_output}" >&2
  echo "KVAF_HCCP_HAL_HEADER_BASELINE_DIAGNOSTIC_END" >&2
  kvaf_fail "installed ascend_hal.h cannot compile in isolation"
fi

kvaf_missing=0
kvaf_nm_output=$(
  LC_ALL=C "${KVAF_HCCP_HAL_NM}" -D --defined-only -- \
    "${kvaf_hal_library_real}" 2>&1
) || kvaf_fail "could not read dynamic symbols from ${kvaf_hal_library_real}"

declare -a kvaf_library_symbols=(
  drvDeviceGetIndexByPhyId
  halQueryDevpid
  halMemBindSibling
  halResAddrMapV2
  halResAddrUnmapV2
)
for kvaf_symbol in "${kvaf_library_symbols[@]}"; do
  if awk -v wanted="${kvaf_symbol}" '
      {
        name = $NF
        sub(/@.*/, "", name)
        if (name == wanted) found = 1
      }
      END { exit !found }
    ' <<<"${kvaf_nm_output}"
  then
    echo "KVAF_HCCP_HAL_LIBRARY_SYMBOL_${kvaf_symbol}=available"
  else
    echo "KVAF_HCCP_HAL_LIBRARY_SYMBOL_${kvaf_symbol}=missing"
    kvaf_missing=$((kvaf_missing + 1))
  fi
done

declare -a kvaf_header_probes=(
  "drvDeviceGetIndexByPhyId:KVAF_HAL_PROBE_DRV_DEVICE_GET_INDEX_BY_PHY_ID"
  "halQueryDevpid:KVAF_HAL_PROBE_QUERY_DEVPID"
  "halMemBindSibling:KVAF_HAL_PROBE_MEM_BIND_SIBLING"
  "halResAddrMapV2:KVAF_HAL_PROBE_RES_ADDR_MAP_V2"
  "halResAddrUnmapV2:KVAF_HAL_PROBE_RES_ADDR_UNMAP_V2"
  "res_map_info_in:KVAF_HAL_PROBE_RES_MAP_INFO_IN"
  "res_map_info_out:KVAF_HAL_PROBE_RES_MAP_INFO_OUT"
  "RES_ADDR_TYPE_HCCP_URMA_JFC:KVAF_HAL_PROBE_RES_TYPE_HCCP_URMA_JFC"
  "RES_ADDR_TYPE_HCCP_URMA_JETTY:KVAF_HAL_PROBE_RES_TYPE_HCCP_URMA_JETTY"
  "RES_ADDR_TYPE_HCCP_URMA_DB:KVAF_HAL_PROBE_RES_TYPE_HCCP_URMA_DB"
  "UdmaVaInfo_private_payload:KVAF_HAL_PROBE_UDMA_VA_INFO_PRIVATE_ABI"
)
for kvaf_probe_entry in "${kvaf_header_probes[@]}"; do
  kvaf_name=${kvaf_probe_entry%%:*}
  kvaf_selector=${kvaf_probe_entry#*:}
  if kvaf_compile_output=$(
    env -u CPATH -u C_INCLUDE_PATH -u CPLUS_INCLUDE_PATH \
      -u OBJC_INCLUDE_PATH \
      "${KVAF_HCCP_HAL_CC}" \
      -std=gnu11 -fsyntax-only \
      "${kvaf_include_args[@]}" \
      -include "${kvaf_hal_header_real}" \
      "-D${kvaf_selector}=1" \
      "${PROBE_SOURCE}" 2>&1
  ); then
    echo "KVAF_HCCP_HAL_HEADER_CAPABILITY_${kvaf_name}=available"
  else
    echo "KVAF_HCCP_HAL_HEADER_CAPABILITY_${kvaf_name}=missing"
    echo "KVAF_HCCP_HAL_HEADER_DIAGNOSTIC_${kvaf_name}_BEGIN" >&2
    sed -n '1,6p' <<<"${kvaf_compile_output}" >&2
    echo "KVAF_HCCP_HAL_HEADER_DIAGNOSTIC_${kvaf_name}_END" >&2
    kvaf_missing=$((kvaf_missing + 1))
  fi
done

if (( kvaf_missing != 0 )); then
  echo "KVAF_HCCP_HAL_SURFACE=incomplete"
  echo "KVAF_HCCP_HAL_MISSING_COUNT=${kvaf_missing}"
  echo "KVAF_HCCP_HAL_RUNTIME_SEMANTICS=not-tested"
  echo "KVAF_HCCP_PRIVATE_PAYLOAD_ABI=source-audited-layout-incompatible-with-selected-public-header"
  echo "KVAF_HCCP_BACKEND_READY=no"
  echo "No HAL function or hardware operation was executed." >&2
  exit 2
fi

echo "KVAF_HCCP_HAL_SURFACE=declarations-complete"
echo "KVAF_HCCP_HAL_MISSING_COUNT=0"
echo "KVAF_HCCP_HAL_RUNTIME_SEMANTICS=unproven-not-called"
echo "KVAF_HCCP_HAL_ABI_COMPATIBILITY=unproven-by-surface-check"
echo "KVAF_HCCP_PRIVATE_PAYLOAD_ABI=source-audited-hcomm-v1-runtime-unproven"
echo "KVAF_HCCP_PRIVATE_LAYOUT=compile-time-size-and-offsets-verified-runtime-unproven"
echo "KVAF_HCCP_RESOURCE_ID_AND_UASID_SEMANTICS=unproven"
echo "KVAF_HCCP_BACKEND_READY=no"
echo "KVAF_A5_HCCP_HAL_DECLARATION_PREFLIGHT_OK"
