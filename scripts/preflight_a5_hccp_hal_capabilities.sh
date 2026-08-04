#!/usr/bin/env bash

# Read-only capability inventory for an HCCP-owned A5 control plane.
#
# The preferred backend is libra.so: HCCP owns Context/MR/CQ/QP/TP lifetime
# and returns the mapped SQ/CQ/doorbell addresses consumed by AIV.  Direct
# private HAL mapping is retained only as a separately reported fallback.
# This script never loads a library, invokes HCCP/HAL, opens a device node,
# creates a resource, or mutates CANN registration/state.

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
  echo "ERROR: A5 HCCP control-plane static preflight: $*" >&2
  echo "No HCCP/HAL function or hardware operation was executed." >&2
  exit 2
}

for kvaf_command in awk dirname realpath sed tr; do
  command -v "${kvaf_command}" >/dev/null 2>&1 ||
    kvaf_fail "${kvaf_command} is required"
done

KVAF_HCCP_HAL_CC=${KVAF_HCCP_HAL_CC:-cc}
KVAF_HCCP_HAL_NM=${KVAF_HCCP_HAL_NM:-nm}
KVAF_HCCP_LDCONFIG=${KVAF_HCCP_LDCONFIG:-ldconfig}
if [[ -n "${KVAF_HCCP_HOST_MACHINE:-}" ]]; then
  kvaf_host_machine=${KVAF_HCCP_HOST_MACHINE}
else
  command -v uname >/dev/null 2>&1 || kvaf_fail "uname is required"
  kvaf_host_machine=$(uname -m) || kvaf_fail "cannot identify host machine"
fi
kvaf_host_machine=$(printf '%s' "${kvaf_host_machine}" | tr '[:upper:]' '[:lower:]')
case "${kvaf_host_machine}" in
  aarch64|arm64) kvaf_ldconfig_arch_pattern=aarch64 ;;
  x86_64|amd64) kvaf_ldconfig_arch_pattern='x86-64|x86_64' ;;
  *) kvaf_ldconfig_arch_pattern=unsupported ;;
esac
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

kvaf_require_under_driver_root()
{
  local kvaf_path=$1
  local kvaf_label=$2
  [[ "${kvaf_path}" == /* ]] ||
    kvaf_fail "${kvaf_label} path must be absolute: ${kvaf_path}"
  [[ -f "${kvaf_path}" && -r "${kvaf_path}" ]] ||
    kvaf_fail "${kvaf_label} is missing or unreadable: ${kvaf_path}"
  local kvaf_real
  kvaf_real=$(realpath "${kvaf_path}") ||
    kvaf_fail "cannot resolve ${kvaf_label}: ${kvaf_path}"
  case "${kvaf_real}" in
    "${kvaf_driver_root_real}"/*) ;;
    *) kvaf_fail "${kvaf_label} resolves outside KVAF_ASCEND_DRIVER_ROOT" ;;
  esac
  printf '%s\n' "${kvaf_real}"
}

kvaf_select_unique_file()
{
  local kvaf_label=$1
  shift
  local -a kvaf_found=()
  local kvaf_candidate kvaf_real kvaf_seen
  for kvaf_candidate in "$@"; do
    [[ -f "${kvaf_candidate}" && -r "${kvaf_candidate}" ]] || continue
    kvaf_real=$(realpath "${kvaf_candidate}") ||
      kvaf_fail "cannot resolve ${kvaf_label} candidate: ${kvaf_candidate}"
    case "${kvaf_real}" in
      "${kvaf_driver_root_real}"/*) ;;
      *) kvaf_fail "${kvaf_label} candidate resolves outside driver root" ;;
    esac
    kvaf_seen=0
    for kvaf_candidate in "${kvaf_found[@]:-}"; do
      if [[ "${kvaf_candidate}" == "${kvaf_real}" ]]; then
        kvaf_seen=1
        break
      fi
    done
    (( kvaf_seen != 0 )) || kvaf_found+=("${kvaf_real}")
  done
  (( ${#kvaf_found[@]} <= 1 )) ||
    kvaf_fail "multiple distinct ${kvaf_label} candidates exist; select explicitly"
  printf '%s\n' "${kvaf_found[0]:-}"
}

if [[ -n "${KVAF_HCCP_HAL_HEADER:-}" ]]; then
  kvaf_hal_header=$(kvaf_require_under_driver_root \
    "${KVAF_HCCP_HAL_HEADER}" "ascend_hal.h")
else
  kvaf_hal_header=$(kvaf_select_unique_file "ascend_hal.h" \
    "${KVAF_ASCEND_DRIVER_ROOT}/include/driver/ascend_hal.h" \
    "${KVAF_ASCEND_DRIVER_ROOT}/include/ascend_hal.h")
fi

if [[ -n "${KVAF_HCCP_HAL_LIBRARY:-}" ]]; then
  kvaf_hal_library=$(kvaf_require_under_driver_root \
    "${KVAF_HCCP_HAL_LIBRARY}" "libascend_hal.so")
else
  kvaf_hal_library=$(kvaf_select_unique_file "libascend_hal.so" \
    "${KVAF_ASCEND_DRIVER_ROOT}/lib64/driver/libascend_hal.so" \
    "${KVAF_ASCEND_DRIVER_ROOT}/lib64/libascend_hal.so")
fi

kvaf_require_absolute_regular_file()
{
  local kvaf_path=$1
  local kvaf_label=$2
  [[ "${kvaf_path}" == /* ]] ||
    kvaf_fail "${kvaf_label} path must be absolute: ${kvaf_path}"
  [[ -f "${kvaf_path}" && -r "${kvaf_path}" ]] ||
    kvaf_fail "${kvaf_label} is missing or unreadable: ${kvaf_path}"
  realpath "${kvaf_path}" ||
    kvaf_fail "cannot resolve ${kvaf_label}: ${kvaf_path}"
}

# libra.so is not confined to the driver tree.  The open HCCP build installs
# it below <CANN>/<arch>-linux/lib64, while deployed images may expose it only
# through LD_LIBRARY_PATH or the dynamic-linker cache. Inventory those
# read-only hints plus fixed layouts, canonicalize every existing file, and
# inspect the first candidate in deterministic discovery order. This is not a
# proof of a future executable's DT_RPATH/DT_RUNPATH resolution. Empty or
# relative loader-path entries are intentionally ignored instead of treating
# the current directory as trusted. No candidate is ever loaded.
declare -a kvaf_ra_candidates=()
declare -a kvaf_ra_candidate_sources=()
kvaf_ra_ld_ignored=0
kvaf_ra_ldconfig_status=unavailable
kvaf_ra_ascend_home_status=unset

kvaf_add_ra_candidate()
{
  local kvaf_path=$1
  local kvaf_source=$2
  [[ "${kvaf_path}" == /* ]] || return 0
  [[ -f "${kvaf_path}" && -r "${kvaf_path}" ]] || return 0
  local kvaf_real kvaf_existing
  kvaf_real=$(realpath "${kvaf_path}") ||
    kvaf_fail "cannot resolve libra.so candidate: ${kvaf_path}"
  for kvaf_existing in "${kvaf_ra_candidates[@]:-}"; do
    [[ "${kvaf_existing}" == "${kvaf_real}" ]] && return 0
  done
  kvaf_ra_candidates+=("${kvaf_real}")
  kvaf_ra_candidate_sources+=("${kvaf_source}")
}

if [[ -n "${KVAF_HCCP_RA_LIBRARY:-}" ]]; then
  kvaf_ra_library=$(kvaf_require_absolute_regular_file \
    "${KVAF_HCCP_RA_LIBRARY}" "libra.so")
  kvaf_ra_discovery_source=explicit
  kvaf_ra_candidates+=("${kvaf_ra_library}")
  kvaf_ra_candidate_sources+=(explicit)
else
  # The ELF loader searches these directories first and in this order.
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    IFS=: read -r -a kvaf_ra_ld_dirs <<<"${LD_LIBRARY_PATH}"
    for kvaf_ra_ld_dir in "${kvaf_ra_ld_dirs[@]}"; do
      if [[ -z "${kvaf_ra_ld_dir}" || "${kvaf_ra_ld_dir}" != /* ||
            ! -d "${kvaf_ra_ld_dir}" ]]; then
        kvaf_ra_ld_ignored=$((kvaf_ra_ld_ignored + 1))
        continue
      fi
      kvaf_add_ra_candidate \
        "${kvaf_ra_ld_dir}/libra.so" "ld-library-path"
    done
  fi

  # The cache is the next source consulted by the ELF loader.  Parsing it is
  # read-only; failures merely reduce discovery coverage.
  if command -v "${KVAF_HCCP_LDCONFIG}" >/dev/null 2>&1; then
    if kvaf_ra_ldconfig_output=$(LC_ALL=C "${KVAF_HCCP_LDCONFIG}" -p 2>/dev/null); then
      kvaf_ra_ldconfig_status=readable
      while IFS= read -r kvaf_ra_cached_path; do
        [[ -n "${kvaf_ra_cached_path}" ]] || continue
        kvaf_add_ra_candidate "${kvaf_ra_cached_path}" "ldconfig-cache"
      done < <(awk -v arch_pattern="${kvaf_ldconfig_arch_pattern}" '
          $1 == "libra.so" {
            descriptor = tolower($0)
            sub(/^[^(]*\(/, "", descriptor)
            sub(/\).*/, "", descriptor)
            if (arch_pattern == "unsupported" ||
                descriptor !~ arch_pattern) next
            for (i = 2; i < NF; ++i) {
              if ($i == "=>") {
                print $(i + 1)
                break
              }
            }
          }
        ' <<<"${kvaf_ra_ldconfig_output}")
    else
      kvaf_ra_ldconfig_status=failed
    fi
  fi

  # Fixed-layout fallbacks make an absolute-path backend possible even when
  # set_env.sh did not add the matching CANN directory to the loader path.
  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    if [[ "${ASCEND_HOME_PATH}" == /* && -d "${ASCEND_HOME_PATH}" ]]; then
      kvaf_ra_ascend_home=$(realpath "${ASCEND_HOME_PATH}") ||
        kvaf_fail "cannot resolve ASCEND_HOME_PATH"
      kvaf_ra_ascend_home_status=usable
      for kvaf_ra_cann_dir in \
        "${kvaf_ra_ascend_home}/lib64" \
        "${kvaf_ra_ascend_home}/aarch64-linux/lib64" \
        "${kvaf_ra_ascend_home}/x86_64-linux/lib64" \
        "${kvaf_ra_ascend_home}/arm64-linux/lib64"; do
        kvaf_add_ra_candidate "${kvaf_ra_cann_dir}/libra.so" \
          "ascend-home-layout"
      done
    else
      kvaf_ra_ascend_home_status=ignored-nonabsolute-or-missing
    fi
  fi

  for kvaf_ra_driver_path in \
    "${KVAF_ASCEND_DRIVER_ROOT}/lib64/common/libra.so" \
    "${KVAF_ASCEND_DRIVER_ROOT}/lib64/driver/libra.so" \
    "${KVAF_ASCEND_DRIVER_ROOT}/lib64/libra.so"; do
    kvaf_add_ra_candidate "${kvaf_ra_driver_path}" "driver-layout"
  done

  kvaf_ra_library=${kvaf_ra_candidates[0]:-}
  kvaf_ra_discovery_source=${kvaf_ra_candidate_sources[0]:-not-found}
fi

# libhcomm.so is a deliberately lower-priority bootstrap candidate.  It owns
# Endpoint/MR/Channel/TP/Jetty lifetime during initialization; the existing
# HcommUrmaResources wrapper then snapshots the SQ/CQ/token metadata consumed
# directly by AIV.  Inventorying it here does not make HCOMM part of the
# per-transfer hot path and does not call any HCOMM entry point.
declare -a kvaf_hcomm_candidates=()
declare -a kvaf_hcomm_candidate_sources=()
kvaf_hcomm_ld_ignored=0
kvaf_hcomm_ldconfig_status=unavailable
kvaf_hcomm_ascend_home_status=unset

kvaf_add_hcomm_candidate()
{
  local kvaf_path=$1
  local kvaf_source=$2
  [[ "${kvaf_path}" == /* ]] || return 0
  [[ -f "${kvaf_path}" && -r "${kvaf_path}" ]] || return 0
  local kvaf_real kvaf_existing
  kvaf_real=$(realpath "${kvaf_path}") ||
    kvaf_fail "cannot resolve libhcomm.so candidate: ${kvaf_path}"
  for kvaf_existing in "${kvaf_hcomm_candidates[@]:-}"; do
    [[ "${kvaf_existing}" == "${kvaf_real}" ]] && return 0
  done
  kvaf_hcomm_candidates+=("${kvaf_real}")
  kvaf_hcomm_candidate_sources+=("${kvaf_source}")
}

if [[ -n "${KVAF_HCCP_HCOMM_LIBRARY:-}" ]]; then
  kvaf_hcomm_library=$(kvaf_require_absolute_regular_file \
    "${KVAF_HCCP_HCOMM_LIBRARY}" "libhcomm.so")
  kvaf_hcomm_discovery_source=explicit
  kvaf_hcomm_candidates+=("${kvaf_hcomm_library}")
  kvaf_hcomm_candidate_sources+=(explicit)
else
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    IFS=: read -r -a kvaf_hcomm_ld_dirs <<<"${LD_LIBRARY_PATH}"
    for kvaf_hcomm_ld_dir in "${kvaf_hcomm_ld_dirs[@]}"; do
      if [[ -z "${kvaf_hcomm_ld_dir}" ||
            "${kvaf_hcomm_ld_dir}" != /* ||
            ! -d "${kvaf_hcomm_ld_dir}" ]]; then
        kvaf_hcomm_ld_ignored=$((kvaf_hcomm_ld_ignored + 1))
        continue
      fi
      kvaf_add_hcomm_candidate \
        "${kvaf_hcomm_ld_dir}/libhcomm.so" "ld-library-path"
    done
  fi

  if command -v "${KVAF_HCCP_LDCONFIG}" >/dev/null 2>&1; then
    if kvaf_hcomm_ldconfig_output=$(LC_ALL=C \
        "${KVAF_HCCP_LDCONFIG}" -p 2>/dev/null); then
      kvaf_hcomm_ldconfig_status=readable
      while IFS= read -r kvaf_hcomm_cached_path; do
        [[ -n "${kvaf_hcomm_cached_path}" ]] || continue
        kvaf_add_hcomm_candidate \
          "${kvaf_hcomm_cached_path}" "ldconfig-cache"
      done < <(awk -v arch_pattern="${kvaf_ldconfig_arch_pattern}" '
          $1 == "libhcomm.so" {
            descriptor = tolower($0)
            sub(/^[^(]*\(/, "", descriptor)
            sub(/\).*/, "", descriptor)
            if (arch_pattern == "unsupported" ||
                descriptor !~ arch_pattern) next
            for (i = 2; i < NF; ++i) {
              if ($i == "=>") {
                print $(i + 1)
                break
              }
            }
          }
        ' <<<"${kvaf_hcomm_ldconfig_output}")
    else
      kvaf_hcomm_ldconfig_status=failed
    fi
  fi

  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    if [[ "${ASCEND_HOME_PATH}" == /* && -d "${ASCEND_HOME_PATH}" ]]; then
      kvaf_hcomm_ascend_home=$(realpath "${ASCEND_HOME_PATH}") ||
        kvaf_fail "cannot resolve ASCEND_HOME_PATH for libhcomm.so"
      kvaf_hcomm_ascend_home_status=usable
      for kvaf_hcomm_cann_dir in \
        "${kvaf_hcomm_ascend_home}/lib64" \
        "${kvaf_hcomm_ascend_home}/aarch64-linux/lib64" \
        "${kvaf_hcomm_ascend_home}/x86_64-linux/lib64" \
        "${kvaf_hcomm_ascend_home}/arm64-linux/lib64"; do
        kvaf_add_hcomm_candidate \
          "${kvaf_hcomm_cann_dir}/libhcomm.so" "ascend-home-layout"
      done
    else
      kvaf_hcomm_ascend_home_status=ignored-nonabsolute-or-missing
    fi
  fi

  for kvaf_hcomm_driver_path in \
    "${KVAF_ASCEND_DRIVER_ROOT}/lib64/common/libhcomm.so" \
    "${KVAF_ASCEND_DRIVER_ROOT}/lib64/driver/libhcomm.so" \
    "${KVAF_ASCEND_DRIVER_ROOT}/lib64/libhcomm.so"; do
    kvaf_add_hcomm_candidate \
      "${kvaf_hcomm_driver_path}" "driver-layout"
  done

  kvaf_hcomm_library=${kvaf_hcomm_candidates[0]:-}
  kvaf_hcomm_discovery_source=${kvaf_hcomm_candidate_sources[0]:-not-found}
fi

echo "EXECUTION_ENV=CALLER_SELECTED_READ_ONLY_PREFLIGHT"
echo "KVAF_HCCP_HAL_HEADER=${kvaf_hal_header:-not-found}"
echo "KVAF_HCCP_HAL_LIBRARY=${kvaf_hal_library:-not-found}"
echo "KVAF_HCCP_RA_LIBRARY=${kvaf_ra_library:-not-found}"
echo "KVAF_HCCP_RA_DISCOVERY_SOURCE=${kvaf_ra_discovery_source}"
echo "KVAF_HCCP_DISCOVERY_HOST_MACHINE=${kvaf_host_machine}"
echo "KVAF_HCCP_DISCOVERY_ORDER=explicit-safe-loader-hints-cann-driver-layouts"
echo "KVAF_HCCP_DISCOVERY_ELF_RUNPATH_COVERAGE=not-inspected"
echo "KVAF_HCCP_DISCOVERY_FILE_ARCH_COMPATIBILITY=ldconfig-filtered-other-sources-unproven"
echo "KVAF_HCCP_RA_DISCOVERY_CANDIDATE_COUNT=${#kvaf_ra_candidates[@]}"
echo "KVAF_HCCP_RA_DISCOVERY_ADDITIONAL_COUNT=$(( ${#kvaf_ra_candidates[@]} > 0 ? ${#kvaf_ra_candidates[@]} - 1 : 0 ))"
echo "KVAF_HCCP_RA_DISCOVERY_LD_LIBRARY_PATH_IGNORED_COUNT=${kvaf_ra_ld_ignored}"
echo "KVAF_HCCP_RA_DISCOVERY_LDCONFIG=${kvaf_ra_ldconfig_status}"
echo "KVAF_HCCP_RA_DISCOVERY_ASCEND_HOME=${kvaf_ra_ascend_home_status}"
for kvaf_ra_index in "${!kvaf_ra_candidates[@]}"; do
  echo "KVAF_HCCP_RA_DISCOVERY_CANDIDATE_${kvaf_ra_index}=${kvaf_ra_candidates[${kvaf_ra_index}]}"
  echo "KVAF_HCCP_RA_DISCOVERY_CANDIDATE_${kvaf_ra_index}_SOURCE=${kvaf_ra_candidate_sources[${kvaf_ra_index}]}"
done
echo "KVAF_HCCP_HCOMM_LIBRARY=${kvaf_hcomm_library:-not-found}"
echo "KVAF_HCCP_HCOMM_DISCOVERY_SOURCE=${kvaf_hcomm_discovery_source}"
echo "KVAF_HCCP_HCOMM_DISCOVERY_CANDIDATE_COUNT=${#kvaf_hcomm_candidates[@]}"
echo "KVAF_HCCP_HCOMM_DISCOVERY_ADDITIONAL_COUNT=$(( ${#kvaf_hcomm_candidates[@]} > 0 ? ${#kvaf_hcomm_candidates[@]} - 1 : 0 ))"
echo "KVAF_HCCP_HCOMM_DISCOVERY_LD_LIBRARY_PATH_IGNORED_COUNT=${kvaf_hcomm_ld_ignored}"
echo "KVAF_HCCP_HCOMM_DISCOVERY_LDCONFIG=${kvaf_hcomm_ldconfig_status}"
echo "KVAF_HCCP_HCOMM_DISCOVERY_ASCEND_HOME=${kvaf_hcomm_ascend_home_status}"
for kvaf_hcomm_index in "${!kvaf_hcomm_candidates[@]}"; do
  echo "KVAF_HCCP_HCOMM_DISCOVERY_CANDIDATE_${kvaf_hcomm_index}=${kvaf_hcomm_candidates[${kvaf_hcomm_index}]}"
  echo "KVAF_HCCP_HCOMM_DISCOVERY_CANDIDATE_${kvaf_hcomm_index}_SOURCE=${kvaf_hcomm_candidate_sources[${kvaf_hcomm_index}]}"
done
echo "KVAF_HCCP_OPERATION_MODE=compile-and-symbol-inspection-only"
echo "KVAF_HCCP_CONTROL_PLANE_DESIGN=host-owned-libra-init-aiv-direct-data-plane"
echo "KVAF_HCCP_HOT_PATH=direct-aiv-simt-sq-cq-no-hccp-per-transfer"
echo "KVAF_HCCP_PRIVATE_PAYLOAD_PUBLIC_API=no"
echo "KVAF_HCCP_PRIVATE_PAYLOAD_SOURCE=HCOMM@2591d7ff74c18fb9829dd5c0c157f97558405169"
echo "KVAF_HCCP_PRIVATE_PAYLOAD_SOURCE_SHA256=9753b9de6dbd8106677672959ffb909708dcd36aab071baa6b67c0a548c5792d"

kvaf_symbol_present()
{
  local kvaf_wanted=$1
  local kvaf_symbols=$2
  awk -v wanted="${kvaf_wanted}" '
      {
        name = $NF
        sub(/@.*/, "", name)
        if (name == wanted) found = 1
      }
      END { exit !found }
    ' <<<"${kvaf_symbols}"
}

kvaf_read_symbols()
{
  local kvaf_library=$1
  LC_ALL=C "${KVAF_HCCP_HAL_NM}" -D --defined-only -- \
    "${kvaf_library}" 2>&1
}

declare -a kvaf_hal_host_symbols=(
  drvDeviceGetIndexByPhyId
  halQueryDevpid
  halResAddrMapV2
  halResAddrUnmapV2
)
declare -a kvaf_hal_device_offline_symbols=(halMemBindSibling)
declare -a kvaf_hal_symbols=(
  "${kvaf_hal_host_symbols[@]}"
  "${kvaf_hal_device_offline_symbols[@]}"
)
kvaf_hal_symbol_missing=0
kvaf_hal_device_offline_symbol_missing=0
if [[ -n "${kvaf_hal_library}" ]]; then
  kvaf_hal_nm_output=$(kvaf_read_symbols "${kvaf_hal_library}") ||
    kvaf_fail "could not read dynamic symbols from ${kvaf_hal_library}"
  for kvaf_symbol in "${kvaf_hal_host_symbols[@]}"; do
    if kvaf_symbol_present "${kvaf_symbol}" "${kvaf_hal_nm_output}"; then
      echo "KVAF_HCCP_HAL_LIBRARY_SYMBOL_${kvaf_symbol}=available"
    else
      echo "KVAF_HCCP_HAL_LIBRARY_SYMBOL_${kvaf_symbol}=missing"
      kvaf_hal_symbol_missing=$((kvaf_hal_symbol_missing + 1))
    fi
  done
  for kvaf_symbol in "${kvaf_hal_device_offline_symbols[@]}"; do
    if kvaf_symbol_present "${kvaf_symbol}" "${kvaf_hal_nm_output}"; then
      echo "KVAF_HCCP_HAL_LIBRARY_SYMBOL_${kvaf_symbol}=available"
    else
      echo "KVAF_HCCP_HAL_LIBRARY_SYMBOL_${kvaf_symbol}=missing"
      kvaf_hal_device_offline_symbol_missing=$((
        kvaf_hal_device_offline_symbol_missing + 1
      ))
    fi
  done
else
  for kvaf_symbol in "${kvaf_hal_symbols[@]}"; do
    echo "KVAF_HCCP_HAL_LIBRARY_SYMBOL_${kvaf_symbol}=unavailable"
  done
  kvaf_hal_symbol_missing=${#kvaf_hal_host_symbols[@]}
  kvaf_hal_device_offline_symbol_missing=${#kvaf_hal_device_offline_symbols[@]}
fi
echo "KVAF_HCCP_HOST_MAP_REQUIRED_SYMBOL_MISSING_COUNT=${kvaf_hal_symbol_missing}"
if (( kvaf_hal_device_offline_symbol_missing == 0 )); then
  echo "KVAF_HCCP_DEVICE_OFFLINE_SHAREPOOL_SURFACE=complete-selected-dso-symbol-only"
else
  echo "KVAF_HCCP_DEVICE_OFFLINE_SHAREPOOL_SURFACE=incomplete-selected-dso-symbol-only"
fi
echo "KVAF_HCCP_DEVICE_OFFLINE_SHAREPOOL_MISSING_COUNT=${kvaf_hal_device_offline_symbol_missing}"
echo "KVAF_HCCP_DEVICE_OFFLINE_SHAREPOOL_DEPENDENCY_CLOSURE=unproven"
echo "KVAF_HCCP_DEVICE_OFFLINE_MISSING_BIND_BEHAVIOR=hcomm-9.1-retries-up-to-120s-if-entered"

kvaf_hal_header_missing=0
kvaf_hal_device_offline_header_missing=0
if [[ -n "${kvaf_hal_header}" ]]; then
  command -v "${KVAF_HCCP_HAL_CC}" >/dev/null 2>&1 ||
    kvaf_fail "C compiler is unavailable: ${KVAF_HCCP_HAL_CC}"
  declare -a kvaf_include_args=("-I${KVAF_PUBLIC_INCLUDE_DIR}")
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
      *) kvaf_fail "HAL include directory resolves outside driver root" ;;
    esac
    kvaf_include_args+=("-I${kvaf_dir}")
  }
  [[ -d "${KVAF_ASCEND_DRIVER_ROOT}/include" ]] &&
    kvaf_add_include_dir "${KVAF_ASCEND_DRIVER_ROOT}/include"
  kvaf_add_include_dir "$(dirname "${kvaf_hal_header}")"
  if [[ -n "${KVAF_HCCP_HAL_INCLUDE_DIRS:-}" ]]; then
    IFS=: read -r -a kvaf_extra_include_dirs \
      <<<"${KVAF_HCCP_HAL_INCLUDE_DIRS}"
    for kvaf_include_dir in "${kvaf_extra_include_dirs[@]}"; do
      kvaf_add_include_dir "${kvaf_include_dir}"
    done
  fi

  if kvaf_baseline_output=$(env -u CPATH -u C_INCLUDE_PATH \
      -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
      "${KVAF_HCCP_HAL_CC}" -std=gnu11 -fsyntax-only \
      "${kvaf_include_args[@]}" -include "${kvaf_hal_header}" \
      -DKVAF_HAL_PROBE_BASELINE=1 "${PROBE_SOURCE}" 2>&1); then
    echo "KVAF_HCCP_HAL_HEADER_BASELINE=usable"
  else
    echo "KVAF_HCCP_HAL_HEADER_BASELINE=unusable"
    echo "KVAF_HCCP_HAL_HEADER_BASELINE_DIAGNOSTIC_BEGIN" >&2
    sed -n '1,12p' <<<"${kvaf_baseline_output}" >&2
    echo "KVAF_HCCP_HAL_HEADER_BASELINE_DIAGNOSTIC_END" >&2
    kvaf_fail "installed ascend_hal.h cannot compile in isolation"
  fi

  declare -a kvaf_host_header_probes=(
    "drvDeviceGetIndexByPhyId:KVAF_HAL_PROBE_DRV_DEVICE_GET_INDEX_BY_PHY_ID"
    "halQueryDevpid:KVAF_HAL_PROBE_QUERY_DEVPID"
    "halResAddrMapV2:KVAF_HAL_PROBE_RES_ADDR_MAP_V2"
    "halResAddrUnmapV2:KVAF_HAL_PROBE_RES_ADDR_UNMAP_V2"
    "res_map_info_in:KVAF_HAL_PROBE_RES_MAP_INFO_IN"
    "res_map_info_out:KVAF_HAL_PROBE_RES_MAP_INFO_OUT"
    "RES_ADDR_TYPE_HCCP_URMA_JFC:KVAF_HAL_PROBE_RES_TYPE_HCCP_URMA_JFC"
    "RES_ADDR_TYPE_HCCP_URMA_JETTY:KVAF_HAL_PROBE_RES_TYPE_HCCP_URMA_JETTY"
    "RES_ADDR_TYPE_HCCP_URMA_DB:KVAF_HAL_PROBE_RES_TYPE_HCCP_URMA_DB"
    "UdmaVaInfo_private_payload:KVAF_HAL_PROBE_UDMA_VA_INFO_PRIVATE_ABI"
  )
  declare -a kvaf_device_offline_header_probes=(
    "halMemBindSibling:KVAF_HAL_PROBE_MEM_BIND_SIBLING"
  )
  for kvaf_probe_entry in "${kvaf_host_header_probes[@]}"; do
    kvaf_name=${kvaf_probe_entry%%:*}
    kvaf_selector=${kvaf_probe_entry#*:}
    if kvaf_compile_output=$(env -u CPATH -u C_INCLUDE_PATH \
        -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
        "${KVAF_HCCP_HAL_CC}" -std=gnu11 -fsyntax-only \
        "${kvaf_include_args[@]}" -include "${kvaf_hal_header}" \
        "-D${kvaf_selector}=1" "${PROBE_SOURCE}" 2>&1); then
      echo "KVAF_HCCP_HAL_HEADER_CAPABILITY_${kvaf_name}=available"
    else
      echo "KVAF_HCCP_HAL_HEADER_CAPABILITY_${kvaf_name}=missing"
      echo "KVAF_HCCP_HAL_HEADER_DIAGNOSTIC_${kvaf_name}_BEGIN" >&2
      sed -n '1,6p' <<<"${kvaf_compile_output}" >&2
      echo "KVAF_HCCP_HAL_HEADER_DIAGNOSTIC_${kvaf_name}_END" >&2
      kvaf_hal_header_missing=$((kvaf_hal_header_missing + 1))
    fi
  done
  for kvaf_probe_entry in "${kvaf_device_offline_header_probes[@]}"; do
    kvaf_name=${kvaf_probe_entry%%:*}
    kvaf_selector=${kvaf_probe_entry#*:}
    if kvaf_compile_output=$(env -u CPATH -u C_INCLUDE_PATH \
        -u CPLUS_INCLUDE_PATH -u OBJC_INCLUDE_PATH \
        "${KVAF_HCCP_HAL_CC}" -std=gnu11 -fsyntax-only \
        "${kvaf_include_args[@]}" -include "${kvaf_hal_header}" \
        "-D${kvaf_selector}=1" "${PROBE_SOURCE}" 2>&1); then
      echo "KVAF_HCCP_HAL_HEADER_CAPABILITY_${kvaf_name}=available"
    else
      echo "KVAF_HCCP_HAL_HEADER_CAPABILITY_${kvaf_name}=missing"
      echo "KVAF_HCCP_HAL_HEADER_DIAGNOSTIC_${kvaf_name}_BEGIN" >&2
      sed -n '1,6p' <<<"${kvaf_compile_output}" >&2
      echo "KVAF_HCCP_HAL_HEADER_DIAGNOSTIC_${kvaf_name}_END" >&2
      kvaf_hal_device_offline_header_missing=$((
        kvaf_hal_device_offline_header_missing + 1
      ))
    fi
  done
else
  echo "KVAF_HCCP_HAL_HEADER_BASELINE=unavailable"
  echo "KVAF_HCCP_HAL_HEADER_SURFACE=unavailable-runtime-image-has-no-sdk-header"
  echo "KVAF_HCCP_PRIVATE_LAYOUT=not-checked-header-unavailable"
  kvaf_hal_header_missing=1
  kvaf_hal_device_offline_header_missing=1
fi

echo "KVAF_HCCP_DEVICE_OFFLINE_SHAREPOOL_HEADER_MISSING_COUNT=${kvaf_hal_device_offline_header_missing}"

if (( kvaf_hal_symbol_missing == 0 && kvaf_hal_header_missing == 0 )) &&
   [[ -n "${kvaf_hal_header}" ]]; then
  echo "KVAF_HCCP_HAL_SURFACE=declarations-complete"
  echo "KVAF_HCCP_HAL_MISSING_COUNT=0"
  echo "KVAF_HCCP_PRIVATE_PAYLOAD_ABI=source-audited-hcomm-v1-runtime-unproven"
  echo "KVAF_HCCP_PRIVATE_LAYOUT=compile-time-size-and-offsets-verified-runtime-unproven"
  echo "KVAF_A5_HCCP_HAL_DECLARATION_PREFLIGHT_OK"
else
  echo "KVAF_HCCP_HAL_SURFACE=incomplete-or-header-unavailable"
  echo "KVAF_HCCP_HAL_MISSING_COUNT=$((kvaf_hal_symbol_missing + kvaf_hal_header_missing))"
  echo "KVAF_HCCP_PRIVATE_PAYLOAD_ABI=not-selected-for-primary-backend"
fi
echo "KVAF_HCCP_HAL_RUNTIME_SEMANTICS=unproven-not-called"
echo "KVAF_HCCP_DIRECT_PRIVATE_HAL_BACKEND=deferred-not-primary"

declare -a kvaf_ra_ctx_symbols=(
  RaCtxInit RaCtxDeinit
  RaCtxLmemRegister RaCtxLmemUnregister
  RaCtxRmemImport RaCtxRmemUnimport
  RaCtxChanCreate RaCtxChanDestroy
  RaCtxCqCreate RaCtxCqDestroy
  RaCtxQpCreate RaCtxQpDestroy
  RaCtxQpImport RaCtxQpUnimport
  RaCtxQpBind RaCtxQpUnbind
  RaCtxGetTpInfoList RaCtxGetTpAttr RaCtxSetTpAttr
)
declare -a kvaf_ra_legacy_symbols=(
  RaInit RaRdevGetHandle RaRdevInitV2
  RaAiQpCreate RaQpDestroy
  RaRegisterMr RaDeregisterMr RaMrReg
  RaQpConnectAsync RaGetQpStatus
)
kvaf_ra_ctx_missing=0
kvaf_ra_legacy_missing=0
if [[ -n "${kvaf_ra_library}" ]]; then
  kvaf_ra_nm_output=$(kvaf_read_symbols "${kvaf_ra_library}") ||
    kvaf_fail "could not read dynamic symbols from ${kvaf_ra_library}"
  for kvaf_symbol in "${kvaf_ra_ctx_symbols[@]}"; do
    if kvaf_symbol_present "${kvaf_symbol}" "${kvaf_ra_nm_output}"; then
      echo "KVAF_HCCP_RA_CTX_SYMBOL_${kvaf_symbol}=available"
    else
      echo "KVAF_HCCP_RA_CTX_SYMBOL_${kvaf_symbol}=missing"
      kvaf_ra_ctx_missing=$((kvaf_ra_ctx_missing + 1))
    fi
  done
  for kvaf_symbol in "${kvaf_ra_legacy_symbols[@]}"; do
    if kvaf_symbol_present "${kvaf_symbol}" "${kvaf_ra_nm_output}"; then
      echo "KVAF_HCCP_RA_LEGACY_SYMBOL_${kvaf_symbol}=available"
    else
      echo "KVAF_HCCP_RA_LEGACY_SYMBOL_${kvaf_symbol}=missing"
      kvaf_ra_legacy_missing=$((kvaf_ra_legacy_missing + 1))
    fi
  done
else
  kvaf_ra_ctx_missing=${#kvaf_ra_ctx_symbols[@]}
  kvaf_ra_legacy_missing=${#kvaf_ra_legacy_symbols[@]}
fi

if [[ -z "${kvaf_ra_library}" ]]; then
  echo "KVAF_HCCP_RA_CTX_SURFACE=unavailable"
  echo "KVAF_HCCP_RA_LEGACY_SURFACE=unavailable"
  kvaf_primary_candidate=none-libra-not-found
elif (( kvaf_ra_ctx_missing == 0 )); then
  echo "KVAF_HCCP_RA_CTX_SURFACE=complete-symbols-only"
  if (( kvaf_ra_legacy_missing == 0 )); then
    echo "KVAF_HCCP_RA_LEGACY_SURFACE=complete-symbols-only"
  else
    echo "KVAF_HCCP_RA_LEGACY_SURFACE=incomplete"
  fi
  kvaf_primary_candidate=libra-ra-ctx
elif (( kvaf_ra_legacy_missing == 0 )); then
  echo "KVAF_HCCP_RA_CTX_SURFACE=incomplete"
  echo "KVAF_HCCP_RA_LEGACY_SURFACE=complete-symbols-only"
  kvaf_primary_candidate=libra-legacy-ai-qp
else
  echo "KVAF_HCCP_RA_CTX_SURFACE=incomplete"
  echo "KVAF_HCCP_RA_LEGACY_SURFACE=incomplete"
  kvaf_primary_candidate=none-required-symbols-missing
fi

echo "KVAF_HCCP_CONTROL_PLANE_CANDIDATE=${kvaf_primary_candidate}"

echo "KVAF_HCCP_RA_CTX_MISSING_COUNT=${kvaf_ra_ctx_missing}"
echo "KVAF_HCCP_RA_LEGACY_MISSING_COUNT=${kvaf_ra_legacy_missing}"
echo "KVAF_HCCP_RA_ABI_COMPATIBILITY=unproven-by-symbol-surface"
echo "KVAF_HCCP_RA_QUEUE_ADDRESS_CONTRACT=requires-version-matched-header-and-runtime-probe"
echo "KVAF_HCCP_RA_PRODUCT_MODE_SUPPORT=unproven-user-ctl-and-custom-sq-must-be-probed"
echo "KVAF_HCCP_RA_OPEN_SOURCE_910_95_CUSTOM_SQ=known-rejection-do-not-infer-installed-runtime"
echo "KVAF_HCCP_RA_LIFECYCLE_CONTRACT=host-owner-reverse-order-return-codes-required"

# Exact dynamic C ABI consumed by
# access_operator/a5/src/hcomm_urma_resources.cpp. EndpointDescInit and
# HcommChannelDescInit are header-defined inline helpers, so requiring them as
# ELF exports would incorrectly reject a valid libhcomm.so.
declare -a kvaf_hcomm_resource_symbols=(
  HcommEndpointCreate HcommEndpointDestroy
  HcommMemReg HcommMemUnreg HcommMemExport HcommMemImport HcommMemUnimport
  HcommChannelCreate HcommChannelGetStatus HcommChannelDestroy
)
kvaf_hcomm_resource_missing=0
if [[ -n "${kvaf_hcomm_library}" ]]; then
  kvaf_hcomm_nm_output=$(kvaf_read_symbols "${kvaf_hcomm_library}") ||
    kvaf_fail "could not read dynamic symbols from ${kvaf_hcomm_library}"
  for kvaf_symbol in "${kvaf_hcomm_resource_symbols[@]}"; do
    if kvaf_symbol_present "${kvaf_symbol}" "${kvaf_hcomm_nm_output}"; then
      echo "KVAF_HCCP_HCOMM_RESOURCE_SYMBOL_${kvaf_symbol}=available"
    else
      echo "KVAF_HCCP_HCOMM_RESOURCE_SYMBOL_${kvaf_symbol}=missing"
      kvaf_hcomm_resource_missing=$((kvaf_hcomm_resource_missing + 1))
    fi
  done
else
  kvaf_hcomm_resource_missing=${#kvaf_hcomm_resource_symbols[@]}
fi

if [[ -z "${kvaf_hcomm_library}" ]]; then
  kvaf_hcomm_fallback_candidate=none-libhcomm-not-found
  echo "KVAF_HCCP_HCOMM_RESOURCE_SURFACE=unavailable"
elif (( kvaf_hcomm_resource_missing == 0 )); then
  kvaf_hcomm_fallback_candidate=libhcomm-resource-api
  echo "KVAF_HCCP_HCOMM_RESOURCE_SURFACE=complete-symbols-only"
else
  kvaf_hcomm_fallback_candidate=none-required-symbols-missing
  echo "KVAF_HCCP_HCOMM_RESOURCE_SURFACE=incomplete"
fi
echo "KVAF_HCCP_HCOMM_RESOURCE_MISSING_COUNT=${kvaf_hcomm_resource_missing}"
echo "KVAF_HCCP_CONTROL_PLANE_FALLBACK_STATIC_SYMBOL_CANDIDATE=${kvaf_hcomm_fallback_candidate}"

case "${kvaf_primary_candidate}" in
  libra-*) kvaf_effective_candidate=${kvaf_primary_candidate} ;;
  *)
    if [[ "${kvaf_hcomm_fallback_candidate}" == libhcomm-resource-api ]]; then
      kvaf_effective_candidate=libhcomm-resource-api
    else
      kvaf_effective_candidate=none-usable-static-symbol-surface
    fi
    ;;
esac
echo "KVAF_HCCP_SELECTED_STATIC_SYMBOL_CANDIDATE=${kvaf_effective_candidate}"
echo "KVAF_HCCP_HCOMM_ROLE=initialization-bootstrap-only-not-aiv-hot-path"
echo "KVAF_HCCP_HCOMM_ABI_COMPATIBILITY=unproven-by-symbol-surface"
echo "KVAF_HCCP_HCOMM_HEADER_ABI=requires-version-matched-build-verification"
echo "KVAF_HCCP_HCOMM_LIFECYCLE_CONTRACT=endpoint-mr-channel-reverse-order-return-codes-required"
echo "KVAF_HCCP_BACKEND_READY=no"
echo "KVAF_A5_HCCP_CONTROL_PLANE_STATIC_AUDIT_OK"
echo "No HCCP/HAL function or hardware operation was executed." >&2
