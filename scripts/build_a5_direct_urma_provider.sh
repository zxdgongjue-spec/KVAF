#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
source "${SCRIPT_DIR}/a5_direct_urma_legacy_guard.sh"
kvaf_configure_direct_urma_state_paths "${ROOT_DIR}" || exit 2

KVAF_URMA_PERFTEST_ROOT=${KVAF_URMA_PERFTEST_ROOT:-/data/c00982601/urma_perftest}
KVAF_DIRECT_PROVIDER_BUILD_DIR=${KVAF_DIRECT_PROVIDER_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma-provider"}
KVAF_DIRECT_PROVIDER_BUILD_JOBS=${KVAF_DIRECT_PROVIDER_BUILD_JOBS:-8}
KVAF_DIRECT_PROVIDER_CC=${KVAF_DIRECT_PROVIDER_CC:-$(command -v gcc || true)}
KVAF_URMA_LIBRARY=${KVAF_URMA_LIBRARY:-}
KVAF_URMA_COMMON_LIBRARY=${KVAF_URMA_COMMON_LIBRARY:-}
KVAF_UMMU_UAPI_ROOT=${KVAF_UMMU_UAPI_ROOT:-}
KVAF_UMMU_UAPI_INCLUDE_DIR=${KVAF_UMMU_UAPI_INCLUDE_DIR:-}
KVAF_REFERENCE_PERFTEST_LIBRARY=${KVAF_REFERENCE_PERFTEST_LIBRARY:-}

die()
{
  echo "ERROR: $*" >&2
  exit 2
}

discover_ldconfig_library()
{
  local soname=$1
  local -a candidates=()
  mapfile -t candidates < <(
    ldconfig -p 2>/dev/null |
      awk -v soname="${soname}" '$1 == soname {print $NF}' |
      LC_ALL=C sort -u
  )
  if (( ${#candidates[@]} != 1 )); then
    die "expected one ${soname} from ldconfig; set its exact KVAF path"
  fi
  realpath -e "${candidates[0]}"
}

discover_needed_library()
{
  local object=$1
  local soname_regex=$2
  local -a candidates=()

  mapfile -t candidates < <(
    ldd "${object}" 2>/dev/null |
      awk -v pattern="${soname_regex}" \
        '$1 ~ pattern && $2 == "=>" {print $3}' |
      LC_ALL=C sort -u
  )
  if (( ${#candidates[@]} != 1 )); then
    die "expected one ${soname_regex} dependency for ${object}; set its exact KVAF path"
  fi
  realpath -e "${candidates[0]}"
}

elf_sha256()
{
  sha256sum "$1" | awk '{print $1}'
}

elf_build_id()
{
  LC_ALL=C readelf -n "$1" 2>/dev/null |
    awk '/Build ID:/ && id == "" {id=$3} END {print id}'
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

if [[ -z "${KVAF_DIRECT_PROVIDER_CC}" ||
      ! -x "${KVAF_DIRECT_PROVIDER_CC}" ]]; then
  die "a system C compiler is required for the Host/AICPU provider"
fi
if ! [[ "${KVAF_DIRECT_PROVIDER_BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  die "KVAF_DIRECT_PROVIDER_BUILD_JOBS must be positive"
fi
for command in \
  cmake cmp find flock ldconfig ldd nm readelf realpath sha256sum sort
do
  command -v "${command}" >/dev/null 2>&1 ||
    die "${command} is required"
done
KVAF_URMA_PERFTEST_ROOT=$(realpath -e "${KVAF_URMA_PERFTEST_ROOT}")
[[ -d "${KVAF_URMA_PERFTEST_ROOT}" ]] ||
  die "KVAF_URMA_PERFTEST_ROOT is not a directory"
if [[ -z "${KVAF_REFERENCE_PERFTEST_LIBRARY}" ]]; then
  ub_microbenchmark_reference="${ROOT_DIR}/../UB_MicroBenchmark_new/3rd/urma_perftest/liburma_perftest.so"
  if [[ -r "${ub_microbenchmark_reference}" ]]; then
    KVAF_REFERENCE_PERFTEST_LIBRARY=${ub_microbenchmark_reference}
  else
    KVAF_REFERENCE_PERFTEST_LIBRARY="${KVAF_URMA_PERFTEST_ROOT}/urma/code/output/tools/urma_perftest/liburma_perftest.so"
  fi
fi
KVAF_REFERENCE_PERFTEST_LIBRARY=$(
  realpath -e "${KVAF_REFERENCE_PERFTEST_LIBRARY}"
) || die "the reference liburma_perftest.so is unavailable"

required_source_paths=(
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/tools/urma_perftest/perftest_resources.h"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/tools/urma_perftest/perftest_resources.c"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/tools/urma_perftest/perftest_parameters.c"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/tools/urma_perftest/perftest_communication.c"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/include/urma_types.h"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/lib/urma/core/urma_main.c"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/lib/urma/core/urma_cmd.c"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/lib/urma/core/urma_cmd_tlv.c"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/lib/urma/core/urma_cp_api.c"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/lib/urma/core/urma_device.c"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/lib/urma/core/urma_dp_api.c"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/lib/urma/core/urma_format_convert.c"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/lib/urma/core/urma_log.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/kernelspace/include/udma/udma_abi.h"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_common.h"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_ops.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_buf.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_ctl.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_ctrlq_tp.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_db.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_hmap.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_jetty.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_tid.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_segment.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_jfs.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_jfc.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_jfr.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_log.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/include/ummu/ummu_api.h"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/ummu/ummu_common.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/ummu/ummu_level_block.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/ummu/ummu_seg_mng.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/ummu/ummu_mapt.c"
)
for path in "${required_source_paths[@]}"; do
  [[ -r "${path}" ]] ||
    die "incomplete urma_perftest source tree, missing ${path}"
done

if [[ -n "${KVAF_UMMU_UAPI_ROOT}" &&
      -n "${KVAF_UMMU_UAPI_INCLUDE_DIR}" ]]; then
  kvaf_ummu_uapi_root_real=$(realpath -e "${KVAF_UMMU_UAPI_ROOT}") ||
    die "KVAF_UMMU_UAPI_ROOT does not exist"
  kvaf_ummu_uapi_include_real=$(
    realpath -e "${KVAF_UMMU_UAPI_INCLUDE_DIR}"
  ) || die "KVAF_UMMU_UAPI_INCLUDE_DIR does not exist"
  [[ "${kvaf_ummu_uapi_root_real}" == \
      "${kvaf_ummu_uapi_include_real}" ]] ||
    die "KVAF_UMMU_UAPI_ROOT and KVAF_UMMU_UAPI_INCLUDE_DIR disagree"
  KVAF_UMMU_UAPI_ROOT=${kvaf_ummu_uapi_root_real}
elif [[ -n "${KVAF_UMMU_UAPI_ROOT}" ]]; then
  KVAF_UMMU_UAPI_ROOT=$(realpath -e "${KVAF_UMMU_UAPI_ROOT}") ||
    die "KVAF_UMMU_UAPI_ROOT does not exist"
elif [[ -n "${KVAF_UMMU_UAPI_INCLUDE_DIR}" ]]; then
  KVAF_UMMU_UAPI_ROOT=$(
    realpath -e "${KVAF_UMMU_UAPI_INCLUDE_DIR}"
  ) || die "KVAF_UMMU_UAPI_INCLUDE_DIR does not exist"
else
  die "missing full A5 UMMU UAPI; set KVAF_UMMU_UAPI_ROOT to a root containing include/uapi/linux/ummu_core.h with struct ummu_tid_info.{hw_cap,va,size}, HW_CAP_EXPAN, HW_CAP_DOUBLE_PLBI, HW_CAP_KCMD_PLBI, UMMU_IOCPLBI_VA and UMMU_IOCPLBI_ALL; the incomplete urma_perftest/drivers/ubus header and system libummu fallback are forbidden"
fi
KVAF_UMMU_UAPI_INCLUDE_DIR=${KVAF_UMMU_UAPI_ROOT}
KVAF_UMMU_UAPI_HEADER="${KVAF_UMMU_UAPI_ROOT}/include/uapi/linux/ummu_core.h"
[[ -r "${KVAF_UMMU_UAPI_HEADER}" ]] ||
  die "KVAF_UMMU_UAPI_ROOT does not contain include/uapi/linux/ummu_core.h"
legacy_ummu_uapi_root=$(
  realpath -e "${KVAF_URMA_PERFTEST_ROOT}/drivers/ubus/kernel"
)
case "${KVAF_UMMU_UAPI_ROOT}/" in
  "${legacy_ummu_uapi_root}/"*)
    die "the incomplete urma_perftest/drivers/ubus UMMU UAPI is forbidden"
    ;;
esac

if ! printf '%s\n' \
    '#include <stddef.h>' \
    '#include <stdint.h>' \
    '#include <sys/ioctl.h>' \
    '#include "ummu_api.h"' \
    'int main(void) {' \
    '  struct ummu_tid_info info = {0};' \
    '  uint64_t caps = info.hw_cap;' \
    '  info.va = 0; info.size = 0;' \
    '  caps |= HW_CAP_EXPAN | HW_CAP_DOUBLE_PLBI | HW_CAP_KCMD_PLBI;' \
    '  (void)info.va; (void)info.size;' \
    '  return (int)(caps ^ (uint64_t)UMMU_IOCPLBI_VA ^' \
    '      (uint64_t)UMMU_IOCPLBI_ALL);' \
    '}' |
    "${KVAF_DIRECT_PROVIDER_CC}" -std=gnu11 -Werror -x c \
      -fsyntax-only \
      -I"${KVAF_UMMU_UAPI_ROOT}" \
      -I"${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/include/ummu" \
      -; then
  die "KVAF_UMMU_UAPI_ROOT failed the full A5 UMMU ABI probe; required: struct ummu_tid_info.{hw_cap,va,size}, HW_CAP_EXPAN, HW_CAP_DOUBLE_PLBI, HW_CAP_KCMD_PLBI, UMMU_IOCPLBI_VA and UMMU_IOCPLBI_ALL; no fallback is permitted"
fi
if ! grep -Eq \
    '^[[:space:]]*#define[[:space:]]+UDMA_TID_SHIFT[[:space:]]+8(U)?[[:space:]]*$' \
    "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/kernelspace/include/udma/udma_abi.h"; then
  die "direct A5 provider requires the TID_SHIFT=8 UDMA source ABI"
fi

if [[ -z "${KVAF_URMA_LIBRARY}" ]]; then
  KVAF_URMA_LIBRARY=$(discover_ldconfig_library liburma.so.0)
fi
if [[ -z "${KVAF_URMA_COMMON_LIBRARY}" ]]; then
  KVAF_URMA_COMMON_LIBRARY=$(
    discover_ldconfig_library liburma_common.so.0
  )
fi
KVAF_URMA_LIBRARY=$(realpath -e "${KVAF_URMA_LIBRARY}")
KVAF_URMA_COMMON_LIBRARY=$(
  realpath -e "${KVAF_URMA_COMMON_LIBRARY}"
)
for library in \
  "${KVAF_URMA_LIBRARY}" \
  "${KVAF_URMA_COMMON_LIBRARY}"
do
  [[ -r "${library}" ]] ||
    die "runtime URMA library is not readable: ${library}"
  elf_header=$(readelf -h "${library}")
  grep -Eq 'Machine:[[:space:]]+AArch64' <<<"${elf_header}" ||
    die "runtime URMA library is not AArch64: ${library}"
done
[[ -r "${KVAF_REFERENCE_PERFTEST_LIBRARY}" ]] ||
  die "reference perftest library is not readable"
reference_perftest_header=$(
  readelf -h "${KVAF_REFERENCE_PERFTEST_LIBRARY}"
)
grep -Eq 'Machine:[[:space:]]+AArch64' \
  <<<"${reference_perftest_header}" ||
  die "reference perftest library is not AArch64"
reference_urma_library=$(
  discover_needed_library \
    "${KVAF_REFERENCE_PERFTEST_LIBRARY}" '^liburma\.so'
)
[[ "${reference_urma_library}" == "${KVAF_URMA_LIBRARY}" ]] ||
  die "reference perftest and KVAF resolve different liburma runtimes"

kvaf_check_legacy_direct_urma_state "${ROOT_DIR}" ||
  die "legacy direct URMA state blocks provider mutation"

if [[ -L "${KVAF_GLOBAL_STATE_DIR}" ||
      ( -e "${KVAF_GLOBAL_STATE_DIR}" &&
        ! -d "${KVAF_GLOBAL_STATE_DIR}" ) ]]; then
  die "global KVAF state path must be a real directory"
fi
mkdir -p "${KVAF_GLOBAL_STATE_DIR}"
chmod 0700 "${KVAF_GLOBAL_STATE_DIR}"
if [[ -L "${KVAF_DIRECT_STATE_DIR}" ||
      ( -e "${KVAF_DIRECT_STATE_DIR}" &&
        ! -d "${KVAF_DIRECT_STATE_DIR}" ) ]]; then
  die "direct URMA state path must be a real directory"
fi
mkdir -p "${KVAF_DIRECT_STATE_DIR}" \
  "$(dirname "${KVAF_DIRECT_DEPLOY_RUN_LOCK}")" \
  "$(dirname "${KVAF_DIRECT_ACTIVE_SESSION_LOCK}")"
chmod 0700 "${KVAF_DIRECT_STATE_DIR}"
for guarded_path in \
  "${KVAF_DIRECT_DEPLOY_RUN_LOCK}" \
  "${KVAF_DIRECT_ACTIVE_SESSION_LOCK}" \
  "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" \
  "${KVAF_DIRECT_QUARANTINE_FILE}"
do
  [[ ! -L "${guarded_path}" ]] ||
    die "persistent state path must not be a symlink: ${guarded_path}"
done
exec 9>"${KVAF_DIRECT_DEPLOY_RUN_LOCK}"
if ! flock -w 30 9; then
  die "another direct URMA deploy/run owns ${KVAF_DIRECT_DEPLOY_RUN_LOCK}"
fi
if [[ -e "${KVAF_DIRECT_QUARANTINE_FILE}" ||
      -L "${KVAF_DIRECT_QUARANTINE_FILE}" ||
      -e "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ||
      -L "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ]]; then
  die "direct URMA active/quarantine state blocks provider mutation"
fi
[[ ! -L "${KVAF_DIRECT_ACTIVE_SESSION_LOCK}" ]] ||
  die "active-session lock path must not be a symlink"
exec 8>"${KVAF_DIRECT_ACTIVE_SESSION_LOCK}"
flock -n 8 || die "a live direct URMA worker blocks provider mutation"

mkdir -p "$(dirname "${KVAF_DIRECT_PROVIDER_BUILD_DIR}")"
mkdir -p "${KVAF_DIRECT_PROVIDER_BUILD_DIR}"
success_stamp="${KVAF_DIRECT_PROVIDER_BUILD_DIR}/build.success"
rm -f "${success_stamp}"

staging_dir=$(
  mktemp -d "$(
    dirname "${KVAF_DIRECT_PROVIDER_BUILD_DIR}"
  )/.kvaf-direct-provider-build.XXXXXX"
)
cleanup()
{
  rm -rf "${staging_dir}"
}
trap cleanup EXIT

source_roots=(
  "${ROOT_DIR}/access_operator/a5/provider"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/tools/urma_perftest"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/include"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/lib/urma/core"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/kernelspace/include/udma"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/kernelspace/urma/code/include"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/kernelspace/urma/code/lib/urma"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/include"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/include/ummu"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/ummu"
  "${KVAF_UMMU_UAPI_ROOT}/include/uapi"
)
enumerate_provider_sources()
{
  local output_file=$1
  find -L "${source_roots[@]}" -type f \
    \( -name '*.c' -o -name '*.h' -o -name 'CMakeLists.txt' \
       -o -name '*.map' \) -print0 |
    LC_ALL=C sort -zu >"${output_file}"
}

hash_provider_source_list()
{
  local source_list=$1
  local output_file=$2
  local path
  local -a source_files=()

  while IFS= read -r -d '' path; do
    source_files+=("${path}")
  done <"${source_list}"
  (( ${#source_files[@]} != 0 )) ||
    die "no provider source files were selected"
  sha256sum "${source_files[@]}" >"${output_file}"
}

source_list_before="${staging_dir}/source.before.list"
source_list_after="${staging_dir}/source.after.list"
enumerate_provider_sources "${source_list_before}"
hash_provider_source_list \
  "${source_list_before}" "${staging_dir}/source.before.sha256"

cmake \
  -S "${ROOT_DIR}/access_operator/a5/provider" \
  -B "${staging_dir}/cmake" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER="${KVAF_DIRECT_PROVIDER_CC}" \
  -DKVAF_URMA_PERFTEST_ROOT="${KVAF_URMA_PERFTEST_ROOT}" \
  -DKVAF_URMA_LIBRARY="${KVAF_URMA_LIBRARY}" \
  -DKVAF_URMA_COMMON_LIBRARY="${KVAF_URMA_COMMON_LIBRARY}" \
  -DKVAF_UMMU_UAPI_ROOT="${KVAF_UMMU_UAPI_ROOT}"
cmake --build "${staging_dir}/cmake" \
  --clean-first \
  --parallel "${KVAF_DIRECT_PROVIDER_BUILD_JOBS}"

host_provider="${staging_dir}/cmake/libkvaf_urma_provider_host.so"
device_provider="${staging_dir}/cmake/libkvaf_urma_provider_device.so"
for provider in "${host_provider}" "${device_provider}"; do
  [[ -r "${provider}" ]] ||
    die "provider build did not produce ${provider}"
done

enumerate_provider_sources "${source_list_after}"
if ! cmp -s "${source_list_before}" "${source_list_after}"; then
  die "provider source file set changed during the build"
fi
hash_provider_source_list \
  "${source_list_after}" "${staging_dir}/source.after.sha256"
if ! cmp -s \
    "${staging_dir}/source.before.sha256" \
    "${staging_dir}/source.after.sha256"; then
  die "provider or urma_perftest source changed during the build"
fi
mv "${staging_dir}/source.after.sha256" \
  "${staging_dir}/source.sha256"

public_provider_symbols=(
  kvaf_urma_provider_open_v2
  kvaf_urma_provider_close_v2
  kvaf_urma_provider_private_abi_v2
  aicpu_kvaf_urma_open_v2
  aicpu_kvaf_urma_close_v2
)
validate_provider()
{
  local provider=$1
  local label=$2
  local dynamic_file="${staging_dir}/${label}.dynamic.txt"
  local ldd_file="${staging_dir}/${label}.ldd.txt"
  local provider_dynamic_symbols provider_all_symbols elf_header
  local symbol resolved_common

  provider_dynamic_symbols=$(nm -D --defined-only "${provider}")
  for symbol in "${public_provider_symbols[@]}"; do
    grep -Eq \
      "[[:space:]]${symbol}(@@KVAF_URMA_PROVIDER_2)?$" \
      <<<"${provider_dynamic_symbols}" ||
      die "${label} provider does not export ${symbol}"
  done
  if grep -Eq \
      '[[:space:]](perftest_|kvaf_perftest_|kvaf_udma_|kvaf_source_ummu_|create_duplex_ctx|destroy_ctx)' \
      <<<"${provider_dynamic_symbols}"; then
    die "${label} provider leaks an internal implementation symbol"
  fi
  provider_all_symbols=$(nm --defined-only "${provider}")
  for symbol in \
    kvaf_perftest_create_duplex_ctx_checked \
    kvaf_perftest_destroy_duplex_ctx_checked \
    kvaf_udma_extract_direct_queue_info \
    kvaf_udma_source_provider_owns_context \
    kvaf_udma_source_provider_owns_device \
    kvaf_udma_source_runtime_init \
    kvaf_udma_source_runtime_uninit \
    kvaf_udma_source_runtime_is_owned \
    kvaf_udma_source_public_abi_signature \
    kvaf_udma_source_public_abi_matches \
    kvaf_urma_core_has_dynamic_provider_scan_disabled \
    kvaf_udma_source_live_context_count \
    kvaf_udma_source_live_segment_count \
    kvaf_udma_source_live_token_count \
    kvaf_udma_source_live_jetty_count \
    kvaf_udma_source_live_jfc_count \
    kvaf_udma_source_orphan_segment_count \
    kvaf_udma_source_orphan_token_count \
    kvaf_udma_source_orphan_jetty_count \
    kvaf_udma_source_retry_orphan_cleanup \
    kvaf_source_ummu_capability \
    kvaf_source_ummu_get_status \
    kvaf_source_ummu_live_tid_count \
    kvaf_source_ummu_pending_operation_count \
    kvaf_source_ummu_retained_mapping_count \
    kvaf_source_ummu_retry_pending \
    kvaf_source_ummu_checked_contract_ready \
    kvaf_source_ummu_seg_tree_is_empty \
    ummu_allocate_tid \
    ummu_grant \
    ummu_ungrant \
    ummu_ungrant_by_token \
    ummu_free_tid
  do
    grep -Eq "[[:space:]]${symbol}$" \
      <<<"${provider_all_symbols}" ||
      die "${label} provider is missing internal symbol ${symbol}"
  done

  readelf -d "${provider}" >"${dynamic_file}"
  elf_header=$(readelf -h "${provider}")
  if ! grep -Eq 'Machine:[[:space:]]+AArch64' <<<"${elf_header}" ||
     ! grep -Eq 'Type:[[:space:]]+DYN' <<<"${elf_header}"; then
    die "${label} provider must be AArch64 ET_DYN"
  fi
  if grep -Eq 'Shared library: \[/|Shared library: \[[A-Za-z]:' \
      "${dynamic_file}"; then
    die "${label} provider records an absolute DT_NEEDED path"
  fi
  if grep -Eq 'lib(hcomm|urma_perftest)' "${dynamic_file}"; then
    die "${label} provider unexpectedly links HCOMM or prebuilt perftest"
  fi
  if grep -Eq '\((RPATH|RUNPATH)\)' "${dynamic_file}"; then
    die "${label} provider must not use RPATH/RUNPATH"
  fi
  if grep -Eq 'lib(urma|ummu|urma-udma)\.so' "${dynamic_file}"; then
    die "${label} unexpectedly links external liburma/UMMU/UDMA code"
  fi
  if readelf --version-info "${provider}" 2>/dev/null |
      grep -Fq GLIBCXX; then
    die "${label} C-only provider unexpectedly requires GLIBCXX"
  fi

  if ! ldd -r "${provider}" >"${ldd_file}" 2>&1; then
    cat "${ldd_file}" >&2
    die "ldd could not load the ${label} provider"
  fi
  if grep -Eq 'not found|undefined symbol' "${ldd_file}"; then
    cat "${ldd_file}" >&2
    die "${label} provider has unresolved runtime dependencies"
  fi
  if grep -Eq 'lib(urma|ummu|urma-udma)\.so' "${ldd_file}"; then
    cat "${ldd_file}" >&2
    die "${label} resolves external liburma/UMMU/UDMA code"
  fi
  resolved_common=$(
    awk '$1 ~ /^liburma_common\.so/ && $2 == "=>" {print $3}' \
      "${ldd_file}"
  )
  [[ -n "${resolved_common}" ]] ||
    die "ldd did not resolve ${label} liburma_common dependency"
  [[ "$(realpath -e "${resolved_common}")" == \
      "${KVAF_URMA_COMMON_LIBRARY}" ]] ||
    die "${label} link-time and runtime liburma_common paths differ"
}

validate_provider "${host_provider}" host-provider
validate_provider "${device_provider}" device-provider

public_abi_probe="${staging_dir}/cmake/kvaf_urma_public_abi_probe"
[[ -x "${public_abi_probe}" ]] ||
  die "provider build did not produce the public-ABI probe"
public_abi_probe_output=$("${public_abi_probe}") ||
  die "source UDMA/perftest public-ABI probe failed"
if ! [[ "${public_abi_probe_output}" =~ \
^KVAF_URMA_PUBLIC_ABI=MATCH:([0-9a-f]{16})$ ]]; then
  die "source UDMA/perftest public-ABI probe returned unexpected output"
fi
urma_public_abi_signature=${BASH_REMATCH[1]}

provider_abi_probe="${staging_dir}/cmake/kvaf_urma_provider_abi_probe"
[[ -x "${provider_abi_probe}" ]] ||
  die "provider build did not produce the private-ABI probe"
host_abi_probe_output=$(
  "${provider_abi_probe}" "${host_provider}" 1
) || die "Host provider private-ABI probe failed"
[[ "${host_abi_probe_output}" == \
    "KVAF_URMA_PROVIDER_PRIVATE_ABI=1" ]] ||
  die "Host provider private-ABI probe returned unexpected output"
device_abi_probe_output=$(
  "${provider_abi_probe}" "${device_provider}" 2
) || die "Device provider private-ABI probe failed"
[[ "${device_abi_probe_output}" == \
    "KVAF_URMA_PROVIDER_PRIVATE_ABI=2" ]] ||
  die "Device provider private-ABI probe returned unexpected output"

aicpu_contract_probe="${staging_dir}/cmake/kvaf_urma_provider_aicpu_contract_probe"
[[ -x "${aicpu_contract_probe}" ]] ||
  die "provider build did not produce the AICPU contract probe"
aicpu_contract_output=$(
  "${aicpu_contract_probe}" "${device_provider}"
) || die "Device provider AICPU return-contract probe failed"
[[ "${aicpu_contract_output}" == \
    "KVAF_URMA_PROVIDER_AICPU_CONTRACT=PASS" ]] ||
  die "Device provider AICPU return-contract probe returned unexpected output"

ummu_uapi_probe="${staging_dir}/cmake/kvaf_ummu_uapi_probe"
[[ -x "${ummu_uapi_probe}" ]] ||
  die "provider build did not produce the UMMU-UAPI probe"
ummu_uapi_probe_output=$("${ummu_uapi_probe}") ||
  die "full A5 UMMU-UAPI runtime probe failed"
if ! [[ "${ummu_uapi_probe_output}" =~ \
^KVAF_UMMU_UAPI_ABI=tid_info_size:[1-9][0-9]*,tid_offset:[0-9]+,hw_cap_offset:[0-9]+,va_offset:[0-9]+,size_offset:[0-9]+,alloc_tid:0x[0-9a-f]{16},free_tid:0x[0-9a-f]{16},plbi_va:0x[0-9a-f]{16},plbi_all:0x[0-9a-f]{16},cap_expan:0x[0-9a-f]{16},cap_double_plbi:0x[0-9a-f]{16},cap_kcmd_plbi:0x[0-9a-f]{16}$ ]]; then
  die "full A5 UMMU-UAPI probe returned unexpected output"
fi
ummu_uapi_abi=${ummu_uapi_probe_output#KVAF_UMMU_UAPI_ABI=}
ummu_uapi_abi_sha=$(
  printf '%s\n' "${ummu_uapi_probe_output}" | sha256sum | awk '{print $1}'
) || die "could not hash the UMMU-UAPI ABI probe"
[[ "${ummu_uapi_abi_sha}" =~ ^[0-9a-f]{64}$ ]] ||
  die "UMMU-UAPI ABI probe SHA-256 output is malformed"
ummu_uapi_probe_binary_sha=$(elf_sha256 "${ummu_uapi_probe}") ||
  die "could not hash the UMMU-UAPI probe executable"
[[ "${ummu_uapi_probe_binary_sha}" =~ ^[0-9a-f]{64}$ ]] ||
  die "UMMU-UAPI probe executable SHA-256 output is malformed"

host_provider_sha=$(elf_sha256 "${host_provider}") ||
  die "could not hash the Host provider"
device_provider_sha=$(elf_sha256 "${device_provider}") ||
  die "could not hash the Device provider"
[[ "${host_provider_sha}" =~ ^[0-9a-f]{64}$ &&
   "${device_provider_sha}" =~ ^[0-9a-f]{64}$ ]] ||
  die "provider SHA-256 output is malformed"
[[ "${host_provider_sha}" != "${device_provider_sha}" ]] ||
  die "Host and Device providers must be distinct binaries"
source_manifest_sha=$(
  elf_sha256 "${staging_dir}/source.sha256"
) || die "could not hash the provider source manifest"
[[ "${source_manifest_sha}" =~ ^[0-9a-f]{64}$ ]] ||
  die "provider source-manifest SHA-256 output is malformed"

capture_elf_identity()
{
  local library=$1
  local sha_before sha_after build_id

  sha_before=$(elf_sha256 "${library}") ||
    die "could not hash ELF provenance input: ${library}"
  build_id=$(elf_build_id "${library}") ||
    die "could not read ELF provenance build ID: ${library}"
  sha_after=$(elf_sha256 "${library}") ||
    die "could not re-hash ELF provenance input: ${library}"
  [[ "${sha_before}" == "${sha_after}" ]] ||
    die "ELF input changed while recording provenance: ${library}"
  [[ "${sha_before}" =~ ^[0-9a-f]{64}$ ]] ||
    die "ELF provenance SHA-256 output is malformed: ${library}"
  [[ "${build_id}" =~ ^[0-9A-Fa-f]+$ ]] ||
    die "ELF provenance input has no valid build ID: ${library}"
  captured_elf_sha=${sha_before}
  captured_elf_build_id=${build_id}
}

capture_elf_identity "${KVAF_URMA_LIBRARY}"
urma_library_sha=${captured_elf_sha}
urma_library_build_id=${captured_elf_build_id}
capture_elf_identity "${KVAF_URMA_COMMON_LIBRARY}"
urma_common_library_sha=${captured_elf_sha}
urma_common_library_build_id=${captured_elf_build_id}
ummu_uapi_header_sha=$(elf_sha256 "${KVAF_UMMU_UAPI_HEADER}") ||
  die "could not hash the pinned full A5 UMMU UAPI header"
[[ "${ummu_uapi_header_sha}" =~ ^[0-9a-f]{64}$ ]] ||
  die "UMMU UAPI header SHA-256 output is malformed"
capture_elf_identity "${KVAF_REFERENCE_PERFTEST_LIBRARY}"
reference_perftest_library_sha=${captured_elf_sha}
reference_perftest_library_build_id=${captured_elf_build_id}

cat >"${staging_dir}/build.manifest" <<EOF
schema=kvaf-direct-urma-provider-v2
host_provider_filename=libkvaf_urma_provider_host.so
host_provider_execution_domain=host-process
host_provider_sha256=${host_provider_sha}
host_private_abi=1
host_private_abi_name=perftest-aiv-v1-host
device_provider_filename=libkvaf_urma_provider_device.so
device_provider_execution_domain=aicpu
device_provider_sha256=${device_provider_sha}
device_private_abi=2
device_private_abi_name=perftest-aiv-v1-device
source_manifest_sha256=${source_manifest_sha}
urma_public_abi_signature=${urma_public_abi_signature}
urma_library_path=${KVAF_URMA_LIBRARY}
urma_library_sha256=${urma_library_sha}
urma_library_build_id=${urma_library_build_id}
urma_library_linkage=abi-reference-only
urma_common_library_path=${KVAF_URMA_COMMON_LIBRARY}
urma_common_library_sha256=${urma_common_library_sha}
urma_common_library_build_id=${urma_common_library_build_id}
udma_provider_mode=source-built-internal
urma_core_mode=source-linked-no-dynamic-providers
udma_source_capability_magic=0x4b56414655444d41
udma_runtime_capability=0x0000000000020001
ummu_provider_mode=source-built-internal
ummu_source_capability_magic=0x4b564146554d4d55
ummu_runtime_capability=0x0000000000010001
ummu_uapi_root=${KVAF_UMMU_UAPI_ROOT}
ummu_uapi_header_path=${KVAF_UMMU_UAPI_HEADER}
ummu_uapi_header_sha256=${ummu_uapi_header_sha}
ummu_uapi_abi=${ummu_uapi_abi}
ummu_uapi_abi_sha256=${ummu_uapi_abi_sha}
ummu_uapi_probe_filename=kvaf_ummu_uapi_probe
ummu_uapi_probe_sha256=${ummu_uapi_probe_binary_sha}
udma_teardown_contract=checked-retryable-v1
reference_perftest_abi=perftest-aiv-v1
reference_perftest_library_path=${KVAF_REFERENCE_PERFTEST_LIBRARY}
reference_perftest_library_sha256=${reference_perftest_library_sha}
reference_perftest_library_build_id=${reference_perftest_library_build_id}
EOF
build_manifest_sha=$(
  elf_sha256 "${staging_dir}/build.manifest"
) || die "could not hash the provider build manifest"
[[ "${build_manifest_sha}" =~ ^[0-9a-f]{64}$ ]] ||
  die "provider build-manifest SHA-256 output is malformed"
cat >"${staging_dir}/build.success" <<EOF
schema=kvaf-direct-urma-provider-success-v2
host_provider_sha256=${host_provider_sha}
device_provider_sha256=${device_provider_sha}
build_manifest_sha256=${build_manifest_sha}
EOF

[[ "$(manifest_value host_provider_sha256 \
      "${staging_dir}/build.success")" == "${host_provider_sha}" &&
    "$(manifest_value device_provider_sha256 \
      "${staging_dir}/build.success")" == "${device_provider_sha}" ]] ||
  die "internal success-stamp validation failed"

publish_dir=$(
  mktemp -d "${KVAF_DIRECT_PROVIDER_BUILD_DIR}/.publish.XXXXXX"
)
cp "${host_provider}" \
  "${publish_dir}/libkvaf_urma_provider_host.so"
cp "${device_provider}" \
  "${publish_dir}/libkvaf_urma_provider_device.so"
cp "${staging_dir}/source.sha256" "${publish_dir}/source.sha256"
cp "${staging_dir}/build.manifest" "${publish_dir}/build.manifest"
cp "${staging_dir}/host-provider.dynamic.txt" \
  "${publish_dir}/host-provider.dynamic.txt"
cp "${staging_dir}/host-provider.ldd.txt" \
  "${publish_dir}/host-provider.ldd.txt"
cp "${staging_dir}/device-provider.dynamic.txt" \
  "${publish_dir}/device-provider.dynamic.txt"
cp "${staging_dir}/device-provider.ldd.txt" \
  "${publish_dir}/device-provider.ldd.txt"
cp "${ummu_uapi_probe}" "${publish_dir}/kvaf_ummu_uapi_probe"
cp "${staging_dir}/build.success" "${publish_dir}/build.success"
chmod 0644 "${publish_dir}/"*
chmod 0755 "${publish_dir}/kvaf_ummu_uapi_probe"

for artifact in \
  libkvaf_urma_provider_host.so \
  libkvaf_urma_provider_device.so \
  source.sha256 \
  build.manifest \
  host-provider.dynamic.txt \
  host-provider.ldd.txt \
  device-provider.dynamic.txt \
  device-provider.ldd.txt \
  kvaf_ummu_uapi_probe
do
  mv -f "${publish_dir}/${artifact}" \
    "${KVAF_DIRECT_PROVIDER_BUILD_DIR}/${artifact}"
done
mv -f "${publish_dir}/build.success" "${success_stamp}"
rmdir "${publish_dir}"

echo "KVAF_A5_DIRECT_URMA_PROVIDER_BUILD_OK"
echo "HOST_PROVIDER=${KVAF_DIRECT_PROVIDER_BUILD_DIR}/libkvaf_urma_provider_host.so"
echo "DEVICE_PROVIDER=${KVAF_DIRECT_PROVIDER_BUILD_DIR}/libkvaf_urma_provider_device.so"
echo "SOURCE_MANIFEST=${KVAF_DIRECT_PROVIDER_BUILD_DIR}/source.sha256"
echo "BUILD_MANIFEST=${KVAF_DIRECT_PROVIDER_BUILD_DIR}/build.manifest"
echo "HOST_PROVIDER_SHA256=${host_provider_sha}"
echo "DEVICE_PROVIDER_SHA256=${device_provider_sha}"
echo "URMA_LIBRARY=${KVAF_URMA_LIBRARY}"
echo "UMMU_UAPI_ROOT=${KVAF_UMMU_UAPI_ROOT}"
echo "UMMU_UAPI_HEADER_SHA256=${ummu_uapi_header_sha}"
echo "UMMU_UAPI_ABI=${ummu_uapi_abi}"
echo "UMMU_UAPI_ABI_SHA256=${ummu_uapi_abi_sha}"
echo "UMMU_UAPI_PROBE_SHA256=${ummu_uapi_probe_binary_sha}"
echo "UDMA_TEARDOWN_CONTRACT=checked-retryable-v1"
echo "REFERENCE_PERFTEST_LIBRARY=${KVAF_REFERENCE_PERFTEST_LIBRARY}"
