#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
source "${SCRIPT_DIR}/a5_direct_urma_legacy_guard.sh"
kvaf_configure_direct_urma_state_paths "${ROOT_DIR}" || exit 2

KVAF_UMDK_ROOT=${KVAF_UMDK_ROOT:-/data/c00982601/umdk}
KVAF_LIBUMMU_ROOT=${KVAF_LIBUMMU_ROOT:-/data/c00982601/libummu}
KVAF_DIRECT_PROVIDER_BUILD_DIR=${KVAF_DIRECT_PROVIDER_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma-provider"}
KVAF_DIRECT_PROVIDER_BUILD_JOBS=${KVAF_DIRECT_PROVIDER_BUILD_JOBS:-8}
KVAF_DIRECT_PROVIDER_CC=${KVAF_DIRECT_PROVIDER_CC:-$(command -v gcc || true)}
KVAF_RUNTIME_UDMA_UAPI_HEADER=${KVAF_RUNTIME_UDMA_UAPI_HEADER:-}
KVAF_RUNTIME_UMMU_UAPI_HEADER=${KVAF_RUNTIME_UMMU_UAPI_HEADER:-}

# This backend private-casts UMDK queue objects.  Compatibility therefore
# means one audited source pair, not merely "some clean UMDK checkout" whose
# public headers happen to compile.
kvaf_expected_umdk_commit=12917421beb93c1351eb2f124bb2f34612d8bf2d
kvaf_expected_umdk_version=26.06.0
kvaf_expected_umdk_remote=https://gitcode.com/openeuler/umdk.git
kvaf_expected_libummu_remote=https://gitcode.com/openeuler/libummu.git
kvaf_expected_libummu_commit=70de020f776d8def0d8a8941cc7cbc5928d66c10
kvaf_expected_libummu_version=1.0.5
kvaf_expected_libummu_abi=1

die()
{
  echo "ERROR: $*" >&2
  exit 2
}

sha256_file()
{
  sha256sum "$1" | awk '{print $1}'
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

starlark_string_value()
{
  local key=$1
  local file=$2
  awk -v key="${key}" '
    $0 ~ "^[[:space:]]*" key "[[:space:]]*=[[:space:]]*\"[^\"]+\"[[:space:]]*$" {
      value = $0
      sub("^[[:space:]]*" key "[[:space:]]*=[[:space:]]*\"", "", value)
      sub("\"[[:space:]]*$", "", value)
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

require_clean_git_source()
{
  local source_root=$1
  local label=$2
  local top status

  git -C "${source_root}" rev-parse --is-inside-work-tree \
    >/dev/null 2>&1 || die "${label} is not a Git worktree"
  top=$(git -C "${source_root}" rev-parse --show-toplevel) ||
    die "could not locate ${label} Git root"
  top=$(realpath -e "${top}") ||
    die "could not canonicalize ${label} Git root"
  [[ "${top}" == "${source_root}" ]] ||
    die "${label} must point at its Git repository root"
  status=$(git -C "${source_root}" status \
    --porcelain=v1 --untracked-files=all) ||
    die "could not inspect ${label} source state"
  [[ -z "${status}" ]] ||
    die "${label} source tree must be clean; commit or remove local changes"
}

require_official_git_origin()
{
  local source_root=$1
  local expected=$2
  local label=$3
  local actual

  actual=$(git -C "${source_root}" remote get-url origin) ||
    die "${label} has no readable origin remote"
  # Do not print the actual URL: a badly configured shared account may have
  # embedded a credential in it.  Reject every non-canonical source instead.
  [[ "${actual}" == "${expected}" ]] ||
    die "${label} origin is not the canonical official GitCode repository"
  printf '%s\n' "${expected}"
}

if [[ -z "${KVAF_DIRECT_PROVIDER_CC}" ||
      ! -x "${KVAF_DIRECT_PROVIDER_CC}" ]]; then
  die "a system C compiler is required for the Host/AICPU provider"
fi
if ! [[ "${KVAF_DIRECT_PROVIDER_BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  die "KVAF_DIRECT_PROVIDER_BUILD_JOBS must be positive"
fi
for command in \
  awk cmake cmp find flock git ldd nm readelf realpath sha256sum sort uname
do
  command -v "${command}" >/dev/null 2>&1 ||
    die "${command} is required"
done

KVAF_UMDK_ROOT=$(realpath -e "${KVAF_UMDK_ROOT}") ||
  die "KVAF_UMDK_ROOT does not exist"
KVAF_LIBUMMU_ROOT=$(realpath -e "${KVAF_LIBUMMU_ROOT}") ||
  die "KVAF_LIBUMMU_ROOT does not exist"
[[ -d "${KVAF_UMDK_ROOT}" ]] ||
  die "KVAF_UMDK_ROOT is not a directory"
[[ -d "${KVAF_LIBUMMU_ROOT}" ]] ||
  die "KVAF_LIBUMMU_ROOT is not a directory"

require_clean_git_source "${KVAF_UMDK_ROOT}" UMDK
require_clean_git_source "${KVAF_LIBUMMU_ROOT}" libummu
umdk_git_origin=$(require_official_git_origin \
  "${KVAF_UMDK_ROOT}" "${kvaf_expected_umdk_remote}" UMDK)
libummu_git_origin=$(require_official_git_origin \
  "${KVAF_LIBUMMU_ROOT}" "${kvaf_expected_libummu_remote}" libummu)

umdk_git_commit=$(git -C "${KVAF_UMDK_ROOT}" rev-parse HEAD) ||
  die "could not read UMDK commit"
umdk_git_tree=$(git -C "${KVAF_UMDK_ROOT}" rev-parse 'HEAD^{tree}') ||
  die "could not read UMDK tree"
libummu_git_commit=$(git -C "${KVAF_LIBUMMU_ROOT}" rev-parse HEAD) ||
  die "could not read libummu commit"
libummu_git_tree=$(git -C "${KVAF_LIBUMMU_ROOT}" \
  rev-parse 'HEAD^{tree}') || die "could not read libummu tree"
for oid in \
  "${umdk_git_commit}" "${umdk_git_tree}" \
  "${libummu_git_commit}" "${libummu_git_tree}"
do
  [[ "${oid}" =~ ^[0-9a-f]{40}$ ]] ||
    die "source Git object ID is malformed: ${oid}"
done
[[ "${umdk_git_commit}" == "${kvaf_expected_umdk_commit}" ]] ||
  die "UMDK HEAD ${umdk_git_commit} is not the audited ${kvaf_expected_umdk_commit}"

umdk_deps_file="${KVAF_UMDK_ROOT}/src/urma/bazel/urma_deps.bzl"
[[ -r "${umdk_deps_file}" ]] ||
  die "UMDK does not contain src/urma/bazel/urma_deps.bzl"
expected_libummu_remote=$(starlark_string_value \
  LIBUMMU_REMOTE "${umdk_deps_file}") ||
  die "UMDK does not declare one LIBUMMU_REMOTE"
expected_libummu_commit=$(starlark_string_value \
  LIBUMMU_COMMIT "${umdk_deps_file}") ||
  die "UMDK does not declare one LIBUMMU_COMMIT"
expected_libummu_version=$(starlark_string_value \
  LIBUMMU_VERSION "${umdk_deps_file}") ||
  die "UMDK does not declare one LIBUMMU_VERSION"
expected_libummu_abi=$(starlark_string_value \
  LIBUMMU_ABI_VERSION "${umdk_deps_file}") ||
  die "UMDK does not declare one LIBUMMU_ABI_VERSION"
[[ "${expected_libummu_commit}" =~ ^[0-9a-f]{40}$ ]] ||
  die "UMDK LIBUMMU_COMMIT is malformed"
[[ "${libummu_git_commit}" == "${expected_libummu_commit}" ]] ||
  die "libummu HEAD ${libummu_git_commit} does not match UMDK pin ${expected_libummu_commit}"
[[ "${expected_libummu_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
  die "UMDK LIBUMMU_VERSION is malformed"
[[ "${expected_libummu_abi}" =~ ^[0-9]+$ ]] ||
  die "UMDK LIBUMMU_ABI_VERSION is malformed"
[[ "${expected_libummu_remote}" == "${kvaf_expected_libummu_remote}" &&
   "${expected_libummu_commit}" == "${kvaf_expected_libummu_commit}" &&
   "${expected_libummu_version}" == "${kvaf_expected_libummu_version}" &&
   "${expected_libummu_abi}" == "${kvaf_expected_libummu_abi}" ]] ||
  die "audited UMDK/libummu dependency tuple changed"

umdk_release_version=$(awk '
  /^[[:space:]]*Version[[:space:]]*:[[:space:]]*/ {
    value = $0
    sub(/^[[:space:]]*Version[[:space:]]*:[[:space:]]*/, "", value)
    count++
  }
  END {
    if (count != 1 || value == "") exit 1
    print value
  }
' "${KVAF_UMDK_ROOT}/umdk.spec") ||
  die "could not read the unique UMDK package version"
[[ "${umdk_release_version}" == "${kvaf_expected_umdk_version}" ]] ||
  die "UMDK release ${umdk_release_version} is not the audited ${kvaf_expected_umdk_version}"

KVAF_UMDK_URMA_ROOT="${KVAF_UMDK_ROOT}/src/urma"
KVAF_URMA_COMMON_ROOT="${KVAF_UMDK_URMA_ROOT}/common"
KVAF_URMA_CORE_ROOT="${KVAF_UMDK_URMA_ROOT}/lib/urma/core"
KVAF_PERFTEST_ROOT="${KVAF_UMDK_URMA_ROOT}/tools/urma_perftest"
KVAF_UDMA_ROOT="${KVAF_UMDK_URMA_ROOT}/hw/udma"
KVAF_UDMA_UAPI_HEADER="${KVAF_UDMA_ROOT}/kernel_headers/udma_abi.h"
KVAF_UMMU_UAPI_HEADER="${KVAF_LIBUMMU_ROOT}/kernel_headers/ummu_core.h"

kernel_release=$(uname -r) || die "could not read the running kernel release"
runtime_udma_uapi_candidates=()
if [[ -n "${KVAF_RUNTIME_UDMA_UAPI_HEADER}" ]]; then
  runtime_udma_uapi_candidates+=("${KVAF_RUNTIME_UDMA_UAPI_HEADER}")
else
  for candidate in \
    "/usr/src/kernels/${kernel_release}/include/uapi/ub/urma/udma/udma_abi.h" \
    "/lib/modules/${kernel_release}/build/include/uapi/ub/urma/udma/udma_abi.h" \
    "${ROOT_DIR}/../kernel-uapi/${kernel_release}/udma_abi.h"
  do
    [[ -r "${candidate}" ]] && runtime_udma_uapi_candidates+=("${candidate}")
  done
fi
(( ${#runtime_udma_uapi_candidates[@]} != 0 )) ||
  die "running-kernel UDMA UAPI not found; set KVAF_RUNTIME_UDMA_UAPI_HEADER"
KVAF_RUNTIME_UDMA_UAPI_HEADER=$(
  realpath -e "${runtime_udma_uapi_candidates[0]}"
) || die "could not canonicalize the running-kernel UDMA UAPI"
for candidate in "${runtime_udma_uapi_candidates[@]}"; do
  candidate=$(realpath -e "${candidate}") ||
    die "could not canonicalize running-kernel UDMA UAPI candidate"
  cmp -s "${KVAF_UDMA_UAPI_HEADER}" "${candidate}" ||
    die "official UMDK and running-kernel UDMA UAPI headers differ: ${candidate}"
done
runtime_udma_uapi_header_sha=$(sha256_file \
  "${KVAF_RUNTIME_UDMA_UAPI_HEADER}")

runtime_ummu_uapi_candidates=()
if [[ -n "${KVAF_RUNTIME_UMMU_UAPI_HEADER}" ]]; then
  runtime_ummu_uapi_candidates+=("${KVAF_RUNTIME_UMMU_UAPI_HEADER}")
else
  for candidate in \
    "/usr/src/kernels/${kernel_release}/include/uapi/linux/ummu_core.h" \
    "/lib/modules/${kernel_release}/build/include/uapi/linux/ummu_core.h" \
    "${ROOT_DIR}/../kernel-uapi/${kernel_release}/ummu_core.h"
  do
    [[ -r "${candidate}" ]] && runtime_ummu_uapi_candidates+=("${candidate}")
  done
fi
(( ${#runtime_ummu_uapi_candidates[@]} != 0 )) ||
  die "running-kernel UMMU UAPI not found; install matching kernel-devel or set KVAF_RUNTIME_UMMU_UAPI_HEADER"
KVAF_RUNTIME_UMMU_UAPI_HEADER=$(
  realpath -e "${runtime_ummu_uapi_candidates[0]}"
) || die "could not canonicalize the running-kernel UMMU UAPI"
for candidate in "${runtime_ummu_uapi_candidates[@]}"; do
  candidate=$(realpath -e "${candidate}") ||
    die "could not canonicalize running-kernel UMMU UAPI candidate"
  cmp -s "${KVAF_UMMU_UAPI_HEADER}" "${candidate}" ||
    die "paired libummu and running-kernel UMMU UAPI headers differ: ${candidate}"
done
runtime_ummu_uapi_header_sha=$(sha256_file \
  "${KVAF_RUNTIME_UMMU_UAPI_HEADER}")
runtime_ummu_uapi_match=verified-exact

required_source_paths=(
  "${KVAF_URMA_COMMON_ROOT}/ub_bitmap.c"
  "${KVAF_URMA_COMMON_ROOT}/ub_dstring.c"
  "${KVAF_URMA_COMMON_ROOT}/ub_get_clock.c"
  "${KVAF_URMA_COMMON_ROOT}/ub_hash.c"
  "${KVAF_URMA_COMMON_ROOT}/ub_hmap.c"
  "${KVAF_URMA_COMMON_ROOT}/ub_util.c"
  "${KVAF_URMA_CORE_ROOT}/urma_main.c"
  "${KVAF_URMA_CORE_ROOT}/urma_cmd.c"
  "${KVAF_URMA_CORE_ROOT}/urma_cmd_tlv.c"
  "${KVAF_URMA_CORE_ROOT}/urma_cp_api.c"
  "${KVAF_URMA_CORE_ROOT}/urma_device.c"
  "${KVAF_URMA_CORE_ROOT}/urma_dp_api.c"
  "${KVAF_URMA_CORE_ROOT}/urma_format_convert.c"
  "${KVAF_URMA_CORE_ROOT}/urma_log.c"
  "${KVAF_URMA_CORE_ROOT}/urma_perf.c"
  "${KVAF_URMA_CORE_ROOT}/ubagg_stats.c"
  "${KVAF_PERFTEST_ROOT}/perftest_log.c"
  "${KVAF_PERFTEST_ROOT}/perftest_parameters.c"
  "${KVAF_PERFTEST_ROOT}/perftest_resources.c"
  "${KVAF_PERFTEST_ROOT}/perftest_mgmt.c"
  "${KVAF_PERFTEST_ROOT}/perftest_mgmt.h"
  "${KVAF_PERFTEST_ROOT}/perftest_mgmt_tcp.c"
  "${KVAF_PERFTEST_ROOT}/perftest_mgmt_tcp.h"
  "${KVAF_PERFTEST_ROOT}/perftest_mgmt_ub.c"
  "${KVAF_PERFTEST_ROOT}/perftest_mgmt_ub.h"
  "${KVAF_UDMA_UAPI_HEADER}"
  "${KVAF_UDMA_ROOT}/udma_u_common.h"
  "${KVAF_UDMA_ROOT}/udma_u_ops.c"
  "${KVAF_UDMA_ROOT}/udma_u_buf.c"
  "${KVAF_UDMA_ROOT}/udma_u_ctl.c"
  "${KVAF_UDMA_ROOT}/udma_u_ctrlq_tp.c"
  "${KVAF_UDMA_ROOT}/udma_u_db.c"
  "${KVAF_UDMA_ROOT}/udma_u_jetty.c"
  "${KVAF_UDMA_ROOT}/udma_u_tid.c"
  "${KVAF_UDMA_ROOT}/udma_u_segment.c"
  "${KVAF_UDMA_ROOT}/udma_u_jfs.c"
  "${KVAF_UDMA_ROOT}/udma_u_jfc.c"
  "${KVAF_UDMA_ROOT}/udma_u_jfr.c"
  "${KVAF_UDMA_ROOT}/udma_u_log.c"
  "${KVAF_LIBUMMU_ROOT}/include/ummu_api.h"
  "${KVAF_UMMU_UAPI_HEADER}"
  "${KVAF_LIBUMMU_ROOT}/ummu_common.c"
  "${KVAF_LIBUMMU_ROOT}/ummu_level_block.c"
  "${KVAF_LIBUMMU_ROOT}/ummu_seg_mng.c"
  "${KVAF_LIBUMMU_ROOT}/ummu_mapt.c"
)
for path in "${required_source_paths[@]}"; do
  [[ -r "${path}" ]] ||
    die "incomplete paired UMDK/libummu tree, missing ${path}"
done
[[ ! -e "${KVAF_UDMA_ROOT}/udma_u_hmap.c" ]] ||
  die "unexpected legacy UDMA hmap source in official UMDK root"

if ! grep -Eq \
    '^[[:space:]]*#define[[:space:]]+UDMA_TID_SHIFT[[:space:]]+8U?[[:space:]]*$' \
    "${KVAF_UDMA_UAPI_HEADER}" ||
   ! grep -Eq \
    '^[[:space:]]*#define[[:space:]]+MAP_INDEX_SHIFT[[:space:]]+4[[:space:]]*$' \
    "${KVAF_UDMA_UAPI_HEADER}" ||
   ! grep -Eq \
    '^[[:space:]]*#define[[:space:]]+UDMA_SEGMENT_ACCESS_GUARD[[:space:]]+0xf[[:space:]]*$' \
    "${KVAF_UDMA_UAPI_HEADER}"; then
  die "official UDMA UAPI constants do not match the required A5 ABI"
fi

if ! printf '%s\n' \
    '#include <stdint.h>' \
    '#include <sys/ioctl.h>' \
    '#include "ummu_api.h"' \
    'int main(void) {' \
    '  struct ummu_tid_info info = {0};' \
    '  uint64_t caps = info.hw_cap;' \
    '  info.va = 0; info.size = 0;' \
    '  caps |= HW_CAP_EXPAN | HW_CAP_DOUBLE_PLBI | HW_CAP_KCMD_PLBI;' \
    '  caps |= HW_CAP_FREE_BIT | HW_CAP_PPLBI;' \
    '  return (int)(caps ^ (uint64_t)UMMU_IOCPLBI_VA ^' \
    '      (uint64_t)UMMU_IOCPLBI_ALL);' \
    '}' |
    "${KVAF_DIRECT_PROVIDER_CC}" -std=gnu11 -Werror -x c \
      -fsyntax-only \
      -I"${KVAF_LIBUMMU_ROOT}/include" \
      -I"${KVAF_LIBUMMU_ROOT}/kernel_headers" \
      -; then
  die "paired libummu failed the full A5 UMMU ABI syntax probe"
fi

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

kvaf_source_roots=(
  "${ROOT_DIR}/access_operator/a5/provider"
)
umdk_source_roots=(
  "${umdk_deps_file}"
  "${KVAF_UMDK_ROOT}/umdk.spec"
  "${KVAF_URMA_COMMON_ROOT}"
  "${KVAF_UMDK_URMA_ROOT}/lib/urma/core"
  "${KVAF_UMDK_URMA_ROOT}/lib/urma/bond/include"
  "${KVAF_PERFTEST_ROOT}"
  "${KVAF_UDMA_ROOT}"
)
libummu_source_roots=(
  "${KVAF_LIBUMMU_ROOT}/CMakeLists.txt"
  "${KVAF_LIBUMMU_ROOT}/include"
  "${KVAF_LIBUMMU_ROOT}/kernel_headers"
  "${KVAF_LIBUMMU_ROOT}/ummu_common.c"
  "${KVAF_LIBUMMU_ROOT}/ummu_common.h"
  "${KVAF_LIBUMMU_ROOT}/ummu_level_block.c"
  "${KVAF_LIBUMMU_ROOT}/ummu_level_block.h"
  "${KVAF_LIBUMMU_ROOT}/ummu_mapt.c"
  "${KVAF_LIBUMMU_ROOT}/ummu_mapt.h"
  "${KVAF_LIBUMMU_ROOT}/ummu_queue.c"
  "${KVAF_LIBUMMU_ROOT}/ummu_queue.h"
  "${KVAF_LIBUMMU_ROOT}/ummu_resource.c"
  "${KVAF_LIBUMMU_ROOT}/ummu_resource.h"
  "${KVAF_LIBUMMU_ROOT}/ummu_seg_mng.c"
  "${KVAF_LIBUMMU_ROOT}/ummu_seg_mng.h"
)

enumerate_source_group()
{
  local output_file=$1
  shift
  find -L "$@" -type f \
    \( -name '*.c' -o -name '*.h' -o -name 'CMakeLists.txt' \
       -o -name '*.map' -o -name '*.bzl' -o -name '*.spec' \) \
    -print0 | LC_ALL=C sort -zu >"${output_file}"
}

hash_source_list()
{
  local source_list=$1
  local output_file=$2
  local path
  local -a source_files=()

  while IFS= read -r -d '' path; do
    source_files+=("${path}")
  done <"${source_list}"
  (( ${#source_files[@]} != 0 )) ||
    die "no source files were selected for ${output_file}"
  sha256sum "${source_files[@]}" >"${output_file}"
}

enumerate_all_sources()
{
  local suffix=$1
  enumerate_source_group "${staging_dir}/kvaf.${suffix}.list" \
    "${kvaf_source_roots[@]}"
  enumerate_source_group "${staging_dir}/umdk.${suffix}.list" \
    "${umdk_source_roots[@]}"
  enumerate_source_group "${staging_dir}/libummu.${suffix}.list" \
    "${libummu_source_roots[@]}"
  cat \
    "${staging_dir}/kvaf.${suffix}.list" \
    "${staging_dir}/umdk.${suffix}.list" \
    "${staging_dir}/libummu.${suffix}.list" |
    LC_ALL=C sort -zu >"${staging_dir}/source.${suffix}.list"
  hash_source_list "${staging_dir}/source.${suffix}.list" \
    "${staging_dir}/source.${suffix}.sha256"
  hash_source_list "${staging_dir}/umdk.${suffix}.list" \
    "${staging_dir}/umdk-source.${suffix}.sha256"
  hash_source_list "${staging_dir}/libummu.${suffix}.list" \
    "${staging_dir}/libummu-source.${suffix}.sha256"
}

enumerate_all_sources before

cmake \
  -S "${ROOT_DIR}/access_operator/a5/provider" \
  -B "${staging_dir}/cmake" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER="${KVAF_DIRECT_PROVIDER_CC}" \
  -DKVAF_UMDK_ROOT="${KVAF_UMDK_ROOT}" \
  -DKVAF_LIBUMMU_ROOT="${KVAF_LIBUMMU_ROOT}"
cmake --build "${staging_dir}/cmake" \
  --clean-first \
  --parallel "${KVAF_DIRECT_PROVIDER_BUILD_JOBS}"

host_provider="${staging_dir}/cmake/libkvaf_urma_provider_host.so"
device_provider="${staging_dir}/cmake/libkvaf_urma_provider_device.so"
for provider in "${host_provider}" "${device_provider}"; do
  [[ -r "${provider}" ]] ||
    die "provider build did not produce ${provider}"
done

enumerate_all_sources after
for group in source umdk-source libummu-source; do
  if ! cmp -s \
      "${staging_dir}/${group}.before.sha256" \
      "${staging_dir}/${group}.after.sha256"; then
    die "${group} inputs changed during the provider build"
  fi
  mv "${staging_dir}/${group}.after.sha256" \
    "${staging_dir}/${group}.sha256"
done
require_clean_git_source "${KVAF_UMDK_ROOT}" UMDK
require_clean_git_source "${KVAF_LIBUMMU_ROOT}" libummu
[[ "$(git -C "${KVAF_UMDK_ROOT}" rev-parse HEAD)" == \
      "${umdk_git_commit}" &&
    "$(git -C "${KVAF_UMDK_ROOT}" rev-parse 'HEAD^{tree}')" == \
      "${umdk_git_tree}" &&
    "$(git -C "${KVAF_LIBUMMU_ROOT}" rev-parse HEAD)" == \
      "${libummu_git_commit}" &&
    "$(git -C "${KVAF_LIBUMMU_ROOT}" rev-parse 'HEAD^{tree}')" == \
      "${libummu_git_tree}" ]] ||
  die "paired UMDK/libummu Git revisions changed during the build"
cmp -s "${KVAF_UDMA_UAPI_HEADER}" \
  "${KVAF_RUNTIME_UDMA_UAPI_HEADER}" ||
  die "running-kernel UDMA UAPI changed during the build"
[[ "$(sha256_file "${KVAF_RUNTIME_UDMA_UAPI_HEADER}")" == \
      "${runtime_udma_uapi_header_sha}" ]] ||
  die "running-kernel UDMA UAPI hash changed during the build"
cmp -s "${KVAF_UMMU_UAPI_HEADER}" \
  "${KVAF_RUNTIME_UMMU_UAPI_HEADER}" ||
  die "running-kernel UMMU UAPI changed during the build"
[[ "$(sha256_file "${KVAF_RUNTIME_UMMU_UAPI_HEADER}")" == \
      "${runtime_ummu_uapi_header_sha}" ]] ||
  die "running-kernel UMMU UAPI hash changed during the build"

public_provider_symbols=(
  kvaf_urma_provider_open_v2
  kvaf_urma_provider_close_v2
  kvaf_urma_provider_private_abi_v2
  aicpu_kvaf_urma_open_v2
  aicpu_kvaf_urma_close_v2
  aicpu_kvaf_urma_owner_probe_v1
  aicpu_kvaf_urma_epoch_owner_v1
)

validate_provider()
{
  local provider=$1
  local label=$2
  local dynamic_file="${staging_dir}/${label}.dynamic.txt"
  local ldd_file="${staging_dir}/${label}.ldd.txt"
  local provider_dynamic_symbols provider_all_symbols elf_header symbol

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
    kvaf_udma_source_get_segment_failure \
    kvaf_urma_core_has_dynamic_provider_scan_disabled \
    kvaf_udma_source_live_context_count \
    kvaf_udma_source_live_segment_count \
    kvaf_udma_source_live_token_count \
    kvaf_udma_source_live_jetty_count \
    kvaf_udma_source_live_jfr_count \
    kvaf_udma_source_live_jfc_count \
    kvaf_udma_source_orphan_segment_count \
    kvaf_udma_source_orphan_token_count \
    kvaf_udma_source_orphan_jetty_count \
    kvaf_udma_source_orphan_jfr_count \
    kvaf_udma_source_retry_orphan_jfrs \
    kvaf_udma_source_retry_orphan_cleanup \
    kvaf_source_ummu_capability \
    kvaf_source_ummu_get_status \
    kvaf_source_ummu_get_thread_status \
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
    grep -Eq "[[:space:]]${symbol}$" <<<"${provider_all_symbols}" ||
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
  if grep -Eq 'lib(urma|urma_common|ummu|urma-udma)\.so' \
      "${dynamic_file}"; then
    die "${label} unexpectedly links external URMA/UMMU code"
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
  if grep -Eq 'lib(urma|urma_common|ummu|urma-udma)\.so' \
      "${ldd_file}"; then
    cat "${ldd_file}" >&2
    die "${label} resolves external URMA/UMMU code"
  fi
}

validate_provider "${host_provider}" host-provider
validate_provider "${device_provider}" device-provider

public_abi_probe="${staging_dir}/cmake/kvaf_urma_public_abi_probe"
[[ -x "${public_abi_probe}" ]] ||
  die "provider build did not produce the public-ABI probe"
public_abi_probe_output=$("${public_abi_probe}") ||
  die "official UDMA/perftest public-ABI probe failed"
if ! [[ "${public_abi_probe_output}" =~ \
^KVAF_URMA_PUBLIC_ABI=MATCH:([0-9a-f]{16})$ ]]; then
  die "official UDMA/perftest public-ABI probe returned unexpected output"
fi
urma_public_abi_signature=${BASH_REMATCH[1]}

provider_abi_probe="${staging_dir}/cmake/kvaf_urma_provider_abi_probe"
[[ -x "${provider_abi_probe}" ]] ||
  die "provider build did not produce the private-ABI API probe"
host_abi_probe_output=$(
  "${provider_abi_probe}" "${host_provider}" 1
) || die "Host provider private-ABI API probe failed"
[[ "${host_abi_probe_output}" == \
    "KVAF_URMA_PROVIDER_PRIVATE_ABI=1" ]] ||
  die "Host provider private-ABI API probe returned unexpected output"
device_abi_probe_output=$(
  "${provider_abi_probe}" "${device_provider}" 2
) || die "Device provider private-ABI API probe failed"
[[ "${device_abi_probe_output}" == \
    "KVAF_URMA_PROVIDER_PRIVATE_ABI=2" ]] ||
  die "Device provider private-ABI API probe returned unexpected output"

aicpu_contract_probe="${staging_dir}/cmake/kvaf_urma_provider_aicpu_contract_probe"
[[ -x "${aicpu_contract_probe}" ]] ||
  die "provider build did not produce the AICPU contract probe"
aicpu_contract_output=$(
  "${aicpu_contract_probe}" "${device_provider}"
) || die "Device provider AICPU return-contract probe failed"
[[ "${aicpu_contract_output}" == \
    "KVAF_URMA_PROVIDER_AICPU_CONTRACT=PASS" ]] ||
  die "Device provider AICPU return-contract probe returned unexpected output"

udma_uapi_probe="${staging_dir}/cmake/kvaf_udma_uapi_probe"
udma_private_abi_probe="${staging_dir}/cmake/kvaf_udma_private_abi_probe"
ummu_uapi_probe="${staging_dir}/cmake/kvaf_ummu_uapi_probe"
for probe in \
  "${udma_uapi_probe}" "${udma_private_abi_probe}" "${ummu_uapi_probe}"
do
  [[ -x "${probe}" ]] || die "provider build did not produce ${probe}"
done

udma_uapi_probe_output=$("${udma_uapi_probe}") ||
  die "official UDMA-UAPI probe failed"
[[ "${udma_uapi_probe_output}" == KVAF_UDMA_UAPI_ABI=* ]] ||
  die "official UDMA-UAPI probe returned unexpected output"
[[ "${udma_uapi_probe_output}" == *\
"map_index_shift:4,tid_shift:8,access_guard:0x0000000f,access_local_only:0x00000001,access_read:0x00000002,access_write:0x00000004,access_atomic:0x00000008"* ]] ||
  die "official UDMA-UAPI access or mmap constants do not match A5"
udma_private_abi_probe_output=$("${udma_private_abi_probe}") ||
  die "official UDMA private-layout probe failed"
[[ "${udma_private_abi_probe_output}" == KVAF_UDMA_PRIVATE_ABI=* ]] ||
  die "official UDMA private-layout probe returned unexpected output"
ummu_uapi_probe_output=$("${ummu_uapi_probe}") ||
  die "paired UMMU-UAPI probe failed"
[[ "${ummu_uapi_probe_output}" == KVAF_UMMU_UAPI_ABI=* ]] ||
  die "paired UMMU-UAPI probe returned unexpected output"
[[ "${ummu_uapi_probe_output}" == *"cap_free_bit:0x0000000000000010,cap_pplbi:0x0000000000000020" ]] ||
  die "paired UMMU-UAPI probe omits FREE_BIT/PPLBI capabilities"

probe_output_sha()
{
  printf '%s\n' "$1" | sha256sum | awk '{print $1}'
}
udma_uapi_abi=${udma_uapi_probe_output#KVAF_UDMA_UAPI_ABI=}
udma_private_abi=${udma_private_abi_probe_output#KVAF_UDMA_PRIVATE_ABI=}
ummu_uapi_abi=${ummu_uapi_probe_output#KVAF_UMMU_UAPI_ABI=}
udma_uapi_abi_sha=$(probe_output_sha "${udma_uapi_probe_output}")
udma_private_abi_sha=$(probe_output_sha "${udma_private_abi_probe_output}")
ummu_uapi_abi_sha=$(probe_output_sha "${ummu_uapi_probe_output}")
udma_uapi_probe_binary_sha=$(sha256_file "${udma_uapi_probe}")
udma_private_abi_probe_binary_sha=$(sha256_file \
  "${udma_private_abi_probe}")
ummu_uapi_probe_binary_sha=$(sha256_file "${ummu_uapi_probe}")
for digest in \
  "${udma_uapi_abi_sha}" "${udma_private_abi_sha}" \
  "${ummu_uapi_abi_sha}" "${udma_uapi_probe_binary_sha}" \
  "${udma_private_abi_probe_binary_sha}" \
  "${ummu_uapi_probe_binary_sha}"
do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] ||
    die "ABI probe SHA-256 output is malformed"
done

host_provider_sha=$(sha256_file "${host_provider}") ||
  die "could not hash the Host provider"
device_provider_sha=$(sha256_file "${device_provider}") ||
  die "could not hash the Device provider"
[[ "${host_provider_sha}" =~ ^[0-9a-f]{64}$ &&
   "${device_provider_sha}" =~ ^[0-9a-f]{64}$ ]] ||
  die "provider SHA-256 output is malformed"
[[ "${host_provider_sha}" != "${device_provider_sha}" ]] ||
  die "Host and Device providers must be distinct binaries"

source_manifest_sha=$(sha256_file "${staging_dir}/source.sha256")
umdk_source_manifest_sha=$(sha256_file \
  "${staging_dir}/umdk-source.sha256")
libummu_source_manifest_sha=$(sha256_file \
  "${staging_dir}/libummu-source.sha256")
udma_uapi_header_sha=$(sha256_file "${KVAF_UDMA_UAPI_HEADER}")
ummu_uapi_header_sha=$(sha256_file "${KVAF_UMMU_UAPI_HEADER}")
[[ "${udma_uapi_header_sha}" == "${runtime_udma_uapi_header_sha}" ]] ||
  die "official and running-kernel UDMA UAPI hashes disagree after cmp"
[[ "${ummu_uapi_header_sha}" == \
    "${runtime_ummu_uapi_header_sha}" ]] ||
  die "paired and running-kernel UMMU UAPI hashes disagree after cmp"
for digest in \
  "${source_manifest_sha}" "${umdk_source_manifest_sha}" \
  "${libummu_source_manifest_sha}" "${udma_uapi_header_sha}" \
  "${ummu_uapi_header_sha}"
do
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] ||
    die "source provenance SHA-256 output is malformed"
done

cat >"${staging_dir}/build.manifest" <<EOF
schema=kvaf-direct-urma-provider-v3
source_layout=official-umdk-libummu-paired-v1
build_kernel_release=${kernel_release}
umdk_root=${KVAF_UMDK_ROOT}
umdk_git_origin=${umdk_git_origin}
umdk_git_commit=${umdk_git_commit}
umdk_git_tree=${umdk_git_tree}
umdk_git_tree_state=clean
umdk_release_version=${umdk_release_version}
umdk_source_manifest_filename=umdk-source.sha256
umdk_source_manifest_sha256=${umdk_source_manifest_sha}
libummu_root=${KVAF_LIBUMMU_ROOT}
libummu_git_origin=${libummu_git_origin}
libummu_git_commit=${libummu_git_commit}
libummu_git_tree=${libummu_git_tree}
libummu_git_tree_state=clean
libummu_expected_remote=${expected_libummu_remote}
libummu_expected_commit=${expected_libummu_commit}
libummu_version=${expected_libummu_version}
libummu_abi_version=${expected_libummu_abi}
libummu_source_manifest_filename=libummu-source.sha256
libummu_source_manifest_sha256=${libummu_source_manifest_sha}
umdk_libummu_pairing=verified
source_manifest_filename=source.sha256
source_manifest_sha256=${source_manifest_sha}
host_provider_filename=libkvaf_urma_provider_host.so
host_provider_execution_domain=host-process
host_provider_sha256=${host_provider_sha}
host_private_abi=1
host_private_abi_name=official-umdk-udma-v1-host
device_provider_filename=libkvaf_urma_provider_device.so
device_provider_execution_domain=aicpu
device_provider_sha256=${device_provider_sha}
device_private_abi=2
device_private_abi_name=official-umdk-udma-v1-device
urma_public_abi_signature=${urma_public_abi_signature}
urma_core_mode=official-umdk-source-linked-no-dynamic-providers
urma_common_mode=official-umdk-source-built-internal
urma_common_dt_needed=absent
perftest_source=official-umdk
perftest_management=official-umdk-ub-default-tcp-fallback-v1
udma_provider_mode=official-umdk-source-built-internal
udma_source_capability_magic=0x4b56414655444d41
udma_runtime_capability=0x0000000000020001
udma_uapi_header_path=${KVAF_UDMA_UAPI_HEADER}
udma_uapi_header_sha256=${udma_uapi_header_sha}
runtime_udma_uapi_header_path=${KVAF_RUNTIME_UDMA_UAPI_HEADER}
runtime_udma_uapi_header_sha256=${runtime_udma_uapi_header_sha}
runtime_udma_uapi_match=verified-exact
udma_uapi_abi=${udma_uapi_abi}
udma_uapi_abi_sha256=${udma_uapi_abi_sha}
udma_uapi_probe_filename=kvaf_udma_uapi_probe
udma_uapi_probe_sha256=${udma_uapi_probe_binary_sha}
udma_private_abi=${udma_private_abi}
udma_private_abi_sha256=${udma_private_abi_sha}
udma_private_abi_probe_filename=kvaf_udma_private_abi_probe
udma_private_abi_probe_sha256=${udma_private_abi_probe_binary_sha}
ummu_provider_mode=paired-libummu-source-built-internal
ummu_source_capability_magic=0x4b564146554d4d55
ummu_runtime_capability=0x0000000000010001
ummu_uapi_header_path=${KVAF_UMMU_UAPI_HEADER}
ummu_uapi_header_sha256=${ummu_uapi_header_sha}
runtime_ummu_uapi_header_path=${KVAF_RUNTIME_UMMU_UAPI_HEADER}
runtime_ummu_uapi_header_sha256=${runtime_ummu_uapi_header_sha}
runtime_ummu_uapi_match=${runtime_ummu_uapi_match}
ummu_uapi_abi=${ummu_uapi_abi}
ummu_uapi_abi_sha256=${ummu_uapi_abi_sha}
ummu_uapi_probe_filename=kvaf_ummu_uapi_probe
ummu_uapi_probe_sha256=${ummu_uapi_probe_binary_sha}
udma_teardown_contract=checked-retryable-v1
EOF

build_manifest_sha=$(sha256_file "${staging_dir}/build.manifest") ||
  die "could not hash the provider build manifest"
[[ "${build_manifest_sha}" =~ ^[0-9a-f]{64}$ ]] ||
  die "provider build-manifest SHA-256 output is malformed"
cat >"${staging_dir}/build.success" <<EOF
schema=kvaf-direct-urma-provider-success-v3
host_provider_sha256=${host_provider_sha}
device_provider_sha256=${device_provider_sha}
build_manifest_sha256=${build_manifest_sha}
source_manifest_sha256=${source_manifest_sha}
umdk_source_manifest_sha256=${umdk_source_manifest_sha}
libummu_source_manifest_sha256=${libummu_source_manifest_sha}
runtime_udma_uapi_header_sha256=${runtime_udma_uapi_header_sha}
runtime_ummu_uapi_header_sha256=${runtime_ummu_uapi_header_sha}
udma_uapi_probe_sha256=${udma_uapi_probe_binary_sha}
udma_private_abi_probe_sha256=${udma_private_abi_probe_binary_sha}
ummu_uapi_probe_sha256=${ummu_uapi_probe_binary_sha}
EOF

[[ "$(manifest_value schema "${staging_dir}/build.success")" == \
      "kvaf-direct-urma-provider-success-v3" &&
    "$(manifest_value host_provider_sha256 \
      "${staging_dir}/build.success")" == "${host_provider_sha}" &&
    "$(manifest_value device_provider_sha256 \
      "${staging_dir}/build.success")" == "${device_provider_sha}" &&
    "$(manifest_value build_manifest_sha256 \
      "${staging_dir}/build.success")" == "${build_manifest_sha}" &&
    "$(manifest_value runtime_udma_uapi_header_sha256 \
      "${staging_dir}/build.success")" == \
      "${runtime_udma_uapi_header_sha}" &&
    "$(manifest_value runtime_ummu_uapi_header_sha256 \
      "${staging_dir}/build.success")" == \
      "${runtime_ummu_uapi_header_sha}" ]] ||
  die "internal v3 success-stamp validation failed"

publish_dir=$(
  mktemp -d "${KVAF_DIRECT_PROVIDER_BUILD_DIR}/.publish.XXXXXX"
)
cp "${host_provider}" \
  "${publish_dir}/libkvaf_urma_provider_host.so"
cp "${device_provider}" \
  "${publish_dir}/libkvaf_urma_provider_device.so"
for artifact in \
  source.sha256 umdk-source.sha256 libummu-source.sha256 \
  build.manifest build.success \
  host-provider.dynamic.txt host-provider.ldd.txt \
  device-provider.dynamic.txt device-provider.ldd.txt
do
  cp "${staging_dir}/${artifact}" "${publish_dir}/${artifact}"
done
cp "${udma_uapi_probe}" "${publish_dir}/kvaf_udma_uapi_probe"
cp "${udma_private_abi_probe}" \
  "${publish_dir}/kvaf_udma_private_abi_probe"
cp "${ummu_uapi_probe}" "${publish_dir}/kvaf_ummu_uapi_probe"
chmod 0644 "${publish_dir}/"*
chmod 0755 \
  "${publish_dir}/kvaf_udma_uapi_probe" \
  "${publish_dir}/kvaf_udma_private_abi_probe" \
  "${publish_dir}/kvaf_ummu_uapi_probe"

for artifact in \
  libkvaf_urma_provider_host.so \
  libkvaf_urma_provider_device.so \
  source.sha256 \
  umdk-source.sha256 \
  libummu-source.sha256 \
  build.manifest \
  host-provider.dynamic.txt \
  host-provider.ldd.txt \
  device-provider.dynamic.txt \
  device-provider.ldd.txt \
  kvaf_udma_uapi_probe \
  kvaf_udma_private_abi_probe \
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
echo "UMDK_ROOT=${KVAF_UMDK_ROOT}"
echo "UMDK_GIT_COMMIT=${umdk_git_commit}"
echo "LIBUMMU_ROOT=${KVAF_LIBUMMU_ROOT}"
echo "LIBUMMU_GIT_COMMIT=${libummu_git_commit}"
echo "UMDK_LIBUMMU_PAIRING=verified"
echo "RUNTIME_UDMA_UAPI_HEADER=${KVAF_RUNTIME_UDMA_UAPI_HEADER}"
echo "RUNTIME_UDMA_UAPI_MATCH=verified-exact"
echo "RUNTIME_UMMU_UAPI_HEADER=${KVAF_RUNTIME_UMMU_UAPI_HEADER}"
echo "RUNTIME_UMMU_UAPI_MATCH=${runtime_ummu_uapi_match}"
echo "UDMA_UAPI_ABI_SHA256=${udma_uapi_abi_sha}"
echo "UDMA_PRIVATE_ABI_SHA256=${udma_private_abi_sha}"
echo "UMMU_UAPI_ABI_SHA256=${ummu_uapi_abi_sha}"
echo "URMA_COMMON_DT_NEEDED=absent"
echo "PERFTEST_MANAGEMENT=official-umdk-ub-default-tcp-fallback-v1"
echo "UDMA_TEARDOWN_CONTRACT=checked-retryable-v1"
