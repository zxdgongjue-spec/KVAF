#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

KVAF_URMA_PERFTEST_ROOT=${KVAF_URMA_PERFTEST_ROOT:-/data/c00982601/urma_perftest}
KVAF_DIRECT_PROVIDER_BUILD_DIR=${KVAF_DIRECT_PROVIDER_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma-provider"}
KVAF_DIRECT_PROVIDER_BUILD_JOBS=${KVAF_DIRECT_PROVIDER_BUILD_JOBS:-8}
KVAF_DIRECT_PROVIDER_CC=${KVAF_DIRECT_PROVIDER_CC:-$(command -v gcc || true)}
KVAF_DIRECT_DEPLOY_RUN_LOCK=${KVAF_DIRECT_DEPLOY_RUN_LOCK:-/run/lock/kvaf-a5-direct-urma.lock}
KVAF_URMA_LIBRARY=${KVAF_URMA_LIBRARY:-}
KVAF_URMA_COMMON_LIBRARY=${KVAF_URMA_COMMON_LIBRARY:-}
KVAF_UDMA_PROVIDER_LIBRARY=${KVAF_UDMA_PROVIDER_LIBRARY:-}
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
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/kernelspace/include/udma/udma_abi.h"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_common.h"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_jetty.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_tid.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_segment.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_jfs.c"
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma/udma_u_jfc.c"
)
for path in "${required_source_paths[@]}"; do
  [[ -r "${path}" ]] ||
    die "incomplete urma_perftest source tree, missing ${path}"
done
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
if [[ -z "${KVAF_UDMA_PROVIDER_LIBRARY}" ]]; then
  default_udma_provider="$(
    dirname "${KVAF_URMA_LIBRARY}"
  )/urma/liburma-udma.so"
  [[ -r "${default_udma_provider}" ]] ||
    die "set KVAF_UDMA_PROVIDER_LIBRARY to the loaded liburma UDMA provider"
  KVAF_UDMA_PROVIDER_LIBRARY=${default_udma_provider}
fi
KVAF_UDMA_PROVIDER_LIBRARY=$(
  realpath -e "${KVAF_UDMA_PROVIDER_LIBRARY}"
)

for library in \
  "${KVAF_URMA_LIBRARY}" \
  "${KVAF_URMA_COMMON_LIBRARY}" \
  "${KVAF_UDMA_PROVIDER_LIBRARY}"
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

mkdir -p "$(dirname "${KVAF_DIRECT_DEPLOY_RUN_LOCK}")"
exec 9>"${KVAF_DIRECT_DEPLOY_RUN_LOCK}"
if ! flock -w 30 9; then
  die "another direct URMA deploy/run owns ${KVAF_DIRECT_DEPLOY_RUN_LOCK}"
fi

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
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi/userspace/udma"
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
  -DKVAF_URMA_COMMON_LIBRARY="${KVAF_URMA_COMMON_LIBRARY}"
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
  local symbol resolved_urma resolved_common

  provider_dynamic_symbols=$(nm -D --defined-only "${provider}")
  for symbol in "${public_provider_symbols[@]}"; do
    grep -Eq \
      "[[:space:]]${symbol}(@@KVAF_URMA_PROVIDER_2)?$" \
      <<<"${provider_dynamic_symbols}" ||
      die "${label} provider does not export ${symbol}"
  done
  if grep -Eq \
      '[[:space:]](perftest_|kvaf_perftest_|kvaf_udma_|create_duplex_ctx|destroy_ctx)' \
      <<<"${provider_dynamic_symbols}"; then
    die "${label} provider leaks an internal implementation symbol"
  fi
  provider_all_symbols=$(nm --defined-only "${provider}")
  for symbol in \
    kvaf_perftest_create_duplex_ctx_checked \
    kvaf_perftest_destroy_duplex_ctx_checked \
    kvaf_udma_extract_direct_queue_info
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
  grep -Fq 'liburma.so' "${dynamic_file}" ||
    die "${label} source-compiled provider does not link liburma"
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
  resolved_urma=$(
    awk '$1 ~ /^liburma\.so/ && $2 == "=>" {print $3}' "${ldd_file}"
  )
  resolved_common=$(
    awk '$1 ~ /^liburma_common\.so/ && $2 == "=>" {print $3}' \
      "${ldd_file}"
  )
  [[ -n "${resolved_urma}" && -n "${resolved_common}" ]] ||
    die "ldd did not resolve ${label} liburma dependencies"
  [[ "$(realpath -e "${resolved_urma}")" == \
      "${KVAF_URMA_LIBRARY}" ]] ||
    die "${label} link-time and runtime liburma paths differ"
  [[ "$(realpath -e "${resolved_common}")" == \
      "${KVAF_URMA_COMMON_LIBRARY}" ]] ||
    die "${label} link-time and runtime liburma_common paths differ"
}

validate_provider "${host_provider}" host-provider
validate_provider "${device_provider}" device-provider

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
capture_elf_identity "${KVAF_UDMA_PROVIDER_LIBRARY}"
udma_provider_library_sha=${captured_elf_sha}
udma_provider_library_build_id=${captured_elf_build_id}
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
urma_library_path=${KVAF_URMA_LIBRARY}
urma_library_sha256=${urma_library_sha}
urma_library_build_id=${urma_library_build_id}
urma_common_library_path=${KVAF_URMA_COMMON_LIBRARY}
urma_common_library_sha256=${urma_common_library_sha}
urma_common_library_build_id=${urma_common_library_build_id}
udma_provider_library_path=${KVAF_UDMA_PROVIDER_LIBRARY}
udma_provider_library_sha256=${udma_provider_library_sha}
udma_provider_library_build_id=${udma_provider_library_build_id}
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
cp "${staging_dir}/build.success" "${publish_dir}/build.success"
chmod 0644 "${publish_dir}/"*

for artifact in \
  libkvaf_urma_provider_host.so \
  libkvaf_urma_provider_device.so \
  source.sha256 \
  build.manifest \
  host-provider.dynamic.txt \
  host-provider.ldd.txt \
  device-provider.dynamic.txt \
  device-provider.ldd.txt
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
echo "UDMA_PROVIDER_LIBRARY=${KVAF_UDMA_PROVIDER_LIBRARY}"
echo "REFERENCE_PERFTEST_LIBRARY=${KVAF_REFERENCE_PERFTEST_LIBRARY}"
