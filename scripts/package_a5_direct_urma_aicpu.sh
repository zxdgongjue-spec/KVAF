#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
source "${SCRIPT_DIR}/a5_direct_urma_legacy_guard.sh"
source "${SCRIPT_DIR}/a5_direct_urma_manifest_v3.sh"
kvaf_configure_direct_urma_state_paths "${ROOT_DIR}" || exit 2

: "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"

KVAF_DIRECT_PROVIDER_BUILD_DIR=${KVAF_DIRECT_PROVIDER_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma-provider"}
KVAF_DIRECT_HOST_PROVIDER_LIBRARY=${KVAF_DIRECT_HOST_PROVIDER_LIBRARY:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/libkvaf_urma_provider_host.so"}
KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY=${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/libkvaf_urma_provider_device.so"}
KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST=${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/source.sha256"}
KVAF_DIRECT_PROVIDER_BUILD_MANIFEST=${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/build.manifest"}
KVAF_DIRECT_PROVIDER_SUCCESS_STAMP=${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/build.success"}
KVAF_DIRECT_AICPU_ARCHIVE=${KVAF_DIRECT_AICPU_ARCHIVE:-"${ASCEND_HOME_PATH}/compat/liburma_perftest.tar.gz"}

die()
{
  echo "ERROR: $*" >&2
  exit 2
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

verify_aicpu_archive()
{
  local archive=$1
  local label=$2
  local archive_entries unexpected_entries
  local archive_provider_sha archive_manifest_sha
  local archive_source_manifest_sha archive_umdk_manifest_sha
  local archive_libummu_manifest_sha archive_probe_sha local_probe_sha
  local entry probe

  [[ -r "${archive}" ]] ||
    die "${label} AICPU archive is not readable: ${archive}"
  archive_entries=$(tar -tzf "${archive}") ||
    die "${label} AICPU archive is not a valid tar.gz: ${archive}"
  for entry in \
    aicpu_kernels_device/libkvaf_urma_provider_device.so \
    aicpu_kernels_device/kvaf-direct-urma-provider.manifest \
    aicpu_kernels_device/kvaf-direct-urma-source.sha256 \
    aicpu_kernels_device/kvaf-direct-urma-umdk-source.sha256 \
    aicpu_kernels_device/kvaf-direct-urma-libummu-source.sha256 \
    aicpu_kernels_device/kvaf_udma_uapi_probe \
    aicpu_kernels_device/kvaf_udma_private_abi_probe \
    aicpu_kernels_device/kvaf_ummu_uapi_probe
  do
    grep -Fxq "${entry}" <<<"${archive_entries}" ||
      die "${label} AICPU archive is missing ${entry}"
  done
  unexpected_entries=$(
    grep -Ev \
      '^aicpu_kernels_device/?$|^aicpu_kernels_device/(libkvaf_urma_provider_device\.so|kvaf-direct-urma-provider\.manifest|kvaf-direct-urma-source\.sha256|kvaf-direct-urma-umdk-source\.sha256|kvaf-direct-urma-libummu-source\.sha256|kvaf_udma_uapi_probe|kvaf_udma_private_abi_probe|kvaf_ummu_uapi_probe)$' \
      <<<"${archive_entries}" || true
  )
  [[ -z "${unexpected_entries}" ]] ||
    die "${label} AICPU archive contains an unexpected entry"

  archive_provider_sha=$(
    tar -xOzf "${archive}" \
      aicpu_kernels_device/libkvaf_urma_provider_device.so |
      sha256sum |
      awk '{print $1}'
  ) || die "could not hash ${label} archived Device provider"
  archive_manifest_sha=$(
    tar -xOzf "${archive}" \
      aicpu_kernels_device/kvaf-direct-urma-provider.manifest |
      sha256sum |
      awk '{print $1}'
  ) || die "could not hash ${label} archived build manifest"
  archive_source_manifest_sha=$(
    tar -xOzf "${archive}" \
      aicpu_kernels_device/kvaf-direct-urma-source.sha256 |
      sha256sum |
      awk '{print $1}'
  ) || die "could not hash ${label} archived source manifest"
  archive_umdk_manifest_sha=$(
    tar -xOzf "${archive}" \
      aicpu_kernels_device/kvaf-direct-urma-umdk-source.sha256 |
      sha256sum | awk '{print $1}'
  ) || die "could not hash ${label} archived UMDK source manifest"
  archive_libummu_manifest_sha=$(
    tar -xOzf "${archive}" \
      aicpu_kernels_device/kvaf-direct-urma-libummu-source.sha256 |
      sha256sum | awk '{print $1}'
  ) || die "could not hash ${label} archived libummu source manifest"
  [[ "${archive_provider_sha}" == "${device_provider_sha}" &&
     "${archive_manifest_sha}" == "${build_manifest_sha}" &&
     "${archive_source_manifest_sha}" == "${source_manifest_sha}" &&
     "${archive_umdk_manifest_sha}" == "${umdk_source_manifest_sha}" &&
     "${archive_libummu_manifest_sha}" == \
       "${libummu_source_manifest_sha}" ]] ||
    die "${label} AICPU provider/provenance hash mismatch"
  for probe in kvaf_udma_uapi_probe kvaf_udma_private_abi_probe \
    kvaf_ummu_uapi_probe; do
    archive_probe_sha=$(
      tar -xOzf "${archive}" "aicpu_kernels_device/${probe}" |
        sha256sum | awk '{print $1}'
    ) || die "could not hash ${label} archived ${probe}"
    local_probe_sha=$(sha256sum \
      "${KVAF_DIRECT_PROVIDER_BUILD_DIR}/${probe}" | awk '{print $1}')
    [[ "${archive_probe_sha}" == "${local_probe_sha}" ]] ||
      die "${label} archived ${probe} does not match the verified build"
  done
}

[[ -e /.dockerenv ]] ||
  die "package this AICPU archive inside vllm-ascend-cs-kvaf"
[[ ${EUID} -eq 0 ]] ||
  die "AICPU compat packaging requires container uid 0"
for command in flock nm readelf sha256sum tar realpath; do
  command -v "${command}" >/dev/null 2>&1 ||
    die "${command} is required"
done

kvaf_check_legacy_direct_urma_state "${ROOT_DIR}" ||
  die "legacy direct URMA state blocks AICPU packaging"

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

# Validate exactly once while holding the same deploy lock as the builder and
# runner.  This closes the check/use race with build.success and the atomically
# published provider artifacts.
kvaf_verify_direct_provider_build_v3 ||
  die "rebuild the paired official UMDK/libummu provider before packaging"
host_provider_sha=${KVAF_VERIFIED_HOST_PROVIDER_SHA}
device_provider_sha=${KVAF_VERIFIED_DEVICE_PROVIDER_SHA}
source_manifest_sha=${KVAF_VERIFIED_SOURCE_MANIFEST_SHA}
umdk_source_manifest_sha=${KVAF_VERIFIED_UMDK_SOURCE_MANIFEST_SHA}
libummu_source_manifest_sha=${KVAF_VERIFIED_LIBUMMU_SOURCE_MANIFEST_SHA}
build_manifest_sha=${KVAF_VERIFIED_BUILD_MANIFEST_SHA}
umdk_source_manifest=${KVAF_VERIFIED_UMDK_SOURCE_MANIFEST}
libummu_source_manifest=${KVAF_VERIFIED_LIBUMMU_SOURCE_MANIFEST}

if [[ -e "${KVAF_DIRECT_QUARANTINE_FILE}" ||
      -L "${KVAF_DIRECT_QUARANTINE_FILE}" ]]; then
  die "an unsafe direct URMA session is quarantined; reset or isolate the NPU before packaging"
fi
if [[ -e "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ||
      -L "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ]]; then
  die "an incomplete direct URMA session blocks AICPU packaging"
fi
[[ ! -L "${KVAF_DIRECT_ACTIVE_SESSION_LOCK}" ]] ||
  die "active-session lock path must not be a symlink"
exec 8>"${KVAF_DIRECT_ACTIVE_SESSION_LOCK}"
flock -n 8 || die "a live direct URMA worker blocks AICPU packaging"

provider_dynamic_symbols=$(
  nm -D --defined-only "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}"
)
for symbol in \
  kvaf_urma_provider_open_v2 \
  kvaf_urma_provider_close_v2 \
  kvaf_urma_provider_private_abi_v2 \
  aicpu_kvaf_urma_open_v2 \
  aicpu_kvaf_urma_close_v2
do
  grep -Eq "[[:space:]]${symbol}(@@KVAF_URMA_PROVIDER_2)?$" \
    <<<"${provider_dynamic_symbols}" ||
    die "provider does not export ${symbol}"
done
if grep -Eq \
    '[[:space:]](perftest_|kvaf_perftest_|create_duplex_ctx|destroy_ctx)' \
    <<<"${provider_dynamic_symbols}"; then
  die "provider leaks an internal perftest symbol"
fi

expected_archive="${ASCEND_HOME_PATH}/compat/liburma_perftest.tar.gz"
if [[ "$(realpath -m "${KVAF_DIRECT_AICPU_ARCHIVE}")" != \
      "$(realpath -m "${expected_archive}")" ]]; then
  die "the AICPU archive must be installed at ${expected_archive}"
fi
compat_dir=$(dirname "${KVAF_DIRECT_AICPU_ARCHIVE}")
mkdir -p "${compat_dir}"

shopt -s nullglob
for archive in "${compat_dir}"/*.tar.gz; do
  if [[ "${archive}" == "${KVAF_DIRECT_AICPU_ARCHIVE}" ]]; then
    continue
  fi
  archive_listing=$(tar -tzf "${archive}" 2>/dev/null || true)
  if grep -Eq \
      '(^|/)(libkvaf_urma_provider(_host|_device)?|liburma_perftest)\.so$' \
      <<<"${archive_listing}"; then
    die "conflicting direct/perftest provider exists in ${archive}"
  fi
done
shopt -u nullglob

temporary_dir=$(mktemp -d /tmp/kvaf-direct-aicpu.XXXXXX)
temporary_install_archive=
cleanup()
{
  if [[ -n "${temporary_install_archive}" ]]; then
    rm -f "${temporary_install_archive}"
  fi
  rm -rf "${temporary_dir}"
}
trap cleanup EXIT

package_dir="${temporary_dir}/aicpu_kernels_device"
mkdir -p "${package_dir}"
cp "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}" \
  "${package_dir}/libkvaf_urma_provider_device.so"
cp "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" \
  "${package_dir}/kvaf-direct-urma-provider.manifest"
cp "${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST}" \
  "${package_dir}/kvaf-direct-urma-source.sha256"
cp "${umdk_source_manifest}" \
  "${package_dir}/kvaf-direct-urma-umdk-source.sha256"
cp "${libummu_source_manifest}" \
  "${package_dir}/kvaf-direct-urma-libummu-source.sha256"
for probe in kvaf_udma_uapi_probe kvaf_udma_private_abi_probe \
  kvaf_ummu_uapi_probe; do
  cp "${KVAF_DIRECT_PROVIDER_BUILD_DIR}/${probe}" \
    "${package_dir}/${probe}"
done
chmod 0644 \
  "${package_dir}/libkvaf_urma_provider_device.so" \
  "${package_dir}/kvaf-direct-urma-provider.manifest" \
  "${package_dir}/kvaf-direct-urma-source.sha256" \
  "${package_dir}/kvaf-direct-urma-umdk-source.sha256" \
  "${package_dir}/kvaf-direct-urma-libummu-source.sha256"
chmod 0755 \
  "${package_dir}/kvaf_udma_uapi_probe" \
  "${package_dir}/kvaf_udma_private_abi_probe" \
  "${package_dir}/kvaf_ummu_uapi_probe"

temporary_archive="${temporary_dir}/liburma_perftest.tar.gz"
tar -C "${temporary_dir}" -czf "${temporary_archive}" \
  aicpu_kernels_device
verify_aicpu_archive "${temporary_archive}" candidate

installed_listing=
if [[ -e "${KVAF_DIRECT_AICPU_ARCHIVE}" ]]; then
  installed_listing=$(
    tar -tzf "${KVAF_DIRECT_AICPU_ARCHIVE}" 2>/dev/null || true
  )
fi
if [[ -e "${KVAF_DIRECT_AICPU_ARCHIVE}" ]] &&
   ! grep -Fxq \
       'aicpu_kernels_device/libkvaf_urma_provider_device.so' \
       <<<"${installed_listing}"; then
  backup="${KVAF_DIRECT_AICPU_ARCHIVE}.before-kvaf-direct"
  if [[ -e "${backup}" ]]; then
    die "refusing to overwrite existing backup: ${backup}"
  fi
  cp -p "${KVAF_DIRECT_AICPU_ARCHIVE}" "${backup}"
  echo "BACKUP=${backup}"
fi

temporary_install_archive=$(
  mktemp "${compat_dir}/.liburma_perftest.tar.gz.XXXXXX"
)
cp "${temporary_archive}" "${temporary_install_archive}"
chmod 0644 "${temporary_install_archive}"
verify_aicpu_archive "${temporary_install_archive}" install-candidate
mv -f "${temporary_install_archive}" "${KVAF_DIRECT_AICPU_ARCHIVE}"
temporary_install_archive=

verify_aicpu_archive "${KVAF_DIRECT_AICPU_ARCHIVE}" installed

echo "KVAF_A5_DIRECT_URMA_AICPU_PACKAGE_OK"
echo "AICPU_ARCHIVE=${KVAF_DIRECT_AICPU_ARCHIVE}"
echo "DEVICE_PROVIDER_SHA256=${device_provider_sha}"
tar -tzf "${KVAF_DIRECT_AICPU_ARCHIVE}"
