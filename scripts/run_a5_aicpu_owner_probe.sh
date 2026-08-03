#!/usr/bin/env bash

# Guarded entry point for the resource-free AICPU/AIV owner-concurrency probe.
# It validates the exact provider build, direct build, installed AICPU archive,
# and CANN package registration before the first ACL/NPU call.

set -Eeuo pipefail
umask 077
unset BASH_ENV ENV
unset -f exit builtin 2>/dev/null || true

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)

echo "KVAF_A5_AICPU_OWNER_DYNAMIC_PROBE_DISABLED" >&2
echo "CAPABILITY_OBSERVATION_LEDGER=not-implemented" >&2
echo "No CANN/package/ACL/AICPU/NPU operation was attempted." >&2
builtin exit 2
if [[ 0 == 1 ]]; then
source "${SCRIPT_DIR}/a5_direct_urma_legacy_guard.sh"
source "${SCRIPT_DIR}/a5_direct_urma_manifest_v3.sh"
source "${SCRIPT_DIR}/a5_direct_urma_runtime_preflight.sh"
kvaf_configure_direct_urma_state_paths "${ROOT_DIR}" || exit 2

: "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"

KVAF_A5_DIRECT_BUILD_DIR=${KVAF_A5_DIRECT_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma"}
KVAF_A5_DIRECT_BINARY=${KVAF_A5_DIRECT_BINARY:-"${KVAF_A5_DIRECT_BUILD_DIR}/access_operator/a5/kvaf_a5_aiv_direct_urma_h2d_smoke"}
KVAF_A5_OWNER_PROBE_BINARY=${KVAF_A5_OWNER_PROBE_BINARY:-"${KVAF_A5_DIRECT_BUILD_DIR}/access_operator/a5/kvaf_a5_aicpu_owner_probe"}
KVAF_A5_DIRECT_BUILD_MANIFEST=${KVAF_A5_DIRECT_BUILD_MANIFEST:-"${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke.manifest"}
KVAF_DIRECT_PROVIDER_BUILD_DIR=${KVAF_DIRECT_PROVIDER_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma-provider"}
KVAF_DIRECT_HOST_PROVIDER_LIBRARY=${KVAF_DIRECT_HOST_PROVIDER_LIBRARY:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/libkvaf_urma_provider_host.so"}
KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY=${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/libkvaf_urma_provider_device.so"}
KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST=${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/source.sha256"}
KVAF_DIRECT_PROVIDER_BUILD_MANIFEST=${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/build.manifest"}
KVAF_DIRECT_PROVIDER_SUCCESS_STAMP=${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/build.success"}
KVAF_DIRECT_AICPU_ARCHIVE=${KVAF_DIRECT_AICPU_ARCHIVE:-"${ASCEND_HOME_PATH}/opp/vendors/cust/op_impl/aicpu/kernel/cann-kvaf-direct-urma-compat.tar.gz"}
KVAF_DIRECT_AICPU_REGISTRATION_FILE=${KVAF_DIRECT_AICPU_REGISTRATION_FILE:-"${ASCEND_HOME_PATH}/conf/ascend_package_load.ini"}
KVAF_ASCEND_HAL_LIBRARY=/usr/local/Ascend/driver/lib64/driver/libascend_hal.so

die()
{
  echo "ERROR: AICPU owner probe preflight: $*" >&2
  exit 2
}

[[ -e /.dockerenv ]] ||
  die "run this probe inside vllm-ascend-cs-kvaf"
[[ ${EUID} -eq 0 ]] ||
  die "container uid 0 is required"
for command in awk chmod dirname flock grep ldd mkdir nm readelf realpath \
  sha256sum tar timeout; do
  command -v "${command}" >/dev/null 2>&1 ||
    die "${command} is required"
done

kvaf_check_legacy_direct_urma_state "${ROOT_DIR}" ||
  die "legacy direct URMA state blocks the probe"
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

# Hold both locks through exec so neither a rebuild/repackage nor a direct
# worker can race the package and binary proofs used by this ACL process.
exec 9>"${KVAF_DIRECT_DEPLOY_RUN_LOCK}"
flock -w 30 9 ||
  die "another direct URMA deploy/run owns ${KVAF_DIRECT_DEPLOY_RUN_LOCK}"
if [[ -e "${KVAF_DIRECT_QUARANTINE_FILE}" ||
      -L "${KVAF_DIRECT_QUARANTINE_FILE}" ]]; then
  die "an unsafe direct URMA session is quarantined"
fi
if [[ -e "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ||
      -L "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ]]; then
  die "an incomplete direct URMA session blocks the probe"
fi
exec 8>"${KVAF_DIRECT_ACTIVE_SESSION_LOCK}"
flock -n 8 || die "a live direct URMA worker blocks the probe"

kvaf_verify_direct_provider_build_v3 ||
  die "provider build verification failed"
host_provider_sha=${KVAF_VERIFIED_HOST_PROVIDER_SHA}
device_provider_sha=${KVAF_VERIFIED_DEVICE_PROVIDER_SHA}
source_manifest_sha=${KVAF_VERIFIED_SOURCE_MANIFEST_SHA}
umdk_source_manifest_sha=${KVAF_VERIFIED_UMDK_SOURCE_MANIFEST_SHA}
libummu_source_manifest_sha=${KVAF_VERIFIED_LIBUMMU_SOURCE_MANIFEST_SHA}
build_manifest_sha=${KVAF_VERIFIED_BUILD_MANIFEST_SHA}
kvaf_verify_direct_runtime_artifacts_v3 ||
  die "runtime/package/registration verification failed"

manifest_owner_probe=$(kvaf_runtime_preflight_manifest_value \
  owner_probe_path "${KVAF_A5_DIRECT_BUILD_MANIFEST}") ||
  die "direct build manifest lacks owner_probe_path"
manifest_owner_probe_sha=$(kvaf_runtime_preflight_manifest_value \
  owner_probe_sha256 "${KVAF_A5_DIRECT_BUILD_MANIFEST}") ||
  die "direct build manifest lacks owner_probe_sha256"
[[ ! -L "${KVAF_A5_OWNER_PROBE_BINARY}" &&
   -f "${KVAF_A5_OWNER_PROBE_BINARY}" &&
   -x "${KVAF_A5_OWNER_PROBE_BINARY}" ]] ||
  die "owner probe binary is missing or unsafe"
[[ "$(realpath -e "${KVAF_A5_OWNER_PROBE_BINARY}")" == \
   "$(realpath -e "${manifest_owner_probe}")" ]] ||
  die "owner probe path differs from its successful build manifest"
actual_owner_probe_sha=$(sha256sum \
  "${KVAF_A5_OWNER_PROBE_BINARY}" | awk '{print $1}') ||
  die "could not hash owner probe binary"
[[ "${actual_owner_probe_sha}" == "${manifest_owner_probe_sha}" ]] ||
  die "owner probe binary changed after its successful build"

owner_probe_dynamic=$(readelf -d "${KVAF_A5_OWNER_PROBE_BINARY}") ||
  die "could not inspect owner probe dynamic linkage"
grep -Fq libhcomm <<<"${owner_probe_dynamic}" &&
  die "owner probe unexpectedly links HCOMM"
owner_probe_ldd=$(ldd -r "${KVAF_A5_OWNER_PROBE_BINARY}" 2>&1) || {
  echo "${owner_probe_ldd}" >&2
  die "owner probe runtime linkage is unresolved"
}
if grep -Eq 'not found|undefined symbol' <<<"${owner_probe_ldd}"; then
  echo "${owner_probe_ldd}" >&2
  die "owner probe runtime linkage is unresolved"
fi
owner_probe_ascendcl=$(kvaf_runtime_preflight_ldd_path \
  libascendcl.so <<<"${owner_probe_ldd}") ||
  die "owner probe did not resolve exactly one libascendcl"
manifest_ascendcl=$(kvaf_runtime_preflight_manifest_value \
  ascendcl_library_path "${KVAF_A5_DIRECT_BUILD_MANIFEST}") ||
  die "direct build manifest lacks ascendcl_library_path"
[[ "$(realpath -e "${owner_probe_ascendcl}")" == \
   "$(realpath -e "${manifest_ascendcl}")" ]] ||
  die "owner probe resolves a different libascendcl than its build"

echo "KVAF_A5_AICPU_OWNER_PROBE_PREFLIGHT_OK"
echo "AICPU_ARCHIVE=${KVAF_DIRECT_AICPU_ARCHIVE}"
echo "AICPU_REGISTRATION=verified-exact"
exec "${KVAF_A5_OWNER_PROBE_BINARY}" "$@"
fi
/bin/sh -c 'exit 2'
