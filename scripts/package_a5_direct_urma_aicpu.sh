#!/usr/bin/env bash

set -Eeuo pipefail
umask 077
unset BASH_ENV ENV
unset -f exit builtin 2>/dev/null || true

# Installing this archive mutates CANN's live package registry and makes the
# Device provider loader-visible.  Exported URMA symbols and a successful
# zero-resource probe do not prove the complete AICPU caller-domain,
# Segment/Jetty/TP, AIV queue-address, cache-ordering, or teardown contract.
# Keep the deployment transaction unreachable until a sealed capability
# ledger proves those properties.  There is deliberately no environment or
# command-line override for this gate.
echo "KVAF_A5_DIRECT_AICPU_PACKAGE_MUTATION_DISABLED" >&2
echo "CAPABILITY_LEDGER=not-implemented" >&2
echo "No CANN/state/helper/lock/archive/registry operation was attempted." >&2
builtin exit 2

# If an inherited shell hook ever shadows `exit`, the complete historical
# deployment transaction remains enclosed by a constant-false branch.
if [[ 0 == 1 ]]; then

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
source "${SCRIPT_DIR}/a5_direct_urma_legacy_guard.sh"
source "${SCRIPT_DIR}/a5_direct_urma_manifest_v3.sh"
source "${SCRIPT_DIR}/a5_aicpu_package_registration.sh"
kvaf_configure_direct_urma_state_paths "${ROOT_DIR}" || exit 2

: "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"

KVAF_DIRECT_PROVIDER_BUILD_DIR=${KVAF_DIRECT_PROVIDER_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma-provider"}
KVAF_DIRECT_HOST_PROVIDER_LIBRARY=${KVAF_DIRECT_HOST_PROVIDER_LIBRARY:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/libkvaf_urma_provider_host.so"}
KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY=${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/libkvaf_urma_provider_device.so"}
KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST=${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/source.sha256"}
KVAF_DIRECT_PROVIDER_BUILD_MANIFEST=${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/build.manifest"}
KVAF_DIRECT_PROVIDER_SUCCESS_STAMP=${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP:-"${KVAF_DIRECT_PROVIDER_BUILD_DIR}/build.success"}
KVAF_DIRECT_AICPU_PACKAGE_NAME=cann-kvaf-direct-urma-compat.tar.gz
KVAF_DIRECT_AICPU_PACKAGE_RELATIVE_DIR=opp/vendors/cust/op_impl/aicpu/kernel
KVAF_DIRECT_AICPU_ARCHIVE=${KVAF_DIRECT_AICPU_ARCHIVE:-"${ASCEND_HOME_PATH}/${KVAF_DIRECT_AICPU_PACKAGE_RELATIVE_DIR}/${KVAF_DIRECT_AICPU_PACKAGE_NAME}"}
KVAF_DIRECT_AICPU_REGISTRATION_FILE=${KVAF_DIRECT_AICPU_REGISTRATION_FILE:-"${ASCEND_HOME_PATH}/conf/ascend_package_load.ini"}
KVAF_DURABLE_STATE_HELPER=${SCRIPT_DIR}/durable_state_file.py
registered_package_audit_sha=
registered_package_audit_b64=

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

elf_soname()
{
  local file=$1
  readelf -d "${file}" | awk '
    /\(SONAME\)/ {
      value = $0
      sub(/^.*\[/, "", value)
      sub(/\].*$/, "", value)
      count++
    }
    END {
      if (count != 1 || value == "") exit 1
      print value
    }
  '
}

pin_registered_package_audit()
{
  local audit_script=${SCRIPT_DIR}/audit_a5_registered_aicpu_packages.sh
  local audit_sha

  [[ -f ${audit_script} && ! -L ${audit_script} && -x ${audit_script} ]] ||
    die "registered AICPU package audit is missing or unsafe"
  # O_NOFOLLOW plus pre/post fstat gives one bounded byte snapshot even in a
  # shared checkout.  Base64 is used because shell variables cannot preserve
  # NUL bytes.  Both transaction audits decode and execute this same snapshot,
  # and its digest must match the provider build manifest.
  registered_package_audit_b64=$(python3 -c '
import base64, os, stat, sys
path = sys.argv[1]
flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
fd = os.open(path, flags)
try:
    before = os.fstat(fd)
    if (not stat.S_ISREG(before.st_mode) or
            before.st_size <= 0 or before.st_size > 1024 * 1024 or
            before.st_mode & 0o111 == 0):
        raise RuntimeError("unsafe audit source")
    chunks = []
    remaining = before.st_size
    while remaining:
        chunk = os.read(fd, min(remaining, 65536))
        if not chunk:
            raise RuntimeError("short audit read")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(fd, 1):
        raise RuntimeError("audit grew during read")
    after = os.fstat(fd)
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns)
    if identity(before) != identity(after):
        raise RuntimeError("audit changed during read")
    sys.stdout.write(base64.b64encode(b"".join(chunks)).decode("ascii"))
finally:
    os.close(fd)
' "${audit_script}") ||
    die "could not pin the registered-package audit bytes"
  [[ -n ${registered_package_audit_b64} ]] ||
    die "registered-package audit snapshot is empty"
  audit_sha=$(printf '%s' "${registered_package_audit_b64}" |
    base64 -d | sha256sum | awk '{print $1}') ||
    die "could not hash the pinned registered-package audit bytes"
  [[ ${audit_sha} == \
     "${KVAF_VERIFIED_REGISTERED_PACKAGE_AUDIT_SHA:-}" ]] ||
    die "registered-package audit differs from the verified provider build"
  registered_package_audit_sha=${audit_sha}
}

run_registered_package_audit()
{
  local audit_rc

  [[ -n ${registered_package_audit_b64} &&
     ${registered_package_audit_sha} =~ ^[0-9a-f]{64}$ ]] ||
    die "registered-package audit snapshot was not pinned"
  if printf '%s' "${registered_package_audit_b64}" | base64 -d | \
    bash --noprofile --norc -s -- \
    --ascend-home "${ASCEND_HOME_PATH}" \
    --registry "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" \
    --replace-archive "${expected_archive}" \
    --candidate-soname \
      "${device_provider_soname}=${device_provider_sha}" \
    --candidate-soname \
      "${runtime_probe_provider_filename}=${runtime_probe_provider_sha}"; then
    audit_rc=0
  else
    audit_rc=$?
  fi
  (( audit_rc == 0 )) ||
    die "registered AICPU package/SONAME audit failed"
}

verify_aicpu_archive()
{
  local archive=$1
  local label=$2
  local archive_entries unexpected_entries
  local archive_provider_sha archive_manifest_sha
  local archive_runtime_probe_provider_sha
  local archive_source_manifest_sha archive_umdk_manifest_sha
  local archive_libummu_manifest_sha archive_probe_sha local_probe_sha
  local entry probe

  [[ -r "${archive}" ]] ||
    die "${label} AICPU archive is not readable: ${archive}"
  archive_entries=$(tar -tzf "${archive}") ||
    die "${label} AICPU archive is not a valid tar.gz: ${archive}"
  for entry in \
    aicpu_kernels_device/libkvaf_urma_provider_device.so \
    "aicpu_kernels_device/${runtime_probe_provider_filename}" \
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
      "^aicpu_kernels_device/?$|^aicpu_kernels_device/(libkvaf_urma_provider_device\\.so|${runtime_probe_provider_filename//./\\.}|kvaf-direct-urma-provider\\.manifest|kvaf-direct-urma-source\\.sha256|kvaf-direct-urma-umdk-source\\.sha256|kvaf-direct-urma-libummu-source\\.sha256|kvaf_udma_uapi_probe|kvaf_udma_private_abi_probe|kvaf_ummu_uapi_probe)$" \
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
  archive_runtime_probe_provider_sha=$(
    tar -xOzf "${archive}" \
      "aicpu_kernels_device/${runtime_probe_provider_filename}" |
      sha256sum | awk '{print $1}'
  ) || die "could not hash ${label} versioned runtime provider"
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
     "${archive_runtime_probe_provider_sha}" == \
       "${runtime_probe_provider_sha}" &&
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
for command in awk base64 bash chmod cp dirname flock grep mkdir mktemp mv nm \
  python3 readelf realpath rm sha256sum tail tar wc; do
  command -v "${command}" >/dev/null 2>&1 ||
    die "${command} is required"
done
[[ -f ${KVAF_DURABLE_STATE_HELPER} &&
   ! -L ${KVAF_DURABLE_STATE_HELPER} &&
   -r ${KVAF_DURABLE_STATE_HELPER} ]] ||
  die "durable filesystem helper is missing or unsafe"
export KVAF_AICPU_DURABLE_STATE_HELPER=${KVAF_DURABLE_STATE_HELPER}

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
  die "provider verification failed; inspect the preceding verifier error"
pin_registered_package_audit
host_provider_sha=${KVAF_VERIFIED_HOST_PROVIDER_SHA}
device_provider_sha=${KVAF_VERIFIED_DEVICE_PROVIDER_SHA}
source_manifest_sha=${KVAF_VERIFIED_SOURCE_MANIFEST_SHA}
umdk_source_manifest_sha=${KVAF_VERIFIED_UMDK_SOURCE_MANIFEST_SHA}
libummu_source_manifest_sha=${KVAF_VERIFIED_LIBUMMU_SOURCE_MANIFEST_SHA}
build_manifest_sha=${KVAF_VERIFIED_BUILD_MANIFEST_SHA}
runtime_probe_provider_filename=${KVAF_VERIFIED_RUNTIME_PROBE_PROVIDER_FILENAME}
runtime_probe_provider_build_id=${KVAF_VERIFIED_RUNTIME_PROBE_PROVIDER_BUILD_ID}
runtime_probe_provider_sha=${KVAF_VERIFIED_RUNTIME_PROBE_PROVIDER_SHA}
runtime_probe_provider_source_config_sha=$(manifest_value \
  runtime_probe_provider_source_config_sha256 \
  "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") ||
  die "provider manifest lacks runtime probe source/config identity"
device_provider_soname=$(elf_soname \
  "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}") ||
  die "Device provider has no unique SONAME"
umdk_source_manifest=${KVAF_VERIFIED_UMDK_SOURCE_MANIFEST}
libummu_source_manifest=${KVAF_VERIFIED_LIBUMMU_SOURCE_MANIFEST}
umdk_root=$(manifest_value umdk_root \
  "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") ||
  die "provider build manifest lacks umdk_root"
for management_source in \
  perftest_mgmt.c perftest_mgmt.h \
  perftest_mgmt_tcp.c perftest_mgmt_tcp.h \
  perftest_mgmt_ub.c perftest_mgmt_ub.h
do
  management_path="${umdk_root}/src/urma/tools/urma_perftest/${management_source}"
  awk -v expected="${management_path}" \
      '$2 == expected { found = 1 } END { exit found ? 0 : 1 }' \
      "${umdk_source_manifest}" ||
    die "UMDK source manifest omits ${management_source}"
done

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
  aicpu_kvaf_urma_close_v2 \
  aicpu_kvaf_urma_owner_probe_v1 \
  aicpu_kvaf_urma_epoch_owner_v1
do
  grep -Eq "[[:space:]]${symbol}(@@KVAF_URMA_PROVIDER_2)?$" \
    <<<"${provider_dynamic_symbols}" ||
    die "provider does not export ${symbol}"
done
if grep -Eq \
    '[[:space:]]aicpu_kvaf_urma_rt_probe_v2(@@[^[:space:]]+)?$' \
    <<<"${provider_dynamic_symbols}"; then
  die "canonical Device provider exports the isolated runtime probe"
fi
if grep -Eq \
    '[[:space:]](perftest_|kvaf_perftest_|create_duplex_ctx|destroy_ctx)' \
    <<<"${provider_dynamic_symbols}"; then
  die "provider leaks an internal perftest symbol"
fi

runtime_probe_dynamic_symbols=$(
  nm -D --defined-only \
    "${KVAF_DIRECT_PROVIDER_BUILD_DIR}/${runtime_probe_provider_filename}"
) || die "could not inspect the isolated runtime probe provider"
grep -Eq \
  '[[:space:]]aicpu_kvaf_urma_rt_probe_v2(@@KVAF_URMA_RUNTIME_PROBE_2)?$' \
  <<<"${runtime_probe_dynamic_symbols}" ||
  die "isolated runtime probe provider lacks its AICPU entry"
for symbol in aicpu_kvaf_urma_open_v2 aicpu_kvaf_urma_close_v2 \
  aicpu_kvaf_urma_owner_probe_v1 aicpu_kvaf_urma_epoch_owner_v1
do
  if grep -Eq "[[:space:]]${symbol}(@@[^[:space:]]+)?$" \
      <<<"${runtime_probe_dynamic_symbols}"; then
    die "isolated runtime probe provider leaks canonical entry ${symbol}"
  fi
done

expected_archive="${ASCEND_HOME_PATH}/${KVAF_DIRECT_AICPU_PACKAGE_RELATIVE_DIR}/${KVAF_DIRECT_AICPU_PACKAGE_NAME}"
if [[ "$(realpath -m "${KVAF_DIRECT_AICPU_ARCHIVE}")" != \
      "$(realpath -m "${expected_archive}")" ]]; then
  die "the AICPU archive must be installed at ${expected_archive}"
fi
expected_registration_file="${ASCEND_HOME_PATH}/conf/ascend_package_load.ini"
if [[ "$(realpath -m "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}")" != \
      "$(realpath -m "${expected_registration_file}")" ]]; then
  die "the AICPU package registry must be ${expected_registration_file}"
fi
[[ ! -L "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" &&
   -f "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" &&
   -r "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" &&
   -w "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" ]] ||
  die "the existing CANN AICPU package registry must be a readable/writable regular file"
registration_real_path=$(realpath -e -- \
  "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}") ||
  die "could not resolve the existing CANN AICPU package registry"
[[ ! -L "${registration_real_path}" &&
   -f "${registration_real_path}" &&
   -r "${registration_real_path}" &&
   -w "${registration_real_path}" ]] ||
  die "the resolved CANN AICPU package registry must be a readable/writable regular file"
grep -Fxq 'name:Ascend-aicpu_legacy.tar.gz' \
  "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" ||
  die "the CANN AICPU package registry lacks its built-in baseline"
kvaf_aicpu_acquire_registry_lock \
  "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" ||
  die "could not acquire the shared CANN AICPU registry lock"
if kvaf_aicpu_registration_is_exact \
    "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" \
    "${KVAF_DIRECT_AICPU_PACKAGE_NAME}" \
    "${KVAF_DIRECT_AICPU_PACKAGE_RELATIVE_DIR}"; then
  registration_state=exact
elif kvaf_aicpu_registration_is_absent \
    "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" \
    "${KVAF_DIRECT_AICPU_PACKAGE_NAME}" \
    "${KVAF_DIRECT_AICPU_PACKAGE_RELATIVE_DIR}"; then
  registration_state=absent
else
  die "the KVAF AICPU package registration is duplicated or malformed"
fi
package_install_dir=$(dirname "${KVAF_DIRECT_AICPU_ARCHIVE}")
mkdir -p "${package_install_dir}"
[[ ! -L "${package_install_dir}" &&
   -d "${package_install_dir}" &&
   -w "${package_install_dir}" ]] ||
  die "the AICPU package directory must be a writable real directory"
[[ ! -L "${KVAF_DIRECT_AICPU_ARCHIVE}" &&
   ( ! -e "${KVAF_DIRECT_AICPU_ARCHIVE}" ||
     -f "${KVAF_DIRECT_AICPU_ARCHIVE}" ) ]] ||
  die "the AICPU archive target must not be a symlink or non-file"
if [[ "${registration_state}" == exact ]] &&
   [[ ! -f "${KVAF_DIRECT_AICPU_ARCHIVE}" ||
      ! -r "${KVAF_DIRECT_AICPU_ARCHIVE}" ]]; then
  die "the exact KVAF registration points to a missing/unreadable archive"
fi

# Parse every registered package_path, not only archives adjacent to KVAF's
# target.  Candidate mode ignores the archive being atomically replaced but
# rejects any foreign registration of either KVAF runtime SONAME, even when
# the foreign bytes happen to be identical (loader/package ownership would
# still be ambiguous).
run_registered_package_audit

# Older KVAF revisions wrote the same Device provider into the borrowed UB
# package name.  It is harmless while unregistered, but two registered
# archives extracting the same SO basename have order-dependent semantics.
legacy_registration_count=$(grep -Fxc \
  'name:liburma_perftest.tar.gz' \
  "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" || true)
(( legacy_registration_count == 0 )) ||
  die "remove the legacy liburma_perftest registration before installing the isolated KVAF package"

shopt -s nullglob
for archive in "${package_install_dir}"/*.tar.gz; do
  if [[ "${archive}" == "${KVAF_DIRECT_AICPU_ARCHIVE}" ]]; then
    continue
  fi
  [[ ! -L "${archive}" && -f "${archive}" && -r "${archive}" ]] ||
    die "sibling AICPU package is missing or unsafe: ${archive}"
  archive_listing=$(tar -tzf "${archive}" 2>/dev/null) ||
    die "sibling AICPU package is invalid: ${archive}"
  if grep -Eq \
      '(^|/)(libkvaf_urma_provider(_host|_device)?|libkvaf_rt_[0-9a-f]{16}|liburma_perftest)\.so$' \
      <<<"${archive_listing}"; then
    die "conflicting direct/perftest provider exists in ${archive}"
  fi
done
shopt -u nullglob

temporary_dir=$(mktemp -d /tmp/kvaf-direct-aicpu.XXXXXX)
temporary_install_archive=
rollback_archive=
rollback_archive_sha=
rollback_registration=
rollback_registration_sha=
archive_preexisting=0
transaction_archive_sha=
transaction_registration_sha=
operator_backup=
operator_backup_candidate=
operator_backup_sha=
operator_backup_created=0
deployment_started=0
deployment_committed=0
registration_update_started=0
cleanup()
{
  local cleanup_status=$?
  local restore_candidate restore_registration_candidate
  local current_archive_sha current_registration_sha current_backup_sha
  local registration_rollback_safe=1
  local archive_rollback_safe=0
  local preserved_archive preserved_registry recovery_parent recovery_sha
  trap - EXIT

  # Preserve a verified rollback image outside temporary_dir whenever an
  # automatic rollback cannot be proven safe.  These files deliberately do
  # not end in .tar.gz, so CANN and the sibling-package scan cannot treat them
  # as loadable packages.  An operator can inspect the printed path before
  # making any manual recovery decision.
  preserve_recovery_copy()
  {
    local source_file=$1
    local expected_sha=$2
    local recovery_template=$3
    local recovery_label=$4
    local recovery_file

    recovery_file=$(mktemp "${recovery_template}") || return 1
    if ! cp -p -- "${source_file}" "${recovery_file}"; then
      rm -f -- "${recovery_file}"
      return 1
    fi
    recovery_sha=$(sha256sum -- "${recovery_file}" 2>/dev/null |
      awk '{print $1}' || true)
    if [[ -z ${recovery_sha} || ${recovery_sha} != "${expected_sha}" ]]; then
      rm -f -- "${recovery_file}"
      return 1
    fi
    recovery_parent=$(dirname -- "${recovery_file}")
    if ! python3 "${KVAF_DURABLE_STATE_HELPER}" sync \
        "${recovery_file}" "${recovery_parent}"; then
      rm -f -- "${recovery_file}"
      return 1
    fi
    printf '%s_RECOVERY_COPY=%s\n' "${recovery_label}" "${recovery_file}" >&2
    printf '%s' "${recovery_file}"
  }

  if (( registration_update_started != 0 && deployment_committed == 0 )) &&
     [[ -n ${rollback_registration} && -f ${rollback_registration} ]]; then
    current_registration_sha=$(sha256sum \
      "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" 2>/dev/null |
      awk '{print $1}' || true)
    if [[ -n ${current_registration_sha} &&
          ${current_registration_sha} == "${rollback_registration_sha}" ]]; then
      echo "KVAF_AICPU_REGISTRATION_ROLLBACK_NOT_NEEDED" >&2
    elif [[ -n ${transaction_registration_sha} &&
            ${current_registration_sha} == \
              "${transaction_registration_sha}" ]]; then
      restore_registration_candidate=$(mktemp \
        "$(dirname -- "${registration_real_path}")/.ascend_package_load.ini.rollback.XXXXXX") ||
        restore_registration_candidate=
      if [[ -z ${restore_registration_candidate} ]] ||
         ! cp -p "${rollback_registration}" \
           "${restore_registration_candidate}" ||
         ! python3 "${KVAF_DURABLE_STATE_HELPER}" publish \
           "${restore_registration_candidate}" \
           "${registration_real_path}"; then
        [[ -z ${restore_registration_candidate} ]] ||
          rm -f "${restore_registration_candidate}" || true
        echo "ERROR: failed to restore the AICPU package registry" >&2
        cleanup_status=3
        registration_rollback_safe=0
      else
        echo "KVAF_AICPU_REGISTRATION_ROLLBACK_OK" >&2
      fi
    else
      echo "ERROR: AICPU registry changed outside this transaction; refusing to overwrite it during rollback" >&2
      echo "REGISTRATION=${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" >&2
      cleanup_status=3
      registration_rollback_safe=0
    fi
  fi
  if (( deployment_started == 0 || deployment_committed != 0 )); then
    archive_rollback_safe=1
  elif (( registration_rollback_safe == 0 )); then
    echo "ERROR: archive rollback skipped because registry rollback is not proven safe" >&2
    cleanup_status=3
  elif (( archive_preexisting != 0 )) &&
       [[ -n "${rollback_archive}" && -f "${rollback_archive}" ]]; then
      current_archive_sha=$(sha256sum "${KVAF_DIRECT_AICPU_ARCHIVE}" \
        2>/dev/null | awk '{print $1}' || true)
      if [[ -n ${current_archive_sha} &&
            ${current_archive_sha} == "${rollback_archive_sha}" ]]; then
        echo "KVAF_AICPU_ARCHIVE_ROLLBACK_NOT_NEEDED" >&2
        archive_rollback_safe=1
      elif [[ -n ${transaction_archive_sha} &&
              ${current_archive_sha} == "${transaction_archive_sha}" ]]; then
        restore_candidate=$(mktemp \
          "${package_install_dir}/.${KVAF_DIRECT_AICPU_PACKAGE_NAME}.rollback.XXXXXX") ||
          restore_candidate=
        if [[ -z "${restore_candidate}" ]] ||
           ! cp -p "${rollback_archive}" "${restore_candidate}" ||
           ! python3 "${KVAF_DURABLE_STATE_HELPER}" publish \
             "${restore_candidate}" "${KVAF_DIRECT_AICPU_ARCHIVE}"; then
          [[ -z "${restore_candidate}" ]] ||
            rm -f "${restore_candidate}" || true
          echo "ERROR: failed to restore the previous KVAF AICPU archive" >&2
          cleanup_status=3
        else
          echo "KVAF_AICPU_ARCHIVE_ROLLBACK_OK" >&2
          archive_rollback_safe=1
        fi
      else
        echo "ERROR: AICPU archive changed outside this transaction; refusing to overwrite it during rollback" >&2
        echo "ARCHIVE=${KVAF_DIRECT_AICPU_ARCHIVE}" >&2
        cleanup_status=3
      fi
  elif (( archive_preexisting == 0 )); then
      current_archive_sha=$(sha256sum "${KVAF_DIRECT_AICPU_ARCHIVE}" \
        2>/dev/null | awk '{print $1}' || true)
      if [[ ! -e ${KVAF_DIRECT_AICPU_ARCHIVE} ]]; then
        echo "KVAF_AICPU_ARCHIVE_ABSENT_STATE_ROLLBACK_NOT_NEEDED" >&2
        archive_rollback_safe=1
      elif [[ -n ${transaction_archive_sha} &&
              ${current_archive_sha} == "${transaction_archive_sha}" ]] &&
           python3 "${KVAF_DURABLE_STATE_HELPER}" remove \
             "${KVAF_DIRECT_AICPU_ARCHIVE}"; then
        echo "KVAF_AICPU_ARCHIVE_ABSENT_STATE_ROLLBACK_OK" >&2
        archive_rollback_safe=1
      else
        echo "ERROR: failed to remove the newly installed uncommitted KVAF archive" >&2
        cleanup_status=3
      fi
  else
    echo "ERROR: previous archive rollback image is missing" >&2
    cleanup_status=3
  fi

  if (( registration_rollback_safe == 0 )) &&
     [[ -n ${rollback_registration} && -f ${rollback_registration} ]]; then
    if ! preserved_registry=$(preserve_recovery_copy \
        "${rollback_registration}" "${rollback_registration_sha}" \
        "$(dirname -- "${registration_real_path}")/.ascend_package_load.ini.kvaf-recovery.XXXXXX" \
        KVAF_AICPU_REGISTRY); then
      echo "ERROR: could not preserve the registry rollback image" >&2
      cleanup_status=3
    fi
  fi
  if (( archive_rollback_safe == 0 && archive_preexisting != 0 )) &&
     [[ -n ${rollback_archive} && -f ${rollback_archive} ]]; then
    if ! preserved_archive=$(preserve_recovery_copy \
        "${rollback_archive}" "${rollback_archive_sha}" \
        "${package_install_dir}/.${KVAF_DIRECT_AICPU_PACKAGE_NAME}.kvaf-recovery.XXXXXX" \
        KVAF_AICPU_ARCHIVE); then
      echo "ERROR: could not preserve the archive rollback image" >&2
      cleanup_status=3
    fi
  fi

  if (( operator_backup_created != 0 && deployment_committed == 0 &&
        archive_rollback_safe != 0 )); then
    current_backup_sha=$(sha256sum "${operator_backup}" 2>/dev/null |
      awk '{print $1}' || true)
    if [[ ${current_backup_sha} == "${operator_backup_sha}" ]] &&
       python3 "${KVAF_DURABLE_STATE_HELPER}" remove \
         "${operator_backup}"; then
      echo "KVAF_AICPU_UNCOMMITTED_BACKUP_REMOVED" >&2
    else
      echo "ERROR: uncommitted operator backup changed; refusing to delete it" >&2
      cleanup_status=3
    fi
  elif (( operator_backup_created != 0 && deployment_committed == 0 )); then
    echo "KVAF_AICPU_OPERATOR_BACKUP_PRESERVED=${operator_backup}" >&2
  fi
  if [[ -n "${temporary_install_archive}" ]]; then
    rm -f "${temporary_install_archive}"
  fi
  if [[ -n "${operator_backup_candidate}" ]]; then
    rm -f -- "${operator_backup_candidate}"
  fi
  rm -rf "${temporary_dir}"
  exit "${cleanup_status}"
}
trap cleanup EXIT

rollback_registration="${temporary_dir}/registration.rollback"
cp -p "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" \
  "${rollback_registration}" ||
  die "could not snapshot the AICPU package registry"
rollback_registration_sha=$(sha256sum "${rollback_registration}" |
  awk '{print $1}') || die "could not hash the registry rollback snapshot"

package_dir="${temporary_dir}/aicpu_kernels_device"
mkdir -p "${package_dir}"
cp "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}" \
  "${package_dir}/libkvaf_urma_provider_device.so"
cp "${KVAF_DIRECT_PROVIDER_BUILD_DIR}/${runtime_probe_provider_filename}" \
  "${package_dir}/${runtime_probe_provider_filename}"
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
  "${package_dir}/${runtime_probe_provider_filename}" \
  "${package_dir}/kvaf-direct-urma-provider.manifest" \
  "${package_dir}/kvaf-direct-urma-source.sha256" \
  "${package_dir}/kvaf-direct-urma-umdk-source.sha256" \
  "${package_dir}/kvaf-direct-urma-libummu-source.sha256"
chmod 0755 \
  "${package_dir}/kvaf_udma_uapi_probe" \
  "${package_dir}/kvaf_udma_private_abi_probe" \
  "${package_dir}/kvaf_ummu_uapi_probe"

temporary_archive="${temporary_dir}/${KVAF_DIRECT_AICPU_PACKAGE_NAME}"
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
  operator_backup="${KVAF_DIRECT_AICPU_ARCHIVE}.before-kvaf-direct"
  if [[ -e "${operator_backup}" || -L "${operator_backup}" ]]; then
    die "refusing to overwrite existing backup: ${operator_backup}"
  fi
  operator_backup_sha=$(sha256sum "${KVAF_DIRECT_AICPU_ARCHIVE}" |
    awk '{print $1}') || die "could not hash the pre-KVAF archive"
  operator_backup_candidate=$(mktemp \
    "${package_install_dir}/.${KVAF_DIRECT_AICPU_PACKAGE_NAME}.operator-backup.XXXXXX") ||
    die "could not stage the pre-KVAF operator backup"
  cp -p -- "${KVAF_DIRECT_AICPU_ARCHIVE}" \
    "${operator_backup_candidate}" ||
    die "could not copy the pre-KVAF operator backup"
  [[ $(sha256sum "${operator_backup_candidate}" | awk '{print $1}') == \
     "${operator_backup_sha}" ]] || die "operator backup verification failed"
  [[ ! -e "${operator_backup}" && ! -L "${operator_backup}" ]] ||
    die "operator backup target appeared during publication"
  python3 "${KVAF_DURABLE_STATE_HELPER}" publish \
    "${operator_backup_candidate}" "${operator_backup}" ||
    die "could not durably publish the pre-KVAF operator backup"
  operator_backup_candidate=
  operator_backup_created=1
  echo "BACKUP=${operator_backup}"
fi

temporary_install_archive=$(
  mktemp \
    "${package_install_dir}/.${KVAF_DIRECT_AICPU_PACKAGE_NAME}.XXXXXX"
)
cp "${temporary_archive}" "${temporary_install_archive}"
chmod 0644 "${temporary_install_archive}"
verify_aicpu_archive "${temporary_install_archive}" install-candidate
if [[ -e "${KVAF_DIRECT_AICPU_ARCHIVE}" ]]; then
  archive_preexisting=1
  rollback_archive="${temporary_dir}/registered-archive.rollback"
  cp -p "${KVAF_DIRECT_AICPU_ARCHIVE}" "${rollback_archive}"
  rollback_archive_sha=$(sha256sum "${rollback_archive}" |
    awk '{print $1}') || die "could not hash the archive rollback snapshot"
fi
transaction_archive_sha=$(sha256sum "${temporary_install_archive}" |
  awk '{print $1}') || die "could not hash the install candidate"

# The common registry lock serializes cooperating KVAF packagers.  Recheck the
# exact archive preimage immediately before publication so an earlier foreign
# edit is never silently overwritten.  A non-cooperating root writer can still
# race the final check/rename window; package installation therefore requires
# an exclusive CANN maintenance window.
if (( archive_preexisting != 0 )); then
  [[ ! -L ${KVAF_DIRECT_AICPU_ARCHIVE} &&
     -f ${KVAF_DIRECT_AICPU_ARCHIVE} &&
     $(sha256sum -- "${KVAF_DIRECT_AICPU_ARCHIVE}" | awk '{print $1}') == \
       "${rollback_archive_sha}" ]] ||
    die "AICPU archive changed before transaction publish"
else
  [[ ! -e ${KVAF_DIRECT_AICPU_ARCHIVE} &&
     ! -L ${KVAF_DIRECT_AICPU_ARCHIVE} ]] ||
    die "AICPU archive appeared before transaction publish"
fi
deployment_started=1
python3 "${KVAF_DURABLE_STATE_HELPER}" publish \
  "${temporary_install_archive}" "${KVAF_DIRECT_AICPU_ARCHIVE}" ||
  die "could not durably publish the KVAF AICPU archive"
temporary_install_archive=

verify_aicpu_archive "${KVAF_DIRECT_AICPU_ARCHIVE}" installed

# A tar file is not automatically discovered by CANN.  Register the unique
# KVAF package only after its verified archive is durably visible.  The helper
# atomically replaces the registry while preserving every other package
# stanza; a crash before this point leaves only an inert, unregistered file.
registration_update_started=1
if kvaf_aicpu_install_registration \
    "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" \
    "${KVAF_DIRECT_AICPU_PACKAGE_NAME}" \
    "${KVAF_DIRECT_AICPU_PACKAGE_RELATIVE_DIR}" \
    "${rollback_registration_sha}"; then
  transaction_registration_sha=${KVAF_AICPU_REGISTRATION_PUBLISHED_SHA256:-}
else
  # The helper exposes the candidate SHA before rename.  If durable sync or a
  # post-publish verification fails, cleanup can still recognize and restore
  # this transaction's bytes instead of misclassifying them as an external edit.
  transaction_registration_sha=${KVAF_AICPU_REGISTRATION_PUBLISHED_SHA256:-}
  die "could not atomically register the KVAF AICPU package"
fi
kvaf_aicpu_registration_is_exact \
  "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" \
  "${KVAF_DIRECT_AICPU_PACKAGE_NAME}" \
  "${KVAF_DIRECT_AICPU_PACKAGE_RELATIVE_DIR}" ||
  die "installed KVAF AICPU package registration is not exact"
[[ ${transaction_registration_sha} =~ ^[0-9a-f]{64}$ ]] ||
  die "registration helper did not expose the installed registry identity"
[[ $(sha256sum "${KVAF_DIRECT_AICPU_REGISTRATION_FILE}" | awk '{print $1}') == \
   "${transaction_registration_sha}" ]] ||
  die "installed AICPU registry differs from the helper-published identity"
# Re-audit the installed archive and final registry before committing the
# transaction.  Any failure restores both the previous archive and registry.
run_registered_package_audit
deployment_committed=1

echo "KVAF_A5_DIRECT_URMA_AICPU_PACKAGE_OK"
echo "AICPU_ARCHIVE=${KVAF_DIRECT_AICPU_ARCHIVE}"
echo "AICPU_PACKAGE_NAME=${KVAF_DIRECT_AICPU_PACKAGE_NAME}"
echo "AICPU_REGISTRATION_FILE=${KVAF_DIRECT_AICPU_REGISTRATION_FILE}"
echo "AICPU_REGISTRATION=verified-exact"
echo "AICPU_REGISTRATION_PREVIOUS_STATE=${registration_state}"
echo "DEVICE_PROVIDER_SHA256=${device_provider_sha}"
echo "RUNTIME_PROBE_PROVIDER_FILENAME=${runtime_probe_provider_filename}"
echo "RUNTIME_PROBE_PROVIDER_BUILD_ID_SHA256=${runtime_probe_provider_build_id}"
echo "RUNTIME_PROBE_PROVIDER_SOURCE_CONFIG_SHA256=${runtime_probe_provider_source_config_sha}"
echo "RUNTIME_PROBE_PROVIDER_SHA256=${runtime_probe_provider_sha}"
echo "PERFTEST_MANAGEMENT=official-umdk-ub-default-tcp-fallback-v1"
echo "DEVICE_QUEUE_MEMORY_POLICY=sva-aware-anonymous-or-kernel-v1"
tar -tzf "${KVAF_DIRECT_AICPU_ARCHIVE}"
fi
/bin/sh -c 'exit 2'
