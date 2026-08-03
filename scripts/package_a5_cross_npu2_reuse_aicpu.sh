#!/usr/bin/env bash

# Install one source- and token-owned AICPU package without replacing the
# canonical KVAF package.  This entry point is deliberately separate from the
# production packager: it is only for the explicit, second-order NPU2 reuse
# experiment while another direct-URMA generation may still be resident.

set -Eeuo pipefail
umask 077
unset BASH_ENV ENV
unset -f exit builtin 2>/dev/null || true

# The cross-NPU reuse experiment is retired.  Its packager publishes a live
# CANN archive and registry stanza before invoking any canonical workflow, so
# it must be independently unreachable while the same capability ledger is
# absent.  No risk-acceptance environment variable can bypass this gate.
echo "KVAF_A5_CROSS_NPU2_AICPU_PACKAGE_MUTATION_DISABLED" >&2
echo "CAPABILITY_LEDGER=not-implemented" >&2
echo "CROSS_NPU2_EXPERIMENT=retired" >&2
echo "No CANN/state/helper/lock/archive/registry operation was attempted." >&2
builtin exit 2

# Preserve the retired implementation only as unreachable reference code.
if [[ 0 == 1 ]]; then

if (( $# != 0 )); then
  echo "ERROR: this packager takes no positional arguments" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
source "${SCRIPT_DIR}/a5_aicpu_package_registration.sh"

kvaf_mode=cross-npu2-reuse-versioned-offline-v1
kvaf_risk_contract=I_ACCEPT_UNKNOWN_NPU2_REUSE_AND_SHARED_UMMU_RISK
kvaf_provider_schema=kvaf-a5-cross-npu2-versioned-provider-v1
kvaf_package_schema=kvaf-a5-cross-npu2-versioned-package-v1
kvaf_package_relative_dir=opp/vendors/cust/op_impl/aicpu/kernel
kvaf_canonical_package_name=cann-kvaf-direct-urma-compat.tar.gz

die()
{
  echo "ERROR: $*" >&2
  exit 2
}

sha256_file()
{
  sha256sum -- "$1" | awk '{print $1}'
}

profile_value()
{
  local key=$1
  local profile=$2

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
  ' "${profile}"
}

require_regular_file()
{
  local path=$1
  local label=$2

  [[ ! -L "${path}" && -f "${path}" && -r "${path}" ]] ||
    die "${label} must be a readable regular non-symlink: ${path}"
}

require_profile_keys()
{
  local profile=$1
  shift

  awk -F= -v expected_keys="$*" '
    BEGIN {
      count = split(expected_keys, expected, " ")
      for (index = 1; index <= count; index++) {
        allowed[expected[index]] = 1
      }
    }
    {
      separator = index($0, "=")
      if (separator <= 1 || separator == length($0)) {
        invalid = 1
        next
      }
      key = substr($0, 1, separator - 1)
      if (!(key in allowed) || ++seen[key] != 1) {
        invalid = 1
      }
    }
    END {
      for (key in allowed) {
        if (seen[key] != 1) {
          invalid = 1
        }
      }
      exit(invalid ? 1 : 0)
    }
  ' "${profile}" || die "sealed profile has missing, duplicate, or foreign keys: ${profile}"
}

verify_provider_profile()
{
  local profile=$1
  local expected_provider_dir=$2
  local expected_token=$3
  local expected_commit=$4
  local profile_mode

  require_regular_file "${profile}" "experiment provider profile"
  profile_mode=$(stat -c '%a' -- "${profile}") ||
    die "could not read experiment provider profile mode"
  [[ "${profile_mode}" == 400 ]] ||
    die "experiment provider profile must be sealed mode 0400"

  require_profile_keys "${profile}" \
    schema mode experiment_profile artifact_token source_commit provider_dir \
    host_provider_filename host_provider_sha256 \
    device_provider_filename device_provider_sha256 \
    source_manifest_filename source_manifest_sha256 \
    build_manifest_filename build_manifest_sha256 \
    success_stamp_filename success_stamp_sha256

  [[ "$(profile_value schema "${profile}")" == \
      "${kvaf_provider_schema}" ]] ||
    die "unexpected experiment provider profile schema"
  [[ "$(profile_value mode "${profile}")" == "${kvaf_mode}" ]] ||
    die "unexpected experiment provider mode"
  [[ "$(profile_value experiment_profile "${profile}")" == \
      "${kvaf_mode}" ]] ||
    die "unexpected experiment provider profile contract"
  [[ "$(profile_value artifact_token "${profile}")" == \
      "${expected_token}" ]] ||
    die "experiment provider token does not match the requested token"
  [[ "$(profile_value source_commit "${profile}")" == \
      "${expected_commit}" ]] ||
    die "experiment provider source commit does not match HEAD"
  [[ "$(profile_value provider_dir "${profile}")" == \
      "${expected_provider_dir}" ]] ||
    die "experiment provider directory is not source-owned"
}

profile_artifact()
{
  local profile=$1
  local provider_dir=$2
  local filename_key=$3
  local sha_key=$4
  local expected_filename=${5:-}
  local filename expected_sha path actual_sha

  filename=$(profile_value "${filename_key}" "${profile}") ||
    die "provider profile lacks ${filename_key}"
  [[ "${filename}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ &&
     "${filename}" != . && "${filename}" != .. ]] ||
    die "provider profile contains an unsafe ${filename_key}"
  if [[ -n "${expected_filename}" &&
        "${filename}" != "${expected_filename}" ]]; then
    die "provider profile has an unexpected ${filename_key}"
  fi
  expected_sha=$(profile_value "${sha_key}" "${profile}") ||
    die "provider profile lacks ${sha_key}"
  [[ "${expected_sha}" =~ ^[0-9a-f]{64}$ ]] ||
    die "provider profile has a malformed ${sha_key}"
  path="${provider_dir}/${filename}"
  require_regular_file "${path}" "provider artifact ${filename}"
  actual_sha=$(sha256_file "${path}") ||
    die "could not hash provider artifact ${filename}"
  [[ "${actual_sha}" == "${expected_sha}" ]] ||
    die "provider artifact hash mismatch: ${filename}"
  printf '%s\n' "${filename}"
}

verify_archive()
{
  local archive=$1
  local label=$2
  local listing expected archive_sha
  local entry actual_sha expected_sha

  require_regular_file "${archive}" "${label} archive"
  listing=$(tar -tzf "${archive}") ||
    die "${label} archive is not a valid tar.gz"
  expected=$(printf '%s\n' \
    aicpu_kernels_device/ \
    "aicpu_kernels_device/${kvaf_device_provider_filename}" \
    "aicpu_kernels_device/${kvaf_archive_provider_profile_name}" \
    "aicpu_kernels_device/${kvaf_archive_source_manifest_name}" \
    "aicpu_kernels_device/${kvaf_archive_build_manifest_name}" \
    "aicpu_kernels_device/${kvaf_archive_success_stamp_name}" |
    sort)
  [[ "$(sort <<<"${listing}")" == "${expected}" ]] ||
    die "${label} archive has missing, colliding, or unexpected entries"

  for entry in \
    "${kvaf_device_provider_filename}:${kvaf_device_provider_sha}" \
    "${kvaf_archive_provider_profile_name}:${kvaf_provider_profile_sha}" \
    "${kvaf_archive_source_manifest_name}:${kvaf_source_manifest_sha}" \
    "${kvaf_archive_build_manifest_name}:${kvaf_build_manifest_sha}" \
    "${kvaf_archive_success_stamp_name}:${kvaf_success_stamp_sha}"
  do
    expected_sha=${entry#*:}
    entry=${entry%%:*}
    actual_sha=$(tar -xOzf "${archive}" \
      "aicpu_kernels_device/${entry}" | sha256sum | awk '{print $1}') ||
      die "could not hash ${label} archive entry ${entry}"
    [[ "${actual_sha}" == "${expected_sha}" ]] ||
      die "${label} archive entry hash mismatch: ${entry}"
  done

  archive_sha=$(sha256_file "${archive}") ||
    die "could not hash ${label} archive"
  [[ "${archive_sha}" =~ ^[0-9a-f]{64}$ ]] ||
    die "${label} archive hash is malformed"
  printf '%s\n' "${archive_sha}"
}

verify_package_profile()
{
  local profile=$1
  local expected_archive_sha=$2
  local profile_mode

  require_regular_file "${profile}" "experiment package profile"
  profile_mode=$(stat -c '%a' -- "${profile}") ||
    die "could not read experiment package profile mode"
  [[ "${profile_mode}" == 400 ]] ||
    die "experiment package profile must be sealed mode 0400"
  require_profile_keys "${profile}" \
    schema mode risk_contract artifact_token source_commit provider_dir \
    provider_profile_path provider_profile_sha256 package_name \
    package_relative_dir archive_path archive_sha256 registration_file \
    device_provider_filename device_provider_sha256 archive_device_entry

  [[ "$(profile_value schema "${profile}")" == \
      "${kvaf_package_schema}" &&
     "$(profile_value mode "${profile}")" == "${kvaf_mode}" &&
     "$(profile_value risk_contract "${profile}")" == \
      "${kvaf_risk_contract}" &&
     "$(profile_value artifact_token "${profile}")" == \
      "${kvaf_artifact_token}" &&
     "$(profile_value source_commit "${profile}")" == \
      "${kvaf_source_commit}" &&
     "$(profile_value provider_dir "${profile}")" == \
      "${kvaf_provider_dir}" &&
     "$(profile_value provider_profile_path "${profile}")" == \
      "${kvaf_provider_profile}" &&
     "$(profile_value provider_profile_sha256 "${profile}")" == \
      "${kvaf_provider_profile_sha}" &&
     "$(profile_value package_name "${profile}")" == \
      "${kvaf_package_name}" &&
     "$(profile_value package_relative_dir "${profile}")" == \
      "${kvaf_package_relative_dir}" &&
     "$(profile_value archive_path "${profile}")" == \
      "${kvaf_archive_path}" &&
     "$(profile_value archive_sha256 "${profile}")" == \
      "${expected_archive_sha}" &&
     "$(profile_value registration_file "${profile}")" == \
      "${kvaf_registration_file}" &&
     "$(profile_value device_provider_filename "${profile}")" == \
      "${kvaf_device_provider_filename}" &&
     "$(profile_value device_provider_sha256 "${profile}")" == \
      "${kvaf_device_provider_sha}" &&
     "$(profile_value archive_device_entry "${profile}")" == \
      "aicpu_kernels_device/${kvaf_device_provider_filename}" ]] ||
    die "experiment package profile does not match the sealed package"
}

verify_canonical_archive_unchanged()
{
  local actual_sha

  if [[ "${kvaf_canonical_archive_state}" == absent ]]; then
    [[ ! -e "${kvaf_canonical_archive}" &&
       ! -L "${kvaf_canonical_archive}" ]] ||
      die "canonical AICPU archive appeared during versioned installation"
    return 0
  fi
  require_regular_file "${kvaf_canonical_archive}" \
    "canonical direct URMA AICPU archive"
  actual_sha=$(sha256_file "${kvaf_canonical_archive}") ||
    die "could not re-hash the canonical AICPU archive"
  [[ "${actual_sha}" == "${kvaf_canonical_archive_sha}" ]] ||
    die "canonical AICPU archive changed during versioned installation"
}

[[ "${KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE:-}" == \
    "${kvaf_risk_contract}" ]] ||
  die "the exact NPU2 reuse/shared UMMU risk phrase is required"
kvaf_artifact_token=${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN:-}
[[ "${kvaf_artifact_token}" =~ ^[0-9a-f]{16}$ ]] ||
  die "KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN must be 16 lowercase hexadecimal digits"

[[ -e /.dockerenv ]] ||
  die "install the versioned AICPU package inside vllm-ascend-cs-kvaf"
[[ ${EUID} -eq 0 ]] ||
  die "versioned AICPU package installation requires container uid 0"
: "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"
for command in awk chmod cmp cp dirname flock git grep ln mkdir mktemp \
  realpath rm sha256sum sort stat tail tar wc; do
  command -v "${command}" >/dev/null 2>&1 ||
    die "${command} is required"
done

kvaf_source_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD) ||
  die "could not pin the KVAF source commit"
[[ "${kvaf_source_commit}" =~ ^[0-9a-f]{40}$ ]] ||
  die "the KVAF source commit is malformed"
kvaf_experiment_root=/data/.kvaf-state/KVAF/experimental/reuse-artifacts
kvaf_artifact_root="${kvaf_experiment_root}/${kvaf_source_commit}/${kvaf_artifact_token}"
kvaf_provider_dir="${kvaf_artifact_root}/provider"
kvaf_provider_profile="${kvaf_provider_dir}/experiment-provider.profile"
kvaf_package_profile="${kvaf_artifact_root}/experiment-package.profile"

[[ ! -L "${kvaf_experiment_root}" &&
   -d "${kvaf_experiment_root}" ]] ||
  die "the fixed versioned experiment root must be a real directory"
[[ ! -L "${kvaf_artifact_root}" &&
   -d "${kvaf_artifact_root}" ]] ||
  die "the source/token-owned artifact root is missing or unsafe"
[[ ! -L "${kvaf_provider_dir}" && -d "${kvaf_provider_dir}" ]] ||
  die "the versioned provider directory is missing or unsafe"
[[ "$(realpath -e "${kvaf_provider_dir}")" == \
    "${kvaf_provider_dir}" ]] ||
  die "the versioned provider directory is not canonical"

verify_provider_profile "${kvaf_provider_profile}" \
  "${kvaf_provider_dir}" "${kvaf_artifact_token}" "${kvaf_source_commit}"
kvaf_provider_profile_sha=$(sha256_file "${kvaf_provider_profile}") ||
  die "could not hash the experiment provider profile"

kvaf_host_provider_filename=$(profile_artifact \
  "${kvaf_provider_profile}" "${kvaf_provider_dir}" \
  host_provider_filename host_provider_sha256 \
  libkvaf_urma_provider_host.so)
kvaf_host_provider_sha=$(profile_value host_provider_sha256 \
  "${kvaf_provider_profile}")
kvaf_device_provider_filename=$(profile_artifact \
  "${kvaf_provider_profile}" "${kvaf_provider_dir}" \
  device_provider_filename device_provider_sha256 \
  "libkvaf_ud_${kvaf_artifact_token}.so")
kvaf_device_provider_sha=$(profile_value device_provider_sha256 \
  "${kvaf_provider_profile}")
(( ${#kvaf_device_provider_filename} < 32 )) ||
  die "the AICPU Device provider basename must be shorter than 32 bytes"
kvaf_source_manifest_filename=$(profile_artifact \
  "${kvaf_provider_profile}" "${kvaf_provider_dir}" \
  source_manifest_filename source_manifest_sha256 source.sha256)
kvaf_source_manifest_sha=$(profile_value source_manifest_sha256 \
  "${kvaf_provider_profile}")
kvaf_build_manifest_filename=$(profile_artifact \
  "${kvaf_provider_profile}" "${kvaf_provider_dir}" \
  build_manifest_filename build_manifest_sha256 build.manifest)
kvaf_build_manifest_sha=$(profile_value build_manifest_sha256 \
  "${kvaf_provider_profile}")
kvaf_success_stamp_filename=$(profile_artifact \
  "${kvaf_provider_profile}" "${kvaf_provider_dir}" \
  success_stamp_filename success_stamp_sha256 build.success)
kvaf_success_stamp_sha=$(profile_value success_stamp_sha256 \
  "${kvaf_provider_profile}")

[[ "$(profile_value device_provider_filename \
      "${kvaf_provider_dir}/${kvaf_build_manifest_filename}")" == \
    "${kvaf_device_provider_filename}" &&
   "$(profile_value device_provider_sha256 \
      "${kvaf_provider_dir}/${kvaf_build_manifest_filename}")" == \
    "${kvaf_device_provider_sha}" ]] ||
  die "provider build manifest does not seal the versioned Device provider"

kvaf_ascend_home=$(realpath -e "${ASCEND_HOME_PATH}") ||
  die "ASCEND_HOME_PATH does not exist"
[[ -d "${kvaf_ascend_home}" ]] ||
  die "ASCEND_HOME_PATH is not a directory"
kvaf_package_name="cann-kvaf-du-${kvaf_artifact_token}.tar.gz"
kvaf_package_dir="${kvaf_ascend_home}/${kvaf_package_relative_dir}"
kvaf_archive_path="${kvaf_package_dir}/${kvaf_package_name}"
kvaf_registration_file="${kvaf_ascend_home}/conf/ascend_package_load.ini"
kvaf_canonical_archive="${kvaf_package_dir}/${kvaf_canonical_package_name}"
[[ ! -L "${kvaf_package_dir}" && -d "${kvaf_package_dir}" &&
   -w "${kvaf_package_dir}" ]] ||
  die "the CANN AICPU package directory must be a writable real directory"
require_regular_file "${kvaf_registration_file}" "CANN AICPU registry"
[[ -w "${kvaf_registration_file}" ]] ||
  die "the CANN AICPU registry is not writable"
grep -Fxq 'name:Ascend-aicpu_legacy.tar.gz' \
  "${kvaf_registration_file}" ||
  die "the CANN AICPU registry lacks its built-in baseline"

kvaf_aicpu_acquire_registry_lock "${kvaf_registration_file}" ||
  die "another CANN AICPU registry transaction is running"

if [[ -e "${kvaf_canonical_archive}" ||
      -L "${kvaf_canonical_archive}" ]]; then
  require_regular_file "${kvaf_canonical_archive}" \
    "canonical direct URMA AICPU archive"
  kvaf_canonical_archive_state=present
  kvaf_canonical_archive_sha=$(sha256_file "${kvaf_canonical_archive}") ||
    die "could not hash the canonical AICPU archive"
else
  kvaf_canonical_archive_state=absent
  kvaf_canonical_archive_sha=absent
fi

kvaf_stage_dir=$(mktemp -d \
  "${kvaf_artifact_root}/.package-stage.XXXXXX") ||
  die "could not create the source-owned package staging directory"
kvaf_install_candidate=
cleanup()
{
  local status=$?
  trap - EXIT
  if [[ -n "${kvaf_install_candidate}" ]]; then
    rm -f -- "${kvaf_install_candidate}" || true
  fi
  rm -rf -- "${kvaf_stage_dir}" || true
  exit "${status}"
}
trap cleanup EXIT

kvaf_archive_provider_profile_name="kvaf-du-provider-${kvaf_artifact_token}.profile"
kvaf_archive_source_manifest_name="kvaf-du-source-${kvaf_artifact_token}.sha256"
kvaf_archive_build_manifest_name="kvaf-du-build-${kvaf_artifact_token}.manifest"
kvaf_archive_success_stamp_name="kvaf-du-success-${kvaf_artifact_token}.stamp"
kvaf_stage_package_dir="${kvaf_stage_dir}/aicpu_kernels_device"
mkdir "${kvaf_stage_package_dir}"
chmod 0755 "${kvaf_stage_package_dir}"
cp -- "${kvaf_provider_dir}/${kvaf_device_provider_filename}" \
  "${kvaf_stage_package_dir}/${kvaf_device_provider_filename}"
cp -- "${kvaf_provider_profile}" \
  "${kvaf_stage_package_dir}/${kvaf_archive_provider_profile_name}"
cp -- "${kvaf_provider_dir}/${kvaf_source_manifest_filename}" \
  "${kvaf_stage_package_dir}/${kvaf_archive_source_manifest_name}"
cp -- "${kvaf_provider_dir}/${kvaf_build_manifest_filename}" \
  "${kvaf_stage_package_dir}/${kvaf_archive_build_manifest_name}"
cp -- "${kvaf_provider_dir}/${kvaf_success_stamp_filename}" \
  "${kvaf_stage_package_dir}/${kvaf_archive_success_stamp_name}"
chmod 0444 "${kvaf_stage_package_dir}/"*

kvaf_candidate_archive="${kvaf_stage_dir}/${kvaf_package_name}"
# A deterministic archive makes a token-owned package genuinely idempotent:
# retrying after a partial registry transaction must reproduce the same bytes,
# rather than fail because tar metadata or a gzip timestamp changed.
tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner \
  -C "${kvaf_stage_dir}" -czf "${kvaf_candidate_archive}" \
  aicpu_kernels_device
kvaf_candidate_archive_sha=$(verify_archive \
  "${kvaf_candidate_archive}" candidate)

if [[ -e "${kvaf_archive_path}" || -L "${kvaf_archive_path}" ]]; then
  require_regular_file "${kvaf_archive_path}" \
    "existing versioned AICPU archive"
  kvaf_archive_sha=$(verify_archive "${kvaf_archive_path}" installed)
  [[ "${kvaf_archive_sha}" == "${kvaf_candidate_archive_sha}" ]] ||
    die "the immutable versioned AICPU archive already exists with different bytes"
else
  kvaf_install_candidate=$(mktemp \
    "${kvaf_package_dir}/.${kvaf_package_name}.XXXXXX") ||
    die "could not stage the versioned AICPU archive for installation"
  cp -- "${kvaf_candidate_archive}" "${kvaf_install_candidate}"
  chmod 0444 "${kvaf_install_candidate}"
  verify_archive "${kvaf_install_candidate}" install-candidate >/dev/null
  ln -- "${kvaf_install_candidate}" "${kvaf_archive_path}" ||
    die "could not atomically publish the immutable versioned AICPU archive"
  rm -f -- "${kvaf_install_candidate}"
  kvaf_install_candidate=
  kvaf_archive_sha=$(verify_archive "${kvaf_archive_path}" installed)
fi

if [[ -e "${kvaf_package_profile}" || -L "${kvaf_package_profile}" ]]; then
  verify_package_profile "${kvaf_package_profile}" "${kvaf_archive_sha}"
else
  kvaf_candidate_profile="${kvaf_stage_dir}/experiment-package.profile"
  printf '%s\n' \
    "schema=${kvaf_package_schema}" \
    "mode=${kvaf_mode}" \
    "risk_contract=${kvaf_risk_contract}" \
    "artifact_token=${kvaf_artifact_token}" \
    "source_commit=${kvaf_source_commit}" \
    "provider_dir=${kvaf_provider_dir}" \
    "provider_profile_path=${kvaf_provider_profile}" \
    "provider_profile_sha256=${kvaf_provider_profile_sha}" \
    "package_name=${kvaf_package_name}" \
    "package_relative_dir=${kvaf_package_relative_dir}" \
    "archive_path=${kvaf_archive_path}" \
    "archive_sha256=${kvaf_archive_sha}" \
    "registration_file=${kvaf_registration_file}" \
    "device_provider_filename=${kvaf_device_provider_filename}" \
    "device_provider_sha256=${kvaf_device_provider_sha}" \
    "archive_device_entry=aicpu_kernels_device/${kvaf_device_provider_filename}" \
    >"${kvaf_candidate_profile}"
  chmod 0400 "${kvaf_candidate_profile}"
  verify_package_profile "${kvaf_candidate_profile}" "${kvaf_archive_sha}"
  ln -- "${kvaf_candidate_profile}" "${kvaf_package_profile}" ||
    die "could not atomically publish the immutable package profile"
  verify_package_profile "${kvaf_package_profile}" "${kvaf_archive_sha}"
fi

kvaf_registry_before="${kvaf_stage_dir}/registry.before"
kvaf_registry_expected="${kvaf_stage_dir}/registry.expected"
cp -p -- "${kvaf_registration_file}" "${kvaf_registry_before}"
kvaf_registry_before_sha=$(sha256_file "${kvaf_registry_before}") ||
  die "could not hash the versioned registry snapshot"
if kvaf_aicpu_registration_is_exact \
    "${kvaf_registration_file}" "${kvaf_package_name}" \
    "${kvaf_package_relative_dir}"; then
  kvaf_registration_previous_state=exact
  cp -p -- "${kvaf_registry_before}" "${kvaf_registry_expected}"
elif kvaf_aicpu_registration_is_absent \
    "${kvaf_registration_file}" "${kvaf_package_name}" \
    "${kvaf_package_relative_dir}"; then
  kvaf_registration_previous_state=absent
  cp -p -- "${kvaf_registry_before}" "${kvaf_registry_expected}"
  if [[ -s "${kvaf_registry_expected}" ]] &&
     [[ $(tail -c 1 -- "${kvaf_registry_expected}" | wc -l) -eq 0 ]]; then
    printf '\n' >>"${kvaf_registry_expected}"
  fi
  kvaf_aicpu_registration_block "${kvaf_package_name}" \
    "${kvaf_package_relative_dir}" >>"${kvaf_registry_expected}"
else
  die "the versioned AICPU registration is duplicated or malformed"
fi

kvaf_aicpu_install_registration "${kvaf_registration_file}" \
  "${kvaf_package_name}" "${kvaf_package_relative_dir}" \
  "${kvaf_registry_before_sha}" ||
  die "could not atomically append the versioned AICPU registration"
cmp -s -- "${kvaf_registry_expected}" "${kvaf_registration_file}" ||
  die "the AICPU registry changed by more than the one versioned stanza"
kvaf_aicpu_registration_is_exact "${kvaf_registration_file}" \
  "${kvaf_package_name}" "${kvaf_package_relative_dir}" ||
  die "the installed versioned AICPU registration is not exact"
verify_canonical_archive_unchanged
verify_archive "${kvaf_archive_path}" installed >/dev/null
verify_package_profile "${kvaf_package_profile}" "${kvaf_archive_sha}"

echo "KVAF_A5_CROSS_NPU2_VERSIONED_AICPU_PACKAGE_OK"
echo "ARTIFACT_TOKEN=${kvaf_artifact_token}"
echo "SOURCE_COMMIT=${kvaf_source_commit}"
echo "PROVIDER_PROFILE=${kvaf_provider_profile}"
echo "PACKAGE_PROFILE=${kvaf_package_profile}"
echo "AICPU_ARCHIVE=${kvaf_archive_path}"
echo "AICPU_PACKAGE_NAME=${kvaf_package_name}"
echo "AICPU_REGISTRATION_FILE=${kvaf_registration_file}"
echo "AICPU_REGISTRATION_PREVIOUS_STATE=${kvaf_registration_previous_state}"
echo "DEVICE_PROVIDER_FILENAME=${kvaf_device_provider_filename}"
echo "DEVICE_PROVIDER_SHA256=${kvaf_device_provider_sha}"
fi
/bin/sh -c 'exit 2'
