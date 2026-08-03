#!/usr/bin/env bash

# Read-only verifier for the source/token-owned provider, direct binaries and
# additive AICPU package used by the second-order NPU2 experiment.  Production
# keeps using a5_direct_urma_runtime_preflight.sh and its canonical filenames.

kvaf_cross_npu2_profile_error()
{
  echo "ERROR: NPU2 versioned runtime preflight: $*" >&2
  return 1
}

kvaf_cross_npu2_profile_value()
{
  local key=$1
  local file=$2
  awk -F= -v key="${key}" '
    $1 == key {
      value = substr($0, length(key) + 2)
      count++
    }
    END {
      if (count != 1 || value == "") exit 1
      print value
    }
  ' "${file}"
}

kvaf_cross_npu2_profile_sha256()
{
  sha256sum -- "$1" | awk '{print $1}'
}

kvaf_cross_npu2_require_regular()
{
  local path=$1
  local label=$2
  [[ ! -L "${path}" && -f "${path}" && -r "${path}" ]] || {
    kvaf_cross_npu2_profile_error \
      "${label} is missing, unreadable, or a symlink: ${path}"
    return 1
  }
}

kvaf_cross_npu2_require_profile_keys()
{
  local profile=$1
  shift
  awk -F= -v expected_keys="$*" '
    BEGIN {
      count = split(expected_keys, expected, " ")
      for (i = 1; i <= count; i++) allowed[expected[i]] = 1
    }
    {
      split_at = index($0, "=")
      if (split_at <= 1 || split_at == length($0)) {
        invalid = 1
        next
      }
      key = substr($0, 1, split_at - 1)
      if (!(key in allowed) || ++seen[key] != 1) invalid = 1
    }
    END {
      for (key in allowed) if (seen[key] != 1) invalid = 1
      exit invalid ? 1 : 0
    }
  ' "${profile}" || {
    kvaf_cross_npu2_profile_error \
      "profile has missing, duplicate, or foreign keys: ${profile}"
    return 1
  }
}

kvaf_cross_npu2_archive_entry_sha()
{
  local archive=$1
  local entry=$2
  tar -xOzf "${archive}" "${entry}" | sha256sum | awk '{print $1}'
}

kvaf_verify_cross_npu2_versioned_runtime_artifacts()
{
  local command source_commit artifact_root provider_dir direct_dir ascend_home
  local provider_profile package_profile direct_profile device_filename
  local package_name package_relative_dir archive registration_file
  local expected_archive expected_registration profile_mode
  local provider_profile_sha package_profile_sha archive_sha
  local host_provider_sha device_provider_sha source_manifest_sha
  local build_manifest_sha success_stamp_sha direct_binary_sha
  local owner_probe_sha direct_manifest_sha direct_source_sha
  local reuse_evidence_sha
  local direct_ldd owner_ldd direct_ascendcl owner_ascendcl manifest_ascendcl
  local listing expected_listing entry expected_sha actual_sha dynamic soname
  local -a package_entries

  for command in awk basename grep ldd nm readelf realpath sha256sum \
    sort stat tar; do
    command -v "${command}" >/dev/null 2>&1 || {
      kvaf_cross_npu2_profile_error "${command} is required"
      return 1
    }
  done
  [[ "${KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE:-}" == \
      I_ACCEPT_UNKNOWN_NPU2_REUSE_AND_SHARED_UMMU_RISK ]] || {
    kvaf_cross_npu2_profile_error "the exact second-order risk phrase is required"
    return 1
  }
  [[ "${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN:-}" =~ \
      ^[0-9a-f]{16}$ ]] || {
    kvaf_cross_npu2_profile_error "artifact token is not 16 lowercase hex digits"
    return 1
  }
  source_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD) || return 1
  [[ "${source_commit}" =~ ^[0-9a-f]{40}$ ]] || return 1
  artifact_root="/data/.kvaf-state/KVAF/experimental/reuse-artifacts/${source_commit}/${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}"
  provider_dir="${artifact_root}/provider"
  direct_dir="${artifact_root}/direct"
  provider_profile="${provider_dir}/experiment-provider.profile"
  package_profile="${artifact_root}/experiment-package.profile"
  direct_profile="${artifact_root}/experiment-direct.profile"

  for profile in "${provider_profile}" "${package_profile}" \
    "${direct_profile}"; do
    kvaf_cross_npu2_require_regular "${profile}" profile || return 1
    profile_mode=$(stat -c '%a' -- "${profile}") || return 1
    [[ "${profile_mode}" == 400 ]] || {
      kvaf_cross_npu2_profile_error "profile is not sealed mode 0400: ${profile}"
      return 1
    }
  done
  [[ ! -L "${artifact_root}" && -d "${artifact_root}" &&
     "$(realpath -e "${artifact_root}")" == "${artifact_root}" ]] || {
    kvaf_cross_npu2_profile_error "artifact root is not a canonical real directory"
    return 1
  }

  kvaf_cross_npu2_require_profile_keys "${provider_profile}" \
    schema experiment_profile mode artifact_token source_commit provider_dir \
    host_provider_filename host_provider_sha256 \
    device_provider_filename device_provider_sha256 \
    source_manifest_filename source_manifest_sha256 \
    build_manifest_filename build_manifest_sha256 \
    success_stamp_filename success_stamp_sha256 || return 1
  [[ "$(kvaf_cross_npu2_profile_value schema "${provider_profile}")" == \
        kvaf-a5-cross-npu2-versioned-provider-v1 &&
     "$(kvaf_cross_npu2_profile_value mode "${provider_profile}")" == \
        cross-npu2-reuse-versioned-offline-v1 &&
     "$(kvaf_cross_npu2_profile_value experiment_profile \
        "${provider_profile}")" == \
        cross-npu2-reuse-versioned-offline-v1 &&
     "$(kvaf_cross_npu2_profile_value artifact_token \
        "${provider_profile}")" == \
        "${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}" &&
     "$(kvaf_cross_npu2_profile_value source_commit \
        "${provider_profile}")" == "${source_commit}" &&
     "$(kvaf_cross_npu2_profile_value provider_dir \
        "${provider_profile}")" == "${provider_dir}" ]] || {
    kvaf_cross_npu2_profile_error "provider profile identity is inconsistent"
    return 1
  }
  device_filename=$(kvaf_cross_npu2_profile_value \
    device_provider_filename "${provider_profile}") || return 1
  [[ "${device_filename}" == \
      "libkvaf_ud_${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}.so" &&
     ${#device_filename} -lt 32 ]] || {
    kvaf_cross_npu2_profile_error "versioned Device provider name is invalid"
    return 1
  }

  KVAF_DIRECT_PROVIDER_BUILD_DIR=${provider_dir}
  KVAF_DIRECT_HOST_PROVIDER_LIBRARY="${provider_dir}/libkvaf_urma_provider_host.so"
  KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY="${provider_dir}/${device_filename}"
  KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST="${provider_dir}/source.sha256"
  KVAF_DIRECT_PROVIDER_BUILD_MANIFEST="${provider_dir}/build.manifest"
  KVAF_DIRECT_PROVIDER_SUCCESS_STAMP="${provider_dir}/build.success"
  kvaf_verify_direct_provider_build_v3 "${device_filename}" || return 1
  host_provider_sha=${KVAF_VERIFIED_HOST_PROVIDER_SHA}
  device_provider_sha=${KVAF_VERIFIED_DEVICE_PROVIDER_SHA}
  source_manifest_sha=${KVAF_VERIFIED_SOURCE_MANIFEST_SHA}
  build_manifest_sha=${KVAF_VERIFIED_BUILD_MANIFEST_SHA}
  success_stamp_sha=$(kvaf_cross_npu2_profile_sha256 \
    "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}") || return 1
  for binding in \
    "host_provider_sha256:${host_provider_sha}" \
    "device_provider_sha256:${device_provider_sha}" \
    "source_manifest_sha256:${source_manifest_sha}" \
    "build_manifest_sha256:${build_manifest_sha}" \
    "success_stamp_sha256:${success_stamp_sha}"; do
    [[ "$(kvaf_cross_npu2_profile_value "${binding%%:*}" \
        "${provider_profile}")" == "${binding#*:}" ]] || {
      kvaf_cross_npu2_profile_error \
        "provider profile does not bind ${binding%%:*}"
      return 1
    }
  done
  dynamic=$(readelf -d "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}") || return 1
  soname=$(awk '/\(SONAME\)/ { value=$0; sub(/^.*\[/,"",value); sub(/\].*$/,"",value); count++ } END { if (count != 1) exit 1; print value }' \
    <<<"${dynamic}") || return 1
  [[ "${soname}" == "${device_filename}" ]] || {
    kvaf_cross_npu2_profile_error "Device provider SONAME is not unique"
    return 1
  }

  kvaf_cross_npu2_require_profile_keys "${direct_profile}" \
    schema artifact_token source_commit direct_build_dir direct_binary \
    direct_binary_sha256 owner_probe owner_probe_sha256 \
    direct_build_manifest direct_build_manifest_sha256 \
    direct_source_manifest direct_source_manifest_sha256 \
    reuse_evidence_sha256 artifact_mutation_policy || return 1
  KVAF_A5_DIRECT_BUILD_DIR=${direct_dir}
  KVAF_A5_DIRECT_BINARY="${direct_dir}/access_operator/a5/kvaf_a5_aiv_direct_urma_h2d_smoke"
  KVAF_A5_OWNER_PROBE_BINARY="${direct_dir}/access_operator/a5/kvaf_a5_aicpu_owner_probe"
  KVAF_A5_DIRECT_BUILD_MANIFEST="${direct_dir}/direct-smoke.manifest"
  kvaf_direct_source_manifest="${direct_dir}/direct-smoke-source.sha256"
  for binding in \
    'schema:kvaf-a5-cross-npu2-versioned-direct-v1' \
    "artifact_token:${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}" \
    "source_commit:${source_commit}" \
    "direct_build_dir:${direct_dir}" \
    "direct_binary:${KVAF_A5_DIRECT_BINARY}" \
    "owner_probe:${KVAF_A5_OWNER_PROBE_BINARY}" \
    "direct_build_manifest:${KVAF_A5_DIRECT_BUILD_MANIFEST}" \
    "direct_source_manifest:${kvaf_direct_source_manifest}" \
    'artifact_mutation_policy:forbidden-during-hardware-run'; do
    [[ "$(kvaf_cross_npu2_profile_value "${binding%%:*}" \
        "${direct_profile}")" == "${binding#*:}" ]] || {
      kvaf_cross_npu2_profile_error \
        "direct profile does not bind ${binding%%:*}"
      return 1
    }
  done
  reuse_evidence_sha=$(kvaf_cross_npu2_profile_value \
    reuse_evidence_sha256 "${direct_profile}") || return 1
  [[ "${reuse_evidence_sha}" =~ ^[0-9a-f]{64}$ ]] || {
    kvaf_cross_npu2_profile_error \
      "direct profile has a malformed reuse evidence digest"
    return 1
  }
  if [[ -n "${KVAF_CROSS_NPU2_REUSE_VALIDATION_SHA256:-}" &&
        "${reuse_evidence_sha}" != \
          "${KVAF_CROSS_NPU2_REUSE_VALIDATION_SHA256}" ]]; then
    kvaf_cross_npu2_profile_error \
      "direct build and runtime evidence generations differ"
    return 1
  fi
  for artifact in "${KVAF_A5_DIRECT_BINARY}" \
    "${KVAF_A5_OWNER_PROBE_BINARY}" "${KVAF_A5_DIRECT_BUILD_MANIFEST}" \
    "${kvaf_direct_source_manifest}"; do
    kvaf_cross_npu2_require_regular "${artifact}" "direct artifact" || return 1
  done
  [[ -x "${KVAF_A5_DIRECT_BINARY}" && -x "${KVAF_A5_OWNER_PROBE_BINARY}" ]] || {
    kvaf_cross_npu2_profile_error "direct binaries are not executable"
    return 1
  }
  direct_binary_sha=$(kvaf_cross_npu2_profile_sha256 \
    "${KVAF_A5_DIRECT_BINARY}") || return 1
  owner_probe_sha=$(kvaf_cross_npu2_profile_sha256 \
    "${KVAF_A5_OWNER_PROBE_BINARY}") || return 1
  direct_manifest_sha=$(kvaf_cross_npu2_profile_sha256 \
    "${KVAF_A5_DIRECT_BUILD_MANIFEST}") || return 1
  direct_source_sha=$(kvaf_cross_npu2_profile_sha256 \
    "${kvaf_direct_source_manifest}") || return 1
  for binding in \
    "direct_binary_sha256:${direct_binary_sha}" \
    "owner_probe_sha256:${owner_probe_sha}" \
    "direct_build_manifest_sha256:${direct_manifest_sha}" \
    "direct_source_manifest_sha256:${direct_source_sha}"; do
    [[ "$(kvaf_cross_npu2_profile_value "${binding%%:*}" \
        "${direct_profile}")" == "${binding#*:}" ]] || {
      kvaf_cross_npu2_profile_error \
        "direct profile does not bind ${binding%%:*}"
      return 1
    }
  done
  [[ "$(kvaf_cross_npu2_profile_value schema \
      "${KVAF_A5_DIRECT_BUILD_MANIFEST}")" == \
      kvaf-a5-direct-smoke-build-v2 &&
     "$(kvaf_cross_npu2_profile_value binary_sha256 \
      "${KVAF_A5_DIRECT_BUILD_MANIFEST}")" == "${direct_binary_sha}" &&
     "$(kvaf_cross_npu2_profile_value owner_probe_sha256 \
      "${KVAF_A5_DIRECT_BUILD_MANIFEST}")" == "${owner_probe_sha}" &&
     "$(kvaf_cross_npu2_profile_value source_manifest_path \
      "${KVAF_A5_DIRECT_BUILD_MANIFEST}")" == \
      "${kvaf_direct_source_manifest}" &&
     "$(kvaf_cross_npu2_profile_value source_manifest_sha256 \
      "${KVAF_A5_DIRECT_BUILD_MANIFEST}")" == "${direct_source_sha}" ]] || {
    kvaf_cross_npu2_profile_error "direct build manifest is inconsistent"
    return 1
  }
  sha256sum --check --strict -- "${kvaf_direct_source_manifest}" \
    >/dev/null || {
    kvaf_cross_npu2_profile_error "direct source changed after build"
    return 1
  }
  direct_ldd=$(ldd -r "${KVAF_A5_DIRECT_BINARY}" 2>&1) || return 1
  owner_ldd=$(ldd -r "${KVAF_A5_OWNER_PROBE_BINARY}" 2>&1) || return 1
  if grep -Eq 'not found|undefined symbol|libhcomm' \
      <<<"${direct_ldd}"$'\n'"${owner_ldd}"; then
    kvaf_cross_npu2_profile_error "direct binary linkage is unresolved"
    return 1
  fi
  direct_ascendcl=$(kvaf_runtime_preflight_ldd_path libascendcl.so \
    <<<"${direct_ldd}") || return 1
  owner_ascendcl=$(kvaf_runtime_preflight_ldd_path libascendcl.so \
    <<<"${owner_ldd}") || return 1
  manifest_ascendcl=$(kvaf_cross_npu2_profile_value ascendcl_library_path \
    "${KVAF_A5_DIRECT_BUILD_MANIFEST}") || return 1
  [[ "$(realpath -e "${direct_ascendcl}")" == \
       "$(realpath -e "${manifest_ascendcl}")" &&
     "$(realpath -e "${owner_ascendcl}")" == \
       "$(realpath -e "${manifest_ascendcl}")" ]] || {
    kvaf_cross_npu2_profile_error "direct binaries resolve another libascendcl"
    return 1
  }

  kvaf_cross_npu2_require_profile_keys "${package_profile}" \
    schema mode risk_contract artifact_token source_commit provider_dir \
    provider_profile_path provider_profile_sha256 package_name \
    package_relative_dir archive_path archive_sha256 registration_file \
    device_provider_filename device_provider_sha256 archive_device_entry \
    || return 1
  ascend_home=$(realpath -e "${ASCEND_HOME_PATH}") || {
    kvaf_cross_npu2_profile_error "ASCEND_HOME_PATH is not canonical/readable"
    return 1
  }
  [[ -d "${ascend_home}" ]] || return 1
  package_name="cann-kvaf-du-${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}.tar.gz"
  package_relative_dir=opp/vendors/cust/op_impl/aicpu/kernel
  archive="${ascend_home}/${package_relative_dir}/${package_name}"
  registration_file="${ascend_home}/conf/ascend_package_load.ini"
  provider_profile_sha=$(kvaf_cross_npu2_profile_sha256 \
    "${provider_profile}") || return 1
  for binding in \
    'schema:kvaf-a5-cross-npu2-versioned-package-v1' \
    'mode:cross-npu2-reuse-versioned-offline-v1' \
    'risk_contract:I_ACCEPT_UNKNOWN_NPU2_REUSE_AND_SHARED_UMMU_RISK' \
    "artifact_token:${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}" \
    "source_commit:${source_commit}" \
    "provider_dir:${provider_dir}" \
    "provider_profile_path:${provider_profile}" \
    "provider_profile_sha256:${provider_profile_sha}" \
    "package_name:${package_name}" \
    "package_relative_dir:${package_relative_dir}" \
    "archive_path:${archive}" \
    "registration_file:${registration_file}" \
    "device_provider_filename:${device_filename}" \
    "device_provider_sha256:${device_provider_sha}" \
    "archive_device_entry:aicpu_kernels_device/${device_filename}"; do
    [[ "$(kvaf_cross_npu2_profile_value "${binding%%:*}" \
        "${package_profile}")" == "${binding#*:}" ]] || {
      kvaf_cross_npu2_profile_error \
        "package profile does not bind ${binding%%:*}"
      return 1
    }
  done
  kvaf_cross_npu2_require_regular "${archive}" "versioned AICPU archive" || return 1
  kvaf_cross_npu2_require_regular "${registration_file}" "AICPU registry" || return 1
  archive_sha=$(kvaf_cross_npu2_profile_sha256 "${archive}") || return 1
  [[ "$(kvaf_cross_npu2_profile_value archive_sha256 \
      "${package_profile}")" == "${archive_sha}" ]] || {
    kvaf_cross_npu2_profile_error "package profile does not bind the archive"
    return 1
  }
  kvaf_aicpu_registration_is_exact "${registration_file}" \
    "${package_name}" "${package_relative_dir}" || {
    kvaf_cross_npu2_profile_error "versioned AICPU package registration is not exact"
    return 1
  }
  package_entries=(
    aicpu_kernels_device/
    "aicpu_kernels_device/${device_filename}"
    "aicpu_kernels_device/kvaf-du-provider-${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}.profile"
    "aicpu_kernels_device/kvaf-du-source-${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}.sha256"
    "aicpu_kernels_device/kvaf-du-build-${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}.manifest"
    "aicpu_kernels_device/kvaf-du-success-${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}.stamp"
  )
  listing=$(tar -tzf "${archive}") || return 1
  expected_listing=$(printf '%s\n' "${package_entries[@]}" | sort)
  [[ "$(sort <<<"${listing}")" == "${expected_listing}" ]] || {
    kvaf_cross_npu2_profile_error "versioned archive entries are inconsistent"
    return 1
  }
  for entry in \
    "${device_filename}:${device_provider_sha}" \
    "kvaf-du-provider-${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}.profile:${provider_profile_sha}" \
    "kvaf-du-source-${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}.sha256:${source_manifest_sha}" \
    "kvaf-du-build-${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}.manifest:${build_manifest_sha}" \
    "kvaf-du-success-${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}.stamp:${success_stamp_sha}"; do
    expected_sha=${entry#*:}
    entry=${entry%%:*}
    actual_sha=$(kvaf_cross_npu2_archive_entry_sha "${archive}" \
      "aicpu_kernels_device/${entry}") || return 1
    [[ "${actual_sha}" == "${expected_sha}" ]] || {
      kvaf_cross_npu2_profile_error "archive entry hash mismatch: ${entry}"
      return 1
    }
  done

  KVAF_DIRECT_PROVIDER_SONAME=${device_filename}
  KVAF_DIRECT_AICPU_ARCHIVE=${archive}
  KVAF_DIRECT_AICPU_REGISTRATION_FILE=${registration_file}
  KVAF_CROSS_NPU2_PROVIDER_PROFILE=${provider_profile}
  KVAF_CROSS_NPU2_PACKAGE_PROFILE=${package_profile}
  KVAF_CROSS_NPU2_DIRECT_PROFILE=${direct_profile}
  KVAF_VERIFIED_DIRECT_BINARY_SHA=${direct_binary_sha}
  KVAF_VERIFIED_DIRECT_BUILD_MANIFEST_SHA=${direct_manifest_sha}
  KVAF_VERIFIED_AICPU_ARCHIVE_SHA=${archive_sha}
  return 0
}
