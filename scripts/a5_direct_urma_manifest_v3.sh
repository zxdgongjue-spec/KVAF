#!/usr/bin/env bash

# Read-only verifier for an atomic direct-URMA provider build.  This file must
# never inspect NPU state or create runtime resources: callers use it as the
# first fail-closed gate before EID, ACL, DDR/HBM, URMA, SQ, or CQ access.

kvaf_v3_error()
{
  echo "ERROR: direct URMA provider v3 verification: $*" >&2
  return 1
}

kvaf_v3_manifest_value()
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

kvaf_v3_require_value()
{
  local key=$1
  local expected=$2
  local file=$3
  local actual

  actual=$(kvaf_v3_manifest_value "${key}" "${file}") || {
    kvaf_v3_error "manifest lacks exactly one non-empty ${key}"
    return 1
  }
  [[ "${actual}" == "${expected}" ]] || {
    kvaf_v3_error "unexpected ${key}: ${actual}"
    return 1
  }
}

kvaf_v3_sha256()
{
  sha256sum -- "$1" | awk '{print $1}'
}

kvaf_v3_verify_hash_binding()
{
  local key=$1
  local expected=$2
  local file=$3
  local actual

  actual=$(kvaf_v3_manifest_value "${key}" "${file}") || {
    kvaf_v3_error "manifest lacks exactly one non-empty ${key}"
    return 1
  }
  [[ "${actual}" =~ ^[0-9a-f]{64}$ && "${actual}" == "${expected}" ]] || {
    kvaf_v3_error "${key} does not bind the verified artifact"
    return 1
  }
}

kvaf_v3_verify_probe()
{
  local prefix=$1
  local required_filename=$2
  local output_prefix=$3
  local abi_key=$4
  local filename path expected_file_sha actual_file_sha
  local expected_abi expected_abi_sha output actual_output_sha

  filename=$(kvaf_v3_manifest_value "${prefix}_probe_filename" \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || {
    kvaf_v3_error "manifest lacks ${prefix}_probe_filename"
    return 1
  }
  [[ "${filename}" == "${required_filename}" ]] || {
    kvaf_v3_error "unexpected ${prefix} probe filename: ${filename}"
    return 1
  }
  path="${KVAF_DIRECT_PROVIDER_BUILD_DIR}/${filename}"
  [[ ! -L "${path}" && -f "${path}" && -x "${path}" ]] || {
    kvaf_v3_error "probe is not a real executable: ${path}"
    return 1
  }
  actual_file_sha=$(kvaf_v3_sha256 "${path}") || return 1
  expected_file_sha=$(kvaf_v3_manifest_value "${prefix}_probe_sha256" \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  [[ "${actual_file_sha}" == "${expected_file_sha}" ]] || {
    kvaf_v3_error "${prefix} probe binary changed after build"
    return 1
  }
  kvaf_v3_verify_hash_binding "${prefix}_probe_sha256" \
    "${actual_file_sha}" "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}" || return 1

  expected_abi=$(kvaf_v3_manifest_value "${abi_key}" \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || {
    kvaf_v3_error "manifest lacks ${abi_key}"
    return 1
  }
  expected_abi_sha=$(kvaf_v3_manifest_value "${abi_key}_sha256" \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  output=$("${path}") || {
    kvaf_v3_error "${prefix} probe execution failed"
    return 1
  }
  [[ "${output}" == "${output_prefix}${expected_abi}" ]] || {
    kvaf_v3_error "${prefix} probe output does not match its manifest ABI"
    return 1
  }
  actual_output_sha=$(printf '%s\n' "${output}" | sha256sum | awk '{print $1}')
  [[ "${actual_output_sha}" == "${expected_abi_sha}" ]] || {
    kvaf_v3_error "${prefix} probe output hash changed"
    return 1
  }
}

kvaf_v3_verify_provider_dynamic()
{
  local label=$1
  local provider=$2
  local dynamic

  dynamic=$(readelf -d "${provider}") || {
    kvaf_v3_error "cannot inspect ${label} provider DT_NEEDED"
    return 1
  }
  if grep -Eq \
      '\(NEEDED\).*\[lib(urma_common|urma|ummu|udma|urma-udma)\.so|\(NEEDED\).*\[lib(hcomm|urma_perftest)\.so' \
      <<<"${dynamic}"; then
    kvaf_v3_error "${label} provider has a forbidden external URMA/UMMU/UDMA/HCOMM DT_NEEDED"
    return 1
  fi
  if grep -Eq '\((RPATH|RUNPATH)\)' <<<"${dynamic}"; then
    kvaf_v3_error "${label} provider contains RPATH/RUNPATH"
    return 1
  fi
}

kvaf_verify_direct_provider_build_v3()
{
  local expected_device_basename=${1:-libkvaf_urma_provider_device.so}
  local artifact source_filename umdk_filename libummu_filename
  local umdk_manifest libummu_manifest
  local host_sha device_sha source_sha umdk_sha libummu_sha build_sha
  local header_path header_sha expected_header_sha
  local runtime_header_path runtime_header_sha expected_runtime_header_sha
  local build_kernel_release
  local umdk_commit libummu_commit expected_libummu_commit
  local umdk_root management_source management_path
  local runtime_probe_provider_filename runtime_probe_provider_path
  local runtime_probe_provider_build_id runtime_probe_provider_sha
  local runtime_probe_provider_source_config_sha
  local runtime_probe_provider_builder_sha current_provider_builder_sha
  local registered_package_audit_sha current_registered_package_audit_sha
  local resource_gate_probe_filename resource_gate_probe_path
  local resource_gate_probe_sha resource_gate_probe_actual_sha
  local resource_gate_output_sha resource_gate_output_actual_sha
  local resource_gate_output
  local binding command key prefix

  if (( $# > 1 )) ||
     [[ -z "${expected_device_basename}" ||
        "${expected_device_basename}" == */* ||
        "${expected_device_basename}" == *\\* ||
        ${#expected_device_basename} -ge 32 ]]; then
    kvaf_v3_error "expected Device provider basename is invalid"
    return 1
  fi

  # Parse the schema before requiring platform-specific ELF tools so legacy
  # v2 artifacts are rejected for the right reason on every control host.
  for command in awk basename; do
    command -v "${command}" >/dev/null 2>&1 || {
      kvaf_v3_error "${command} is required"
      return 1
    }
  done
  for artifact in \
    "${KVAF_DIRECT_HOST_PROVIDER_LIBRARY}" \
    "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}" \
    "${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST}" \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" \
    "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}"
  do
    [[ ! -L "${artifact}" && -f "${artifact}" && -r "${artifact}" ]] || {
      kvaf_v3_error "required atomic-build artifact is missing or a symlink: ${artifact}"
      return 1
    }
  done
  [[ "$(basename "${KVAF_DIRECT_HOST_PROVIDER_LIBRARY}")" == \
      libkvaf_urma_provider_host.so &&
     "$(basename "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}")" == \
      "${expected_device_basename}" ]] || {
    kvaf_v3_error "unexpected Host/Device provider basename"
    return 1
  }

  kvaf_v3_require_value schema kvaf-direct-urma-provider-v3 \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" || return 1
  kvaf_v3_require_value schema kvaf-direct-urma-provider-success-v3 \
    "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}" || return 1
  kvaf_v3_require_value source_layout official-umdk-libummu-paired-v1 \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" || return 1
  for command in grep readelf sha256sum uname; do
    command -v "${command}" >/dev/null 2>&1 || {
      kvaf_v3_error "${command} is required"
      return 1
    }
  done

  build_kernel_release=$(kvaf_v3_manifest_value build_kernel_release \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  [[ "${build_kernel_release}" == "$(uname -r)" ]] || {
    kvaf_v3_error "provider was built for kernel ${build_kernel_release}, current kernel is $(uname -r)"
    return 1
  }

  source_filename=$(kvaf_v3_manifest_value source_manifest_filename \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  umdk_filename=$(kvaf_v3_manifest_value umdk_source_manifest_filename \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  libummu_filename=$(kvaf_v3_manifest_value libummu_source_manifest_filename \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  [[ "${source_filename}" == source.sha256 &&
     "${umdk_filename}" == umdk-source.sha256 &&
     "${libummu_filename}" == libummu-source.sha256 &&
     "$(basename "${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST}")" == \
       "${source_filename}" ]] || {
    kvaf_v3_error "source-manifest filenames are not the v3 contract"
    return 1
  }
  umdk_manifest="${KVAF_DIRECT_PROVIDER_BUILD_DIR}/${umdk_filename}"
  libummu_manifest="${KVAF_DIRECT_PROVIDER_BUILD_DIR}/${libummu_filename}"
  for artifact in "${umdk_manifest}" "${libummu_manifest}"; do
    [[ ! -L "${artifact}" && -f "${artifact}" && -r "${artifact}" ]] || {
      kvaf_v3_error "source manifest is missing or a symlink: ${artifact}"
      return 1
    }
  done

  host_sha=$(kvaf_v3_sha256 "${KVAF_DIRECT_HOST_PROVIDER_LIBRARY}") || return 1
  device_sha=$(kvaf_v3_sha256 "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}") || return 1
  source_sha=$(kvaf_v3_sha256 "${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST}") || return 1
  umdk_sha=$(kvaf_v3_sha256 "${umdk_manifest}") || return 1
  libummu_sha=$(kvaf_v3_sha256 "${libummu_manifest}") || return 1
  build_sha=$(kvaf_v3_sha256 "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  [[ "${host_sha}" != "${device_sha}" ]] || {
    kvaf_v3_error "Host and Device providers are not distinct binaries"
    return 1
  }
  for binding in \
    "host_provider_sha256:${host_sha}" \
    "device_provider_sha256:${device_sha}" \
    "source_manifest_sha256:${source_sha}" \
    "umdk_source_manifest_sha256:${umdk_sha}" \
    "libummu_source_manifest_sha256:${libummu_sha}"
  do
    kvaf_v3_verify_hash_binding "${binding%%:*}" "${binding#*:}" \
      "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" || return 1
    kvaf_v3_verify_hash_binding "${binding%%:*}" "${binding#*:}" \
      "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}" || return 1
  done
  kvaf_v3_verify_hash_binding build_manifest_sha256 "${build_sha}" \
    "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}" || return 1

  runtime_probe_provider_filename=$(kvaf_v3_manifest_value \
    runtime_probe_provider_filename \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  runtime_probe_provider_build_id=$(kvaf_v3_manifest_value \
    runtime_probe_provider_build_id_sha256 \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  runtime_probe_provider_source_config_sha=$(kvaf_v3_manifest_value \
    runtime_probe_provider_source_config_sha256 \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  runtime_probe_provider_builder_sha=$(kvaf_v3_manifest_value \
    runtime_probe_provider_builder_script_sha256 \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  [[ -n ${ROOT_DIR:-} &&
     -f ${ROOT_DIR}/scripts/build_a5_direct_urma_provider.sh &&
     ! -L ${ROOT_DIR}/scripts/build_a5_direct_urma_provider.sh ]] || {
    kvaf_v3_error "current provider build script is missing or unsafe"
    return 1
  }
  current_provider_builder_sha=$(kvaf_v3_sha256 \
    "${ROOT_DIR}/scripts/build_a5_direct_urma_provider.sh") || return 1
  [[ ${runtime_probe_provider_builder_sha} =~ ^[0-9a-f]{64}$ &&
     ${runtime_probe_provider_builder_sha} == \
       "${current_provider_builder_sha}" ]] || {
    kvaf_v3_error \
      "provider build script changed after the verified provider build"
    return 1
  }
  registered_package_audit_sha=$(kvaf_v3_manifest_value \
    registered_package_audit_script_sha256 \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  [[ -f ${ROOT_DIR}/scripts/audit_a5_registered_aicpu_packages.sh &&
     ! -L ${ROOT_DIR}/scripts/audit_a5_registered_aicpu_packages.sh &&
     -x ${ROOT_DIR}/scripts/audit_a5_registered_aicpu_packages.sh ]] || {
    kvaf_v3_error "registered AICPU package audit is missing or unsafe"
    return 1
  }
  current_registered_package_audit_sha=$(kvaf_v3_sha256 \
    "${ROOT_DIR}/scripts/audit_a5_registered_aicpu_packages.sh") || return 1
  [[ ${registered_package_audit_sha} =~ ^[0-9a-f]{64}$ &&
     ${registered_package_audit_sha} == \
       "${current_registered_package_audit_sha}" ]] || {
    kvaf_v3_error \
      "registered AICPU package audit changed after the verified provider build"
    return 1
  }
  kvaf_v3_verify_hash_binding registered_package_audit_script_sha256 \
    "${registered_package_audit_sha}" \
    "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}" || return 1
  [[ ${runtime_probe_provider_build_id} =~ ^[0-9a-f]{64}$ &&
     ${runtime_probe_provider_filename} == \
       "libkvaf_rt_${runtime_probe_provider_build_id:0:16}.so" &&
     ${#runtime_probe_provider_filename} -lt 32 ]] || {
    kvaf_v3_error "runtime probe provider build-instance identity is malformed"
    return 1
  }
  runtime_probe_provider_path="${KVAF_DIRECT_PROVIDER_BUILD_DIR}/${runtime_probe_provider_filename}"
  [[ ! -L ${runtime_probe_provider_path} &&
     -f ${runtime_probe_provider_path} &&
     -r ${runtime_probe_provider_path} ]] || {
    kvaf_v3_error "runtime probe provider alias is missing or unsafe"
    return 1
  }
  runtime_probe_provider_sha=$(kvaf_v3_sha256 \
    "${runtime_probe_provider_path}") || return 1
  kvaf_v3_require_value runtime_probe_provider_build_id_schema \
    random-build-instance-sha256-v1 \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" || return 1
  kvaf_v3_require_value runtime_probe_provider_source_config_schema \
    audited-inputs-sha256-v1 \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" || return 1
  [[ ${runtime_probe_provider_source_config_sha} =~ ^[0-9a-f]{64}$ ]] || {
    kvaf_v3_error "runtime probe provider source/config identity is malformed"
    return 1
  }
  for binding in \
    "runtime_probe_provider_build_id_sha256:${runtime_probe_provider_build_id}" \
    "runtime_probe_provider_source_config_sha256:${runtime_probe_provider_source_config_sha}" \
    "runtime_probe_provider_artifact_sha256:${runtime_probe_provider_sha}"
  do
    kvaf_v3_verify_hash_binding "${binding%%:*}" "${binding#*:}" \
      "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" || return 1
    kvaf_v3_verify_hash_binding "${binding%%:*}" "${binding#*:}" \
      "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}" || return 1
  done

  for binding in \
    'resource_entry_gate_schema:capability-ledger-fail-closed-v1' \
    'capability_ledger_state:not-implemented' \
    'provider_open_gate:disabled-before-config-validation-v1' \
    'owner_probe_gate:disabled-before-command-clock-sleep-v1' \
    'epoch_owner_gate:disabled-before-claim-v1' \
    'runtime_probe_gate:disabled-before-dlopen-v1' \
    'resource_gate_contract:verified-valid-request-v1'
  do
    kvaf_v3_require_value "${binding%%:*}" "${binding#*:}" \
      "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" || return 1
    kvaf_v3_require_value "${binding%%:*}" "${binding#*:}" \
      "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}" || return 1
  done
  resource_gate_probe_filename=$(kvaf_v3_manifest_value \
    resource_gate_contract_probe_filename \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  [[ ${resource_gate_probe_filename} == \
       kvaf_urma_provider_aicpu_contract_probe ]] || {
    kvaf_v3_error "unexpected resource-gate contract probe filename"
    return 1
  }
  resource_gate_probe_path="${KVAF_DIRECT_PROVIDER_BUILD_DIR}/${resource_gate_probe_filename}"
  [[ ! -L ${resource_gate_probe_path} &&
     -f ${resource_gate_probe_path} &&
     -x ${resource_gate_probe_path} ]] || {
    kvaf_v3_error "resource-gate contract probe is missing or unsafe"
    return 1
  }
  resource_gate_probe_actual_sha=$(kvaf_v3_sha256 \
    "${resource_gate_probe_path}") || return 1
  resource_gate_probe_sha=$(kvaf_v3_manifest_value \
    resource_gate_contract_probe_sha256 \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  resource_gate_output_sha=$(kvaf_v3_manifest_value \
    resource_gate_contract_output_sha256 \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  [[ ${resource_gate_probe_sha} =~ ^[0-9a-f]{64}$ &&
     ${resource_gate_output_sha} =~ ^[0-9a-f]{64}$ &&
     ${resource_gate_probe_actual_sha} == "${resource_gate_probe_sha}" ]] || {
    kvaf_v3_error "resource-gate contract proof changed after build"
    return 1
  }
  kvaf_v3_verify_hash_binding resource_gate_contract_probe_sha256 \
    "${resource_gate_probe_sha}" \
    "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}" || return 1
  kvaf_v3_verify_hash_binding resource_gate_contract_output_sha256 \
    "${resource_gate_output_sha}" \
    "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}" || return 1

  for binding in \
    'host_provider_filename:libkvaf_urma_provider_host.so' \
    'host_provider_execution_domain:host-process' \
    'host_private_abi:1' \
    'host_private_abi_name:official-umdk-udma-v1-host' \
    "device_provider_filename:${expected_device_basename}" \
    'device_provider_execution_domain:aicpu' \
    'device_private_abi:2' \
    'device_private_abi_name:official-umdk-udma-v1-device' \
    'urma_core_mode:official-umdk-source-linked-no-dynamic-providers' \
    'urma_common_mode:official-umdk-source-built-internal' \
    'urma_common_dt_needed:absent' \
    'perftest_source:official-umdk' \
    'perftest_management:official-umdk-ub-default-tcp-fallback-v1' \
    'device_queue_memory_policy:sva-aware-anonymous-or-kernel-v1' \
    'udma_provider_mode:official-umdk-source-built-internal' \
    'udma_source_capability_magic:0x4b56414655444d41' \
    'udma_runtime_capability:0x0000000000020001' \
    'ummu_provider_mode:paired-libummu-source-built-internal' \
    'ummu_context_fd_contract:private-open-or-unique-process-dup-v1' \
    'ummu_source_capability_magic:0x4b564146554d4d55' \
    'ummu_runtime_capability:0x0000000000010001' \
    'umdk_git_tree_state:clean' \
    'libummu_git_tree_state:clean' \
    'umdk_libummu_pairing:verified' \
    'umdk_git_commit:12917421beb93c1351eb2f124bb2f34612d8bf2d' \
    'umdk_git_origin:https://gitcode.com/openeuler/umdk.git' \
    'umdk_release_version:26.06.0' \
    'libummu_git_origin:https://gitcode.com/openeuler/libummu.git' \
    'libummu_expected_remote:https://gitcode.com/openeuler/libummu.git' \
    'libummu_expected_commit:70de020f776d8def0d8a8941cc7cbc5928d66c10' \
    'libummu_git_commit:70de020f776d8def0d8a8941cc7cbc5928d66c10' \
    'libummu_version:1.0.5' \
    'libummu_abi_version:1' \
    'udma_teardown_contract:checked-retryable-v1'
  do
    kvaf_v3_require_value "${binding%%:*}" "${binding#*:}" \
      "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" || return 1
  done
  if grep -Eq '^(urma|urma_common|udma|ummu)_provider_library_' \
      "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}"; then
    kvaf_v3_error "v3 manifest contains a forbidden external-provider binding"
    return 1
  fi

  umdk_commit=$(kvaf_v3_manifest_value umdk_git_commit \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  libummu_commit=$(kvaf_v3_manifest_value libummu_git_commit \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  expected_libummu_commit=$(kvaf_v3_manifest_value libummu_expected_commit \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  [[ "${umdk_commit}" =~ ^[0-9a-f]{40,64}$ &&
     "${libummu_commit}" =~ ^[0-9a-f]{40,64}$ &&
     "${expected_libummu_commit}" == "${libummu_commit}" ]] || {
    kvaf_v3_error "UMDK/libummu commit pairing is malformed or inconsistent"
    return 1
  }
  for key in umdk_git_tree libummu_git_tree urma_public_abi_signature; do
    kvaf_v3_manifest_value "${key}" \
      "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" >/dev/null || {
      kvaf_v3_error "manifest lacks ${key}"
      return 1
    }
  done

  for prefix in udma_uapi ummu_uapi; do
    header_path=$(kvaf_v3_manifest_value "${prefix}_header_path" \
      "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
    expected_header_sha=$(kvaf_v3_manifest_value "${prefix}_header_sha256" \
      "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
    [[ ! -L "${header_path}" && -f "${header_path}" && -r "${header_path}" ]] || {
      kvaf_v3_error "pinned ${prefix} header is missing or a symlink"
      return 1
    }
    header_sha=$(kvaf_v3_sha256 "${header_path}") || return 1
    [[ "${header_sha}" == "${expected_header_sha}" ]] || {
      kvaf_v3_error "pinned ${prefix} header changed after build"
      return 1
    }

    runtime_header_path=$(kvaf_v3_manifest_value \
      "runtime_${prefix}_header_path" \
      "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
    expected_runtime_header_sha=$(kvaf_v3_manifest_value \
      "runtime_${prefix}_header_sha256" \
      "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
    [[ ! -L "${runtime_header_path}" && -f "${runtime_header_path}" &&
       -r "${runtime_header_path}" ]] || {
      kvaf_v3_error "running-kernel ${prefix} header is missing or a symlink"
      return 1
    }
    runtime_header_sha=$(kvaf_v3_sha256 "${runtime_header_path}") || return 1
    [[ "${runtime_header_sha}" == "${expected_runtime_header_sha}" &&
       "${runtime_header_sha}" == "${header_sha}" ]] || {
      kvaf_v3_error "running-kernel ${prefix} header no longer matches the pinned source"
      return 1
    }
    kvaf_v3_require_value "runtime_${prefix}_match" verified-exact \
      "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" || return 1
    kvaf_v3_verify_hash_binding "runtime_${prefix}_header_sha256" \
      "${runtime_header_sha}" "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}" ||
      return 1
  done

  kvaf_v3_verify_probe udma_uapi kvaf_udma_uapi_probe \
    KVAF_UDMA_UAPI_ABI= udma_uapi_abi || return 1
  kvaf_v3_verify_probe udma_private_abi kvaf_udma_private_abi_probe \
    KVAF_UDMA_PRIVATE_ABI= udma_private_abi || return 1
  kvaf_v3_verify_probe ummu_uapi kvaf_ummu_uapi_probe \
    KVAF_UMMU_UAPI_ABI= ummu_uapi_abi || return 1

  umdk_root=$(kvaf_v3_manifest_value umdk_root \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}") || return 1
  for management_source in \
    perftest_mgmt.c perftest_mgmt.h \
    perftest_mgmt_tcp.c perftest_mgmt_tcp.h \
    perftest_mgmt_ub.c perftest_mgmt_ub.h
  do
    management_path="${umdk_root}/src/urma/tools/urma_perftest/${management_source}"
    awk -v expected="${management_path}" \
        '$2 == expected { found = 1 } END { exit found ? 0 : 1 }' \
        "${umdk_manifest}" || {
      kvaf_v3_error "UMDK source manifest omits ${management_source}"
      return 1
    }
  done

  for artifact in \
    "${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST}" \
    "${umdk_manifest}" \
    "${libummu_manifest}"
  do
    sha256sum --check --strict -- "${artifact}" >/dev/null || {
      kvaf_v3_error "source verification failed: ${artifact}"
      return 1
    }
  done
  kvaf_v3_verify_provider_dynamic Host \
    "${KVAF_DIRECT_HOST_PROVIDER_LIBRARY}" || return 1
  kvaf_v3_verify_provider_dynamic Device \
    "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}" || return 1
  kvaf_v3_verify_provider_dynamic Runtime-probe \
    "${runtime_probe_provider_path}" || return 1

  # The probe sends valid, identity-bound requests through each exported
  # resource-bearing entry.  A PASS proves that the exact Device and runtime
  # probe DSOs in this build return the fixed ledger-unavailable result and
  # leave no provider session, before any URMA/UMMU resource or dlopen path.
  resource_gate_output=$(
    "${resource_gate_probe_path}" \
      "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}" \
      "${runtime_probe_provider_path}"
  ) || {
    kvaf_v3_error "resource-gate contract probe execution failed"
    return 1
  }
  [[ ${resource_gate_output} == \
       KVAF_URMA_PROVIDER_AICPU_CONTRACT=PASS ]] || {
    kvaf_v3_error "resource-gate contract probe returned an unexpected result"
    return 1
  }
  resource_gate_output_actual_sha=$(
    printf '%s\n' "${resource_gate_output}" | sha256sum | awk '{print $1}'
  ) || return 1
  [[ ${resource_gate_output_actual_sha} == "${resource_gate_output_sha}" ]] || {
    kvaf_v3_error "resource-gate contract result changed after build"
    return 1
  }

  KVAF_VERIFIED_HOST_PROVIDER_SHA=${host_sha}
  KVAF_VERIFIED_DEVICE_PROVIDER_SHA=${device_sha}
  KVAF_VERIFIED_SOURCE_MANIFEST_SHA=${source_sha}
  KVAF_VERIFIED_UMDK_SOURCE_MANIFEST_SHA=${umdk_sha}
  KVAF_VERIFIED_LIBUMMU_SOURCE_MANIFEST_SHA=${libummu_sha}
  KVAF_VERIFIED_BUILD_MANIFEST_SHA=${build_sha}
  KVAF_VERIFIED_RUNTIME_PROBE_PROVIDER_FILENAME=${runtime_probe_provider_filename}
  KVAF_VERIFIED_RUNTIME_PROBE_PROVIDER_BUILD_ID=${runtime_probe_provider_build_id}
  KVAF_VERIFIED_RUNTIME_PROBE_PROVIDER_SHA=${runtime_probe_provider_sha}
  KVAF_VERIFIED_REGISTERED_PACKAGE_AUDIT_SHA=${registered_package_audit_sha}
  KVAF_VERIFIED_UMDK_SOURCE_MANIFEST=${umdk_manifest}
  KVAF_VERIFIED_LIBUMMU_SOURCE_MANIFEST=${libummu_manifest}
}
