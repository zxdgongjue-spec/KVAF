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
  local artifact source_filename umdk_filename libummu_filename
  local umdk_manifest libummu_manifest
  local host_sha device_sha source_sha umdk_sha libummu_sha build_sha
  local header_path header_sha expected_header_sha
  local runtime_header_path runtime_header_sha expected_runtime_header_sha
  local build_kernel_release
  local umdk_commit libummu_commit expected_libummu_commit
  local umdk_root management_source management_path
  local binding command key prefix

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
      libkvaf_urma_provider_device.so ]] || {
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

  for binding in \
    'host_provider_filename:libkvaf_urma_provider_host.so' \
    'host_provider_execution_domain:host-process' \
    'host_private_abi:1' \
    'host_private_abi_name:official-umdk-udma-v1-host' \
    'device_provider_filename:libkvaf_urma_provider_device.so' \
    'device_provider_execution_domain:aicpu' \
    'device_private_abi:2' \
    'device_private_abi_name:official-umdk-udma-v1-device' \
    'urma_core_mode:official-umdk-source-linked-no-dynamic-providers' \
    'urma_common_mode:official-umdk-source-built-internal' \
    'urma_common_dt_needed:absent' \
    'perftest_source:official-umdk' \
    'perftest_management:official-umdk-ub-default-tcp-fallback-v1' \
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

  KVAF_VERIFIED_HOST_PROVIDER_SHA=${host_sha}
  KVAF_VERIFIED_DEVICE_PROVIDER_SHA=${device_sha}
  KVAF_VERIFIED_SOURCE_MANIFEST_SHA=${source_sha}
  KVAF_VERIFIED_UMDK_SOURCE_MANIFEST_SHA=${umdk_sha}
  KVAF_VERIFIED_LIBUMMU_SOURCE_MANIFEST_SHA=${libummu_sha}
  KVAF_VERIFIED_BUILD_MANIFEST_SHA=${build_sha}
  KVAF_VERIFIED_UMDK_SOURCE_MANIFEST=${umdk_manifest}
  KVAF_VERIFIED_LIBUMMU_SOURCE_MANIFEST=${libummu_manifest}
}
