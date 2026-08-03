#!/usr/bin/env bash

# File-only validation for the direct smoke executable and the installed
# AICPU bundle.  Callers must already hold the deploy lock and must have
# successfully run kvaf_verify_direct_provider_build_v3.

KVAF_RUNTIME_PREFLIGHT_SCRIPT_DIR=$(
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P
)
source "${KVAF_RUNTIME_PREFLIGHT_SCRIPT_DIR}/a5_aicpu_package_registration.sh"

kvaf_runtime_preflight_error()
{
  echo "ERROR: direct URMA runtime-artifact preflight: $*" >&2
  return 1
}

kvaf_runtime_preflight_manifest_value()
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

kvaf_runtime_preflight_sha256()
{
  sha256sum -- "$1" | awk '{print $1}'
}

kvaf_runtime_preflight_ldd_path()
{
  local soname=$1
  awk -v soname="${soname}" '
    ($1 == soname || index($1, soname ".") == 1) && $2 == "=>" {
      path = $3
      count++
    }
    END {
      if (count != 1 || path == "") exit 1
      print path
    }
  '
}

kvaf_runtime_preflight_archive_sha()
{
  local archive=$1
  local entry=$2
  tar -xOzf "${archive}" "${entry}" | sha256sum | awk '{print $1}'
}

kvaf_verify_direct_runtime_artifacts_v3()
{
  local command artifact direct_binary_sha direct_runtime_header
  local direct_acl_header direct_ascendcl direct_source_manifest
  local direct_ldd resolved_direct_ascendcl direct_build_manifest_sha
  local direct_source_manifest_sha expected_archive archive_entries
  local expected_package_name expected_package_relative_dir
  local expected_registration_file runtime_registration_file
  local legacy_registration_count sibling_archive sibling_entries
  local entry entry_count unexpected_entries archive_provider_sha
  local runtime_probe_provider_filename runtime_probe_provider_sha
  local device_provider_soname registered_package_audit
  local registered_package_audit_b64 registered_package_audit_sha
  local registered_package_audit_rc
  local archive_runtime_probe_provider_sha
  local archive_build_manifest_sha archive_source_manifest_sha
  local archive_umdk_source_manifest_sha archive_libummu_source_manifest_sha
  local probe archive_probe_sha local_probe_sha binary_dynamic
  local provider provider_dynamic provider_symbols symbol
  local -a required_archive_entries=(
    aicpu_kernels_device/
    aicpu_kernels_device/libkvaf_urma_provider_device.so
    aicpu_kernels_device/kvaf-direct-urma-provider.manifest
    aicpu_kernels_device/kvaf-direct-urma-source.sha256
    aicpu_kernels_device/kvaf-direct-urma-umdk-source.sha256
    aicpu_kernels_device/kvaf-direct-urma-libummu-source.sha256
    aicpu_kernels_device/kvaf_udma_uapi_probe
    aicpu_kernels_device/kvaf_udma_private_abi_probe
    aicpu_kernels_device/kvaf_ummu_uapi_probe
  )
  runtime_probe_provider_filename=${KVAF_VERIFIED_RUNTIME_PROBE_PROVIDER_FILENAME:-}
  runtime_probe_provider_sha=${KVAF_VERIFIED_RUNTIME_PROBE_PROVIDER_SHA:-}
  [[ ${runtime_probe_provider_filename} =~ ^libkvaf_rt_[0-9a-f]{16}\.so$ &&
     ${runtime_probe_provider_sha} =~ ^[0-9a-f]{64}$ ]] || {
    kvaf_runtime_preflight_error \
      "verified runtime probe provider identity is unavailable"
    return 1
  }
  required_archive_entries+=(
    "aicpu_kernels_device/${runtime_probe_provider_filename}")

  for command in awk base64 bash cp grep ldd nm python3 readelf readlink realpath sha256sum tar timeout; do
    command -v "${command}" >/dev/null 2>&1 || {
      kvaf_runtime_preflight_error "${command} is required"
      return 1
    }
  done

  [[ ! -L "${KVAF_ASCEND_HAL_LIBRARY}" &&
     -f "${KVAF_ASCEND_HAL_LIBRARY}" &&
     -r "${KVAF_ASCEND_HAL_LIBRARY}" ]] || {
    kvaf_runtime_preflight_error \
      "pinned real Ascend HAL library is missing: ${KVAF_ASCEND_HAL_LIBRARY}"
    return 1
  }
  [[ ! -L "${KVAF_A5_DIRECT_BINARY}" &&
     -f "${KVAF_A5_DIRECT_BINARY}" &&
     -x "${KVAF_A5_DIRECT_BINARY}" ]] || {
    kvaf_runtime_preflight_error \
      "direct smoke binary is missing or unsafe: ${KVAF_A5_DIRECT_BINARY}"
    return 1
  }
  for artifact in \
    "${KVAF_A5_DIRECT_BUILD_MANIFEST}" \
    "${KVAF_DIRECT_HOST_PROVIDER_LIBRARY}" \
    "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}" \
    "${KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST}" \
    "${KVAF_DIRECT_PROVIDER_BUILD_MANIFEST}" \
    "${KVAF_DIRECT_PROVIDER_SUCCESS_STAMP}"
  do
    [[ ! -L "${artifact}" && -f "${artifact}" && -r "${artifact}" ]] || {
      kvaf_runtime_preflight_error \
        "required build artifact is missing or unsafe: ${artifact}"
      return 1
    }
  done

  direct_binary_sha=$(kvaf_runtime_preflight_sha256 \
    "${KVAF_A5_DIRECT_BINARY}") || return 1
  direct_runtime_header=$(kvaf_runtime_preflight_manifest_value \
    runtime_header_path "${KVAF_A5_DIRECT_BUILD_MANIFEST}") || return 1
  direct_acl_header=$(kvaf_runtime_preflight_manifest_value \
    acl_header_path "${KVAF_A5_DIRECT_BUILD_MANIFEST}") || return 1
  direct_ascendcl=$(kvaf_runtime_preflight_manifest_value \
    ascendcl_library_path "${KVAF_A5_DIRECT_BUILD_MANIFEST}") || return 1
  direct_source_manifest=$(kvaf_runtime_preflight_manifest_value \
    source_manifest_path "${KVAF_A5_DIRECT_BUILD_MANIFEST}") || return 1
  for artifact in "${direct_runtime_header}" "${direct_acl_header}" \
    "${direct_ascendcl}" "${direct_source_manifest}"
  do
    [[ ! -L "${artifact}" && -f "${artifact}" && -r "${artifact}" ]] || {
      kvaf_runtime_preflight_error \
        "direct-smoke dependency is missing or unsafe: ${artifact}"
      return 1
    }
  done
  [[ "$(kvaf_runtime_preflight_manifest_value schema \
        "${KVAF_A5_DIRECT_BUILD_MANIFEST}")" == \
        kvaf-a5-direct-smoke-build-v2 &&
     "$(kvaf_runtime_preflight_manifest_value binary_sha256 \
        "${KVAF_A5_DIRECT_BUILD_MANIFEST}")" == "${direct_binary_sha}" &&
     "$(kvaf_runtime_preflight_sha256 "${direct_runtime_header}")" == \
        "$(kvaf_runtime_preflight_manifest_value runtime_header_sha256 \
          "${KVAF_A5_DIRECT_BUILD_MANIFEST}")" &&
     "$(kvaf_runtime_preflight_sha256 "${direct_acl_header}")" == \
        "$(kvaf_runtime_preflight_manifest_value acl_header_sha256 \
          "${KVAF_A5_DIRECT_BUILD_MANIFEST}")" &&
     "$(kvaf_runtime_preflight_sha256 "${direct_ascendcl}")" == \
        "$(kvaf_runtime_preflight_manifest_value ascendcl_library_sha256 \
          "${KVAF_A5_DIRECT_BUILD_MANIFEST}")" &&
     "$(kvaf_runtime_preflight_sha256 "${direct_source_manifest}")" == \
        "$(kvaf_runtime_preflight_manifest_value source_manifest_sha256 \
          "${KVAF_A5_DIRECT_BUILD_MANIFEST}")" ]] || {
    kvaf_runtime_preflight_error \
      "direct smoke binary or official CANN runtime ABI changed after build"
    return 1
  }
  sha256sum --check --strict -- "${direct_source_manifest}" >/dev/null || {
    kvaf_runtime_preflight_error \
      "direct smoke source changed after the successful build"
    return 1
  }
  direct_ldd=$(ldd -r "${KVAF_A5_DIRECT_BINARY}" 2>&1) || {
    echo "${direct_ldd}" >&2
    kvaf_runtime_preflight_error "direct smoke runtime linkage is unresolved"
    return 1
  }
  if grep -Eq 'not found|undefined symbol' <<<"${direct_ldd}"; then
    echo "${direct_ldd}" >&2
    kvaf_runtime_preflight_error "direct smoke runtime linkage is unresolved"
    return 1
  fi
  resolved_direct_ascendcl=$(kvaf_runtime_preflight_ldd_path libascendcl.so \
    <<<"${direct_ldd}") || {
    kvaf_runtime_preflight_error \
      "direct smoke did not resolve exactly one libascendcl"
    return 1
  }
  [[ "$(realpath -e "${resolved_direct_ascendcl}")" == \
     "$(realpath -e "${direct_ascendcl}")" ]] || {
    kvaf_runtime_preflight_error \
      "direct smoke resolves a different libascendcl than its build"
    return 1
  }
  direct_build_manifest_sha=$(kvaf_runtime_preflight_sha256 \
    "${KVAF_A5_DIRECT_BUILD_MANIFEST}") || return 1
  direct_source_manifest_sha=$(kvaf_runtime_preflight_sha256 \
    "${direct_source_manifest}") || return 1

  [[ ! -L "${KVAF_DIRECT_AICPU_ARCHIVE}" &&
     -f "${KVAF_DIRECT_AICPU_ARCHIVE}" &&
     -r "${KVAF_DIRECT_AICPU_ARCHIVE}" ]] || {
    kvaf_runtime_preflight_error \
      "AICPU provider archive is missing or unsafe: ${KVAF_DIRECT_AICPU_ARCHIVE}"
    return 1
  }
  expected_package_name=cann-kvaf-direct-urma-compat.tar.gz
  expected_package_relative_dir=opp/vendors/cust/op_impl/aicpu/kernel
  expected_archive="${ASCEND_HOME_PATH}/${expected_package_relative_dir}/${expected_package_name}"
  [[ "$(realpath -m "${KVAF_DIRECT_AICPU_ARCHIVE}")" == \
     "$(realpath -m "${expected_archive}")" ]] || {
    kvaf_runtime_preflight_error \
      "runtime AICPU archive must be installed at ${expected_archive}"
    return 1
  }
  expected_registration_file="${ASCEND_HOME_PATH}/conf/ascend_package_load.ini"
  runtime_registration_file=${KVAF_DIRECT_AICPU_REGISTRATION_FILE:-${expected_registration_file}}
  [[ "$(realpath -m "${runtime_registration_file}")" == \
     "$(realpath -m "${expected_registration_file}")" ]] || {
    kvaf_runtime_preflight_error \
      "runtime AICPU package registry must be ${expected_registration_file}"
    return 1
  }
  kvaf_aicpu_acquire_registry_lock "${runtime_registration_file}" || {
    kvaf_runtime_preflight_error \
      "could not acquire the shared CANN AICPU registry lock"
    return 1
  }
  kvaf_aicpu_registration_is_exact \
    "${runtime_registration_file}" \
    "${expected_package_name}" \
    "${expected_package_relative_dir}" || {
    kvaf_runtime_preflight_error \
      "KVAF AICPU package is not registered exactly once for CANN loading"
    return 1
  }
  grep -Fxq 'name:Ascend-aicpu_legacy.tar.gz' \
    "${runtime_registration_file}" || {
    kvaf_runtime_preflight_error \
      "CANN AICPU package registry lacks its built-in baseline"
    return 1
  }
  legacy_registration_count=$(grep -Fxc \
    'name:liburma_perftest.tar.gz' \
    "${runtime_registration_file}" || true)
  if (( legacy_registration_count != 0 )); then
    kvaf_runtime_preflight_error \
      "legacy liburma_perftest registration conflicts with the isolated KVAF package"
    return 1
  fi
  registered_package_audit="${KVAF_RUNTIME_PREFLIGHT_SCRIPT_DIR}/audit_a5_registered_aicpu_packages.sh"
  [[ -f ${registered_package_audit} &&
     ! -L ${registered_package_audit} &&
     -x ${registered_package_audit} ]] || {
    kvaf_runtime_preflight_error \
      "registered AICPU package audit is missing or unsafe"
    return 1
  }
  device_provider_soname=$(readelf -d \
    "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}" | awk '
      /\(SONAME\)/ {
        value = $0
        sub(/^.*\[/, "", value)
        sub(/\].*$/, "", value)
        count++
      }
      END { if (count != 1 || value == "") exit 1; print value }
    ') || {
    kvaf_runtime_preflight_error "Device provider has no unique SONAME"
    return 1
  }
  # Securely snapshot the checked script with O_NOFOLLOW and pre/post fstat.
  # Base64 preserves every byte (including NUL) in a shell variable.  The
  # digest must match the provider build manifest, so a pre-open replacement
  # cannot silently weaken this gate.
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
' "${registered_package_audit}") || {
    kvaf_runtime_preflight_error \
      "could not pin the registered AICPU package audit bytes"
    return 1
  }
  [[ -n ${registered_package_audit_b64} ]] || {
    kvaf_runtime_preflight_error \
      "registered AICPU package audit is empty"
    return 1
  }
  registered_package_audit_sha=$(printf '%s' \
    "${registered_package_audit_b64}" | base64 -d | \
    sha256sum | awk '{print $1}') || {
    kvaf_runtime_preflight_error \
      "could not hash the pinned registered AICPU package audit"
    return 1
  }
  [[ ${registered_package_audit_sha} == \
     "${KVAF_VERIFIED_REGISTERED_PACKAGE_AUDIT_SHA:-}" ]] || {
    kvaf_runtime_preflight_error \
      "registered AICPU package audit differs from the verified provider build"
    return 1
  }
  if printf '%s' "${registered_package_audit_b64}" | base64 -d | \
    bash --noprofile --norc -s -- \
      --ascend-home "${ASCEND_HOME_PATH}" \
      --registry "${runtime_registration_file}" \
      --replace-archive "${expected_archive}" \
      --candidate-soname \
        "${device_provider_soname}=${KVAF_VERIFIED_DEVICE_PROVIDER_SHA}" \
      --candidate-soname \
        "${runtime_probe_provider_filename}=${runtime_probe_provider_sha}"; then
    registered_package_audit_rc=0
  else
    registered_package_audit_rc=$?
  fi
  unset registered_package_audit_b64
  if (( registered_package_audit_rc != 0 )); then
    kvaf_runtime_preflight_error \
      "registered AICPU package/SONAME audit failed"
    return 1
  fi
  echo "REGISTERED_AICPU_PACKAGE_AUDIT_SHA256=${registered_package_audit_sha}"
  for sibling_archive in "${expected_archive%/*}"/*.tar.gz; do
    [[ -e "${sibling_archive}" ]] || continue
    [[ "${sibling_archive}" != "${KVAF_DIRECT_AICPU_ARCHIVE}" ]] ||
      continue
    [[ ! -L "${sibling_archive}" &&
       -f "${sibling_archive}" &&
       -r "${sibling_archive}" ]] || {
      kvaf_runtime_preflight_error \
        "sibling AICPU package is missing or unsafe: ${sibling_archive}"
      return 1
    }
    sibling_entries=$(tar -tzf "${sibling_archive}" 2>/dev/null) || {
      kvaf_runtime_preflight_error \
        "sibling AICPU package is invalid: ${sibling_archive}"
      return 1
    }
    if grep -Eq \
        '(^|/)(libkvaf_urma_provider(_host|_device)?|libkvaf_rt_[0-9a-f]{16}|liburma_perftest)\.so$' \
        <<<"${sibling_entries}"; then
      kvaf_runtime_preflight_error \
        "sibling AICPU package contains a conflicting URMA provider: ${sibling_archive}"
      return 1
    fi
  done
  archive_entries=$(tar -tzf "${KVAF_DIRECT_AICPU_ARCHIVE}") || {
    kvaf_runtime_preflight_error "AICPU archive is not a valid tar.gz"
    return 1
  }
  for entry in "${required_archive_entries[@]}"; do
    entry_count=$(grep -Fxc -- "${entry}" <<<"${archive_entries}" || true)
    [[ "${entry_count}" == 1 ]] || {
      kvaf_runtime_preflight_error \
        "AICPU archive must contain exactly one ${entry}"
      return 1
    }
  done
  unexpected_entries=$(grep -Ev \
    "^aicpu_kernels_device/?$|^aicpu_kernels_device/(libkvaf_urma_provider_device\\.so|${runtime_probe_provider_filename//./\\.}|kvaf-direct-urma-provider\\.manifest|kvaf-direct-urma-source\\.sha256|kvaf-direct-urma-umdk-source\\.sha256|kvaf-direct-urma-libummu-source\\.sha256|kvaf_udma_uapi_probe|kvaf_udma_private_abi_probe|kvaf_ummu_uapi_probe)$" \
    <<<"${archive_entries}" || true)
  [[ -z "${unexpected_entries}" ]] || {
    kvaf_runtime_preflight_error \
      "AICPU archive contains a stale or unexpected entry"
    return 1
  }

  archive_provider_sha=$(kvaf_runtime_preflight_archive_sha \
    "${KVAF_DIRECT_AICPU_ARCHIVE}" \
    aicpu_kernels_device/libkvaf_urma_provider_device.so) || return 1
  archive_build_manifest_sha=$(kvaf_runtime_preflight_archive_sha \
    "${KVAF_DIRECT_AICPU_ARCHIVE}" \
    aicpu_kernels_device/kvaf-direct-urma-provider.manifest) || return 1
  archive_runtime_probe_provider_sha=$(kvaf_runtime_preflight_archive_sha \
    "${KVAF_DIRECT_AICPU_ARCHIVE}" \
    "aicpu_kernels_device/${runtime_probe_provider_filename}") || return 1
  archive_source_manifest_sha=$(kvaf_runtime_preflight_archive_sha \
    "${KVAF_DIRECT_AICPU_ARCHIVE}" \
    aicpu_kernels_device/kvaf-direct-urma-source.sha256) || return 1
  archive_umdk_source_manifest_sha=$(kvaf_runtime_preflight_archive_sha \
    "${KVAF_DIRECT_AICPU_ARCHIVE}" \
    aicpu_kernels_device/kvaf-direct-urma-umdk-source.sha256) || return 1
  archive_libummu_source_manifest_sha=$(kvaf_runtime_preflight_archive_sha \
    "${KVAF_DIRECT_AICPU_ARCHIVE}" \
    aicpu_kernels_device/kvaf-direct-urma-libummu-source.sha256) || return 1
  [[ "${archive_provider_sha}" == "${device_provider_sha}" &&
     "${archive_runtime_probe_provider_sha}" == \
       "${runtime_probe_provider_sha}" &&
     "${archive_build_manifest_sha}" == "${build_manifest_sha}" &&
     "${archive_source_manifest_sha}" == "${source_manifest_sha}" &&
     "${archive_umdk_source_manifest_sha}" == \
       "${umdk_source_manifest_sha}" &&
     "${archive_libummu_source_manifest_sha}" == \
       "${libummu_source_manifest_sha}" ]] || {
    kvaf_runtime_preflight_error \
      "AICPU provider or provenance does not match the verified build"
    return 1
  }
  for probe in kvaf_udma_uapi_probe kvaf_udma_private_abi_probe \
    kvaf_ummu_uapi_probe; do
    archive_probe_sha=$(kvaf_runtime_preflight_archive_sha \
      "${KVAF_DIRECT_AICPU_ARCHIVE}" "aicpu_kernels_device/${probe}") ||
      return 1
    local_probe_sha=$(kvaf_runtime_preflight_sha256 \
      "${KVAF_DIRECT_PROVIDER_BUILD_DIR}/${probe}") || return 1
    [[ "${archive_probe_sha}" == "${local_probe_sha}" ]] || {
      kvaf_runtime_preflight_error \
        "archived ${probe} does not match the verified build"
      return 1
    }
  done

  binary_dynamic=$(readelf -d "${KVAF_A5_DIRECT_BINARY}") || return 1
  grep -Fq libhcomm <<<"${binary_dynamic}" && {
    kvaf_runtime_preflight_error "direct smoke unexpectedly links HCOMM"
    return 1
  }
  for provider in "${KVAF_DIRECT_HOST_PROVIDER_LIBRARY}" \
    "${KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY}"
  do
    provider_dynamic=$(readelf -d "${provider}") || return 1
    if grep -Eq \
        'lib(hcomm|urma_perftest|urma_common|urma|urma-udma|ummu|udma)\.so' \
        <<<"${provider_dynamic}" ||
       grep -Eq '\((RPATH|RUNPATH)\)' <<<"${provider_dynamic}"; then
      kvaf_runtime_preflight_error \
        "provider links a forbidden external library or contains RPATH"
      return 1
    fi
    provider_symbols=$(nm -D --defined-only "${provider}") || return 1
    for symbol in kvaf_urma_provider_open_v2 kvaf_urma_provider_close_v2 \
      kvaf_urma_provider_private_abi_v2 aicpu_kvaf_urma_open_v2 \
      aicpu_kvaf_urma_close_v2 aicpu_kvaf_urma_owner_probe_v1 \
      aicpu_kvaf_urma_epoch_owner_v1
    do
      grep -Eq "[[:space:]]${symbol}(@@KVAF_URMA_PROVIDER_2)?$" \
        <<<"${provider_symbols}" || {
        kvaf_runtime_preflight_error \
          "provider does not export ${symbol}: ${provider}"
        return 1
      }
    done
    if grep -Eq \
        '[[:space:]]aicpu_kvaf_urma_rt_probe_v2(@@[^[:space:]]+)?$' \
        <<<"${provider_symbols}"; then
      kvaf_runtime_preflight_error \
        "canonical provider exports the isolated runtime probe: ${provider}"
      return 1
    fi
  done

  provider="${KVAF_DIRECT_PROVIDER_BUILD_DIR}/${runtime_probe_provider_filename}"
  [[ ! -L ${provider} && -f ${provider} && -r ${provider} ]] || {
    kvaf_runtime_preflight_error \
      "isolated runtime probe provider is missing or unsafe"
    return 1
  }
  provider_dynamic=$(readelf -d "${provider}") || return 1
  if grep -Eq \
      'lib(hcomm|urma_perftest|urma_common|urma|urma-udma|ummu|udma)\.so|\((RPATH|RUNPATH)\)' \
      <<<"${provider_dynamic}"; then
    kvaf_runtime_preflight_error \
      "isolated runtime probe provider contains stateful linkage or RPATH"
    return 1
  fi
  provider_symbols=$(nm -D --defined-only "${provider}") || return 1
  grep -Eq \
    '[[:space:]]aicpu_kvaf_urma_rt_probe_v2(@@KVAF_URMA_RUNTIME_PROBE_2)?$' \
    <<<"${provider_symbols}" || {
    kvaf_runtime_preflight_error \
      "isolated runtime probe provider lacks its sole AICPU entry"
    return 1
  }
  for symbol in aicpu_kvaf_urma_open_v2 aicpu_kvaf_urma_close_v2 \
    aicpu_kvaf_urma_owner_probe_v1 aicpu_kvaf_urma_epoch_owner_v1
  do
    if grep -Eq "[[:space:]]${symbol}(@@[^[:space:]]+)?$" \
        <<<"${provider_symbols}"; then
      kvaf_runtime_preflight_error \
        "isolated runtime probe provider leaks canonical entry ${symbol}"
      return 1
    fi
  done

  KVAF_VERIFIED_DIRECT_BINARY_SHA=${direct_binary_sha}
  KVAF_VERIFIED_DIRECT_BUILD_MANIFEST_SHA=${direct_build_manifest_sha}
  KVAF_VERIFIED_DIRECT_SOURCE_MANIFEST=${direct_source_manifest}
  KVAF_VERIFIED_DIRECT_SOURCE_MANIFEST_SHA=${direct_source_manifest_sha}
  KVAF_VERIFIED_DIRECT_RUNTIME_HEADER=${direct_runtime_header}
}
