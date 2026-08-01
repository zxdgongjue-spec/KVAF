#!/usr/bin/env bash

# Shared, file-only helpers for the CANN AICPU package registry.  This file is
# sourced by the installer and runtime preflight; it deliberately does not
# change the caller's shell options.

kvaf_aicpu_registration_block()
{
  local package_name=$1
  local package_path=$2

  printf '%s\n' \
    "name:${package_name}" \
    "install_path:2" \
    "optional:true" \
    "package_path:${package_path}" \
    "load_as_per_soc:false"
}

kvaf_aicpu_registration_arguments_are_valid()
{
  local package_name=$1
  local package_path=$2

  [[ "${package_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*\.tar\.gz$ &&
     "${package_path}" =~ ^[A-Za-z0-9._/-]+$ &&
     "${package_path}" != /* &&
     "${package_path}" != *..* &&
     "${package_path}" != */ ]] \
    || return 1
}

kvaf_aicpu_registration_is_absent()
{
  local registry_file=$1
  local package_name=$2
  local package_path=$3

  kvaf_aicpu_registration_arguments_are_valid \
    "${package_name}" "${package_path}" || return 1
  [[ ! -L "${registry_file}" &&
     -f "${registry_file}" &&
     -r "${registry_file}" ]] || return 1

  awk -v target="name:${package_name}" '
    $0 == target { count++ }
    END { exit(count == 0 ? 0 : 1) }
  ' "${registry_file}"
}

kvaf_aicpu_registration_is_exact()
{
  local registry_file=$1
  local package_name=$2
  local package_path=$3

  kvaf_aicpu_registration_arguments_are_valid \
    "${package_name}" "${package_path}" || return 1
  [[ ! -L "${registry_file}" &&
     -f "${registry_file}" &&
     -r "${registry_file}" ]] || return 1

  awk -v package_name="${package_name}" \
      -v package_path="${package_path}" '
    BEGIN {
      expected[1] = "name:" package_name
      expected[2] = "install_path:2"
      expected[3] = "optional:true"
      expected[4] = "package_path:" package_path
      expected[5] = "load_as_per_soc:false"
      target_count = 0
      target_active = 0
      target_line = 0
      invalid = 0
    }
    function close_target() {
      if (target_active && target_line != 5) {
        invalid = 1
      }
      target_active = 0
      target_line = 0
    }
    /^name:/ {
      close_target()
      if ($0 == expected[1]) {
        target_count++
        target_active = 1
        target_line = 1
      }
      next
    }
    {
      if (target_active) {
        if (target_line < 5) {
          target_line++
          if ($0 != expected[target_line]) {
            invalid = 1
          }
        } else if ($0 !~ /^[[:space:]]*$/ &&
                   $0 !~ /^[[:space:]]*[#;]/) {
          invalid = 1
        }
      }
    }
    END {
      close_target()
      exit(target_count == 1 && invalid == 0 ? 0 : 1)
    }
  ' "${registry_file}"
}

kvaf_aicpu_install_registration()
{
  local registry_file=$1
  local package_name=$2
  local package_path=$3
  local registry_dir temporary_registry

  kvaf_aicpu_registration_arguments_are_valid \
    "${package_name}" "${package_path}" || return 1

  # Never synthesize CANN's base registry.  A missing registry can omit the
  # built-in AICPU packages and is a deployment error, not an empty starting
  # point.  Likewise, do not silently repair a duplicate or malformed KVAF
  # stanza: fail closed so an operator can inspect the foreign edit.
  [[ ! -L "${registry_file}" &&
     -f "${registry_file}" &&
     -r "${registry_file}" &&
     -w "${registry_file}" ]] || return 1
  if kvaf_aicpu_registration_is_exact \
      "${registry_file}" "${package_name}" "${package_path}"; then
    return 0
  fi
  kvaf_aicpu_registration_is_absent \
    "${registry_file}" "${package_name}" "${package_path}" || return 1

  registry_dir=$(dirname -- "${registry_file}") || return 1
  [[ ! -L "${registry_dir}" &&
     -d "${registry_dir}" &&
     -w "${registry_dir}" ]] || return 1
  temporary_registry=$(mktemp \
    "${registry_dir}/.ascend_package_load.ini.kvaf.XXXXXX") || return 1

  # cp -p preserves CANN's owner/mode.  Add exactly one terminating newline
  # only when a non-empty foreign registry omitted it; otherwise preserve the
  # existing bytes before appending the KVAF stanza.
  if ! cp -p -- "${registry_file}" "${temporary_registry}"; then
    rm -f -- "${temporary_registry}"
    return 1
  fi
  if [[ -s "${temporary_registry}" ]] &&
     [[ $(tail -c 1 -- "${temporary_registry}" | wc -l) -eq 0 ]] &&
     ! printf '\n' >>"${temporary_registry}"; then
    rm -f -- "${temporary_registry}"
    return 1
  fi
  if ! kvaf_aicpu_registration_block \
      "${package_name}" "${package_path}" >>"${temporary_registry}"; then
    rm -f -- "${temporary_registry}"
    return 1
  fi
  if ! kvaf_aicpu_registration_is_exact \
       "${temporary_registry}" "${package_name}" "${package_path}"; then
    rm -f -- "${temporary_registry}"
    return 1
  fi
  if ! mv -f -- "${temporary_registry}" "${registry_file}"; then
    rm -f -- "${temporary_registry}"
    return 1
  fi
  kvaf_aicpu_registration_is_exact \
    "${registry_file}" "${package_name}" "${package_path}"
}
