#!/usr/bin/env bash

# Shared, file-only helpers for the CANN AICPU package registry.  This file is
# sourced by the installer and runtime preflight; it deliberately does not
# change the caller's shell options.

KVAF_AICPU_REGISTRY_LOCK_HELD=0
KVAF_AICPU_REGISTRY_LOCK_PATH=

kvaf_aicpu_acquire_registry_lock()
{
  local registry_file=$1
  local registry_real registry_lock

  [[ ! -L "${registry_file}" && -f "${registry_file}" &&
     -r "${registry_file}" && -w "${registry_file}" ]] || {
    kvaf_aicpu_registration_error \
      "stage=validate-registry-for-lock path=${registry_file}"
    return 1
  }
  registry_real=$(realpath -- "${registry_file}") || {
    kvaf_aicpu_registration_error \
      "stage=resolve-registry-for-lock path=${registry_file}"
    return 1
  }
  registry_lock="${registry_real}.kvaf.lock"
  [[ ! -L "${registry_lock}" &&
     ( ! -e "${registry_lock}" || -f "${registry_lock}" ) ]] || {
    kvaf_aicpu_registration_error \
      "stage=validate-registry-lock path=${registry_lock}"
    return 1
  }
  if (( KVAF_AICPU_REGISTRY_LOCK_HELD != 0 )); then
    [[ ${KVAF_AICPU_REGISTRY_LOCK_PATH} == "${registry_lock}" ]] || {
      kvaf_aicpu_registration_error \
        "stage=registry-lock-split-brain expected=${KVAF_AICPU_REGISTRY_LOCK_PATH} actual=${registry_lock}"
      return 1
    }
    return 0
  fi
  # FD 11 is reserved for the CANN registry transaction.  Direct runners use
  # 5..10 for their long-lived compatibility/session locks.
  # Read/write open creates the lock without truncating an existing path.  A
  # symlink or hardlink is opened without modifying its target, then rejected
  # from the descriptor metadata before fchmod/flock.  Revalidate the pathname
  # after locking so all cooperating KVAF processes converge on one inode.
  exec 11<>"${registry_lock}" || {
    kvaf_aicpu_registration_error \
      "stage=open-registry-lock path=${registry_lock}"
    return 1
  }
  if ! python3 -c '
import os, stat, sys
path = sys.argv[1]
opened = os.fstat(11)
named = os.lstat(path)
if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or
        opened.st_uid != os.geteuid() or stat.S_ISLNK(named.st_mode) or
        opened.st_dev != named.st_dev or opened.st_ino != named.st_ino):
    raise SystemExit(1)
os.fchmod(11, 0o600)
' "${registry_lock}"; then
    exec 11>&-
    kvaf_aicpu_registration_error \
      "stage=revalidate-registry-lock path=${registry_lock}"
    return 1
  fi
  flock -w 30 11 || {
    exec 11>&-
    kvaf_aicpu_registration_error \
      "stage=acquire-registry-lock path=${registry_lock}"
    return 1
  }
  if ! python3 -c '
import os, stat, sys
path = sys.argv[1]
opened = os.fstat(11)
named = os.lstat(path)
if (not stat.S_ISREG(named.st_mode) or named.st_nlink != 1 or
        named.st_uid != os.geteuid() or opened.st_dev != named.st_dev or
        opened.st_ino != named.st_ino):
    raise SystemExit(1)
' "${registry_lock}"; then
    exec 11>&-
    kvaf_aicpu_registration_error \
      "stage=post-lock-identity path=${registry_lock}"
    return 1
  fi
  KVAF_AICPU_REGISTRY_LOCK_HELD=1
  KVAF_AICPU_REGISTRY_LOCK_PATH=${registry_lock}
}

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

kvaf_aicpu_registration_error()
{
  echo "ERROR: AICPU package registration: $*" >&2
  return 1
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
  local expected_old_sha=${4:-}
  local registry_real registry_dir temporary_registry
  local current_registry_sha candidate_registry_sha

  KVAF_AICPU_REGISTRATION_PUBLISHED_SHA256=

  kvaf_aicpu_registration_arguments_are_valid \
    "${package_name}" "${package_path}" || {
    kvaf_aicpu_registration_error "stage=validate-arguments"
    return 1
  }

  # Never synthesize CANN's base registry.  A missing registry can omit the
  # built-in AICPU packages and is a deployment error, not an empty starting
  # point.  Likewise, do not silently repair a duplicate or malformed KVAF
  # stanza: fail closed so an operator can inspect the foreign edit.
  [[ ! -L "${registry_file}" &&
     -f "${registry_file}" &&
     -r "${registry_file}" &&
     -w "${registry_file}" ]] || {
    kvaf_aicpu_registration_error \
      "stage=validate-registry path=${registry_file}"
    return 1
  }
  kvaf_aicpu_acquire_registry_lock "${registry_file}" || return 1
  current_registry_sha=$(sha256sum -- "${registry_file}" | awk '{print $1}') || {
    kvaf_aicpu_registration_error "stage=hash-registry-before-update"
    return 1
  }
  if [[ -n ${expected_old_sha} ]]; then
    [[ ${expected_old_sha} =~ ^[0-9a-f]{64}$ &&
       ${current_registry_sha} == "${expected_old_sha}" ]] || {
      kvaf_aicpu_registration_error \
        "stage=compare-registry-before-update expected=${expected_old_sha} actual=${current_registry_sha}"
      return 1
    }
  fi
  if kvaf_aicpu_registration_is_exact \
      "${registry_file}" "${package_name}" "${package_path}"; then
    KVAF_AICPU_REGISTRATION_PUBLISHED_SHA256=${current_registry_sha}
    return 0
  fi
  kvaf_aicpu_registration_is_absent \
    "${registry_file}" "${package_name}" "${package_path}" || {
    kvaf_aicpu_registration_error \
      "stage=validate-existing-registration package=${package_name}"
    return 1
  }

  # CANN may expose conf through a legitimate directory symlink.  Resolve the
  # existing regular file once and create/rename the candidate in its real
  # parent directory.  This keeps the same-filesystem atomic replace while
  # avoiding both a false rejection and a symlink-parent check/use race.  The
  # registry file itself remains forbidden from being a symlink.
  registry_real=$(realpath -- "${registry_file}") || {
    kvaf_aicpu_registration_error \
      "stage=resolve-registry path=${registry_file}"
    return 1
  }
  [[ ! -L "${registry_real}" &&
     -f "${registry_real}" &&
     -r "${registry_real}" &&
     -w "${registry_real}" ]] || {
    kvaf_aicpu_registration_error \
      "stage=validate-resolved-registry path=${registry_real}"
    return 1
  }
  registry_dir=$(dirname -- "${registry_real}") || {
    kvaf_aicpu_registration_error "stage=resolve-parent"
    return 1
  }
  [[ -d "${registry_dir}" && -w "${registry_dir}" ]] || {
    kvaf_aicpu_registration_error \
      "stage=validate-parent path=${registry_dir}"
    return 1
  }
  temporary_registry=$(mktemp \
    "${registry_dir}/.ascend_package_load.ini.kvaf.XXXXXX") || {
    kvaf_aicpu_registration_error \
      "stage=create-candidate parent=${registry_dir}"
    return 1
  }

  # cp -p preserves CANN's owner/mode.  Add exactly one terminating newline
  # only when a non-empty foreign registry omitted it; otherwise preserve the
  # existing bytes before appending the KVAF stanza.
  if ! cp -p -- "${registry_real}" "${temporary_registry}"; then
    rm -f -- "${temporary_registry}"
    kvaf_aicpu_registration_error "stage=copy-registry"
    return 1
  fi
  if [[ -s "${temporary_registry}" ]] &&
     [[ $(tail -c 1 -- "${temporary_registry}" | wc -l) -eq 0 ]] &&
     ! printf '\n' >>"${temporary_registry}"; then
    rm -f -- "${temporary_registry}"
    kvaf_aicpu_registration_error "stage=terminate-existing-registry"
    return 1
  fi
  if ! kvaf_aicpu_registration_block \
      "${package_name}" "${package_path}" >>"${temporary_registry}"; then
    rm -f -- "${temporary_registry}"
    kvaf_aicpu_registration_error "stage=append-registration"
    return 1
  fi
  if ! kvaf_aicpu_registration_is_exact \
       "${temporary_registry}" "${package_name}" "${package_path}"; then
    rm -f -- "${temporary_registry}"
    kvaf_aicpu_registration_error "stage=verify-candidate"
    return 1
  fi
  candidate_registry_sha=$(sha256sum -- "${temporary_registry}" |
    awk '{print $1}') || {
    rm -f -- "${temporary_registry}"
    kvaf_aicpu_registration_error "stage=hash-candidate"
    return 1
  }
  # Recheck the exact old bytes immediately before publish.  The shared lock
  # serializes every cooperating KVAF packager.  The SHA comparison detects a
  # foreign write that completed before this check, but it is not an atomic CAS
  # against a non-cooperating root writer in the final check/rename window.
  # Callers must therefore install packages in an exclusive CANN maintenance
  # window; the code deliberately does not claim a stronger global guarantee.
  current_registry_sha=$(sha256sum -- "${registry_real}" |
    awk '{print $1}') || {
    rm -f -- "${temporary_registry}"
    kvaf_aicpu_registration_error "stage=rehash-registry-before-publish"
    return 1
  }
  if [[ -n ${expected_old_sha} &&
        ${current_registry_sha} != "${expected_old_sha}" ]]; then
    rm -f -- "${temporary_registry}"
    kvaf_aicpu_registration_error \
      "stage=compare-registry-before-publish expected=${expected_old_sha} actual=${current_registry_sha}"
    return 1
  fi
  KVAF_AICPU_REGISTRATION_PUBLISHED_SHA256=${candidate_registry_sha}
  if ! mv -f -- "${temporary_registry}" "${registry_real}"; then
    rm -f -- "${temporary_registry}"
    kvaf_aicpu_registration_error \
      "stage=publish-registration target=${registry_real}"
    return 1
  fi
  if [[ -n ${KVAF_AICPU_DURABLE_STATE_HELPER:-} ]] &&
     ! python3 "${KVAF_AICPU_DURABLE_STATE_HELPER}" sync \
       "${registry_real}" "${registry_dir}"; then
    kvaf_aicpu_registration_error \
      "stage=sync-published-registration target=${registry_real}"
    return 1
  fi
  if [[ "$(realpath -- "${registry_file}")" != "${registry_real}" ]]; then
    kvaf_aicpu_registration_error "stage=verify-stable-path"
    return 1
  fi
  kvaf_aicpu_registration_is_exact \
    "${registry_file}" "${package_name}" "${package_path}" || {
    kvaf_aicpu_registration_error "stage=verify-installed-registration"
    return 1
  }
}
