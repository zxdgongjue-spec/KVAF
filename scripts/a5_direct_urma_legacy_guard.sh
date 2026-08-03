#!/usr/bin/env bash

# Compatibility barrier for state locations used by older direct-URMA
# runners. This file is sourced by build/package/run entry points.

kvaf_configure_direct_urma_state_paths()
{
  local root_dir=$1
  local expected_global_state expected_direct_state
  local variable expected current
  local -a fixed_paths

  # /data is the host/container shared machine namespace on Server B.  This
  # root must not depend on the checkout path: every clone manipulates the
  # same /proc/asdrv_ub selector and the same NPU resources.
  expected_global_state=/data/.kvaf-state
  expected_direct_state="${expected_global_state}/KVAF"
  fixed_paths=(
    "KVAF_GLOBAL_STATE_DIR=${expected_global_state}"
    "KVAF_DIRECT_STATE_DIR=${expected_direct_state}"
    "KVAF_DIRECT_RESULT_ROOT=${expected_direct_state}/results"
    "KVAF_EID_LOCK_FILE=${expected_global_state}/asdrv-ub-pair.lock"
    "KVAF_EID_LEGACY_LOCK_FILE=/run/lock/kvaf_asdrv_ub_pair.lock"
    "KVAF_DIRECT_DEPLOY_RUN_LOCK=${expected_direct_state}/direct-urma.lock"
    "KVAF_DIRECT_ACTIVE_SESSION_LOCK=${expected_direct_state}/direct-urma.active.lock"
    "KVAF_DIRECT_ACTIVE_SESSION_FILE=${expected_direct_state}/direct-urma.active"
    "KVAF_DIRECT_QUARANTINE_FILE=${expected_direct_state}/direct-urma.quarantine"
  )

  # These paths form one recovery transaction. Allowing an old exported
  # per-file override would split locks, proof, and quarantine across roots.
  # Fail with an actionable message instead of silently ignoring the terminal.
  for variable in "${fixed_paths[@]}"; do
    expected=${variable#*=}
    variable=${variable%%=*}
    if declare -p "${variable}" >/dev/null 2>&1; then
      current=${!variable}
      if [[ "${current}" != "${expected}" ]]; then
        echo "ERROR: ${variable} cannot override the fixed direct URMA recovery root" >&2
        echo "EXPECTED=${expected}" >&2
        echo "ACTUAL=${current}" >&2
        echo "Unset the stale variable and retry; no hardware was accessed." >&2
        return 1
      fi
    fi
    printf -v "${variable}" '%s' "${expected}"
  done
}

# Configure the deliberately isolated state transaction used by the one
# explicit NPU7-quarantine -> NPU2 experiment.  This is not a configurable
# alternative production root: both the target NPU and the accepted source
# run are part of the directory identity, and every per-file override is
# rejected just as it is for the production transaction above.
kvaf_configure_cross_npu2_risk_state_paths()
{
  local root_dir=$1
  local source_run_id=$2
  local expected_global_state expected_experiment_root
  local expected_direct_state variable expected current
  local -a fixed_paths

  if [[ ! "${source_run_id}" =~ ^[0-9a-f]{32,128}$ ]]; then
    echo "ERROR: cross-NPU source run ID must be 32-128 lowercase hex digits" >&2
    return 1
  fi

  # Keep ROOT_DIR in the interface so callers cannot accidentally substitute
  # this function for a general state-root setter in another checkout.
  [[ -n "${root_dir}" ]] || return 1
  expected_global_state=/data/.kvaf-state
  expected_experiment_root="${expected_global_state}/KVAF/experimental"
  expected_direct_state="${expected_experiment_root}/npu-2/${source_run_id}"
  fixed_paths=(
    "KVAF_GLOBAL_STATE_DIR=${expected_global_state}"
    "KVAF_CROSS_NPU_EXPERIMENT_ROOT=${expected_experiment_root}"
    "KVAF_DIRECT_STATE_DIR=${expected_direct_state}"
    "KVAF_DIRECT_RESULT_ROOT=${expected_direct_state}/results"
    "KVAF_EID_LOCK_FILE=${expected_global_state}/asdrv-ub-pair.lock"
    "KVAF_EID_LEGACY_LOCK_FILE=/run/lock/kvaf_asdrv_ub_pair.lock"
    "KVAF_DIRECT_DEPLOY_RUN_LOCK=${expected_direct_state}/direct-urma.lock"
    "KVAF_DIRECT_ACTIVE_SESSION_LOCK=${expected_direct_state}/direct-urma.active.lock"
    "KVAF_DIRECT_ACTIVE_SESSION_FILE=${expected_direct_state}/direct-urma.active"
    "KVAF_DIRECT_QUARANTINE_FILE=${expected_direct_state}/direct-urma.quarantine"
    "KVAF_CROSS_NPU_LOCK_FILE=${expected_experiment_root}/cross-npu-risk.lock"
    "KVAF_CROSS_NPU_SOURCE_QUARANTINE_FILE=${expected_global_state}/KVAF/direct-urma.quarantine"
    "KVAF_CROSS_NPU_SOURCE_RESULT_ROOT=${expected_global_state}/KVAF/results"
  )

  for variable in "${fixed_paths[@]}"; do
    expected=${variable#*=}
    variable=${variable%%=*}
    if declare -p "${variable}" >/dev/null 2>&1; then
      current=${!variable}
      if [[ "${current}" != "${expected}" ]]; then
        echo "ERROR: ${variable} cannot override the fixed cross-NPU2 risk root" >&2
        echo "EXPECTED=${expected}" >&2
        echo "ACTUAL=${current}" >&2
        return 1
      fi
    fi
    printf -v "${variable}" '%s' "${expected}"
  done
}

kvaf_check_legacy_direct_urma_state()
{
  local root_dir=$1
  local legacy_path
  local -a legacy_state_roots=(
    "${root_dir}/.kvaf-state"
    "$(dirname "${root_dir}")/.kvaf-state"
  )
  local -a legacy_state_files=(
    "${root_dir}/.kvaf-state/direct-urma.quarantine"
    "$(dirname "${root_dir}")/.kvaf-state/KVAF/direct-urma.quarantine"
    "/run/lock/kvaf-a5-direct-urma.quarantine"
    "${root_dir}/.kvaf-state/direct-urma.active"
    "$(dirname "${root_dir}")/.kvaf-state/KVAF/direct-urma.active"
    "/run/lock/kvaf-a5-direct-urma.active"
  )
  local -a legacy_lock_files=(
    "${root_dir}/.kvaf-state/direct-urma.active.lock"
    "$(dirname "${root_dir}")/.kvaf-state/KVAF/direct-urma.active.lock"
    "/run/lock/kvaf-a5-direct-urma.active.lock"
  )

  for legacy_path in "${legacy_state_roots[@]}"; do
    if [[ -L "${legacy_path}" ]]; then
      echo "ERROR: legacy direct URMA state root is a symlink" >&2
      echo "LEGACY_PATH=${legacy_path}" >&2
      return 2
    fi
  done

  # Keep all historical deploy locks for this shell's entire lifetime. The
  # run entry point deliberately lets workers inherit FDs 5/6/7, so an old
  # runner cannot start after the new supervising shell detaches.
  if ! mkdir -p \
      "${root_dir}/.kvaf-state" \
      "$(dirname "${root_dir}")/.kvaf-state/KVAF"; then
    echo "ERROR: could not create legacy compatibility state root" >&2
    return 2
  fi
  for legacy_path in \
    "/run/lock/kvaf-a5-direct-urma.lock" \
    "${root_dir}/.kvaf-state/direct-urma.lock" \
    "$(dirname "${root_dir}")/.kvaf-state/KVAF/direct-urma.lock"
  do
    if [[ -L "${legacy_path}" ||
          ( -e "${legacy_path}" && ! -f "${legacy_path}" ) ]]; then
      echo "ERROR: legacy direct URMA deploy lock is not a regular file" >&2
      echo "LEGACY_PATH=${legacy_path}" >&2
      return 2
    fi
  done
  if ! exec 7<>"/run/lock/kvaf-a5-direct-urma.lock"; then
    echo "ERROR: could not open legacy /run deploy lock" >&2
    return 2
  fi
  if ! chmod 0600 "/run/lock/kvaf-a5-direct-urma.lock"; then
    echo "ERROR: could not protect legacy /run deploy lock" >&2
    return 2
  fi
  if ! flock -n 7; then
    echo "ERROR: a legacy direct URMA process owns /run/lock/kvaf-a5-direct-urma.lock" >&2
    return 5
  fi
  if ! exec 6<>"${root_dir}/.kvaf-state/direct-urma.lock"; then
    echo "ERROR: could not open legacy worktree deploy lock" >&2
    return 2
  fi
  if ! chmod 0600 "${root_dir}/.kvaf-state/direct-urma.lock"; then
    echo "ERROR: could not protect legacy worktree deploy lock" >&2
    return 2
  fi
  if ! flock -n 6; then
    echo "ERROR: a legacy direct URMA process owns ${root_dir}/.kvaf-state/direct-urma.lock" >&2
    return 5
  fi
  if ! exec 5<>"$(dirname "${root_dir}")/.kvaf-state/KVAF/direct-urma.lock"; then
    echo "ERROR: could not open previous parent-scoped deploy lock" >&2
    return 2
  fi
  if ! chmod 0600 "$(dirname "${root_dir}")/.kvaf-state/KVAF/direct-urma.lock"; then
    echo "ERROR: could not protect previous parent-scoped deploy lock" >&2
    return 2
  fi
  if ! flock -n 5; then
    echo "ERROR: an older direct URMA process owns the parent-scoped deploy lock" >&2
    return 5
  fi

  # Recheck state only after the compatibility locks are held, closing the
  # check/start race with an older runner.
  for legacy_path in "${legacy_state_files[@]}"; do
    if [[ -e "${legacy_path}" || -L "${legacy_path}" ]]; then
      echo "ERROR: legacy direct URMA active/quarantine state exists" >&2
      echo "LEGACY_PATH=${legacy_path}" >&2
      echo "Migrate or resolve it before using the new persistent state root." >&2
      if [[ "${legacy_path}" == *.quarantine ]]; then
        return 4
      fi
      return 5
    fi
  done

  for legacy_path in "${legacy_lock_files[@]}"; do
    if [[ ! -e "${legacy_path}" && ! -L "${legacy_path}" ]]; then
      continue
    fi
    if [[ -L "${legacy_path}" || ! -f "${legacy_path}" ]]; then
      echo "ERROR: legacy direct URMA lock is not a regular file" >&2
      echo "LEGACY_PATH=${legacy_path}" >&2
      return 2
    fi
    if ! flock -n "${legacy_path}" -c true; then
      echo "ERROR: a legacy direct URMA process still owns its old lock" >&2
      echo "LEGACY_PATH=${legacy_path}" >&2
      echo "No topology/EID/NPU operation was attempted." >&2
      return 5
    fi
  done
}
