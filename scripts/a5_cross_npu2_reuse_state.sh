#!/usr/bin/env bash

# Fixed state layout for the second, explicitly risk-accepted NPU2 attempt.
# The first NPU2 incident remains evidence in experimental/npu-2; a new
# generation must never reuse, rename, or remove that transaction.

kvaf_configure_cross_npu2_reuse_state_paths()
{
  local root_dir=$1
  local source_run_id=$2
  local prior_target_run_id=$3
  local artifact_token=$4
  local expected_global_state expected_experiment_root expected_reuse_root
  local expected_direct_state variable expected current path
  local -a fixed_paths protected_paths

  [[ -n "${root_dir}" ]] || return 1
  if [[ "${source_run_id}" != 530f5998a0b646d48db858170a9996ff ]]; then
    echo "ERROR: NPU2 reuse accepts only the pinned NPU7 source incident" >&2
    return 1
  fi
  if [[ "${prior_target_run_id}" != 30ef7336d212459a9317fd6647f6fa15 ]]; then
    echo "ERROR: NPU2 reuse accepts only the pinned first NPU2 incident" >&2
    return 1
  fi
  if [[ ! "${artifact_token}" =~ ^[0-9a-f]{16}$ ]]; then
    echo "ERROR: NPU2 reuse artifact token must be 16 lowercase hex digits" >&2
    return 1
  fi

  expected_global_state=/data/.kvaf-state
  expected_experiment_root="${expected_global_state}/KVAF/experimental"
  expected_reuse_root="${expected_experiment_root}/npu-2-reuse"
  expected_direct_state="${expected_reuse_root}/${source_run_id}/${prior_target_run_id}/${artifact_token}"
  fixed_paths=(
    "KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN=${artifact_token}"
    "KVAF_GLOBAL_STATE_DIR=${expected_global_state}"
    "KVAF_CROSS_NPU2_REUSE_ROOT=${expected_reuse_root}"
    "KVAF_CROSS_NPU2_REUSE_STATE_DIR=${expected_direct_state}"
    "KVAF_DIRECT_STATE_DIR=${expected_direct_state}"
    "KVAF_DIRECT_RESULT_ROOT=${expected_direct_state}/results"
    "KVAF_EID_LOCK_FILE=${expected_global_state}/asdrv-ub-pair.lock"
    "KVAF_EID_LEGACY_LOCK_FILE=/run/lock/kvaf_asdrv_ub_pair.lock"
    "KVAF_DIRECT_DEPLOY_RUN_LOCK=${expected_direct_state}/direct-urma.lock"
    "KVAF_DIRECT_ACTIVE_SESSION_LOCK=${expected_direct_state}/direct-urma.active.lock"
    "KVAF_DIRECT_ACTIVE_SESSION_FILE=${expected_direct_state}/direct-urma.active"
    "KVAF_DIRECT_QUARANTINE_FILE=${expected_direct_state}/direct-urma.quarantine"
    "KVAF_CROSS_NPU2_REUSE_LOCK_FILE=${expected_reuse_root}/reuse-risk.lock"
    "KVAF_CROSS_NPU2_REUSE_SOURCE_QUARANTINE_FILE=${expected_global_state}/KVAF/direct-urma.quarantine"
    "KVAF_CROSS_NPU2_REUSE_SOURCE_RESULT_ROOT=${expected_global_state}/KVAF/results"
    "KVAF_CROSS_NPU2_REUSE_PRIOR_STATE_DIR=${expected_experiment_root}/npu-2/${source_run_id}"
    "KVAF_CROSS_NPU2_REUSE_PRIOR_ACTIVE_FILE=${expected_experiment_root}/npu-2/${source_run_id}/direct-urma.active"
  )

  for variable in "${fixed_paths[@]}"; do
    expected=${variable#*=}
    variable=${variable%%=*}
    if declare -p "${variable}" >/dev/null 2>&1; then
      current=${!variable}
      if [[ "${current}" != "${expected}" ]]; then
        echo "ERROR: ${variable} cannot override the fixed NPU2 reuse root" >&2
        echo "EXPECTED=${expected}" >&2
        echo "ACTUAL=${current}" >&2
        return 1
      fi
    fi
    printf -v "${variable}" '%s' "${expected}"
  done

  # Existing components may be real directories/files, but no component of
  # either the immutable evidence lane or the new generation may be a link.
  protected_paths=(
    "${expected_global_state}"
    "${expected_global_state}/KVAF"
    "${expected_experiment_root}"
    "${expected_experiment_root}/npu-2"
    "${expected_experiment_root}/npu-2/${source_run_id}"
    "${expected_reuse_root}"
    "${expected_reuse_root}/${source_run_id}"
    "${expected_reuse_root}/${source_run_id}/${prior_target_run_id}"
    "${expected_direct_state}"
  )
  for path in "${protected_paths[@]}"; do
    if [[ -L "${path}" || ( -e "${path}" && ! -d "${path}" ) ]]; then
      echo "ERROR: NPU2 reuse state path is not a real directory" >&2
      echo "PATH=${path}" >&2
      return 1
    fi
  done
}
