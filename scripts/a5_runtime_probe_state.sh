#!/usr/bin/env bash

# Pure parsing helpers shared by the guarded AICPU runtime-URMA runner and its
# host-only tests.  This file intentionally performs no I/O at source time.

kvaf_runtime_probe_parse_proc_stat_line()
{
  local kvaf_proc_line=$1
  local kvaf_proc_tail kvaf_proc_state
  local kvaf_proc_ppid kvaf_proc_pgrp kvaf_proc_session kvaf_proc_tty_nr
  local kvaf_proc_tpgid kvaf_proc_flags kvaf_proc_minflt kvaf_proc_cminflt
  local kvaf_proc_majflt kvaf_proc_cmajflt kvaf_proc_utime kvaf_proc_stime
  local kvaf_proc_cutime kvaf_proc_cstime kvaf_proc_priority kvaf_proc_nice
  local kvaf_proc_threads kvaf_proc_itrealvalue kvaf_proc_start_ticks
  local kvaf_proc_remainder

  # The comm field is parenthesized and may itself contain spaces or closing
  # parentheses.  Strip through the final ") " before counting fields.
  [[ ${kvaf_proc_line} == *') '* ]] || return 1
  kvaf_proc_tail=${kvaf_proc_line##*) }
  read -r kvaf_proc_state kvaf_proc_ppid kvaf_proc_pgrp \
    kvaf_proc_session kvaf_proc_tty_nr kvaf_proc_tpgid kvaf_proc_flags \
    kvaf_proc_minflt kvaf_proc_cminflt kvaf_proc_majflt kvaf_proc_cmajflt \
    kvaf_proc_utime kvaf_proc_stime kvaf_proc_cutime kvaf_proc_cstime \
    kvaf_proc_priority kvaf_proc_nice kvaf_proc_threads \
    kvaf_proc_itrealvalue kvaf_proc_start_ticks kvaf_proc_remainder \
    <<<"${kvaf_proc_tail}" || return 1
  [[ ${kvaf_proc_state} =~ ^[A-Za-z]$ &&
     ${kvaf_proc_start_ticks} =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s %s\n' "${kvaf_proc_state}" "${kvaf_proc_start_ticks}"
}

kvaf_runtime_probe_read_proc_state_and_start_ticks()
{
  local kvaf_proc_pid=$1
  local kvaf_proc_root=${2:-/proc}
  local kvaf_proc_line

  [[ ${kvaf_proc_pid} =~ ^[1-9][0-9]*$ &&
     -r ${kvaf_proc_root}/${kvaf_proc_pid}/stat ]] || return 1
  IFS= read -r kvaf_proc_line \
    <"${kvaf_proc_root}/${kvaf_proc_pid}/stat" || return 1
  kvaf_runtime_probe_parse_proc_stat_line "${kvaf_proc_line}"
}

kvaf_runtime_probe_read_single_value()
{
  local kvaf_state_file=$1
  local kvaf_state_key=$2
  local kvaf_state_value

  kvaf_state_value=$(
    awk -F= -v wanted="${kvaf_state_key}" '
      $1 == wanted {
        sub(/^[^=]*=/, "")
        value = $0
        count++
      }
      END {
        if (count != 1) exit 1
        print value
      }
    ' "${kvaf_state_file}"
  ) || return 1
  printf '%s\n' "${kvaf_state_value}"
}

kvaf_runtime_probe_file_has_exact_keys()
{
  local kvaf_state_file=$1
  shift
  local kvaf_state_line kvaf_state_key kvaf_expected_key
  local kvaf_key_expected kvaf_observed_keys=' '
  local kvaf_expected_count=$# kvaf_observed_count=0

  for kvaf_expected_key in "$@"; do
    [[ ${kvaf_expected_key} =~ ^[a-z][a-z0-9_]*$ ]] || return 1
    [[ ${kvaf_observed_keys} != *" ${kvaf_expected_key} "* ]] || return 1
    kvaf_observed_keys+="${kvaf_expected_key} "
  done
  kvaf_observed_keys=' '
  while IFS= read -r kvaf_state_line || [[ -n ${kvaf_state_line} ]]; do
    [[ ${kvaf_state_line} == *=* ]] || return 1
    kvaf_state_key=${kvaf_state_line%%=*}
    kvaf_key_expected=0
    for kvaf_expected_key in "$@"; do
      if [[ ${kvaf_state_key} == "${kvaf_expected_key}" ]]; then
        kvaf_key_expected=1
        break
      fi
    done
    (( kvaf_key_expected == 1 )) || return 1
    [[ ${kvaf_observed_keys} != *" ${kvaf_state_key} "* ]] || return 1
    kvaf_observed_keys+="${kvaf_state_key} "
    (( kvaf_observed_count += 1 ))
  done <"${kvaf_state_file}" || return 1
  (( kvaf_observed_count == kvaf_expected_count ))
}

kvaf_runtime_probe_log_is_safe_terminal()
{
  local kvaf_terminal_rc=$1
  local kvaf_terminal_log=$2
  local kvaf_expected_mode=$3

  [[ ${kvaf_expected_mode} == discover ||
     ${kvaf_expected_mode} == init-device ||
     ${kvaf_expected_mode} == create-delete ]] || return 1
  [[ -f ${kvaf_terminal_log} && ! -L ${kvaf_terminal_log} ]] || return 1
  if [[ ${kvaf_terminal_rc} == 0 ]] &&
     grep -Fxq KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK \
       "${kvaf_terminal_log}"; then
    if [[ ${kvaf_expected_mode} == init-device ]] &&
       grep -Fxq KVAF_A5_RUNTIME_ACCEPTED_INIT_DEVICE \
         "${kvaf_terminal_log}" &&
       grep -Eq \
         '^KVAF_A5_RUNTIME_URMA_PROBE_RESULT mode=init-device .* cleanup_proven=1 ' \
         "${kvaf_terminal_log}"; then
      return 0
    fi
    if [[ ${kvaf_expected_mode} == create-delete ]] &&
       grep -Fxq KVAF_A5_RUNTIME_ACCEPTED_MINIMAL_CREATE_DELETE \
         "${kvaf_terminal_log}" &&
       grep -Eq \
         '^KVAF_A5_RUNTIME_URMA_PROBE_RESULT mode=create-delete .* cleanup_proven=1 ' \
         "${kvaf_terminal_log}"; then
      return 0
    fi
  fi
  if [[ ${kvaf_terminal_rc} == 1 ]] &&
     grep -Fxq \
       "KVAF_A5_RUNTIME_URMA_PRELAUNCH_SAFE mode=${kvaf_expected_mode}" \
       "${kvaf_terminal_log}"; then
    return 0
  fi
  if [[ ${kvaf_terminal_rc} == 3 &&
        ${kvaf_expected_mode} != discover ]] &&
     grep -Fxq KVAF_A5_RUNTIME_URMA_HOST_TEARDOWN_OK \
       "${kvaf_terminal_log}" &&
     grep -Fxq KVAF_A5_RUNTIME_URMA_PROBE_NON_ACCEPTED \
       "${kvaf_terminal_log}" &&
     grep -Eq \
       "^KVAF_A5_RUNTIME_URMA_PROBE_RESULT mode=${kvaf_expected_mode} .* cleanup_proven=1 " \
       "${kvaf_terminal_log}"; then
    return 0
  fi
  return 1
}

# Validate the sealed terminal record against persisted session identity.  The
# worker executable is intentionally not opened here: recovery must keep
# working after the checkout is rebuilt or the old binary is removed.
kvaf_runtime_probe_terminal_evidence_matches()
{
  local kvaf_terminal_file=$1
  local kvaf_expected_run_id=$2
  local kvaf_expected_mode=$3
  local kvaf_expected_boot_id=$4
  local kvaf_expected_logical_id=$5
  local kvaf_expected_physical_id=$6
  local kvaf_expected_worker_path=$7
  local kvaf_expected_worker_sha=$8
  local kvaf_expected_log=$9
  shift 9
  local kvaf_expected_supervisor_pid=$1
  local kvaf_expected_supervisor_start_ticks=$2
  local kvaf_expected_provider_soname=$3
  local kvaf_expected_provider_build_id=$4
  local kvaf_expected_provider_artifact_sha=$5
  local kvaf_schema kvaf_run_id kvaf_mode kvaf_boot_id
  local kvaf_logical_id kvaf_physical_id kvaf_worker_path kvaf_worker_sha
  local kvaf_log kvaf_log_sha kvaf_actual_log_sha kvaf_rc kvaf_reason
  local kvaf_terminal_contract kvaf_lock_contract kvaf_supervisor_pid
  local kvaf_supervisor_start_ticks
  local kvaf_provider_soname kvaf_provider_build_id
  local kvaf_provider_artifact_sha

  kvaf_runtime_probe_file_has_exact_keys "${kvaf_terminal_file}" \
    schema run_id mode boot_id logical_device_id physical_device_id \
    supervisor_pid supervisor_start_ticks worker_path worker_sha256 log \
    provider_soname provider_build_id_sha256 \
    expected_packaged_provider_artifact_sha256 \
    log_sha256 rc reason terminal_contract lock_contract \
    caller_domain_contract || return 1
  kvaf_schema=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" schema) || return 1
  kvaf_run_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" run_id) || return 1
  kvaf_mode=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" mode) || return 1
  kvaf_boot_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" boot_id) || return 1
  kvaf_logical_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" logical_device_id) || return 1
  kvaf_physical_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" physical_device_id) || return 1
  kvaf_supervisor_pid=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" supervisor_pid) || return 1
  kvaf_supervisor_start_ticks=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" supervisor_start_ticks) || return 1
  kvaf_worker_path=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" worker_path) || return 1
  kvaf_worker_sha=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" worker_sha256) || return 1
  kvaf_provider_soname=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" provider_soname) || return 1
  kvaf_provider_build_id=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" provider_build_id_sha256) || return 1
  kvaf_provider_artifact_sha=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" \
    expected_packaged_provider_artifact_sha256) || return 1
  kvaf_log=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" log) || return 1
  kvaf_log_sha=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" log_sha256) || return 1
  kvaf_rc=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" rc) || return 1
  kvaf_reason=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" reason) || return 1
  kvaf_terminal_contract=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" terminal_contract) || return 1
  kvaf_lock_contract=$(kvaf_runtime_probe_read_single_value \
    "${kvaf_terminal_file}" lock_contract) || return 1

  [[ ( ${kvaf_expected_mode} == discover ||
       ${kvaf_expected_mode} == init-device ||
       ${kvaf_expected_mode} == create-delete ) &&
     ${kvaf_schema} == kvaf-a5-runtime-urma-probe-terminal-v1 &&
     ${kvaf_run_id} == "${kvaf_expected_run_id}" &&
     ${kvaf_mode} == "${kvaf_expected_mode}" &&
     ${kvaf_boot_id} == "${kvaf_expected_boot_id}" &&
     ${kvaf_logical_id} == "${kvaf_expected_logical_id}" &&
     ${kvaf_physical_id} == "${kvaf_expected_physical_id}" &&
     ${kvaf_supervisor_pid} == "${kvaf_expected_supervisor_pid}" &&
     ${kvaf_supervisor_start_ticks} == \
       "${kvaf_expected_supervisor_start_ticks}" &&
     ${kvaf_worker_path} == "${kvaf_expected_worker_path}" &&
     ${kvaf_worker_sha} == "${kvaf_expected_worker_sha}" &&
     ${kvaf_provider_soname} == "${kvaf_expected_provider_soname}" &&
     ${kvaf_provider_build_id} == "${kvaf_expected_provider_build_id}" &&
     ${kvaf_provider_artifact_sha} == \
       "${kvaf_expected_provider_artifact_sha}" &&
     ${kvaf_provider_build_id} =~ ^[0-9a-f]{64}$ &&
     ${kvaf_provider_artifact_sha} =~ ^[0-9a-f]{64}$ &&
     ${kvaf_provider_soname} == \
       "libkvaf_rt_${kvaf_provider_build_id:0:16}.so" &&
     ${kvaf_log} == "${kvaf_expected_log}" &&
     ${kvaf_worker_sha} =~ ^[0-9a-f]{64}$ &&
     ${kvaf_log_sha} =~ ^[0-9a-f]{64}$ &&
     ${kvaf_rc} =~ ^(0|1|3|5)$ &&
     ( ${kvaf_reason} == worker_exited_log_synced ||
       ${kvaf_reason} == supervisor_prelaunch_failed_log_synced ) &&
     ${kvaf_terminal_contract} == runtime-probe-exact-terminal-v2 &&
     ${kvaf_lock_contract} == inherited-supervisor-worker-v4-held-flock-pinned-worker-fd &&
     $(kvaf_runtime_probe_read_single_value \
       "${kvaf_terminal_file}" caller_domain_contract) == \
       undocumented-aicpu-urma-abi-source-caller-cleanup-risk-accepted-v1 ]] || return 1
  if [[ ${kvaf_reason} == supervisor_prelaunch_failed_log_synced ]]; then
    [[ ${kvaf_rc} == 1 ]] || return 1
  fi
  [[ -f ${kvaf_log} && ! -L ${kvaf_log} ]] || return 1
  kvaf_actual_log_sha=$(sha256sum -- "${kvaf_log}" | awk '{print $1}') ||
    return 1
  [[ ${kvaf_actual_log_sha} == "${kvaf_log_sha}" ]]
}
