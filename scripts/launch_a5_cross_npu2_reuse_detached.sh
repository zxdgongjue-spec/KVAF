#!/usr/bin/env bash

# Short, host-side operator entry for the single second-order NPU2 reuse
# experiment.  The experiment source is a reviewed detached commit containing
# an external expected-commit guard; this newer wrapper only makes launching
# and observing it reliable when an SSH client cannot paste multiline commands.

set -Ee
set -o pipefail
umask 077

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
LAUNCHER_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd -P)

KVAF_CONTAINER=vllm-ascend-cs-kvaf
KVAF_STATE_ROOT=/data/.kvaf-state/KVAF
KVAF_EXPERIMENT_COMMIT=cc3cc0e8e08bf4882c290dc3ed0a5b40a45312ce
KVAF_EXPERIMENT_WORKTREE="/data/c00982601/KVAF-cross-npu2-${KVAF_EXPERIMENT_COMMIT}"
KVAF_SOURCE_RUN_ID=530f5998a0b646d48db858170a9996ff
KVAF_PRIOR_TARGET_RUN_ID=30ef7336d212459a9317fd6647f6fa15
KVAF_ARTIFACT_TOKEN=0000000000000002
KVAF_RISK_PHRASE=I_ACCEPT_UNKNOWN_NPU2_REUSE_AND_SHARED_UMMU_RISK
KVAF_CLAIM="${KVAF_STATE_ROOT}/experimental/npu-2-reuse/${KVAF_SOURCE_RUN_ID}/${KVAF_PRIOR_TARGET_RUN_ID}/attempt.selected"
KVAF_RESERVATION="${KVAF_STATE_ROOT}/npu2-reuse-detached-launch.selected"
KVAF_LOG="${KVAF_STATE_ROOT}/npu2-reuse-detached.console.log"
KVAF_WORKER_RC="${KVAF_STATE_ROOT}/npu2-reuse-detached-worker.rc"
KVAF_DISPATCH_UNPROVEN="${KVAF_STATE_ROOT}/npu2-reuse-detached-dispatch.unproven"
KVAF_WORKER_SELECTED="${KVAF_STATE_ROOT}/npu2-reuse-detached-worker.selected"
KVAF_WORKER_BOOTSTRAP_UNPROVEN="${KVAF_STATE_ROOT}/npu2-reuse-detached-worker-bootstrap.unproven"

usage()
{
  cat >&2 <<'EOF'
Usage:
  launch_a5_cross_npu2_reuse_detached.sh start --accept-npu2-reuse-risk
  launch_a5_cross_npu2_reuse_detached.sh status
  launch_a5_cross_npu2_reuse_detached.sh watch
EOF
  exit 2
}

die()
{
  echo "ERROR: $*" >&2
  exit 2
}

require_command()
{
  command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

require_host()
{
  [[ ! -e /.dockerenv ]] || die "run this command on server B host"
  (( EUID == 0 )) || die "run this command as root"
  for kvaf_command in awk cat date docker git grep head ln mktemp realpath \
    sha256sum stat tail tr; do
    require_command "${kvaf_command}"
  done
}

require_canonical_state_root()
{
  [[ ! -L "${KVAF_STATE_ROOT}" && -d "${KVAF_STATE_ROOT}" &&
     -w "${KVAF_STATE_ROOT}" ]] ||
    die "state root is missing, unsafe, or unwritable: ${KVAF_STATE_ROOT}"
  [[ "$(realpath -e "${KVAF_STATE_ROOT}")" == "${KVAF_STATE_ROOT}" ]] ||
    die "state root is not canonical"
}

require_clean_git_tree()
{
  local root=$1
  if ! git -C "${root}" diff-index --quiet HEAD -- ||
     [[ -n "$(git -C "${root}" ls-files --others --exclude-standard)" ]]; then
    git -C "${root}" status --short >&2 || true
    die "git checkout is not clean: ${root}"
  fi
}

require_launcher_checkout()
{
  KVAF_LAUNCHER_COMMIT=$(git -C "${LAUNCHER_ROOT}" rev-parse HEAD) ||
    die "cannot identify the launcher checkout"
  [[ "${KVAF_LAUNCHER_COMMIT}" =~ ^[0-9a-f]{40}$ ]] ||
    die "launcher commit is not canonical"
  [[ "$(git -C "${LAUNCHER_ROOT}" rev-parse --abbrev-ref HEAD)" == A5_AIV_URMA_cs ]] ||
    die "launcher checkout is not A5_AIV_URMA_cs"
  require_clean_git_tree "${LAUNCHER_ROOT}"
}

prepare_pinned_worktree()
{
  git -C "${LAUNCHER_ROOT}" cat-file -e \
    "${KVAF_EXPERIMENT_COMMIT}^{commit}" ||
    die "reviewed experiment commit is unavailable"
  if [[ ! -e "${KVAF_EXPERIMENT_WORKTREE}" &&
        ! -L "${KVAF_EXPERIMENT_WORKTREE}" ]]; then
    git -C "${LAUNCHER_ROOT}" worktree add --detach \
      "${KVAF_EXPERIMENT_WORKTREE}" "${KVAF_EXPERIMENT_COMMIT}" ||
      die "could not create the pinned experiment worktree"
  fi
  [[ ! -L "${KVAF_EXPERIMENT_WORKTREE}" &&
     -d "${KVAF_EXPERIMENT_WORKTREE}" ]] ||
    die "pinned experiment worktree is missing or unsafe"
  [[ "$(realpath -e "${KVAF_EXPERIMENT_WORKTREE}")" == "${KVAF_EXPERIMENT_WORKTREE}" ]] ||
    die "pinned experiment worktree is not canonical"
  [[ "$(git -C "${KVAF_EXPERIMENT_WORKTREE}" rev-parse HEAD)" == "${KVAF_EXPERIMENT_COMMIT}" ]] ||
    die "pinned experiment worktree has the wrong commit"
  require_clean_git_tree "${KVAF_EXPERIMENT_WORKTREE}"
}

publish_file_once()
{
  local path=$1
  local mode=$2
  local content=$3
  local parent temporary

  parent=$(dirname "${path}")
  [[ "$(realpath -e "${parent}")" == "${parent}" ]] || return 1
  [[ ! -e "${path}" && ! -L "${path}" ]] || return 1
  temporary=$(mktemp "${parent}/.$(basename "${path}").XXXXXX") || return 1
  if ! printf '%s\n' "${content}" >"${temporary}" ||
     ! chmod "${mode}" "${temporary}" ||
     ! ln "${temporary}" "${path}"; then
    rm -f "${temporary}"
    return 1
  fi
  rm -f "${temporary}"
  [[ "$(stat -c '%h:%a' "${path}")" == "1:${mode}" ]]
}

reservation_content()
{
  printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-detached-launch-v2' \
    "source_run_id=${KVAF_SOURCE_RUN_ID}" \
    "prior_target_run_id=${KVAF_PRIOR_TARGET_RUN_ID}" \
    "experiment_commit=${KVAF_EXPERIMENT_COMMIT}" \
    "launcher_commit=${KVAF_LAUNCHER_COMMIT}" \
    "launch_nonce=${KVAF_LAUNCH_NONCE}" \
    "artifact_token=${KVAF_ARTIFACT_TOKEN}" \
    "experiment_worktree=${KVAF_EXPERIMENT_WORKTREE}" \
    "log=${KVAF_LOG}" \
    "worker_rc=${KVAF_WORKER_RC}" \
    "dispatch_unproven=${KVAF_DISPATCH_UNPROVEN}" \
    "worker_selected=${KVAF_WORKER_SELECTED}" \
    "worker_bootstrap_unproven=${KVAF_WORKER_BOOTSTRAP_UNPROVEN}" \
    'execution_contract=risk-accepted-cross-npu2-reuse-v2'
}

log_header()
{
  printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-detached-log-v2' \
    "source_run_id=${KVAF_SOURCE_RUN_ID}" \
    "prior_target_run_id=${KVAF_PRIOR_TARGET_RUN_ID}" \
    "experiment_commit=${KVAF_EXPERIMENT_COMMIT}" \
    "launcher_commit=${KVAF_LAUNCHER_COMMIT}" \
    "launch_nonce=${KVAF_LAUNCH_NONCE}" \
    "artifact_token=${KVAF_ARTIFACT_TOKEN}" \
    'stage=dispatching'
}

require_regular_identity()
{
  local path=$1
  local mode=$2
  [[ ! -L "${path}" && -f "${path}" && -r "${path}" ]] || return 1
  [[ "$(stat -c '%h:%a' "${path}")" == "1:${mode}" ]]
}

read_reservation()
{
  require_regular_identity "${KVAF_RESERVATION}" 400 ||
    die "detached launch reservation is missing or unsafe"
  KVAF_RECORDED_RESERVATION=$(cat "${KVAF_RESERVATION}") ||
    die "cannot read detached launch reservation"
  KVAF_RECORDED_LAUNCHER_COMMIT=$(awk -F= '
    $1 == "launcher_commit" { value=$2; count++ }
    END { if (count != 1) exit 1; print value }
  ' "${KVAF_RESERVATION}") || die "reservation launcher identity is malformed"
  [[ "${KVAF_RECORDED_LAUNCHER_COMMIT}" =~ ^[0-9a-f]{40}$ ]] ||
    die "reservation launcher commit is malformed"
  KVAF_RECORDED_LAUNCH_NONCE=$(awk -F= '
    $1 == "launch_nonce" { value=$2; count++ }
    END { if (count != 1) exit 1; print value }
  ' "${KVAF_RESERVATION}") || die "reservation launch nonce is malformed"
  [[ "${KVAF_RECORDED_LAUNCH_NONCE}" =~ ^[0-9a-f]{32}$ ]] ||
    die "reservation launch nonce is malformed"
  KVAF_LAUNCHER_COMMIT=${KVAF_RECORDED_LAUNCHER_COMMIT}
  KVAF_LAUNCH_NONCE=${KVAF_RECORDED_LAUNCH_NONCE}
  kvaf_expected_reservation_sha=$(reservation_content | sha256sum | awk '{print $1}')
  kvaf_actual_reservation_sha=$(sha256sum "${KVAF_RESERVATION}" | awk '{print $1}')
  [[ "${kvaf_actual_reservation_sha}" == "${kvaf_expected_reservation_sha}" ]] ||
    die "detached launch reservation content changed"
}

require_log_identity()
{
  require_regular_identity "${KVAF_LOG}" 600 ||
    die "detached launch log is missing or unsafe"
  kvaf_expected_log_header_sha=$(log_header | sha256sum | awk '{print $1}')
  kvaf_actual_log_header_sha=$(head -n 8 "${KVAF_LOG}" | sha256sum | awk '{print $1}')
  [[ "${kvaf_actual_log_header_sha}" == "${kvaf_expected_log_header_sha}" ]] ||
    die "detached launch log identity changed"
}

worker_selection_content()
{
  printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-detached-worker-v1' \
    "experiment_commit=${KVAF_EXPERIMENT_COMMIT}" \
    "launcher_commit=${KVAF_LAUNCHER_COMMIT}" \
    "launch_nonce=${KVAF_LAUNCH_NONCE}" \
    "artifact_token=${KVAF_ARTIFACT_TOKEN}"
}

require_worker_selection_identity()
{
  require_regular_identity "${KVAF_WORKER_SELECTED}" 400 || return 1
  kvaf_expected_worker_sha=$(worker_selection_content | sha256sum | awk '{print $1}')
  kvaf_actual_worker_sha=$(sha256sum "${KVAF_WORKER_SELECTED}" | awk '{print $1}')
  [[ "${kvaf_actual_worker_sha}" == "${kvaf_expected_worker_sha}" ]]
}

run_read_only_preflight()
{
  echo "KVAF_DETACHED_STAGE=read-only-preflight"
  docker exec -u 0 \
    -e USER=root \
    -e "KVAF_DETACHED_EXPECTED_LAUNCHER_ROOT=${LAUNCHER_ROOT}" \
    -e "KVAF_DETACHED_EXPECTED_LAUNCHER_COMMIT=${KVAF_LAUNCHER_COMMIT}" \
    -e "KVAF_DETACHED_EXPECTED_WORKTREE=${KVAF_EXPERIMENT_WORKTREE}" \
    -e "KVAF_DETACHED_EXPECTED_EXPERIMENT_COMMIT=${KVAF_EXPERIMENT_COMMIT}" \
    -e "KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN=${KVAF_ARTIFACT_TOKEN}" \
    -e "KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE=${KVAF_RISK_PHRASE}" \
    -w "${LAUNCHER_ROOT}" \
    "${KVAF_CONTAINER}" \
    bash --noprofile --norc -c '
set -Ee
set -o pipefail
test "$(pwd -P)" = "${KVAF_DETACHED_EXPECTED_LAUNCHER_ROOT}"
test "$(git rev-parse HEAD)" = "${KVAF_DETACHED_EXPECTED_LAUNCHER_COMMIT}"
git diff-index --quiet HEAD --
test -z "$(git ls-files --others --exclude-standard)"
cd "${KVAF_DETACHED_EXPECTED_WORKTREE}"
test "$(pwd -P)" = "${KVAF_DETACHED_EXPECTED_WORKTREE}"
test "$(git rev-parse HEAD)" = "${KVAF_DETACHED_EXPECTED_EXPERIMENT_COMMIT}"
git diff-index --quiet HEAD --
test -z "$(git ls-files --others --exclude-standard)"
kvaf_evidence=$(python3 scripts/validate_a5_cross_npu2_reuse_risk.py --new-artifact-token "${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}")
printf "%s\n" "${kvaf_evidence}"
kvaf_inventory=$(timeout 15s npu-smi info)
kvaf_idle=$(python3 scripts/check_a5_npu_idle.py --physical-device-id 2 <<<"${kvaf_inventory}")
printf "%s\n" "${kvaf_idle}"
grep -Fxq KVAF_NPU_IDLE_STATUS=idle <<<"${kvaf_idle}"
echo KVAF_NPU2_DETACHED_PREFLIGHT_OK
'
}

reserve_launch_generation()
{
  [[ ! -e "${KVAF_RESERVATION}" && ! -L "${KVAF_RESERVATION}" ]] ||
    die "a detached launch generation was already reserved"
  [[ ! -e "${KVAF_LOG}" && ! -L "${KVAF_LOG}" ]] ||
    die "detached launch log already exists"
  [[ ! -e "${KVAF_WORKER_RC}" && ! -L "${KVAF_WORKER_RC}" ]] ||
    die "detached worker result already exists"
  [[ ! -e "${KVAF_DISPATCH_UNPROVEN}" &&
     ! -L "${KVAF_DISPATCH_UNPROVEN}" ]] ||
    die "detached dispatch evidence already exists"
  [[ ! -e "${KVAF_WORKER_SELECTED}" &&
     ! -L "${KVAF_WORKER_SELECTED}" ]] ||
    die "detached worker selection already exists"
  [[ ! -e "${KVAF_WORKER_BOOTSTRAP_UNPROVEN}" &&
     ! -L "${KVAF_WORKER_BOOTSTRAP_UNPROVEN}" ]] ||
    die "detached worker bootstrap evidence already exists"

  publish_file_once "${KVAF_LOG}" 600 "$(log_header)" ||
    die "could not reserve the detached log"
  publish_file_once "${KVAF_RESERVATION}" 400 "$(reservation_content)" ||
    die "could not atomically reserve the detached launch generation"
}

record_dispatch_unproven()
{
  local rc=$1
  local evidence
  evidence=$(printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-dispatch-unproven-v1' \
    "docker_cli_rc=${rc}" \
    'coordinator_completion=unproven')
  publish_file_once "${KVAF_DISPATCH_UNPROVEN}" 400 "${evidence}" || true
  printf '%s\n' "DISPATCH_UNPROVEN=docker_cli_rc_${rc}" >>"${KVAF_LOG}" || true
}

start_experiment()
{
  (( $# == 1 )) && [[ "$1" == --accept-npu2-reuse-risk ]] || usage
  require_host
  require_canonical_state_root
  require_launcher_checkout
  prepare_pinned_worktree
  [[ -r /proc/sys/kernel/random/uuid ]] ||
    die "cannot generate the detached launch nonce"
  KVAF_LAUNCH_NONCE=$(tr -d '-' </proc/sys/kernel/random/uuid)
  [[ "${KVAF_LAUNCH_NONCE}" =~ ^[0-9a-f]{32}$ ]] ||
    die "detached launch nonce is malformed"

  [[ ! -e "${KVAF_CLAIM}" && ! -L "${KVAF_CLAIM}" ]] || {
    echo "ATTEMPT_CLAIM=${KVAF_CLAIM}" >&2
    if [[ ! -L "${KVAF_CLAIM}" && -f "${KVAF_CLAIM}" &&
          -r "${KVAF_CLAIM}" ]]; then
      cat "${KVAF_CLAIM}" >&2
    fi
    die "the one permitted second-order NPU2 attempt was already selected"
  }
  [[ "$(docker inspect -f '{{.State.Status}}' "${KVAF_CONTAINER}")" == running ]] ||
    die "${KVAF_CONTAINER} is not running"

  run_read_only_preflight
  reserve_launch_generation
  if docker exec -d -u 0 \
      -e USER=root \
      -e "KVAF_DETACHED_LAUNCHER_COMMIT=${KVAF_LAUNCHER_COMMIT}" \
      -e "KVAF_DETACHED_LAUNCH_NONCE=${KVAF_LAUNCH_NONCE}" \
      -e "KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN=${KVAF_ARTIFACT_TOKEN}" \
      -e "KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE=${KVAF_RISK_PHRASE}" \
      -w "${LAUNCHER_ROOT}" \
      "${KVAF_CONTAINER}" \
      bash --noprofile --norc \
        "${SCRIPT_DIR}/launch_a5_cross_npu2_reuse_detached.sh" worker; then
    :
  else
    kvaf_dispatch_rc=$?
    record_dispatch_unproven "${kvaf_dispatch_rc}"
    die "Docker dispatch is unproven; do not retry"
  fi

  echo "KVAF_NPU2_REUSE_LAUNCHED_DETACHED"
  echo "EXPERIMENT_COMMIT=${KVAF_EXPERIMENT_COMMIT}"
  echo "LAUNCHER_COMMIT=${KVAF_LAUNCHER_COMMIT}"
  echo "ARTIFACT_TOKEN=${KVAF_ARTIFACT_TOKEN}"
  echo "LOG=${KVAF_LOG}"
  echo "WORKER_RC=${KVAF_WORKER_RC}"
  echo "STATUS_COMMAND=bash scripts/launch_a5_cross_npu2_reuse_detached.sh status"
  echo "WATCH_COMMAND=bash scripts/launch_a5_cross_npu2_reuse_detached.sh watch"
}

publish_worker_rc()
{
  local rc=$1
  local content
  if (( rc < 0 || rc > 255 )); then
    rc=255
  fi
  content=${rc}
  if ! publish_file_once "${KVAF_WORKER_RC}" 400 "${content}"; then
    printf '%s\n' 'ERROR: could not atomically publish detached worker rc' \
      >>"${KVAF_LOG}" 2>/dev/null || true
  fi
}

publish_worker_bootstrap_unproven()
{
  local rc=$1
  local evidence
  evidence=$(printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-worker-bootstrap-unproven-v1' \
    "worker_shell_rc=${rc}" \
    'coordinator_completion=unproven')
  publish_file_once "${KVAF_WORKER_BOOTSTRAP_UNPROVEN}" 400 \
    "${evidence}" || true
}

record_worker_exit()
{
  kvaf_worker_rc=$?
  if [[ "${KVAF_WORKER_OWNS_GENERATION:-0}" == 1 ]]; then
    publish_worker_rc "${kvaf_worker_rc}"
  elif [[ ! -e "${KVAF_WORKER_SELECTED}" &&
          ! -L "${KVAF_WORKER_SELECTED}" ]]; then
    publish_worker_bootstrap_unproven "${kvaf_worker_rc}"
  fi
}

run_worker()
{
  (( $# == 0 )) || usage

  # Reject accidental/manual worker invocation before it can publish a result
  # into the globally reserved generation.  The host launcher supplies all
  # three values in the same docker exec request that selects this mode.
  [[ -e /.dockerenv ]] || die "worker mode is container-only"
  (( EUID == 0 )) || die "worker mode requires uid 0"
  [[ "${KVAF_DETACHED_LAUNCHER_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] ||
    die "detached worker authorization is missing"
  [[ "${KVAF_DETACHED_LAUNCH_NONCE:-}" =~ ^[0-9a-f]{32}$ ]] ||
    die "detached worker nonce is missing"
  [[ "${KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE:-}" == "${KVAF_RISK_PHRASE}" ]] ||
    die "worker risk identity is missing"
  [[ "${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN:-}" == "${KVAF_ARTIFACT_TOKEN}" ]] ||
    die "worker artifact token is not pinned"

  # The paths used by the EXIT recorder are compile-time constants in this
  # tracked wrapper, not operator-provided environment variables.
  KVAF_WORKER_OWNS_GENERATION=0
  trap record_worker_exit EXIT
  require_canonical_state_root
  read_reservation
  require_log_identity
  [[ "${KVAF_DETACHED_LAUNCHER_COMMIT:-}" == "${KVAF_RECORDED_LAUNCHER_COMMIT}" ]] ||
    die "launcher commit changed before worker start"
  [[ "${KVAF_DETACHED_LAUNCH_NONCE:-}" == "${KVAF_RECORDED_LAUNCH_NONCE}" ]] ||
    die "launcher nonce changed before worker start"
  [[ ! -e "${KVAF_WORKER_RC}" && ! -L "${KVAF_WORKER_RC}" &&
     ! -e "${KVAF_DISPATCH_UNPROVEN}" &&
     ! -L "${KVAF_DISPATCH_UNPROVEN}" ]] ||
    die "worker generation already has terminal or dispatch evidence"
  publish_file_once "${KVAF_WORKER_SELECTED}" 400 \
    "$(worker_selection_content)" ||
    die "another detached worker already selected this generation"
  KVAF_WORKER_OWNS_GENERATION=1
  exec >>"${KVAF_LOG}" 2>&1

  echo "stage=container-worker-started"
  echo "worker_pid=$$"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  [[ "$(git -C "${LAUNCHER_ROOT}" rev-parse HEAD)" == "${KVAF_RECORDED_LAUNCHER_COMMIT}" ]] ||
    die "launcher checkout moved before worker start"
  require_clean_git_tree "${LAUNCHER_ROOT}"
  prepare_pinned_worktree

  export USER=${USER:-root}
  KVAF_ASCEND_SET_ENV=/usr/local/Ascend/ascend-toolkit/set_env.sh
  [[ ! -L "${KVAF_ASCEND_SET_ENV}" && -f "${KVAF_ASCEND_SET_ENV}" &&
     -r "${KVAF_ASCEND_SET_ENV}" ]] || die "CANN set_env.sh is missing"
  source "${KVAF_ASCEND_SET_ENV}"
  if [[ -r /data/c00982601/asc-devkit-9.1.0-20260722/include/simt_api/asc_simt.h ]]; then
    export KVAF_ASC_DEVKIT_PATH=/data/c00982601/asc-devkit-9.1.0-20260722
  elif [[ -r /data/c00982601/asc-devkit/include/simt_api/asc_simt.h ]]; then
    export KVAF_ASC_DEVKIT_PATH=/data/c00982601/asc-devkit
  else
    die "AscendC SIMT headers are missing"
  fi
  export KVAF_UMDK_ROOT=/data/c00982601/umdk
  export KVAF_LIBUMMU_ROOT=/data/c00982601/libummu
  export KVAF_CROSS_NPU2_REUSE_EXPECTED_SOURCE_COMMIT=${KVAF_EXPERIMENT_COMMIT}

  cd "${KVAF_EXPERIMENT_WORKTREE}"
  [[ "$(git rev-parse HEAD)" == "${KVAF_EXPERIMENT_COMMIT}" ]] ||
    die "pinned experiment commit changed immediately before coordinator start"
  require_clean_git_tree "${KVAF_EXPERIMENT_WORKTREE}"
  bash scripts/run_a5_cross_npu2_reuse_experiment.sh zero-wqe
}

read_canonical_worker_rc()
{
  require_regular_identity "${KVAF_WORKER_RC}" 400 || return 1
  mapfile -t kvaf_rc_lines <"${KVAF_WORKER_RC}" || return 1
  (( ${#kvaf_rc_lines[@]} == 1 )) || return 1
  [[ "${kvaf_rc_lines[0]}" =~ ^(0|[1-9][0-9]{0,2})$ ]] || return 1
  (( 10#${kvaf_rc_lines[0]} <= 255 )) || return 1
  KVAF_RECORDED_WORKER_RC=${kvaf_rc_lines[0]}
}

require_dispatch_unproven_identity()
{
  require_regular_identity "${KVAF_DISPATCH_UNPROVEN}" 400 || return 1
  mapfile -t kvaf_dispatch_lines <"${KVAF_DISPATCH_UNPROVEN}" || return 1
  (( ${#kvaf_dispatch_lines[@]} == 3 )) || return 1
  [[ "${kvaf_dispatch_lines[0]}" == schema=kvaf-a5-cross-npu2-dispatch-unproven-v1 ]] ||
    return 1
  [[ "${kvaf_dispatch_lines[1]}" =~ ^docker_cli_rc=(0|[1-9][0-9]{0,2})$ ]] ||
    return 1
  [[ "${kvaf_dispatch_lines[2]}" == coordinator_completion=unproven ]]
}

require_bootstrap_unproven_identity()
{
  require_regular_identity "${KVAF_WORKER_BOOTSTRAP_UNPROVEN}" 400 || return 1
  mapfile -t kvaf_bootstrap_lines <"${KVAF_WORKER_BOOTSTRAP_UNPROVEN}" ||
    return 1
  (( ${#kvaf_bootstrap_lines[@]} == 3 )) || return 1
  [[ "${kvaf_bootstrap_lines[0]}" == schema=kvaf-a5-cross-npu2-worker-bootstrap-unproven-v1 ]] ||
    return 1
  [[ "${kvaf_bootstrap_lines[1]}" =~ ^worker_shell_rc=(0|[1-9][0-9]{0,2})$ ]] ||
    return 1
  [[ "${kvaf_bootstrap_lines[2]}" == coordinator_completion=unproven ]]
}

print_key_log_evidence()
{
  grep -E \
    '^(STAGE=|KVAF_CROSS_NPU2_REUSE_ATTEMPT_SELECTED=|KVAF_A5_AICPU_OWNER_PROBE_OK|KVAF_NPU2_REUSE_ZERO_WQE_RC=|KVAF_NPU2_REUSE_ZERO_WQE_ACCEPTANCE=|HARD_STOP:|ERROR:)' \
    "${KVAF_LOG}" || true
  echo "LOG_TAIL:"
  tail -n 40 "${KVAF_LOG}" || true
}

show_status()
{
  (( $# == 0 )) || usage
  require_host
  require_canonical_state_root
  if [[ ! -e "${KVAF_RESERVATION}" && ! -L "${KVAF_RESERVATION}" ]]; then
    for kvaf_partial in "${KVAF_LOG}" "${KVAF_WORKER_RC}" \
      "${KVAF_DISPATCH_UNPROVEN}" "${KVAF_WORKER_SELECTED}" \
      "${KVAF_WORKER_BOOTSTRAP_UNPROVEN}"; do
      if [[ -e "${kvaf_partial}" || -L "${kvaf_partial}" ]]; then
        echo "KVAF_DETACHED_STATUS=partial-launch-unproven"
        echo "PARTIAL_PATH=${kvaf_partial}"
        return 5
      fi
    done
    echo "KVAF_DETACHED_STATUS=not-started"
    return 1
  fi
  read_reservation
  require_log_identity
  echo "EXPERIMENT_COMMIT=${KVAF_EXPERIMENT_COMMIT}"
  echo "LAUNCHER_COMMIT=${KVAF_RECORDED_LAUNCHER_COMMIT}"
  echo "ARTIFACT_TOKEN=${KVAF_ARTIFACT_TOKEN}"
  echo "LOG=${KVAF_LOG}"
  echo "WORKER_RC=${KVAF_WORKER_RC}"

  if [[ -e "${KVAF_DISPATCH_UNPROVEN}" ||
        -L "${KVAF_DISPATCH_UNPROVEN}" ]]; then
    require_dispatch_unproven_identity || die "dispatch evidence is unsafe"
    if [[ -e "${KVAF_WORKER_RC}" || -L "${KVAF_WORKER_RC}" ]]; then
      read_canonical_worker_rc || die "detached worker rc is malformed or unsafe"
      echo "KVAF_DETACHED_STATUS=finished-after-dispatch-unproven"
      echo "KVAF_COORDINATOR_RC=${KVAF_RECORDED_WORKER_RC}"
    else
      echo "KVAF_DETACHED_STATUS=dispatch-unproven"
    fi
    cat "${KVAF_DISPATCH_UNPROVEN}"
    print_key_log_evidence
    return 5
  fi
  if [[ -e "${KVAF_WORKER_BOOTSTRAP_UNPROVEN}" ||
        -L "${KVAF_WORKER_BOOTSTRAP_UNPROVEN}" ]]; then
    require_bootstrap_unproven_identity ||
      die "worker bootstrap evidence is unsafe"
    echo "KVAF_DETACHED_STATUS=worker-bootstrap-unproven"
    cat "${KVAF_WORKER_BOOTSTRAP_UNPROVEN}"
    print_key_log_evidence
    return 5
  fi
  if [[ -e "${KVAF_WORKER_SELECTED}" || -L "${KVAF_WORKER_SELECTED}" ]]; then
    require_worker_selection_identity || die "worker selection is unsafe"
  else
    echo "KVAF_DETACHED_STATUS=dispatched-worker-not-selected"
    print_key_log_evidence
    return 5
  fi
  if [[ -e "${KVAF_WORKER_RC}" || -L "${KVAF_WORKER_RC}" ]]; then
    read_canonical_worker_rc || die "detached worker rc is malformed or unsafe"
    echo "KVAF_DETACHED_STATUS=finished"
    echo "KVAF_COORDINATOR_RC=${KVAF_RECORDED_WORKER_RC}"
    kvaf_status_rc=${KVAF_RECORDED_WORKER_RC}
    if (( kvaf_status_rc == 0 )); then
      if ! grep -Fxq 'KVAF_NPU2_REUSE_ZERO_WQE_RC=0' "${KVAF_LOG}" ||
         ! grep -Fxq 'KVAF_NPU2_REUSE_ZERO_WQE_ACCEPTANCE=PASS' "${KVAF_LOG}"; then
        echo "KVAF_DETACHED_STATUS=finished-inconsistent"
        print_key_log_evidence
        return 5
      fi
    fi
  else
    echo "KVAF_DETACHED_STATUS=running-or-unproven"
    print_key_log_evidence
    return 5
  fi
  print_key_log_evidence
  return "${kvaf_status_rc}"
}

watch_status()
{
  (( $# == 0 )) || usage
  require_host
  require_canonical_state_root
  read_reservation
  require_log_identity
  echo "Monitoring only; Ctrl+C does not signal the detached worker."
  while [[ ! -e "${KVAF_WORKER_RC}" &&
           ! -L "${KVAF_WORKER_RC}" &&
           ! -e "${KVAF_DISPATCH_UNPROVEN}" &&
           ! -L "${KVAF_DISPATCH_UNPROVEN}" &&
           ! -e "${KVAF_WORKER_BOOTSTRAP_UNPROVEN}" &&
           ! -L "${KVAF_WORKER_BOOTSTRAP_UNPROVEN}" ]]; do
    date '+WAITING %F %T'
    tail -n 3 "${KVAF_LOG}" || true
    sleep 15
  done
  show_status
}

(( $# >= 1 )) || usage
kvaf_mode=$1
shift
case "${kvaf_mode}" in
  start) start_experiment "$@" ;;
  worker) run_worker "$@" ;;
  status) show_status "$@" ;;
  watch) watch_status "$@" ;;
  *) usage ;;
esac
