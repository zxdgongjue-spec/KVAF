#!/usr/bin/env bash

# Short, host-side operator entry for the single second-order NPU2 reuse
# experiment.  The experiment source is a reviewed detached commit containing
# an external expected-commit guard; this newer wrapper only makes launching
# and observing it reliable when an SSH client cannot paste multiline commands.

unset BASH_ENV ENV
unset -f exit builtin 2>/dev/null || true

echo "KVAF_A5_CROSS_NPU2_REUSE_DETACHED_COORDINATOR_DISABLED" >&2
echo "CAPABILITY_LEDGER=not-implemented" >&2
echo "No argument/environment/source/state/CANN/docker/NPU operation was attempted." >&2
builtin exit 2
if [[ 0 == 1 ]]; then

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
KVAF_KNOWN_CANN_SYMLINK_FAILURE_COMMIT=5615967b639f49e787221c317a95a0a1b4739015
KVAF_RECOVERY_RESERVATION="${KVAF_STATE_ROOT}/npu2-reuse-detached-cann-recovery.selected"
KVAF_RECOVERY_LOG="${KVAF_STATE_ROOT}/npu2-reuse-detached-cann-recovery.console.log"
KVAF_RECOVERY_WORKER_RC="${KVAF_STATE_ROOT}/npu2-reuse-detached-cann-recovery-worker.rc"
KVAF_RECOVERY_DISPATCH_UNPROVEN="${KVAF_STATE_ROOT}/npu2-reuse-detached-cann-recovery-dispatch.unproven"
KVAF_RECOVERY_WORKER_SELECTED="${KVAF_STATE_ROOT}/npu2-reuse-detached-cann-recovery-worker.selected"
KVAF_RECOVERY_BOOTSTRAP_UNPROVEN="${KVAF_STATE_ROOT}/npu2-reuse-detached-cann-recovery-worker-bootstrap.unproven"
KVAF_ARTIFACT_ROOT="${KVAF_STATE_ROOT}/experimental/reuse-artifacts/${KVAF_EXPERIMENT_COMMIT}/${KVAF_ARTIFACT_TOKEN}"
KVAF_REUSE_STATE_DIR="${KVAF_STATE_ROOT}/experimental/npu-2-reuse/${KVAF_SOURCE_RUN_ID}/${KVAF_PRIOR_TARGET_RUN_ID}/${KVAF_ARTIFACT_TOKEN}"

usage()
{
  cat >&2 <<'EOF'
Usage:
  launch_a5_cross_npu2_reuse_detached.sh start --accept-npu2-reuse-risk
  launch_a5_cross_npu2_reuse_detached.sh status
  launch_a5_cross_npu2_reuse_detached.sh watch
  launch_a5_cross_npu2_reuse_detached.sh recover-cann-bootstrap --accept-npu2-reuse-risk
  launch_a5_cross_npu2_reuse_detached.sh recovery-status
  launch_a5_cross_npu2_reuse_detached.sh recovery-watch
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
  for kvaf_command in awk cat cmp date dirname docker git grep head ln \
    mktemp realpath sha256sum stat tail tr wc; do
    require_command "${kvaf_command}"
  done
}

require_canonical_state_root()
{
  local kvaf_state_mode

  [[ ! -L "${KVAF_STATE_ROOT}" && -d "${KVAF_STATE_ROOT}" &&
     -w "${KVAF_STATE_ROOT}" ]] ||
    die "state root is missing, unsafe, or unwritable: ${KVAF_STATE_ROOT}"
  [[ "$(realpath -e "${KVAF_STATE_ROOT}")" == "${KVAF_STATE_ROOT}" ]] ||
    die "state root is not canonical"
  [[ "$(stat -c '%u' "${KVAF_STATE_ROOT}")" == 0 ]] ||
    die "state root is not root-owned"
  kvaf_state_mode=$(stat -c '%a' "${KVAF_STATE_ROOT}") ||
    die "cannot inspect state root mode"
  (( (8#${kvaf_state_mode} & 8#022) == 0 )) ||
    die "state root is group/world writable"
}

inspect_pinned_cann_installation()
{
  local kvaf_cann_alias=/usr/local/Ascend/ascend-toolkit/set_env.sh
  local kvaf_cann_entry=/usr/local/Ascend/cann-9.1.0/set_env.sh
  local kvaf_cann_root=/usr/local/Ascend/cann-9.1.0
  local kvaf_alias_real kvaf_entry_real kvaf_root_real
  local kvaf_entry_sha kvaf_entry_identity
  local kvaf_path kvaf_mode kvaf_owner kvaf_links

  kvaf_root_real=$(realpath -e -- "${kvaf_cann_root}") ||
    die "pinned CANN 9.1.0 root is missing"
  [[ "${kvaf_root_real}" == "${kvaf_cann_root}" &&
     ! -L "${kvaf_cann_root}" && -d "${kvaf_cann_root}" ]] ||
    die "pinned CANN 9.1.0 root is not canonical"
  kvaf_alias_real=$(realpath -e -- "${kvaf_cann_alias}") ||
    die "CANN set_env.sh alias is unresolved"
  kvaf_entry_real=$(realpath -e -- "${kvaf_cann_entry}") ||
    die "pinned CANN set_env.sh is missing"
  [[ "${kvaf_alias_real}" == "${kvaf_entry_real}" ]] ||
    die "CANN set_env.sh alias does not select pinned CANN 9.1.0"
  [[ ! -L "${kvaf_entry_real}" && -f "${kvaf_entry_real}" &&
     -r "${kvaf_entry_real}" ]] ||
    die "resolved CANN set_env.sh is not a readable regular file"
  case "${kvaf_entry_real}" in
    "${kvaf_root_real}"/*) ;;
    *) die "resolved CANN set_env.sh escapes pinned CANN 9.1.0" ;;
  esac
  kvaf_path=${kvaf_entry_real}
  while :; do
    [[ ! -L "${kvaf_path}" ]] ||
      die "resolved CANN path contains an unexpected symlink"
    kvaf_owner=$(stat -c '%u' "${kvaf_path}") ||
      die "cannot inspect resolved CANN path owner"
    kvaf_mode=$(stat -c '%a' "${kvaf_path}") ||
      die "cannot inspect resolved CANN path mode"
    [[ "${kvaf_owner}" == 0 ]] ||
      die "resolved CANN path is not root-owned"
    (( (8#${kvaf_mode} & 8#022) == 0 )) ||
      die "resolved CANN path is group/world writable"
    if [[ "${kvaf_path}" == "${kvaf_entry_real}" ]]; then
      kvaf_links=$(stat -c '%h' "${kvaf_path}") ||
        die "cannot inspect resolved CANN set_env.sh link count"
      [[ "${kvaf_links}" == 1 ]] ||
        die "resolved CANN set_env.sh has multiple hard links"
    fi
    [[ "${kvaf_path}" != "${kvaf_root_real}" ]] || break
    kvaf_path=$(dirname -- "${kvaf_path}") ||
      die "cannot walk resolved CANN path"
    case "${kvaf_path}" in
      "${kvaf_root_real}"|"${kvaf_root_real}"/*) ;;
      *) die "resolved CANN parent escapes pinned CANN 9.1.0" ;;
    esac
  done

  kvaf_entry_sha=$(sha256sum -- "${kvaf_entry_real}" | awk '{print $1}') ||
    die "cannot hash pinned CANN set_env.sh"
  kvaf_entry_identity=$(stat -c '%d:%i:%s:%a:%h:%u' \
    "${kvaf_entry_real}") || die "cannot inspect pinned CANN set_env.sh"
  KVAF_PINNED_CANN_ROOT=${kvaf_root_real}
  KVAF_PINNED_CANN_SET_ENV=${kvaf_entry_real}
  KVAF_PINNED_CANN_SET_ENV_SEAL="${kvaf_entry_sha}:${kvaf_entry_identity}"
}

source_pinned_cann_environment()
{
  local kvaf_before_root kvaf_before_path kvaf_before_seal
  local kvaf_bisheng_real kvaf_bisheng_mode kvaf_bisheng_owner
  local kvaf_bisheng_links kvaf_bisheng_sha kvaf_bisheng_identity

  inspect_pinned_cann_installation
  kvaf_before_root=${KVAF_PINNED_CANN_ROOT}
  kvaf_before_path=${KVAF_PINNED_CANN_SET_ENV}
  kvaf_before_seal=${KVAF_PINNED_CANN_SET_ENV_SEAL}
  unset ASCEND_HOME_PATH
  source "${kvaf_before_path}"
  inspect_pinned_cann_installation
  [[ "${KVAF_PINNED_CANN_ROOT}" == "${kvaf_before_root}" &&
     "${KVAF_PINNED_CANN_SET_ENV}" == "${kvaf_before_path}" &&
     "${KVAF_PINNED_CANN_SET_ENV_SEAL}" == "${kvaf_before_seal}" ]] ||
    die "CANN set_env.sh identity changed while sourcing"
  [[ -n "${ASCEND_HOME_PATH:-}" &&
     "$(realpath -e -- "${ASCEND_HOME_PATH}")" == "${kvaf_before_root}" ]] ||
    die "CANN environment selected an unexpected ASCEND_HOME_PATH"
  kvaf_bisheng_real=$(realpath -e -- "$(command -v bisheng)") ||
    die "pinned CANN environment did not provide bisheng"
  case "${kvaf_bisheng_real}" in
    "${kvaf_before_root}"/*) ;;
    *) die "CANN environment selected bisheng outside pinned CANN 9.1.0" ;;
  esac
  [[ ! -L "${kvaf_bisheng_real}" && -f "${kvaf_bisheng_real}" &&
     -r "${kvaf_bisheng_real}" && -x "${kvaf_bisheng_real}" ]] ||
    die "resolved bisheng is not a readable executable regular file"
  kvaf_bisheng_owner=$(stat -c '%u' "${kvaf_bisheng_real}") ||
    die "cannot inspect resolved bisheng owner"
  kvaf_bisheng_mode=$(stat -c '%a' "${kvaf_bisheng_real}") ||
    die "cannot inspect resolved bisheng mode"
  kvaf_bisheng_links=$(stat -c '%h' "${kvaf_bisheng_real}") ||
    die "cannot inspect resolved bisheng link count"
  [[ "${kvaf_bisheng_owner}" == 0 && "${kvaf_bisheng_links}" == 1 ]] ||
    die "resolved bisheng ownership or hard-link identity is unsafe"
  (( (8#${kvaf_bisheng_mode} & 8#022) == 0 )) ||
    die "resolved bisheng is group/world writable"
  kvaf_bisheng_sha=$(sha256sum -- "${kvaf_bisheng_real}" |
    awk '{print $1}') || die "cannot hash resolved bisheng"
  kvaf_bisheng_identity=$(stat -c '%d:%i:%s:%a:%h:%u' \
    "${kvaf_bisheng_real}") || die "cannot seal resolved bisheng"
  KVAF_PINNED_CANN_BISHENG=${kvaf_bisheng_real}
  KVAF_PINNED_CANN_BISHENG_SEAL="${kvaf_bisheng_sha}:${kvaf_bisheng_identity}"
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

sealed_root_file_identity()
{
  local path=$1
  local mode=$2
  local identity digest

  require_regular_identity "${path}" "${mode}" || return 1
  identity=$(stat -c '%d:%i:%s:%a:%h:%u' "${path}") || return 1
  [[ "${identity}" == *":${mode}:1:0" ]] || return 1
  digest=$(sha256sum -- "${path}" | awk '{print $1}') || return 1
  [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] || return 1
  printf '%s:%s\n' "${digest}" "${identity}"
}

verify_original_cann_symlink_failure()
{
  local kvaf_require_fresh=${1:-1}
  local -a kvaf_failed_log_lines
  local kvaf_last_byte_is_newline

  [[ "${kvaf_require_fresh}" == 0 || "${kvaf_require_fresh}" == 1 ]] ||
    die "invalid original-failure verification mode"

  sealed_root_file_identity "${KVAF_RESERVATION}" 400 >/dev/null ||
    die "original detached reservation is not sealed root-owned evidence"
  sealed_root_file_identity "${KVAF_LOG}" 600 >/dev/null ||
    die "original detached log is not sealed root-owned evidence"
  sealed_root_file_identity "${KVAF_WORKER_SELECTED}" 400 >/dev/null ||
    die "original worker selection is not sealed root-owned evidence"
  sealed_root_file_identity "${KVAF_WORKER_RC}" 400 >/dev/null ||
    die "original worker result is not sealed root-owned evidence"
  [[ ! -e "${KVAF_DISPATCH_UNPROVEN}" &&
     ! -L "${KVAF_DISPATCH_UNPROVEN}" &&
     ! -e "${KVAF_WORKER_BOOTSTRAP_UNPROVEN}" &&
     ! -L "${KVAF_WORKER_BOOTSTRAP_UNPROVEN}" ]] ||
    die "original dispatch or worker bootstrap is ambiguous"

  read_reservation
  [[ "${KVAF_RECORDED_LAUNCHER_COMMIT}" == \
     "${KVAF_KNOWN_CANN_SYMLINK_FAILURE_COMMIT}" ]] ||
    die "original launch is not the reviewed CANN symlink failure generation"
  require_log_identity
  require_worker_selection_identity ||
    die "original worker selection identity changed"
  [[ "$(stat -c '%s' "${KVAF_WORKER_RC}")" == 2 ]] ||
    die "original bootstrap result length changed"
  cmp -s <(printf '2\n') "${KVAF_WORKER_RC}" ||
    die "original bootstrap result is not exact rc=2"

  mapfile -t kvaf_failed_log_lines <"${KVAF_LOG}" ||
    die "cannot read the original detached log"
  (( ${#kvaf_failed_log_lines[@]} == 12 )) ||
    die "original detached log has unexpected extra or missing lines"
  kvaf_last_byte_is_newline=$(tail -c 1 -- "${KVAF_LOG}" | wc -l) ||
    die "cannot inspect original detached log terminator"
  [[ "${kvaf_last_byte_is_newline}" == 1 ]] ||
    die "original detached log is not newline terminated"
  [[ "${kvaf_failed_log_lines[8]}" == stage=container-worker-started ]] ||
    die "original detached log lacks the exact worker-start marker"
  [[ "${kvaf_failed_log_lines[9]}" =~ ^worker_pid=([1-9][0-9]*)$ ]] ||
    die "original detached worker pid is malformed"
  KVAF_FAILED_WORKER_PID=${BASH_REMATCH[1]}
  [[ "${kvaf_failed_log_lines[10]}" =~ \
     ^started_at=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] ||
    die "original detached worker timestamp is malformed"
  [[ "${kvaf_failed_log_lines[11]}" == \
     'ERROR: CANN set_env.sh is missing' ]] ||
    die "original detached log is not the exact reviewed CANN failure"

  if [[ "${kvaf_require_fresh}" == 1 ]]; then
    [[ ! -e "${KVAF_CLAIM}" && ! -L "${KVAF_CLAIM}" ]] ||
      die "the hardware attempt claim already exists"
    [[ ! -e "${KVAF_ARTIFACT_ROOT}" && ! -L "${KVAF_ARTIFACT_ROOT}" ]] ||
      die "the failed generation already created coordinator artifacts"
    [[ ! -e "${KVAF_REUSE_STATE_DIR}" && ! -L "${KVAF_REUSE_STATE_DIR}" ]] ||
      die "the failed generation already created coordinator state"
  fi

  KVAF_FAILED_RESERVATION_SEAL=$(sealed_root_file_identity \
    "${KVAF_RESERVATION}" 400) || die "cannot seal original reservation"
  KVAF_FAILED_LOG_SEAL=$(sealed_root_file_identity \
    "${KVAF_LOG}" 600) || die "cannot seal original log"
  KVAF_FAILED_WORKER_SELECTION_SEAL=$(sealed_root_file_identity \
    "${KVAF_WORKER_SELECTED}" 400) || die "cannot seal original worker selection"
  KVAF_FAILED_WORKER_RC_SEAL=$(sealed_root_file_identity \
    "${KVAF_WORKER_RC}" 400) || die "cannot seal original worker result"
  KVAF_FAILED_LAUNCH_NONCE=${KVAF_RECORDED_LAUNCH_NONCE}
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

run_container_recovery_gates()
{
  local kvaf_expected_launcher_commit=$1
  local kvaf_expected_cann_path=${2:-}
  local kvaf_expected_cann_seal=${3:-}
  local kvaf_expected_bisheng_path=${4:-}
  local kvaf_expected_bisheng_seal=${5:-}
  local kvaf_proc_entry kvaf_proc_pid
  local kvaf_proc_cmdline kvaf_proc_environment
  local kvaf_claim_parent kvaf_package_name kvaf_package_dir
  local kvaf_package_path kvaf_registration_file kvaf_registration_real
  local kvaf_registration_dir
  local kvaf_inventory kvaf_idle
  local -a kvaf_claim_temporaries kvaf_package_temporaries
  local -a kvaf_registry_temporaries kvaf_token_residues

  for kvaf_command in awk cat cmp dirname git grep npu-smi python3 realpath \
    sha256sum stat timeout tr; do
    require_command "${kvaf_command}"
  done

  [[ -e /.dockerenv ]] || die "CANN recovery gates are container-only"
  (( EUID == 0 )) || die "CANN recovery gates require uid 0"
  [[ "${kvaf_expected_launcher_commit}" =~ ^[0-9a-f]{40}$ ]] ||
    die "CANN recovery launcher identity is malformed"
  if [[ -n "${kvaf_expected_cann_path}${kvaf_expected_cann_seal}${kvaf_expected_bisheng_path}${kvaf_expected_bisheng_seal}" ]]; then
    [[ -n "${kvaf_expected_cann_path}" &&
       -n "${kvaf_expected_cann_seal}" &&
       -n "${kvaf_expected_bisheng_path}" &&
       -n "${kvaf_expected_bisheng_seal}" ]] ||
      die "CANN recovery expected environment identity is incomplete"
  fi
  [[ "$(git -C "${LAUNCHER_ROOT}" rev-parse HEAD)" == \
     "${kvaf_expected_launcher_commit}" ]] ||
    die "recovery launcher checkout moved"
  require_clean_git_tree "${LAUNCHER_ROOT}"
  prepare_pinned_worktree
  verify_original_cann_symlink_failure

  for kvaf_proc_entry in /proc/[0-9]*; do
    [[ -d "${kvaf_proc_entry}" ]] || continue
    kvaf_proc_pid=${kvaf_proc_entry##*/}
    [[ "${kvaf_proc_pid}" != "$$" ]] || continue
    if [[ ! -r "${kvaf_proc_entry}/environ" ||
          ! -r "${kvaf_proc_entry}/cmdline" ]]; then
      [[ ! -e "${kvaf_proc_entry}" ]] && continue
      die "a persistent process is unreadable during recovery proof"
    fi
    if ! kvaf_proc_environment=$(tr '\0' '\n' \
        <"${kvaf_proc_entry}/environ" 2>/dev/null); then
      [[ ! -e "${kvaf_proc_entry}" ]] && continue
      die "a process environment became unreadable during recovery proof"
    fi
    if grep -Fxq \
        "KVAF_DETACHED_LAUNCH_NONCE=${KVAF_FAILED_LAUNCH_NONCE}" \
        <<<"${kvaf_proc_environment}"; then
      die "a process still retains the failed launch nonce"
    fi
    if grep -Fxq \
        "KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN=${KVAF_ARTIFACT_TOKEN}" \
        <<<"${kvaf_proc_environment}"; then
      die "another process already retains the recovery artifact token"
    fi
    if ! kvaf_proc_cmdline=$(tr '\0' ' ' \
        <"${kvaf_proc_entry}/cmdline" 2>/dev/null); then
      [[ ! -e "${kvaf_proc_entry}" ]] && continue
      die "a process command line became unreadable during recovery proof"
    fi
    case "${kvaf_proc_cmdline}" in
      *"launch_a5_cross_npu2_reuse_detached.sh worker"*|\
      *"run_a5_cross_npu2_reuse_experiment.sh"*|\
      *"build_a5_direct_urma_provider.sh"*|\
      *"package_a5_cross_npu2_reuse_aicpu.sh"*|\
      *"kvaf_a5_aicpu_owner_probe"*|\
      *"run_a5_direct_urma_h2d_smoke.sh"*|\
      *"${KVAF_EXPERIMENT_WORKTREE}"*)
        die "a coordinator, builder, probe, or direct runner is still live"
        ;;
    esac
  done

  kvaf_claim_parent=$(dirname -- "${KVAF_CLAIM}")
  shopt -s nullglob
  kvaf_claim_temporaries=("${kvaf_claim_parent}"/.attempt.selected.*)
  (( ${#kvaf_claim_temporaries[@]} == 0 )) ||
    die "an attempt-claim temporary exists; coordinator entry is ambiguous"
  [[ ! -e "${KVAF_CLAIM}" && ! -L "${KVAF_CLAIM}" &&
     ! -e "${KVAF_ARTIFACT_ROOT}" && ! -L "${KVAF_ARTIFACT_ROOT}" &&
     ! -e "${KVAF_REUSE_STATE_DIR}" && ! -L "${KVAF_REUSE_STATE_DIR}" ]] ||
    die "coordinator claim, artifacts, or state already exist"

  inspect_pinned_cann_installation
  if [[ -n "${kvaf_expected_cann_path}" ]]; then
    [[ "${KVAF_PINNED_CANN_SET_ENV}" == "${kvaf_expected_cann_path}" &&
       "${KVAF_PINNED_CANN_SET_ENV_SEAL}" == \
         "${kvaf_expected_cann_seal}" ]] ||
      die "pinned CANN set_env.sh changed before source"
  fi
  source_pinned_cann_environment
  if [[ -n "${kvaf_expected_cann_path}" ]]; then
    [[ "${KVAF_PINNED_CANN_SET_ENV}" == "${kvaf_expected_cann_path}" &&
       "${KVAF_PINNED_CANN_SET_ENV_SEAL}" == \
         "${kvaf_expected_cann_seal}" &&
       "${KVAF_PINNED_CANN_BISHENG}" == "${kvaf_expected_bisheng_path}" &&
       "${KVAF_PINNED_CANN_BISHENG_SEAL}" == \
         "${kvaf_expected_bisheng_seal}" ]] ||
      die "pinned CANN environment changed after recovery reservation"
  fi
  kvaf_package_name="cann-kvaf-du-${KVAF_ARTIFACT_TOKEN}.tar.gz"
  kvaf_package_dir="${KVAF_PINNED_CANN_ROOT}/opp/vendors/cust/op_impl/aicpu/kernel"
  kvaf_package_path="${kvaf_package_dir}/${kvaf_package_name}"
  kvaf_registration_file="${KVAF_PINNED_CANN_ROOT}/conf/ascend_package_load.ini"
  [[ ! -L "${kvaf_package_dir}" && -d "${kvaf_package_dir}" &&
     -w "${kvaf_package_dir}" &&
     "$(realpath -e -- "${kvaf_package_dir}")" == "${kvaf_package_dir}" ]] ||
    die "the pinned CANN AICPU package directory is missing or unsafe"
  [[ ! -e "${kvaf_package_path}" && ! -L "${kvaf_package_path}" ]] ||
    die "the token-owned AICPU package already exists"
  kvaf_package_temporaries=("$(dirname -- "${kvaf_package_path}")"/."${kvaf_package_name}".*)
  (( ${#kvaf_package_temporaries[@]} == 0 )) ||
    die "a token-owned AICPU package temporary already exists"
  [[ ! -L "${kvaf_registration_file}" &&
     -f "${kvaf_registration_file}" && -r "${kvaf_registration_file}" &&
     -w "${kvaf_registration_file}" ]] ||
    die "the pinned CANN AICPU registry is missing or unsafe"
  grep -Fxq 'name:Ascend-aicpu_legacy.tar.gz' \
    "${kvaf_registration_file}" ||
    die "the pinned CANN AICPU registry lacks its built-in baseline"
  if grep -Fq "${KVAF_ARTIFACT_TOKEN}" "${kvaf_registration_file}" ||
     grep -Fq "libkvaf_ud_${KVAF_ARTIFACT_TOKEN}.so" \
       "${kvaf_registration_file}"; then
    die "the token-owned AICPU registration or partial residue exists"
  fi
  kvaf_registration_real=$(realpath -e -- "${kvaf_registration_file}") ||
    die "cannot resolve the pinned CANN AICPU registry"
  kvaf_registration_dir=$(dirname -- "${kvaf_registration_real}") ||
    die "cannot resolve the pinned CANN AICPU registry parent"
  kvaf_registry_temporaries=(
    "${kvaf_registration_dir}"/.ascend_package_load.ini.kvaf.*
  )
  (( ${#kvaf_registry_temporaries[@]} == 0 )) ||
    die "an AICPU registry transaction temporary already exists"
  kvaf_token_residues=(
    "${kvaf_package_dir}"/*"${KVAF_ARTIFACT_TOKEN}"*
    "${kvaf_package_dir}"/.*"${KVAF_ARTIFACT_TOKEN}"*
  )
  (( ${#kvaf_token_residues[@]} == 0 )) ||
    die "a token-owned AICPU package residue already exists"

  cd "${KVAF_EXPERIMENT_WORKTREE}"
  [[ "$(git rev-parse HEAD)" == "${KVAF_EXPERIMENT_COMMIT}" ]] ||
    die "pinned experiment commit moved during recovery proof"
  require_clean_git_tree "${KVAF_EXPERIMENT_WORKTREE}"
  python3 scripts/validate_a5_cross_npu2_reuse_risk.py \
    --new-artifact-token "${KVAF_ARTIFACT_TOKEN}"
  kvaf_inventory=$(timeout 15s npu-smi info) ||
    die "npu-smi could not prove NPU2 idle during recovery proof"
  kvaf_idle=$(python3 scripts/check_a5_npu_idle.py \
    --physical-device-id 2 <<<"${kvaf_inventory}") ||
    die "NPU2 idle evidence is malformed during recovery proof"
  printf '%s\n' "${kvaf_idle}"
  grep -Fxq KVAF_NPU_IDLE_STATUS=idle <<<"${kvaf_idle}" ||
    die "NPU2 is not strictly idle during recovery proof"

  KVAF_ACTUAL_RECOVERY_CANN_PATH=${KVAF_PINNED_CANN_SET_ENV}
  KVAF_ACTUAL_RECOVERY_CANN_SEAL=${KVAF_PINNED_CANN_SET_ENV_SEAL}
  KVAF_ACTUAL_RECOVERY_BISHENG_PATH=${KVAF_PINNED_CANN_BISHENG}
  KVAF_ACTUAL_RECOVERY_BISHENG_SEAL=${KVAF_PINNED_CANN_BISHENG_SEAL}
  echo "KVAF_RECOVERY_CANN_PATH=${KVAF_PINNED_CANN_SET_ENV}"
  echo "KVAF_RECOVERY_CANN_SEAL=${KVAF_PINNED_CANN_SET_ENV_SEAL}"
  echo "KVAF_RECOVERY_BISHENG_PATH=${KVAF_PINNED_CANN_BISHENG}"
  echo "KVAF_RECOVERY_BISHENG_SEAL=${KVAF_PINNED_CANN_BISHENG_SEAL}"
  echo "KVAF_NPU2_CANN_BOOTSTRAP_RECOVERY_PREFLIGHT_OK"
}

run_recovery_container_preflight()
{
  local kvaf_output

  kvaf_output=$(docker exec -u 0 \
    -e USER=root \
    -e "KVAF_RECOVERY_EXPECTED_LAUNCHER_COMMIT=${KVAF_RECOVERY_LAUNCHER_COMMIT}" \
    -e "KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN=${KVAF_ARTIFACT_TOKEN}" \
    -e "KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE=${KVAF_RISK_PHRASE}" \
    -w "${LAUNCHER_ROOT}" \
    "${KVAF_CONTAINER}" \
    bash --noprofile --norc \
      "${SCRIPT_DIR}/launch_a5_cross_npu2_reuse_detached.sh" \
      recovery-preflight) ||
    die "container CANN bootstrap recovery preflight failed"
  printf '%s\n' "${kvaf_output}"
  KVAF_RECOVERY_CANN_PATH=$(awk -F= '
    $1 == "KVAF_RECOVERY_CANN_PATH" { value=$2; count++ }
    END { if (count != 1) exit 1; print value }
  ' <<<"${kvaf_output}") || die "recovery CANN path evidence is malformed"
  KVAF_RECOVERY_CANN_SEAL=$(awk -F= '
    $1 == "KVAF_RECOVERY_CANN_SEAL" { value=substr($0, index($0, "=") + 1); count++ }
    END { if (count != 1) exit 1; print value }
  ' <<<"${kvaf_output}") || die "recovery CANN seal evidence is malformed"
  KVAF_RECOVERY_BISHENG_PATH=$(awk -F= '
    $1 == "KVAF_RECOVERY_BISHENG_PATH" { value=$2; count++ }
    END { if (count != 1) exit 1; print value }
  ' <<<"${kvaf_output}") || die "recovery bisheng path evidence is malformed"
  KVAF_RECOVERY_BISHENG_SEAL=$(awk -F= '
    $1 == "KVAF_RECOVERY_BISHENG_SEAL" { value=substr($0, index($0, "=") + 1); count++ }
    END { if (count != 1) exit 1; print value }
  ' <<<"${kvaf_output}") || die "recovery bisheng seal evidence is malformed"
  [[ "${KVAF_RECOVERY_CANN_PATH}" == \
       /usr/local/Ascend/cann-9.1.0/* &&
     "${KVAF_RECOVERY_CANN_SEAL}" =~ \
       ^[0-9a-f]{64}:[0-9]+:[0-9]+:[0-9]+:[0-9]+:1:0$ &&
     "${KVAF_RECOVERY_BISHENG_PATH}" == \
       /usr/local/Ascend/cann-9.1.0/* &&
     "${KVAF_RECOVERY_BISHENG_SEAL}" =~ \
       ^[0-9a-f]{64}:[0-9]+:[0-9]+:[0-9]+:[0-9]+:1:0$ ]] ||
    die "recovery CANN evidence is not canonical"
}

run_pinned_cann_read_only_preflight()
{
  docker exec -u 0 \
    -e USER=root \
    -e "KVAF_CANN_PREFLIGHT_EXPECTED_LAUNCHER_COMMIT=${KVAF_LAUNCHER_COMMIT}" \
    -w "${LAUNCHER_ROOT}" \
    "${KVAF_CONTAINER}" \
    bash --noprofile --norc \
      "${SCRIPT_DIR}/launch_a5_cross_npu2_reuse_detached.sh" \
      cann-preflight || die "pinned CANN environment preflight failed"
}

run_cann_preflight_mode()
{
  (( $# == 0 )) || usage
  [[ -e /.dockerenv ]] || die "cann-preflight mode is container-only"
  (( EUID == 0 )) || die "cann-preflight mode requires uid 0"
  [[ "${KVAF_CANN_PREFLIGHT_EXPECTED_LAUNCHER_COMMIT:-}" =~ \
     ^[0-9a-f]{40}$ ]] || die "CANN preflight authorization is missing"
  [[ "$(git -C "${LAUNCHER_ROOT}" rev-parse HEAD)" == \
     "${KVAF_CANN_PREFLIGHT_EXPECTED_LAUNCHER_COMMIT}" ]] ||
    die "launcher checkout moved before CANN preflight"
  require_clean_git_tree "${LAUNCHER_ROOT}"
  source_pinned_cann_environment
  echo "KVAF_PINNED_CANN_ROOT=${KVAF_PINNED_CANN_ROOT}"
  echo "KVAF_PINNED_CANN_SET_ENV=${KVAF_PINNED_CANN_SET_ENV}"
  echo "KVAF_PINNED_CANN_PREFLIGHT_OK"
}

run_recovery_preflight_mode()
{
  (( $# == 0 )) || usage
  [[ -e /.dockerenv ]] || die "recovery-preflight mode is container-only"
  (( EUID == 0 )) || die "recovery-preflight mode requires uid 0"
  [[ "${KVAF_RECOVERY_EXPECTED_LAUNCHER_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] ||
    die "recovery-preflight authorization is missing"
  [[ "${KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE:-}" == \
     "${KVAF_RISK_PHRASE}" ]] || die "recovery-preflight risk identity is missing"
  [[ "${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN:-}" == \
     "${KVAF_ARTIFACT_TOKEN}" ]] || die "recovery-preflight token is not pinned"
  require_canonical_state_root
  run_container_recovery_gates \
    "${KVAF_RECOVERY_EXPECTED_LAUNCHER_COMMIT}"
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

recovery_reservation_content()
{
  printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-cann-bootstrap-recovery-v1' \
    "source_run_id=${KVAF_SOURCE_RUN_ID}" \
    "prior_target_run_id=${KVAF_PRIOR_TARGET_RUN_ID}" \
    "experiment_commit=${KVAF_EXPERIMENT_COMMIT}" \
    "artifact_token=${KVAF_ARTIFACT_TOKEN}" \
    "failed_launcher_commit=${KVAF_KNOWN_CANN_SYMLINK_FAILURE_COMMIT}" \
    "failed_launch_nonce=${KVAF_FAILED_LAUNCH_NONCE}" \
    "failed_reservation_seal=${KVAF_FAILED_RESERVATION_SEAL}" \
    "failed_log_seal=${KVAF_FAILED_LOG_SEAL}" \
    "failed_worker_selection_seal=${KVAF_FAILED_WORKER_SELECTION_SEAL}" \
    "failed_worker_rc_seal=${KVAF_FAILED_WORKER_RC_SEAL}" \
    "cann_set_env_path=${KVAF_RECOVERY_CANN_PATH}" \
    "cann_set_env_seal=${KVAF_RECOVERY_CANN_SEAL}" \
    "cann_bisheng_path=${KVAF_RECOVERY_BISHENG_PATH}" \
    "cann_bisheng_seal=${KVAF_RECOVERY_BISHENG_SEAL}" \
    "recovery_launcher_commit=${KVAF_RECOVERY_LAUNCHER_COMMIT}" \
    "recovery_nonce=${KVAF_RECOVERY_NONCE}" \
    "experiment_worktree=${KVAF_EXPERIMENT_WORKTREE}" \
    "recovery_log=${KVAF_RECOVERY_LOG}" \
    "recovery_worker_rc=${KVAF_RECOVERY_WORKER_RC}" \
    "recovery_dispatch_unproven=${KVAF_RECOVERY_DISPATCH_UNPROVEN}" \
    "recovery_worker_selected=${KVAF_RECOVERY_WORKER_SELECTED}" \
    "recovery_bootstrap_unproven=${KVAF_RECOVERY_BOOTSTRAP_UNPROVEN}" \
    'execution_contract=single-pre-coordinator-cann-bootstrap-recovery-v1'
}

recovery_log_header()
{
  printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-cann-bootstrap-recovery-log-v1' \
    "source_run_id=${KVAF_SOURCE_RUN_ID}" \
    "prior_target_run_id=${KVAF_PRIOR_TARGET_RUN_ID}" \
    "experiment_commit=${KVAF_EXPERIMENT_COMMIT}" \
    "artifact_token=${KVAF_ARTIFACT_TOKEN}" \
    "failed_launcher_commit=${KVAF_KNOWN_CANN_SYMLINK_FAILURE_COMMIT}" \
    "failed_launch_nonce=${KVAF_FAILED_LAUNCH_NONCE}" \
    "failed_reservation_seal=${KVAF_FAILED_RESERVATION_SEAL}" \
    "failed_log_seal=${KVAF_FAILED_LOG_SEAL}" \
    "cann_set_env_path=${KVAF_RECOVERY_CANN_PATH}" \
    "cann_set_env_seal=${KVAF_RECOVERY_CANN_SEAL}" \
    "cann_bisheng_path=${KVAF_RECOVERY_BISHENG_PATH}" \
    "cann_bisheng_seal=${KVAF_RECOVERY_BISHENG_SEAL}" \
    "recovery_launcher_commit=${KVAF_RECOVERY_LAUNCHER_COMMIT}" \
    "recovery_nonce=${KVAF_RECOVERY_NONCE}" \
    'stage=dispatching-recovery'
}

recovery_worker_selection_content()
{
  printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-cann-bootstrap-recovery-worker-v1' \
    "experiment_commit=${KVAF_EXPERIMENT_COMMIT}" \
    "artifact_token=${KVAF_ARTIFACT_TOKEN}" \
    "failed_launch_nonce=${KVAF_FAILED_LAUNCH_NONCE}" \
    "recovery_launcher_commit=${KVAF_RECOVERY_LAUNCHER_COMMIT}" \
    "recovery_nonce=${KVAF_RECOVERY_NONCE}"
}

read_recovery_field()
{
  local key=$1
  awk -F= -v key="${key}" '
    $1 == key { value=substr($0, index($0, "=") + 1); count++ }
    END { if (count != 1) exit 1; print value }
  ' "${KVAF_RECOVERY_RESERVATION}"
}

read_recovery_reservation()
{
  local kvaf_expected_sha kvaf_actual_sha

  sealed_root_file_identity "${KVAF_RECOVERY_RESERVATION}" 400 >/dev/null ||
    die "CANN recovery reservation is missing or unsafe"
  KVAF_FAILED_LAUNCH_NONCE=$(read_recovery_field failed_launch_nonce) ||
    die "recovery failed-launch nonce is malformed"
  KVAF_FAILED_RESERVATION_SEAL=$(read_recovery_field failed_reservation_seal) ||
    die "recovery original-reservation seal is malformed"
  KVAF_FAILED_LOG_SEAL=$(read_recovery_field failed_log_seal) ||
    die "recovery original-log seal is malformed"
  KVAF_FAILED_WORKER_SELECTION_SEAL=$(read_recovery_field \
    failed_worker_selection_seal) ||
    die "recovery original-worker seal is malformed"
  KVAF_FAILED_WORKER_RC_SEAL=$(read_recovery_field failed_worker_rc_seal) ||
    die "recovery original-result seal is malformed"
  KVAF_RECOVERY_CANN_PATH=$(read_recovery_field cann_set_env_path) ||
    die "recovery CANN path is malformed"
  KVAF_RECOVERY_CANN_SEAL=$(read_recovery_field cann_set_env_seal) ||
    die "recovery CANN seal is malformed"
  KVAF_RECOVERY_BISHENG_PATH=$(read_recovery_field cann_bisheng_path) ||
    die "recovery bisheng path is malformed"
  KVAF_RECOVERY_BISHENG_SEAL=$(read_recovery_field cann_bisheng_seal) ||
    die "recovery bisheng seal is malformed"
  KVAF_RECOVERY_LAUNCHER_COMMIT=$(read_recovery_field \
    recovery_launcher_commit) || die "recovery launcher commit is malformed"
  KVAF_RECOVERY_NONCE=$(read_recovery_field recovery_nonce) ||
    die "recovery nonce is malformed"
  [[ "${KVAF_FAILED_LAUNCH_NONCE}" =~ ^[0-9a-f]{32}$ &&
     "${KVAF_RECOVERY_LAUNCHER_COMMIT}" =~ ^[0-9a-f]{40}$ &&
     "${KVAF_RECOVERY_NONCE}" =~ ^[0-9a-f]{32}$ &&
     "${KVAF_RECOVERY_CANN_PATH}" == \
       /usr/local/Ascend/cann-9.1.0/* &&
     "${KVAF_RECOVERY_CANN_SEAL}" =~ \
       ^[0-9a-f]{64}:[0-9]+:[0-9]+:[0-9]+:[0-9]+:1:0$ &&
     "${KVAF_RECOVERY_BISHENG_PATH}" == \
       /usr/local/Ascend/cann-9.1.0/* &&
     "${KVAF_RECOVERY_BISHENG_SEAL}" =~ \
       ^[0-9a-f]{64}:[0-9]+:[0-9]+:[0-9]+:[0-9]+:1:0$ ]] ||
    die "CANN recovery reservation fields are not canonical"
  for kvaf_seal in \
    "${KVAF_FAILED_RESERVATION_SEAL}" \
    "${KVAF_FAILED_LOG_SEAL}" \
    "${KVAF_FAILED_WORKER_SELECTION_SEAL}" \
    "${KVAF_FAILED_WORKER_RC_SEAL}"
  do
    [[ "${kvaf_seal}" =~ \
      ^[0-9a-f]{64}:[0-9]+:[0-9]+:[0-9]+:(400|600):1:0$ ]] ||
      die "an original evidence seal in the recovery reservation is malformed"
  done
  kvaf_expected_sha=$(recovery_reservation_content | sha256sum |
    awk '{print $1}')
  kvaf_actual_sha=$(sha256sum -- "${KVAF_RECOVERY_RESERVATION}" |
    awk '{print $1}')
  [[ "${kvaf_actual_sha}" == "${kvaf_expected_sha}" ]] ||
    die "CANN recovery reservation content changed"
}

require_recovery_log_identity()
{
  local kvaf_expected_sha kvaf_actual_sha

  sealed_root_file_identity "${KVAF_RECOVERY_LOG}" 600 >/dev/null ||
    die "CANN recovery log is missing or unsafe"
  kvaf_expected_sha=$(recovery_log_header | sha256sum | awk '{print $1}')
  kvaf_actual_sha=$(head -n 16 -- "${KVAF_RECOVERY_LOG}" |
    sha256sum | awk '{print $1}')
  [[ "${kvaf_actual_sha}" == "${kvaf_expected_sha}" ]] ||
    die "CANN recovery log identity changed"
}

require_recovery_worker_selection_identity()
{
  local kvaf_expected_sha kvaf_actual_sha

  sealed_root_file_identity "${KVAF_RECOVERY_WORKER_SELECTED}" 400 >/dev/null ||
    return 1
  kvaf_expected_sha=$(recovery_worker_selection_content |
    sha256sum | awk '{print $1}')
  kvaf_actual_sha=$(sha256sum -- "${KVAF_RECOVERY_WORKER_SELECTED}" |
    awk '{print $1}')
  [[ "${kvaf_actual_sha}" == "${kvaf_expected_sha}" ]]
}

reserve_recovery_generation()
{
  local kvaf_path

  for kvaf_path in \
    "${KVAF_RECOVERY_RESERVATION}" \
    "${KVAF_RECOVERY_LOG}" \
    "${KVAF_RECOVERY_WORKER_RC}" \
    "${KVAF_RECOVERY_DISPATCH_UNPROVEN}" \
    "${KVAF_RECOVERY_WORKER_SELECTED}" \
    "${KVAF_RECOVERY_BOOTSTRAP_UNPROVEN}"
  do
    [[ ! -e "${kvaf_path}" && ! -L "${kvaf_path}" ]] ||
      die "the single CANN bootstrap recovery generation was already used"
  done
  publish_file_once "${KVAF_RECOVERY_LOG}" 600 \
    "$(recovery_log_header)" || die "could not reserve the recovery log"
  publish_file_once "${KVAF_RECOVERY_RESERVATION}" 400 \
    "$(recovery_reservation_content)" ||
    die "could not atomically reserve the CANN bootstrap recovery"
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

record_recovery_dispatch_unproven()
{
  local rc=$1
  local evidence

  evidence=$(printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-cann-recovery-dispatch-unproven-v1' \
    "docker_cli_rc=${rc}" \
    'coordinator_completion=unproven')
  publish_file_once "${KVAF_RECOVERY_DISPATCH_UNPROVEN}" 400 \
    "${evidence}" || true
  printf '%s\n' "DISPATCH_UNPROVEN=docker_cli_rc_${rc}" \
    >>"${KVAF_RECOVERY_LOG}" || true
}

start_cann_bootstrap_recovery()
{
  local kvaf_current_launcher_commit kvaf_path kvaf_dispatch_rc

  (( $# == 1 )) && [[ "$1" == --accept-npu2-reuse-risk ]] || usage
  require_host
  require_canonical_state_root
  require_launcher_checkout
  kvaf_current_launcher_commit=${KVAF_LAUNCHER_COMMIT}
  KVAF_RECOVERY_LAUNCHER_COMMIT=${kvaf_current_launcher_commit}
  [[ "${KVAF_RECOVERY_LAUNCHER_COMMIT}" != \
     "${KVAF_KNOWN_CANN_SYMLINK_FAILURE_COMMIT}" ]] ||
    die "pull the reviewed CANN bootstrap recovery commit first"
  prepare_pinned_worktree
  verify_original_cann_symlink_failure
  for kvaf_path in \
    "${KVAF_RECOVERY_RESERVATION}" \
    "${KVAF_RECOVERY_LOG}" \
    "${KVAF_RECOVERY_WORKER_RC}" \
    "${KVAF_RECOVERY_DISPATCH_UNPROVEN}" \
    "${KVAF_RECOVERY_WORKER_SELECTED}" \
    "${KVAF_RECOVERY_BOOTSTRAP_UNPROVEN}"
  do
    [[ ! -e "${kvaf_path}" && ! -L "${kvaf_path}" ]] ||
      die "the single CANN bootstrap recovery generation was already selected"
  done
  [[ "$(docker inspect -f '{{.State.Status}}' "${KVAF_CONTAINER}")" == \
     running ]] || die "${KVAF_CONTAINER} is not running"

  KVAF_LAUNCHER_COMMIT=${kvaf_current_launcher_commit}
  run_read_only_preflight
  KVAF_RECOVERY_LAUNCHER_COMMIT=${kvaf_current_launcher_commit}
  run_recovery_container_preflight
  verify_original_cann_symlink_failure
  KVAF_RECOVERY_LAUNCHER_COMMIT=${kvaf_current_launcher_commit}
  [[ -r /proc/sys/kernel/random/uuid ]] ||
    die "cannot generate the CANN recovery nonce"
  KVAF_RECOVERY_NONCE=$(tr -d '-' </proc/sys/kernel/random/uuid)
  [[ "${KVAF_RECOVERY_NONCE}" =~ ^[0-9a-f]{32}$ ]] ||
    die "CANN recovery nonce is malformed"
  reserve_recovery_generation

  if docker exec -d -u 0 \
      -e USER=root \
      -e "KVAF_RECOVERY_LAUNCHER_COMMIT=${KVAF_RECOVERY_LAUNCHER_COMMIT}" \
      -e "KVAF_RECOVERY_NONCE=${KVAF_RECOVERY_NONCE}" \
      -e "KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN=${KVAF_ARTIFACT_TOKEN}" \
      -e "KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE=${KVAF_RISK_PHRASE}" \
      -w "${LAUNCHER_ROOT}" \
      "${KVAF_CONTAINER}" \
      bash --noprofile --norc \
        "${SCRIPT_DIR}/launch_a5_cross_npu2_reuse_detached.sh" \
        recovery-worker; then
    :
  else
    kvaf_dispatch_rc=$?
    record_recovery_dispatch_unproven "${kvaf_dispatch_rc}"
    die "CANN recovery Docker dispatch is unproven; do not retry"
  fi

  echo "KVAF_NPU2_CANN_BOOTSTRAP_RECOVERY_LAUNCHED"
  echo "EXPERIMENT_COMMIT=${KVAF_EXPERIMENT_COMMIT}"
  echo "RECOVERY_LAUNCHER_COMMIT=${KVAF_RECOVERY_LAUNCHER_COMMIT}"
  echo "ARTIFACT_TOKEN=${KVAF_ARTIFACT_TOKEN}"
  echo "RECOVERY_LOG=${KVAF_RECOVERY_LOG}"
  echo "RECOVERY_WORKER_RC=${KVAF_RECOVERY_WORKER_RC}"
  echo "STATUS_COMMAND=bash scripts/launch_a5_cross_npu2_reuse_detached.sh recovery-status"
  echo "WATCH_COMMAND=bash scripts/launch_a5_cross_npu2_reuse_detached.sh recovery-watch"
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
  run_pinned_cann_read_only_preflight
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

publish_recovery_worker_rc()
{
  local rc=$1

  if (( rc < 0 || rc > 255 )); then
    rc=255
  fi
  if ! publish_file_once "${KVAF_RECOVERY_WORKER_RC}" 400 "${rc}"; then
    printf '%s\n' 'ERROR: could not atomically publish CANN recovery worker rc' \
      >>"${KVAF_RECOVERY_LOG}" 2>/dev/null || true
  fi
}

publish_recovery_bootstrap_unproven()
{
  local rc=$1
  local evidence

  evidence=$(printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-cann-recovery-worker-bootstrap-unproven-v1' \
    "worker_shell_rc=${rc}" \
    'coordinator_completion=unproven')
  publish_file_once "${KVAF_RECOVERY_BOOTSTRAP_UNPROVEN}" 400 \
    "${evidence}" || true
}

record_recovery_worker_exit()
{
  kvaf_recovery_worker_rc=$?
  if [[ "${KVAF_RECOVERY_WORKER_OWNS_GENERATION:-0}" == 1 ]]; then
    publish_recovery_worker_rc "${kvaf_recovery_worker_rc}"
  elif [[ ! -e "${KVAF_RECOVERY_WORKER_SELECTED}" &&
          ! -L "${KVAF_RECOVERY_WORKER_SELECTED}" ]]; then
    publish_recovery_bootstrap_unproven "${kvaf_recovery_worker_rc}"
  fi
}

verify_recovery_bound_original_evidence()
{
  local kvaf_require_fresh=${1:-1}
  local kvaf_recorded_failed_nonce
  local kvaf_recorded_reservation_seal kvaf_recorded_log_seal
  local kvaf_recorded_selection_seal kvaf_recorded_rc_seal

  read_recovery_reservation
  kvaf_recorded_failed_nonce=${KVAF_FAILED_LAUNCH_NONCE}
  kvaf_recorded_reservation_seal=${KVAF_FAILED_RESERVATION_SEAL}
  kvaf_recorded_log_seal=${KVAF_FAILED_LOG_SEAL}
  kvaf_recorded_selection_seal=${KVAF_FAILED_WORKER_SELECTION_SEAL}
  kvaf_recorded_rc_seal=${KVAF_FAILED_WORKER_RC_SEAL}
  verify_original_cann_symlink_failure "${kvaf_require_fresh}"
  [[ "${KVAF_FAILED_LAUNCH_NONCE}" == "${kvaf_recorded_failed_nonce}" &&
     "${KVAF_FAILED_RESERVATION_SEAL}" == \
       "${kvaf_recorded_reservation_seal}" &&
     "${KVAF_FAILED_LOG_SEAL}" == "${kvaf_recorded_log_seal}" &&
     "${KVAF_FAILED_WORKER_SELECTION_SEAL}" == \
       "${kvaf_recorded_selection_seal}" &&
     "${KVAF_FAILED_WORKER_RC_SEAL}" == "${kvaf_recorded_rc_seal}" ]] ||
    die "original failed generation changed after recovery authorization"
}

run_recovery_worker()
{
  local kvaf_authorized_launcher_commit kvaf_authorized_nonce
  local kvaf_recorded_cann_path kvaf_recorded_cann_seal
  local kvaf_recorded_bisheng_path kvaf_recorded_bisheng_seal
  local kvaf_current_cann_identity kvaf_current_cann_sha
  local kvaf_current_cann_seal kvaf_current_bisheng_identity
  local kvaf_current_bisheng_sha kvaf_current_bisheng_seal

  (( $# == 0 )) || usage
  [[ -e /.dockerenv ]] || die "recovery-worker mode is container-only"
  (( EUID == 0 )) || die "recovery-worker mode requires uid 0"
  [[ "${KVAF_RECOVERY_LAUNCHER_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] ||
    die "CANN recovery worker authorization is missing"
  [[ "${KVAF_RECOVERY_NONCE:-}" =~ ^[0-9a-f]{32}$ ]] ||
    die "CANN recovery worker nonce is missing"
  [[ "${KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE:-}" == \
     "${KVAF_RISK_PHRASE}" ]] || die "CANN recovery risk identity is missing"
  [[ "${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN:-}" == \
     "${KVAF_ARTIFACT_TOKEN}" ]] || die "CANN recovery token is not pinned"

  kvaf_authorized_launcher_commit=${KVAF_RECOVERY_LAUNCHER_COMMIT}
  kvaf_authorized_nonce=${KVAF_RECOVERY_NONCE}
  KVAF_RECOVERY_WORKER_OWNS_GENERATION=0
  trap record_recovery_worker_exit EXIT
  require_canonical_state_root
  verify_recovery_bound_original_evidence
  [[ "${kvaf_authorized_launcher_commit}" == \
     "${KVAF_RECOVERY_LAUNCHER_COMMIT}" ]] ||
    die "CANN recovery launcher commit changed before worker selection"
  [[ "${kvaf_authorized_nonce}" == "${KVAF_RECOVERY_NONCE}" ]] ||
    die "CANN recovery nonce changed before worker selection"
  require_recovery_log_identity
  [[ ! -e "${KVAF_RECOVERY_WORKER_RC}" &&
     ! -L "${KVAF_RECOVERY_WORKER_RC}" &&
     ! -e "${KVAF_RECOVERY_DISPATCH_UNPROVEN}" &&
     ! -L "${KVAF_RECOVERY_DISPATCH_UNPROVEN}" &&
     ! -e "${KVAF_RECOVERY_BOOTSTRAP_UNPROVEN}" &&
     ! -L "${KVAF_RECOVERY_BOOTSTRAP_UNPROVEN}" ]] ||
    die "CANN recovery generation already has terminal or dispatch evidence"
  publish_file_once "${KVAF_RECOVERY_WORKER_SELECTED}" 400 \
    "$(recovery_worker_selection_content)" ||
    die "another worker already selected the CANN recovery generation"
  KVAF_RECOVERY_WORKER_OWNS_GENERATION=1
  exec >>"${KVAF_RECOVERY_LOG}" 2>&1

  echo "stage=container-recovery-worker-started"
  echo "worker_pid=$$"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  [[ "$(git -C "${LAUNCHER_ROOT}" rev-parse HEAD)" == \
     "${KVAF_RECOVERY_LAUNCHER_COMMIT}" ]] ||
    die "CANN recovery launcher checkout moved before worker start"
  require_clean_git_tree "${LAUNCHER_ROOT}"
  prepare_pinned_worktree
  kvaf_recorded_cann_path=${KVAF_RECOVERY_CANN_PATH}
  kvaf_recorded_cann_seal=${KVAF_RECOVERY_CANN_SEAL}
  kvaf_recorded_bisheng_path=${KVAF_RECOVERY_BISHENG_PATH}
  kvaf_recorded_bisheng_seal=${KVAF_RECOVERY_BISHENG_SEAL}
  run_container_recovery_gates \
    "${KVAF_RECOVERY_LAUNCHER_COMMIT}" \
    "${kvaf_recorded_cann_path}" \
    "${kvaf_recorded_cann_seal}" \
    "${kvaf_recorded_bisheng_path}" \
    "${kvaf_recorded_bisheng_seal}"
  [[ "${KVAF_ACTUAL_RECOVERY_CANN_PATH}" == \
     "${kvaf_recorded_cann_path}" &&
     "${KVAF_ACTUAL_RECOVERY_CANN_SEAL}" == \
       "${kvaf_recorded_cann_seal}" &&
     "${KVAF_ACTUAL_RECOVERY_BISHENG_PATH}" == \
       "${kvaf_recorded_bisheng_path}" &&
     "${KVAF_ACTUAL_RECOVERY_BISHENG_SEAL}" == \
       "${kvaf_recorded_bisheng_seal}" ]] ||
    die "pinned CANN set_env.sh changed after recovery reservation"
  verify_recovery_bound_original_evidence
  require_recovery_log_identity

  kvaf_current_cann_identity=$(stat -c '%d:%i:%s:%a:%h:%u' \
    "${kvaf_recorded_cann_path}") ||
    die "cannot recheck pinned CANN metadata before coordinator"
  kvaf_current_cann_sha=$(sha256sum -- "${kvaf_recorded_cann_path}" |
    awk '{print $1}') || die "cannot recheck pinned CANN hash before coordinator"
  kvaf_current_cann_seal="${kvaf_current_cann_sha}:${kvaf_current_cann_identity}"
  kvaf_current_bisheng_identity=$(stat -c '%d:%i:%s:%a:%h:%u' \
    "${kvaf_recorded_bisheng_path}") ||
    die "cannot recheck pinned bisheng metadata before coordinator"
  kvaf_current_bisheng_sha=$(sha256sum -- "${kvaf_recorded_bisheng_path}" |
    awk '{print $1}') || die "cannot recheck pinned bisheng hash before coordinator"
  kvaf_current_bisheng_seal="${kvaf_current_bisheng_sha}:${kvaf_current_bisheng_identity}"
  [[ "$(realpath -e -- /usr/local/Ascend/ascend-toolkit/set_env.sh)" == \
       "${kvaf_recorded_cann_path}" &&
     "${kvaf_current_cann_seal}" == "${kvaf_recorded_cann_seal}" &&
     "$(realpath -e -- "$(command -v bisheng)")" == \
       "${kvaf_recorded_bisheng_path}" &&
     "${kvaf_current_bisheng_seal}" == \
       "${kvaf_recorded_bisheng_seal}" ]] ||
    die "pinned CANN identity changed immediately before coordinator"

  export USER=${USER:-root}
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
    die "pinned experiment commit changed immediately before recovery coordinator"
  require_clean_git_tree "${KVAF_EXPERIMENT_WORKTREE}"
  echo "stage=coordinator-starting"
  bash scripts/run_a5_cross_npu2_reuse_experiment.sh zero-wqe
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
  source_pinned_cann_environment
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

read_canonical_recovery_worker_rc()
{
  sealed_root_file_identity "${KVAF_RECOVERY_WORKER_RC}" 400 >/dev/null ||
    return 1
  mapfile -t kvaf_recovery_rc_lines <"${KVAF_RECOVERY_WORKER_RC}" ||
    return 1
  (( ${#kvaf_recovery_rc_lines[@]} == 1 )) || return 1
  [[ "${kvaf_recovery_rc_lines[0]}" =~ ^(0|[1-9][0-9]{0,2})$ ]] ||
    return 1
  (( 10#${kvaf_recovery_rc_lines[0]} <= 255 )) || return 1
  KVAF_RECORDED_RECOVERY_RC=${kvaf_recovery_rc_lines[0]}
}

require_recovery_dispatch_unproven_identity()
{
  sealed_root_file_identity "${KVAF_RECOVERY_DISPATCH_UNPROVEN}" 400 \
    >/dev/null || return 1
  mapfile -t kvaf_recovery_dispatch_lines \
    <"${KVAF_RECOVERY_DISPATCH_UNPROVEN}" || return 1
  (( ${#kvaf_recovery_dispatch_lines[@]} == 3 )) || return 1
  [[ "${kvaf_recovery_dispatch_lines[0]}" == \
     schema=kvaf-a5-cross-npu2-cann-recovery-dispatch-unproven-v1 &&
     "${kvaf_recovery_dispatch_lines[1]}" =~ \
       ^docker_cli_rc=(0|[1-9][0-9]{0,2})$ &&
     "${kvaf_recovery_dispatch_lines[2]}" == \
       coordinator_completion=unproven ]]
}

require_recovery_bootstrap_unproven_identity()
{
  sealed_root_file_identity "${KVAF_RECOVERY_BOOTSTRAP_UNPROVEN}" 400 \
    >/dev/null || return 1
  mapfile -t kvaf_recovery_bootstrap_lines \
    <"${KVAF_RECOVERY_BOOTSTRAP_UNPROVEN}" || return 1
  (( ${#kvaf_recovery_bootstrap_lines[@]} == 3 )) || return 1
  [[ "${kvaf_recovery_bootstrap_lines[0]}" == \
     schema=kvaf-a5-cross-npu2-cann-recovery-worker-bootstrap-unproven-v1 &&
     "${kvaf_recovery_bootstrap_lines[1]}" =~ \
       ^worker_shell_rc=(0|[1-9][0-9]{0,2})$ &&
     "${kvaf_recovery_bootstrap_lines[2]}" == \
       coordinator_completion=unproven ]]
}

require_exact_recovery_attempt_claim()
{
  local kvaf_expected kvaf_expected_sha kvaf_actual_sha

  sealed_root_file_identity "${KVAF_CLAIM}" 400 >/dev/null || return 1
  kvaf_expected=$(printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-reuse-attempt-v1' \
    "source_run_id=${KVAF_SOURCE_RUN_ID}" \
    "prior_target_run_id=${KVAF_PRIOR_TARGET_RUN_ID}" \
    "artifact_token=${KVAF_ARTIFACT_TOKEN}" \
    "source_commit=${KVAF_EXPERIMENT_COMMIT}" \
    'execution_contract=risk-accepted-cross-npu2-reuse-v2')
  kvaf_expected_sha=$(printf '%s\n' "${kvaf_expected}" |
    sha256sum | awk '{print $1}')
  kvaf_actual_sha=$(sha256sum -- "${KVAF_CLAIM}" | awk '{print $1}')
  [[ "${kvaf_actual_sha}" == "${kvaf_expected_sha}" ]]
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

print_recovery_key_log_evidence()
{
  grep -E \
    '^(stage=coordinator-starting|STAGE=|KVAF_CROSS_NPU2_REUSE_ATTEMPT_SELECTED=|KVAF_A5_AICPU_OWNER_PROBE_OK|KVAF_NPU2_REUSE_ZERO_WQE_RC=|KVAF_NPU2_REUSE_ZERO_WQE_ACCEPTANCE=|HARD_STOP:|ERROR:)' \
    "${KVAF_RECOVERY_LOG}" || true
  echo "RECOVERY_LOG_TAIL:"
  tail -n 60 "${KVAF_RECOVERY_LOG}" || true
}

show_recovery_status()
{
  local kvaf_path kvaf_status_rc

  (( $# == 0 )) || usage
  require_host
  require_canonical_state_root
  if [[ ! -e "${KVAF_RECOVERY_RESERVATION}" &&
        ! -L "${KVAF_RECOVERY_RESERVATION}" ]]; then
    for kvaf_path in \
      "${KVAF_RECOVERY_LOG}" \
      "${KVAF_RECOVERY_WORKER_RC}" \
      "${KVAF_RECOVERY_DISPATCH_UNPROVEN}" \
      "${KVAF_RECOVERY_WORKER_SELECTED}" \
      "${KVAF_RECOVERY_BOOTSTRAP_UNPROVEN}"
    do
      if [[ -e "${kvaf_path}" || -L "${kvaf_path}" ]]; then
        echo "KVAF_CANN_RECOVERY_STATUS=partial-launch-unproven"
        echo "PARTIAL_PATH=${kvaf_path}"
        return 5
      fi
    done
    verify_original_cann_symlink_failure
    echo "KVAF_ORIGINAL_BOOTSTRAP_RC=2"
    echo "KVAF_ORIGINAL_COORDINATOR_STARTED=no"
    echo "KVAF_CANN_RECOVERY_STATUS=candidate-not-selected"
    echo "KVAF_CANN_RECOVERY_ELIGIBILITY=unproven-until-container-preflight"
    return 1
  fi

  verify_recovery_bound_original_evidence 0
  require_recovery_log_identity
  echo "KVAF_ORIGINAL_BOOTSTRAP_RC=2"
  echo "KVAF_ORIGINAL_COORDINATOR_STARTED=no"
  echo "EXPERIMENT_COMMIT=${KVAF_EXPERIMENT_COMMIT}"
  echo "RECOVERY_LAUNCHER_COMMIT=${KVAF_RECOVERY_LAUNCHER_COMMIT}"
  echo "ARTIFACT_TOKEN=${KVAF_ARTIFACT_TOKEN}"
  echo "RECOVERY_LOG=${KVAF_RECOVERY_LOG}"
  echo "RECOVERY_WORKER_RC=${KVAF_RECOVERY_WORKER_RC}"

  if [[ -e "${KVAF_RECOVERY_DISPATCH_UNPROVEN}" ||
        -L "${KVAF_RECOVERY_DISPATCH_UNPROVEN}" ]]; then
    require_recovery_dispatch_unproven_identity ||
      die "CANN recovery dispatch evidence is unsafe"
    cat "${KVAF_RECOVERY_DISPATCH_UNPROVEN}"
    if [[ -e "${KVAF_RECOVERY_WORKER_RC}" ||
          -L "${KVAF_RECOVERY_WORKER_RC}" ]]; then
      require_recovery_worker_selection_identity ||
        die "completed CANN recovery worker selection is unsafe"
      read_canonical_recovery_worker_rc ||
        die "completed CANN recovery worker result is unsafe"
      kvaf_status_rc=${KVAF_RECORDED_RECOVERY_RC}
      echo "KVAF_CANN_RECOVERY_STATUS=finished-after-dispatch-unproven"
      echo "KVAF_RECOVERY_WORKER_RESULT=${kvaf_status_rc}"
      if grep -Fxq stage=coordinator-starting "${KVAF_RECOVERY_LOG}"; then
        echo "KVAF_RECOVERY_COORDINATOR_STARTED=yes"
        echo "KVAF_COORDINATOR_RC=${kvaf_status_rc}"
      else
        echo "KVAF_RECOVERY_COORDINATOR_STARTED=no"
        echo "KVAF_RECOVERY_BOOTSTRAP_RC=${kvaf_status_rc}"
      fi
      if (( kvaf_status_rc == 0 )); then
        if ! grep -Fxq 'KVAF_NPU2_REUSE_ZERO_WQE_RC=0' \
              "${KVAF_RECOVERY_LOG}" ||
           ! grep -Fxq 'KVAF_NPU2_REUSE_ZERO_WQE_ACCEPTANCE=PASS' \
              "${KVAF_RECOVERY_LOG}" ||
           ! require_exact_recovery_attempt_claim; then
          echo "KVAF_CANN_RECOVERY_STATUS=finished-inconsistent"
          print_recovery_key_log_evidence
          return 5
        fi
        echo "KVAF_CANN_RECOVERY_ACCEPTANCE=PASS"
      fi
      print_recovery_key_log_evidence
      return "${kvaf_status_rc}"
    fi
    echo "KVAF_CANN_RECOVERY_STATUS=dispatch-unproven"
    print_recovery_key_log_evidence
    return 5
  fi
  if [[ -e "${KVAF_RECOVERY_BOOTSTRAP_UNPROVEN}" ||
        -L "${KVAF_RECOVERY_BOOTSTRAP_UNPROVEN}" ]]; then
    require_recovery_bootstrap_unproven_identity ||
      die "CANN recovery worker-bootstrap evidence is unsafe"
    echo "KVAF_CANN_RECOVERY_STATUS=worker-bootstrap-unproven"
    cat "${KVAF_RECOVERY_BOOTSTRAP_UNPROVEN}"
    print_recovery_key_log_evidence
    return 5
  fi
  if [[ -e "${KVAF_RECOVERY_WORKER_SELECTED}" ||
        -L "${KVAF_RECOVERY_WORKER_SELECTED}" ]]; then
    require_recovery_worker_selection_identity ||
      die "CANN recovery worker selection is unsafe"
  else
    echo "KVAF_CANN_RECOVERY_STATUS=dispatched-worker-not-selected"
    print_recovery_key_log_evidence
    return 5
  fi
  if [[ ! -e "${KVAF_RECOVERY_WORKER_RC}" &&
        ! -L "${KVAF_RECOVERY_WORKER_RC}" ]]; then
    echo "KVAF_CANN_RECOVERY_STATUS=running-or-unproven"
    print_recovery_key_log_evidence
    return 5
  fi

  read_canonical_recovery_worker_rc ||
    die "CANN recovery worker result is malformed or unsafe"
  kvaf_status_rc=${KVAF_RECORDED_RECOVERY_RC}
  echo "KVAF_CANN_RECOVERY_STATUS=finished"
  echo "KVAF_RECOVERY_WORKER_RESULT=${kvaf_status_rc}"
  if grep -Fxq stage=coordinator-starting "${KVAF_RECOVERY_LOG}"; then
    echo "KVAF_RECOVERY_COORDINATOR_STARTED=yes"
    echo "KVAF_COORDINATOR_RC=${kvaf_status_rc}"
  else
    echo "KVAF_RECOVERY_COORDINATOR_STARTED=no"
    echo "KVAF_RECOVERY_BOOTSTRAP_RC=${kvaf_status_rc}"
  fi
  if (( kvaf_status_rc == 0 )); then
    if ! grep -Fxq 'KVAF_NPU2_REUSE_ZERO_WQE_RC=0' \
          "${KVAF_RECOVERY_LOG}" ||
       ! grep -Fxq 'KVAF_NPU2_REUSE_ZERO_WQE_ACCEPTANCE=PASS' \
          "${KVAF_RECOVERY_LOG}" ||
       ! require_exact_recovery_attempt_claim; then
      echo "KVAF_CANN_RECOVERY_STATUS=finished-inconsistent"
      print_recovery_key_log_evidence
      return 5
    fi
    echo "KVAF_CANN_RECOVERY_ACCEPTANCE=PASS"
  fi
  print_recovery_key_log_evidence
  return "${kvaf_status_rc}"
}

watch_recovery_status()
{
  (( $# == 0 )) || usage
  require_host
  require_canonical_state_root
  read_recovery_reservation
  require_recovery_log_identity
  echo "Monitoring only; Ctrl+C does not signal the detached recovery worker."
  while [[ ! -e "${KVAF_RECOVERY_WORKER_RC}" &&
           ! -L "${KVAF_RECOVERY_WORKER_RC}" &&
           ! -e "${KVAF_RECOVERY_DISPATCH_UNPROVEN}" &&
           ! -L "${KVAF_RECOVERY_DISPATCH_UNPROVEN}" &&
           ! -e "${KVAF_RECOVERY_BOOTSTRAP_UNPROVEN}" &&
           ! -L "${KVAF_RECOVERY_BOOTSTRAP_UNPROVEN}" ]]; do
    date '+WAITING %F %T'
    tail -n 3 "${KVAF_RECOVERY_LOG}" || true
    sleep 15
  done
  show_recovery_status
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
      echo "KVAF_WORKER_RC=${KVAF_RECORDED_WORKER_RC}"
      if grep -qE \
          '^(WARNING: starting one second-order|STAGE=|KVAF_CROSS_NPU2_REUSE_ATTEMPT_SELECTED=)' \
          "${KVAF_LOG}"; then
        echo "KVAF_COORDINATOR_STARTED=yes"
        echo "KVAF_COORDINATOR_RC=${KVAF_RECORDED_WORKER_RC}"
      else
        echo "KVAF_COORDINATOR_STARTED=no"
        echo "KVAF_BOOTSTRAP_RC=${KVAF_RECORDED_WORKER_RC}"
      fi
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
    echo "KVAF_WORKER_RC=${KVAF_RECORDED_WORKER_RC}"
    if grep -qE \
        '^(WARNING: starting one second-order|STAGE=|KVAF_CROSS_NPU2_REUSE_ATTEMPT_SELECTED=)' \
        "${KVAF_LOG}"; then
      echo "KVAF_COORDINATOR_STARTED=yes"
      echo "KVAF_COORDINATOR_RC=${KVAF_RECORDED_WORKER_RC}"
    else
      echo "KVAF_COORDINATOR_STARTED=no"
      echo "KVAF_BOOTSTRAP_RC=${KVAF_RECORDED_WORKER_RC}"
    fi
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
  recover-cann-bootstrap) start_cann_bootstrap_recovery "$@" ;;
  recovery-preflight) run_recovery_preflight_mode "$@" ;;
  cann-preflight) run_cann_preflight_mode "$@" ;;
  recovery-worker) run_recovery_worker "$@" ;;
  recovery-status) show_recovery_status "$@" ;;
  recovery-watch) watch_recovery_status "$@" ;;
  *) usage ;;
esac
fi
/bin/sh -c 'exit 2'
