#!/usr/bin/env bash

# Prepare and consume one boot-bound maintenance proof for the canonical
# direct-URMA active/quarantine state.  A reboot resets process and device
# state, but /data persists; this helper archives the obsolete filesystem
# barrier only after it proves that the kernel boot ID changed.

set -Eeuo pipefail
umask 077

if (( $# != 1 )) ||
   [[ "$1" != prepare && "$1" != retire ]]; then
  echo "Usage: $0 prepare|retire" >&2
  exit 2
fi
if (( EUID != 0 )); then
  echo "ERROR: reboot state reconciliation must run as root" >&2
  exit 2
fi
if [[ -e /.dockerenv ]]; then
  echo "ERROR: run reboot state reconciliation on Server B host, not in Docker" >&2
  exit 2
fi

action=$1
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
KVAF_DURABLE_STATE_HELPER=${SCRIPT_DIR}/durable_state_file.py
state_dir=/data/.kvaf-state/KVAF
evidence_dir=${state_dir}/reboot-evidence
pending_file=${evidence_dir}/pending.manifest
retired_root=${state_dir}/retired-after-reboot
boot_id_file=/proc/sys/kernel/random/boot_id
deploy_lock=${state_dir}/direct-urma.lock
active_lock=${state_dir}/direct-urma.active.lock
active_file=${state_dir}/direct-urma.active
quarantine_file=${state_dir}/direct-urma.quarantine
cross_lock=${state_dir}/experimental/cross-npu-risk.lock
reuse_lock=${state_dir}/experimental/npu-2-reuse/reuse-risk.lock
reconcile_lock=${state_dir}/reboot-reconcile.lock
legacy_run_deploy_lock=/run/lock/kvaf-a5-direct-urma.lock
legacy_worktree_deploy_lock=${ROOT_DIR}/.kvaf-state/direct-urma.lock
legacy_parent_deploy_lock=$(dirname "${ROOT_DIR}")/.kvaf-state/KVAF/direct-urma.lock
legacy_state_files=(
  "${ROOT_DIR}/.kvaf-state/direct-urma.active"
  "${ROOT_DIR}/.kvaf-state/direct-urma.quarantine"
  "$(dirname "${ROOT_DIR}")/.kvaf-state/KVAF/direct-urma.active"
  "$(dirname "${ROOT_DIR}")/.kvaf-state/KVAF/direct-urma.quarantine"
  /run/lock/kvaf-a5-direct-urma.active
  /run/lock/kvaf-a5-direct-urma.quarantine
)
legacy_active_locks=(
  "${ROOT_DIR}/.kvaf-state/direct-urma.active.lock"
  "$(dirname "${ROOT_DIR}")/.kvaf-state/KVAF/direct-urma.active.lock"
  /run/lock/kvaf-a5-direct-urma.active.lock
)
temporary_paths=()
archived_marker_paths=()

die()
{
  echo "ERROR: $*" >&2
  exit 2
}

cleanup_temporary_paths()
{
  local path
  for path in "${temporary_paths[@]}"; do
    if [[ -n "${path}" && -f "${path}" && ! -L "${path}" ]]; then
      rm -f -- "${path}"
    fi
  done
}

new_evidence_temporary()
{
  local label=$1
  kvaf_temporary_path=$(mktemp "${evidence_dir}/${label}.XXXXXX") ||
    die "could not create temporary evidence"
  temporary_paths+=("${kvaf_temporary_path}")
}

trap cleanup_temporary_paths EXIT

read_boot_id()
{
  local value
  [[ ! -L "${boot_id_file}" && -r "${boot_id_file}" ]] ||
    die "kernel boot ID is unavailable"
  IFS= read -r value <"${boot_id_file}" ||
    die "could not read kernel boot ID"
  [[ "${value}" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
    die "kernel boot ID is malformed"
  printf '%s\n' "${value}"
}

manifest_value()
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

require_owned_regular_or_absent()
{
  local path=$1
  local metadata
  if [[ ! -e "${path}" && ! -L "${path}" ]]; then
    return 0
  fi
  [[ ! -L "${path}" && -f "${path}" ]] ||
    die "state entry is not a regular file: ${path}"
  metadata=$(stat -c '%u:%h:%a' "${path}") ||
    die "could not inspect state entry: ${path}"
  [[ "${metadata}" == 0:1:* ]] ||
    die "state entry is not uniquely root-owned: ${path}"
}

require_safe_directory()
{
  local path=$1
  if [[ -L "${path}" || ( -e "${path}" && ! -d "${path}" ) ]]; then
    die "state directory is unsafe: ${path}"
  fi
  mkdir -p "${path}"
  [[ "$(stat -c '%u' "${path}")" == 0 ]] ||
    die "state directory is not root-owned: ${path}"
  chmod 0700 "${path}"
}

require_existing_root_directory()
{
  local path=$1
  [[ ! -L "${path}" && -d "${path}" ]] ||
    die "required directory is unsafe: ${path}"
  [[ "$(stat -c '%u' "${path}")" == 0 ]] ||
    die "required directory is not root-owned: ${path}"
}

snapshot_marker()
{
  local prefix=$1
  local path=$2
  local before after digest attempt

  for attempt in 1 2 3; do
    if [[ ! -e "${path}" && ! -L "${path}" ]]; then
      printf 'marker_%s_present=0\n' "${prefix}"
      return 0
    fi
    require_owned_regular_or_absent "${path}"
    before=$(stat -c '%d:%i:%s:%u:%g:%f:%h:%Y' "${path}") || return 1
    digest=$(sha256sum "${path}" | awk '{print $1}') || return 1
    after=$(stat -c '%d:%i:%s:%u:%g:%f:%h:%Y' "${path}") || return 1
    if [[ "${before}" == "${after}" ]]; then
      printf 'marker_%s_present=1\n' "${prefix}"
      printf 'marker_%s_stat=%s\n' "${prefix}" "${before}"
      printf 'marker_%s_sha256=%s\n' "${prefix}" "${digest}"
      return 0
    fi
  done
  die "state entry changed while it was being snapshotted: ${path}"
}

for command in \
  awk basename cat chmod cmp cp date flock git mkdir mktemp mv pgrep python3 \
  rm sha256sum stat tr
do
  command -v "${command}" >/dev/null 2>&1 ||
    die "${command} is required"
done
[[ ! -L "${KVAF_DURABLE_STATE_HELPER}" &&
   -f "${KVAF_DURABLE_STATE_HELPER}" &&
   -r "${KVAF_DURABLE_STATE_HELPER}" ]] ||
  die "durable state helper is missing or unsafe"

require_safe_directory /data/.kvaf-state
require_safe_directory "${state_dir}"
require_safe_directory "${evidence_dir}"
require_owned_regular_or_absent "${pending_file}"
require_owned_regular_or_absent "${active_file}"
require_owned_regular_or_absent "${quarantine_file}"
require_owned_regular_or_absent "${reconcile_lock}"
exec 5>"${reconcile_lock}"
chmod 0600 "${reconcile_lock}"
flock -n 5 || die "another reboot reconciliation is active"

current_boot_id=$(read_boot_id)
reconcile_script_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')
durable_helper_sha256=$(sha256sum "${KVAF_DURABLE_STATE_HELPER}" | awk '{print $1}')
kvaf_git_head=$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null) ||
  die "could not identify the KVAF checkout"
[[ "${kvaf_git_head}" =~ ^[0-9a-f]{40}$ ]] ||
  die "KVAF checkout HEAD is malformed"

if [[ "${action}" == prepare ]]; then
  if [[ -e "${pending_file}" || -L "${pending_file}" ]]; then
    die "an unconsumed reboot proof already exists: ${pending_file}"
  fi
  new_evidence_temporary pending.manifest.tmp
  temporary=${kvaf_temporary_path}
  {
    printf 'schema=kvaf-a5-direct-reboot-proof-v1\n'
    printf 'before_boot_id=%s\n' "${current_boot_id}"
    printf 'kvaf_git_head=%s\n' "${kvaf_git_head}"
    printf 'reconcile_script_sha256=%s\n' "${reconcile_script_sha256}"
    printf 'durable_helper_sha256=%s\n' "${durable_helper_sha256}"
    printf 'prepared_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    snapshot_marker active "${active_file}"
    snapshot_marker quarantine "${quarantine_file}"
  } >"${temporary}"
  chmod 0600 "${temporary}"
  python3 "${KVAF_DURABLE_STATE_HELPER}" publish \
    "${temporary}" "${pending_file}"
  python3 "${KVAF_DURABLE_STATE_HELPER}" sync \
    "${pending_file}" "${evidence_dir}" "${state_dir}"
  echo "KVAF_A5_REBOOT_PREPARED"
  echo "BEFORE_BOOT_ID=${current_boot_id}"
  echo "EVIDENCE=${pending_file}"
  exit 0
fi

[[ -f "${pending_file}" && ! -L "${pending_file}" ]] ||
  die "prepare proof is missing: ${pending_file}"
[[ "$(manifest_value schema "${pending_file}")" == kvaf-a5-direct-reboot-proof-v1 ]] ||
  die "prepare proof schema is invalid"
before_boot_id=$(manifest_value before_boot_id "${pending_file}") ||
  die "prepare proof lacks before_boot_id"
[[ "${before_boot_id}" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] ||
  die "prepare proof boot ID is malformed"
if [[ "${before_boot_id}" == "${current_boot_id}" ]]; then
  die "OS reboot is not proven; boot ID did not change"
fi
[[ "$(manifest_value kvaf_git_head "${pending_file}")" == "${kvaf_git_head}" ]] ||
  die "KVAF checkout changed between reboot prepare and retire"
[[ "$(manifest_value reconcile_script_sha256 "${pending_file}")" == "${reconcile_script_sha256}" ]] ||
  die "reconcile script changed between prepare and retire"
[[ "$(manifest_value durable_helper_sha256 "${pending_file}")" == "${durable_helper_sha256}" ]] ||
  die "durable state helper changed between prepare and retire"

# Lock both the canonical deploy transaction and the active-owner identity.
# A reboot releases every old file description; failure here therefore means
# a new process started and retirement must not race it.
# Match the compatibility guard's lock order first.  An old checkout may know
# only these three deploy locks and must not start between pgrep and archive.
require_existing_root_directory /run/lock
require_safe_directory "$(dirname "${legacy_worktree_deploy_lock}")"
require_safe_directory "$(dirname "${legacy_parent_deploy_lock}")"
for lock_path in \
  "${legacy_run_deploy_lock}" \
  "${legacy_worktree_deploy_lock}" \
  "${legacy_parent_deploy_lock}"
do
  require_owned_regular_or_absent "${lock_path}"
done
exec 4<>"${legacy_run_deploy_lock}"
chmod 0600 "${legacy_run_deploy_lock}"
flock -n 4 || die "a legacy direct-URMA run owns ${legacy_run_deploy_lock}"
exec 3<>"${legacy_worktree_deploy_lock}"
chmod 0600 "${legacy_worktree_deploy_lock}"
flock -n 3 || die "a legacy direct-URMA run owns ${legacy_worktree_deploy_lock}"
exec 11<>"${legacy_parent_deploy_lock}"
chmod 0600 "${legacy_parent_deploy_lock}"
flock -n 11 || die "a legacy direct-URMA run owns ${legacy_parent_deploy_lock}"

for guarded in "${legacy_state_files[@]}"; do
  if [[ -e "${guarded}" || -L "${guarded}" ]]; then
    die "legacy active/quarantine state must be resolved separately: ${guarded}"
  fi
done
for lock_path in "${legacy_active_locks[@]}"; do
  [[ -e "${lock_path}" || -L "${lock_path}" ]] || continue
  require_owned_regular_or_absent "${lock_path}"
  flock -n "${lock_path}" -c true ||
    die "a legacy direct-URMA worker owns ${lock_path}"
done

for lock_path in "${deploy_lock}" "${active_lock}"; do
  require_owned_regular_or_absent "${lock_path}"
done
exec 9>"${deploy_lock}"
chmod 0600 "${deploy_lock}"
flock -n 9 || die "a direct-URMA deploy/run owns ${deploy_lock}"
exec 8>"${active_lock}"
chmod 0600 "${active_lock}"
flock -n 8 || die "a direct-URMA worker owns ${active_lock}"

# Cross-NPU diagnostic launchers use their own coordinator locks while
# reading canonical quarantine state.  Hold both so the process scan and
# marker comparison cannot race a new experimental dispatch.
require_safe_directory "$(dirname "${cross_lock}")"
require_safe_directory "$(dirname "${reuse_lock}")"
for lock_path in "${cross_lock}" "${reuse_lock}"; do
  require_owned_regular_or_absent "${lock_path}"
done
exec 7>"${cross_lock}"
chmod 0600 "${cross_lock}"
flock -n 7 || die "a cross-NPU experiment owns ${cross_lock}"
exec 6>"${reuse_lock}"
chmod 0600 "${reuse_lock}"
flock -n 6 || die "a cross-NPU reuse experiment owns ${reuse_lock}"

new_evidence_temporary unexpected-processes.tmp
process_snapshot=${kvaf_temporary_path}
set +e
pgrep -af \
    '[l]aunch_a5_cross_npu2|[r]un_a5_cross_npu2|[r]un_a5_direct_urma|[k]vaf_a5_aiv_direct' \
    >"${process_snapshot}" 2>/dev/null
pgrep_rc=$?
set -e
case "${pgrep_rc}" in
  0)
    cat "${process_snapshot}" >&2
    rm -f "${process_snapshot}"
    die "a KVAF direct-URMA process exists after reboot"
    ;;
  1)
    rm -f "${process_snapshot}"
    ;;
  *)
    rm -f "${process_snapshot}"
    die "could not inspect KVAF processes after reboot"
    ;;
esac

# The boot proof is valid only for the exact marker identities captured by
# prepare.  A marker created or replaced on the new boot must never be
# mistaken for pre-reboot state merely because the old proof still exists.
new_evidence_temporary expected-markers.tmp
expected_snapshot=${kvaf_temporary_path}
new_evidence_temporary actual-markers.tmp
actual_snapshot=${kvaf_temporary_path}
awk '/^marker_(active|quarantine)_/' "${pending_file}" \
  >"${expected_snapshot}"
{
  snapshot_marker active "${active_file}"
  snapshot_marker quarantine "${quarantine_file}"
} >"${actual_snapshot}"
if ! cmp -s "${expected_snapshot}" "${actual_snapshot}"; then
  echo "EXPECTED_MARKERS:" >&2
  cat "${expected_snapshot}" >&2
  echo "CURRENT_MARKERS:" >&2
  cat "${actual_snapshot}" >&2
  rm -f "${expected_snapshot}" "${actual_snapshot}"
  die "canonical state changed after reboot proof was prepared"
fi
rm -f "${expected_snapshot}" "${actual_snapshot}"

require_safe_directory "${retired_root}"
archive_dir=${retired_root}/${before_boot_id}-to-${current_boot_id}-$(date -u +%Y%m%dT%H%M%SZ)
[[ ! -e "${archive_dir}" && ! -L "${archive_dir}" ]] ||
  die "reboot retirement archive already exists"
mkdir "${archive_dir}"
chmod 0700 "${archive_dir}"

manifest_tmp=$(mktemp "${archive_dir}/retirement.manifest.tmp.XXXXXX") ||
  die "could not create retirement manifest"
temporary_paths+=("${manifest_tmp}")
{
  printf 'schema=kvaf-a5-direct-reboot-retirement-v1\n'
  printf 'before_boot_id=%s\n' "${before_boot_id}"
  printf 'current_boot_id=%s\n' "${current_boot_id}"
  printf 'retired_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"${manifest_tmp}"

for guarded in "${active_file}" "${quarantine_file}"; do
  [[ -e "${guarded}" || -L "${guarded}" ]] || continue
  require_owned_regular_or_absent "${guarded}"
  name=$(basename "${guarded}")
  source_sha=$(sha256sum "${guarded}" | awk '{print $1}')
  cp -p --no-preserve=ownership "${guarded}" "${archive_dir}/${name}"
  chmod 0600 "${archive_dir}/${name}"
  archived_marker_paths+=("${archive_dir}/${name}")
  archive_sha=$(sha256sum "${archive_dir}/${name}" | awk '{print $1}')
  [[ "${archive_sha}" == "${source_sha}" ]] ||
    die "archived state checksum changed: ${guarded}"
  printf 'retired_%s_sha256=%s\n' \
    "$(printf '%s' "${name}" | tr '.-' '__')" "${archive_sha}" \
    >>"${manifest_tmp}"
  printf 'retired_%s_stat=%s\n' \
    "$(printf '%s' "${name}" | tr '.-' '__')" \
    "$(stat -c '%d:%i:%s:%u:%g:%f:%h:%Y' "${archive_dir}/${name}")" \
    >>"${manifest_tmp}"
done

cp -p --no-preserve=ownership "${pending_file}" \
  "${archive_dir}/prepare.manifest"
chmod 0600 "${archive_dir}/prepare.manifest"
chmod 0600 "${manifest_tmp}"
mv "${manifest_tmp}" "${archive_dir}/retirement.manifest"
archive_sync_paths=(
  "${archive_dir}/prepare.manifest"
  "${archive_dir}/retirement.manifest"
  "${archived_marker_paths[@]}"
  "${archive_dir}"
  "${retired_root}"
  "${state_dir}"
)
python3 "${KVAF_DURABLE_STATE_HELPER}" sync \
  "${archive_sync_paths[@]}"

# Remove only after the archive and manifest are durable.  Lock files,
# results and every experimental/risk artifact are deliberately retained.
for guarded in "${active_file}" "${quarantine_file}" "${pending_file}"; do
  python3 "${KVAF_DURABLE_STATE_HELPER}" remove "${guarded}"
done
python3 "${KVAF_DURABLE_STATE_HELPER}" sync \
  "${evidence_dir}" "${state_dir}"

echo "KVAF_A5_REBOOT_STATE_RETIRED"
echo "BEFORE_BOOT_ID=${before_boot_id}"
echo "CURRENT_BOOT_ID=${current_boot_id}"
echo "ARCHIVE=${archive_dir}"
