#!/usr/bin/env bash

# One-paste build/package/run entry for the clean NPU2 validation path.
# Every delegated build/run command retains its normal machine-wide guards;
# this coordinator never removes locks or recovery evidence.

unset BASH_ENV ENV
unset -f exit builtin 2>/dev/null || true

echo "KVAF_A5_NPU2_CLEAN_PREPARE_COORDINATOR_DISABLED" >&2
echo "CAPABILITY_LEDGER=not-implemented" >&2
echo "No argument/environment/source/state/CANN/docker/NPU operation was attempted." >&2
builtin exit 2
if [[ 0 == 1 ]]; then

set -Ee
set -o pipefail
umask 077

if (( $# != 0 )); then
  echo "Usage: $0" >&2
  exit 2
fi
if (( EUID != 0 )); then
  echo "ERROR: run this entry as root inside vllm-ascend-cs-kvaf" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
KVAF_ASCEND_SET_ENV=${KVAF_ASCEND_SET_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
kvaf_cann_root=/usr/local/Ascend/cann-9.1.0
kvaf_cann_entry=${kvaf_cann_root}/set_env.sh
kvaf_cann_root_real=$(realpath -e -- "${kvaf_cann_root}") || {
  echo "ERROR: pinned CANN 9.1.0 root is missing" >&2
  exit 2
}
kvaf_cann_alias_real=$(realpath -e -- "${KVAF_ASCEND_SET_ENV}") || {
  echo "ERROR: CANN set_env.sh alias is unresolved" >&2
  exit 2
}
kvaf_cann_entry_real=$(realpath -e -- "${kvaf_cann_entry}") || {
  echo "ERROR: pinned CANN set_env.sh is missing" >&2
  exit 2
}
if [[ "${kvaf_cann_root_real}" != "${kvaf_cann_root}" ||
      -L "${kvaf_cann_root}" ||
      ! -d "${kvaf_cann_root}" ||
      "${kvaf_cann_alias_real}" != "${kvaf_cann_entry_real}" ||
      ! -f "${kvaf_cann_entry_real}" ||
      ! -r "${kvaf_cann_entry_real}" ||
      "${kvaf_cann_entry_real}" != "${kvaf_cann_root_real}/"* ]]; then
  echo "ERROR: CANN set_env.sh does not resolve inside pinned CANN 9.1.0" >&2
  echo "ALIAS=${KVAF_ASCEND_SET_ENV}" >&2
  echo "RESOLVED=${kvaf_cann_alias_real}" >&2
  exit 2
fi
kvaf_cann_entry_mode=$(stat -c '%a' "${kvaf_cann_entry_real}") || exit 2
if [[ "$(stat -c '%u:%h' "${kvaf_cann_entry_real}")" != 0:1 ]] ||
   (( (8#${kvaf_cann_entry_mode} & 8#022) != 0 )); then
  echo "ERROR: resolved CANN set_env.sh ownership or mode is unsafe" >&2
  exit 2
fi

# Some shared root shells do not export USER.  CANN setup may inspect it, so
# provide only the conventional root value for this root-only coordinator.
export USER=${USER:-root}
source "${kvaf_cann_entry_real}"
if [[ -z "${ASCEND_HOME_PATH:-}" ||
      "$(realpath -e -- "${ASCEND_HOME_PATH}")" != \
        "${kvaf_cann_root_real}" ]]; then
  echo "ERROR: CANN environment selected an unexpected ASCEND_HOME_PATH" >&2
  exit 2
fi
kvaf_bisheng_real=$(realpath -e -- "$(command -v bisheng)") || {
  echo "ERROR: pinned CANN environment did not provide bisheng" >&2
  exit 2
}
if [[ "${kvaf_bisheng_real}" != "${kvaf_cann_root_real}/"* ||
      ! -f "${kvaf_bisheng_real}" ||
      ! -x "${kvaf_bisheng_real}" ]]; then
  echo "ERROR: CANN environment selected an unexpected bisheng" >&2
  exit 2
fi

export KVAF_UMDK_ROOT=${KVAF_UMDK_ROOT:-/data/c00982601/umdk}
export KVAF_LIBUMMU_ROOT=${KVAF_LIBUMMU_ROOT:-/data/c00982601/libummu}
kvaf_pin_clean_path()
{
  local variable=$1
  local expected=$2
  local current=${!variable:-}

  if [[ -n "${current}" && "${current}" != "${expected}" ]]; then
    echo "ERROR: ${variable} cannot override the clean NPU2 artifact path" >&2
    echo "EXPECTED=${expected}" >&2
    echo "ACTUAL=${current}" >&2
    exit 2
  fi
  printf -v "${variable}" '%s' "${expected}"
  export "${variable}"
}

kvaf_pin_clean_path KVAF_DIRECT_PROVIDER_BUILD_DIR \
  "${ROOT_DIR}/build-a5-direct-urma-provider"
kvaf_pin_clean_path KVAF_A5_DIRECT_BUILD_DIR \
  "${ROOT_DIR}/build-a5-direct-urma"
if [[ -z "${KVAF_ASC_DEVKIT_PATH:-}" ]]; then
  if [[ -r /data/c00982601/asc-devkit-9.1.0-20260722/include/simt_api/asc_simt.h ]]; then
    export KVAF_ASC_DEVKIT_PATH=/data/c00982601/asc-devkit-9.1.0-20260722
  else
    export KVAF_ASC_DEVKIT_PATH=/data/c00982601/asc-devkit
  fi
fi
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}

for kvaf_required_path in \
  "${KVAF_UMDK_ROOT}" \
  "${KVAF_LIBUMMU_ROOT}" \
  "${KVAF_ASC_DEVKIT_PATH}"
do
  if [[ -L "${kvaf_required_path}" || ! -d "${kvaf_required_path}" ]]; then
    echo "ERROR: required fixed source/devkit directory is missing or unsafe" >&2
    echo "PATH=${kvaf_required_path}" >&2
    exit 2
  fi
done

kvaf_source_tag=$(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null) || {
  echo "ERROR: cannot identify the KVAF checkout" >&2
  exit 2
}
kvaf_log_tag="${kvaf_source_tag}_$(date +%Y%m%d_%H%M%S)_$$"

echo "KVAF_NPU2_PREPARE_STAGE=provider"
bash "${SCRIPT_DIR}/build_a5_direct_urma_provider.sh" \
  2>&1 | tee "/tmp/kvaf_npu2_provider_${kvaf_log_tag}.log"

echo "KVAF_NPU2_PREPARE_STAGE=aicpu-package"
bash "${SCRIPT_DIR}/package_a5_direct_urma_aicpu.sh" \
  2>&1 | tee "/tmp/kvaf_npu2_package_${kvaf_log_tag}.log"

echo "KVAF_NPU2_PREPARE_STAGE=direct-binary"
bash "${SCRIPT_DIR}/build_a5_direct_urma.sh" \
  2>&1 | tee "/tmp/kvaf_npu2_direct_${kvaf_log_tag}.log"

echo "KVAF_NPU2_PREPARE_STAGE=resource-free-owner-probe"
bash "${SCRIPT_DIR}/run_a5_aicpu_owner_probe.sh" \
  --logical-device-id 2 \
  --timeout-seconds 5 \
  2>&1 | tee "/tmp/kvaf_npu2_owner_probe_${kvaf_log_tag}.log"

echo "KVAF_NPU2_PREPARE=PASS"
# The coordinator is explicitly NPU2-owned.  Remove stale selectors from a
# prior interactive NPU7 shell; the fixed wrapper below reconstructs and
# verifies every value.  Risk-acceptance variables are intentionally retained
# so the clean wrapper rejects them instead of silently changing semantics.
unset KVAF_LOGICAL_DEVICE_ID
unset KVAF_PHYSICAL_DEVICE_ID
unset KVAF_HOST_EID
unset KVAF_DEVICE_EID
unset KVAF_PAIR_CHANNEL
unset KVAF_DIRECT_MANAGEMENT_MODE
unset KVAF_DIRECT_MANAGEMENT_JETTY_ID
unset KVAF_DIRECT_ALLOW_BUSY_NPU
exec bash "${SCRIPT_DIR}/run_a5_direct_urma_npu2_clean_validation.sh"
fi
/bin/sh -c 'exit 2'
