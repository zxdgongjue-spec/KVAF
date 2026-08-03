#!/usr/bin/env bash

# One-paste build/package/run entry for the clean NPU2 validation path.
# Every delegated build/run command retains its normal machine-wide guards;
# this coordinator never removes locks or recovery evidence.

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
if [[ -L "${KVAF_ASCEND_SET_ENV}" ||
      ! -f "${KVAF_ASCEND_SET_ENV}" ||
      ! -r "${KVAF_ASCEND_SET_ENV}" ]]; then
  echo "ERROR: CANN environment script is missing or unsafe" >&2
  echo "PATH=${KVAF_ASCEND_SET_ENV}" >&2
  exit 2
fi

# Some shared root shells do not export USER.  CANN setup may inspect it, so
# provide only the conventional root value for this root-only coordinator.
export USER=${USER:-root}
source "${KVAF_ASCEND_SET_ENV}"

export KVAF_UMDK_ROOT=${KVAF_UMDK_ROOT:-/data/c00982601/umdk}
export KVAF_LIBUMMU_ROOT=${KVAF_LIBUMMU_ROOT:-/data/c00982601/libummu}
export KVAF_DIRECT_PROVIDER_BUILD_DIR=${KVAF_DIRECT_PROVIDER_BUILD_DIR:-${ROOT_DIR}/build-a5-direct-urma-provider}
export KVAF_A5_DIRECT_BUILD_DIR=${KVAF_A5_DIRECT_BUILD_DIR:-${ROOT_DIR}/build-a5-direct-urma}
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
