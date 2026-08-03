#!/usr/bin/env bash

# Explicitly dangerous, narrowly scoped recovery experiment.
#
# This entry does not reclassify or remove the NPU7 quarantine.  It accepts
# one exact legacy zero-WQE incident, leaves its workers/backing untouched,
# and targets only the disjoint NPU2 UDMA endpoints.  UMMU/AICPU isolation
# across NPUs is still unproven, so this is not a production bypass.

unset BASH_ENV ENV
unset -f exit builtin 2>/dev/null || true

echo "KVAF_A5_CROSS_NPU2_RISK_ACCEPTED_COORDINATOR_DISABLED" >&2
echo "CAPABILITY_LEDGER=not-implemented" >&2
echo "No argument/environment/source/state/CANN/docker/NPU operation was attempted." >&2
builtin exit 2
if [[ 0 == 1 ]]; then

set -Eeuo pipefail
umask 077

kvaf_stage=${1:-}
if (( $# != 1 )) ||
   [[ "${kvaf_stage}" != zero-wqe && "${kvaf_stage}" != one-kib ]]; then
  echo "Usage: $0 {zero-wqe|one-kib}" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)

readonly kvaf_acceptance_phrase=I_ACCEPT_UNKNOWN_CROSS_NPU_DMA_RISK
if [[ "${KVAF_CROSS_NPU_RISK_ACCEPTANCE:-}" != \
      "${kvaf_acceptance_phrase}" ]]; then
  echo "ERROR: cross-NPU execution requires the exact acceptance phrase" >&2
  echo "This experiment may still corrupt shared UMMU state and require an OS reboot." >&2
  exit 2
fi
: "${KVAF_CROSS_NPU_SOURCE_RUN_ID:?set the pinned NPU7 run ID}"
: "${KVAF_CROSS_NPU_SOURCE_QUARANTINE_SHA256:?set the pinned NPU7 quarantine SHA256}"
: "${KVAF_CROSS_NPU_ARTIFACT_ID:?set the offline artifact ID}"

if [[ ! "${KVAF_CROSS_NPU_SOURCE_RUN_ID}" =~ ^[0-9a-f]{32,128}$ ||
      ! "${KVAF_CROSS_NPU_SOURCE_QUARANTINE_SHA256}" =~ ^[0-9a-f]{64}$ ||
      ! "${KVAF_CROSS_NPU_ARTIFACT_ID}" =~ ^[0-9a-f]{32}$ ]]; then
  echo "ERROR: run ID, quarantine SHA256, or artifact ID is not canonical" >&2
  exit 2
fi

kvaf_source_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD) || exit 2
if [[ ! "${kvaf_source_commit}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: could not pin the KVAF source commit" >&2
  exit 2
fi
kvaf_expected_build_dir="/data/.kvaf-state/KVAF/experimental/artifacts/${kvaf_source_commit}/${KVAF_CROSS_NPU_ARTIFACT_ID}"
if [[ -n "${KVAF_A5_DIRECT_BUILD_DIR+x}" &&
      "${KVAF_A5_DIRECT_BUILD_DIR}" != "${kvaf_expected_build_dir}" ]]; then
  echo "ERROR: KVAF_A5_DIRECT_BUILD_DIR does not select the pinned offline artifact" >&2
  echo "EXPECTED=${kvaf_expected_build_dir}" >&2
  echo "ACTUAL=${KVAF_A5_DIRECT_BUILD_DIR}" >&2
  exit 2
fi
export KVAF_A5_DIRECT_BUILD_DIR=${kvaf_expected_build_dir}
unset KVAF_A5_DIRECT_BINARY
unset KVAF_A5_DIRECT_BUILD_MANIFEST

kvaf_expected_provider_dir="${ROOT_DIR}/build-a5-direct-urma-provider"
if [[ -n "${KVAF_DIRECT_PROVIDER_BUILD_DIR+x}" &&
      "${KVAF_DIRECT_PROVIDER_BUILD_DIR}" != \
        "${kvaf_expected_provider_dir}" ]]; then
  echo "ERROR: risk mode requires the source-verified default provider build" >&2
  echo "EXPECTED=${kvaf_expected_provider_dir}" >&2
  echo "ACTUAL=${KVAF_DIRECT_PROVIDER_BUILD_DIR}" >&2
  exit 2
fi
export KVAF_DIRECT_PROVIDER_BUILD_DIR=${kvaf_expected_provider_dir}
unset KVAF_DIRECT_HOST_PROVIDER_LIBRARY
unset KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY
unset KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST
unset KVAF_DIRECT_PROVIDER_BUILD_MANIFEST
unset KVAF_DIRECT_PROVIDER_SUCCESS_STAMP
unset KVAF_DIRECT_AICPU_ARCHIVE
unset KVAF_DIRECT_AICPU_REGISTRATION_FILE

kvaf_artifact_manifest="${KVAF_A5_DIRECT_BUILD_DIR}/cross-npu2-artifact.manifest"
if [[ -L "${kvaf_artifact_manifest}" ||
      ! -f "${kvaf_artifact_manifest}" ||
      ! -r "${kvaf_artifact_manifest}" ]]; then
  echo "ERROR: build the fresh cross-NPU2 offline artifact first" >&2
  echo "MANIFEST=${kvaf_artifact_manifest}" >&2
  exit 2
fi
if ! grep -Fxq "schema=kvaf-a5-cross-npu2-offline-artifact-v1" \
       "${kvaf_artifact_manifest}" ||
   ! grep -Fxq "source_commit=${kvaf_source_commit}" \
       "${kvaf_artifact_manifest}" ||
   ! grep -Fxq "artifact_id=${KVAF_CROSS_NPU_ARTIFACT_ID}" \
       "${kvaf_artifact_manifest}" ||
   ! grep -Fxq "source_run_id=${KVAF_CROSS_NPU_SOURCE_RUN_ID}" \
       "${kvaf_artifact_manifest}" ||
   ! grep -Fxq \
       "source_quarantine_sha256=${KVAF_CROSS_NPU_SOURCE_QUARANTINE_SHA256}" \
       "${kvaf_artifact_manifest}"; then
  echo "ERROR: offline artifact identity does not match this acceptance" >&2
  exit 2
fi

python3 "${SCRIPT_DIR}/validate_a5_cross_npu_risk_acceptance.py" \
  --quarantine /data/.kvaf-state/KVAF/direct-urma.quarantine \
  --quarantine-sha256 \
    "${KVAF_CROSS_NPU_SOURCE_QUARANTINE_SHA256}" \
  --run-id "${KVAF_CROSS_NPU_SOURCE_RUN_ID}" \
  --source-logical-device-id 7 \
  --result-root /data/.kvaf-state/KVAF/results

# This wrapper owns every target selector.  Exported values cannot redirect
# the risk path to NPU7, another UDMA endpoint, TCP management, or a busy NPU.
export KVAF_LOGICAL_DEVICE_ID=2
export KVAF_PHYSICAL_DEVICE_ID=2
export KVAF_HOST_EID=00000000007f040000100000df0a0b00
export KVAF_DEVICE_EID=000000000000020000100000df004100
export KVAF_PAIR_CHANNEL=0
# Device names are scoped to the selected Host/AICPU inventory.  Do not
# infer them from the Host /dev/uburma namespace or preserve caller
# overrides: the main runner resolves both fixed EIDs live and then applies
# the pinned NPU2 risk contract before creating any resource.
unset KVAF_HOST_URMA_DEVICE
unset KVAF_HOST_EID_INDEX
unset KVAF_DEVICE_URMA_DEVICE
unset KVAF_DEVICE_EID_INDEX
export KVAF_DIRECT_MANAGEMENT_MODE=ub
export KVAF_DIRECT_MANAGEMENT_JETTY_ID=502
export KVAF_DIRECT_ALLOW_BUSY_NPU=0
export KVAF_H2D_BYTES_PER_TRANSFER=1024
export KVAF_H2D_BATCH_ITERATIONS=1
export KVAF_H2D_DESCRIPTOR_CAPACITY=1
export KVAF_H2D_TIMEOUT_SECONDS=${KVAF_H2D_TIMEOUT_SECONDS:-60}
export KVAF_H2D_RUN_TIMEOUT_SECONDS=${KVAF_H2D_RUN_TIMEOUT_SECONDS:-120}
export KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS=${KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS:-30}
export KVAF_A5_RUN_TAG="npu2_${kvaf_stage}_risk_${KVAF_CROSS_NPU_ARTIFACT_ID}_$(date +%Y%m%d_%H%M%S)_$$"

unset KVAF_H2D_BYTES
unset KVAF_A5_RESULT_DIR
unset KVAF_DIRECT_CONTROL_PORT
unset KVAF_DIRECT_CONTROL_SERVER_ADDRESS
unset KVAF_DIRECT_CONTROL_BIND_ADDRESS
unset KVAF_DIRECT_SERVER_START_DELAY_SECONDS
export KVAF_EID_PROC_ROOT=/proc/asdrv_ub
export KVAF_UBCORE_SYSFS_ROOT=/sys/class/ubcore
export KVAF_HCCN_TOOL=/usr/local/Ascend/driver/tools/hccn_tool
unset ASCEND_RT_VISIBLE_DEVICES

if [[ "${kvaf_stage}" == zero-wqe ]]; then
  export KVAF_H2D_TRANSFER_COUNT=0
else
  : "${KVAF_CROSS_NPU_ZERO_WQE_RESULT_DIR:?set the proven NPU2 zero-WQE result directory}"
  python3 "${SCRIPT_DIR}/validate_a5_cross_npu_zero_wqe_pass.py" \
    --result-dir "${KVAF_CROSS_NPU_ZERO_WQE_RESULT_DIR}" \
    --source-run-id "${KVAF_CROSS_NPU_SOURCE_RUN_ID}" \
    --source-quarantine-sha256 \
      "${KVAF_CROSS_NPU_SOURCE_QUARANTINE_SHA256}" \
    --artifact-id "${KVAF_CROSS_NPU_ARTIFACT_ID}" \
    --source-commit "${kvaf_source_commit}"
  export KVAF_H2D_TRANSFER_COUNT=1
fi

echo "WARNING: accepting unproven cross-NPU UMMU/AICPU isolation" >&2
echo "SOURCE_NPU=7 TARGET_NPU=2 STAGE=${kvaf_stage}" >&2
echo "The NPU7 quarantine and workers will not be modified or signalled." >&2

exec bash "${SCRIPT_DIR}/run_a5_direct_urma_h2d_smoke.sh" \
  risk-accepted-cross-npu2-v1
fi
/bin/sh -c 'exit 2'
