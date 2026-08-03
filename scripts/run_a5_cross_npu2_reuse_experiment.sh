#!/usr/bin/env bash

# One-shot coordinator for the second, explicitly risk-accepted NPU2 reuse
# experiment.  It preserves both earlier incidents byte-for-byte and creates
# a source/token-owned provider, AICPU package, direct binary and state lane.

set -Eeuo pipefail
umask 077

kvaf_stage=${1:-}
if (( $# != 1 )) || [[ "${kvaf_stage}" != zero-wqe ]]; then
  echo "Usage: $0 zero-wqe" >&2
  exit 2
fi

kvaf_risk_phrase=I_ACCEPT_UNKNOWN_NPU2_REUSE_AND_SHARED_UMMU_RISK
if [[ "${KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE:-}" != \
      "${kvaf_risk_phrase}" ]]; then
  echo "ERROR: export the exact NPU2 reuse/shared UMMU risk phrase" >&2
  exit 2
fi
if [[ ! -e /.dockerenv || ${EUID} -ne 0 ]]; then
  echo "ERROR: run this entry point as uid 0 inside vllm-ascend-cs-kvaf" >&2
  exit 2
fi
: "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
source "${SCRIPT_DIR}/a5_cross_npu2_reuse_state.sh"
source "${SCRIPT_DIR}/a5_direct_urma_manifest_v3.sh"
source "${SCRIPT_DIR}/a5_direct_urma_runtime_preflight.sh"
source "${SCRIPT_DIR}/a5_cross_npu2_versioned_runtime_preflight.sh"

for kvaf_command in awk cat chmod git grep ln mkdir mktemp npu-smi python3 \
  realpath rm sha256sum stat timeout tr; do
  command -v "${kvaf_command}" >/dev/null 2>&1 || {
    echo "ERROR: ${kvaf_command} is required" >&2
    exit 2
  }
done

kvaf_token=${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN:-}
if [[ -z "${kvaf_token}" ]]; then
  if [[ ! -r /proc/sys/kernel/random/uuid ]]; then
    echo "ERROR: cannot generate a one-shot artifact token" >&2
    exit 2
  fi
  kvaf_uuid=$(tr -d '-' </proc/sys/kernel/random/uuid)
  kvaf_token=${kvaf_uuid:0:16}
fi
if [[ ! "${kvaf_token}" =~ ^[0-9a-f]{16}$ ]]; then
  echo "ERROR: KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN must be 16 lowercase hex digits" >&2
  exit 2
fi
export KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN=${kvaf_token}

kvaf_source_run_id=530f5998a0b646d48db858170a9996ff
kvaf_prior_run_id=30ef7336d212459a9317fd6647f6fa15
kvaf_configure_cross_npu2_reuse_state_paths \
  "${ROOT_DIR}" "${kvaf_source_run_id}" "${kvaf_prior_run_id}" \
  "${kvaf_token}" || exit 2

kvaf_source_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD) || exit 2
if [[ ! "${kvaf_source_commit}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "ERROR: current KVAF commit is not canonical" >&2
  exit 2
fi

kvaf_require_clean_source()
{
  if ! git -C "${ROOT_DIR}" diff-index --quiet HEAD -- ||
     [[ -n "$(git -C "${ROOT_DIR}" ls-files --others --exclude-standard)" ]]; then
    echo "ERROR: the one-shot experiment requires a clean KVAF checkout" >&2
    return 2
  fi
  if [[ "$(git -C "${ROOT_DIR}" rev-parse HEAD)" != \
        "${kvaf_source_commit}" ]]; then
    echo "ERROR: the KVAF source commit changed during the experiment" >&2
    return 2
  fi
}

kvaf_require_clean_source
kvaf_attempt_claim="/data/.kvaf-state/KVAF/experimental/npu-2-reuse/${kvaf_source_run_id}/${kvaf_prior_run_id}/attempt.selected"
if [[ -e "${kvaf_attempt_claim}" || -L "${kvaf_attempt_claim}" ]]; then
  echo "ERROR: the one permitted second-order NPU2 attempt was already selected" >&2
  echo "ATTEMPT_CLAIM=${kvaf_attempt_claim}" >&2
  if [[ ! -L "${kvaf_attempt_claim}" &&
        -f "${kvaf_attempt_claim}" &&
        -r "${kvaf_attempt_claim}" ]]; then
    cat "${kvaf_attempt_claim}" >&2
  fi
  echo "A new token and a same-token retry are both forbidden." >&2
  exit 2
fi
kvaf_artifact_root="/data/.kvaf-state/KVAF/experimental/reuse-artifacts/${kvaf_source_commit}/${kvaf_token}"
kvaf_package_name="cann-kvaf-du-${kvaf_token}.tar.gz"
kvaf_ascend_home=$(realpath -e "${ASCEND_HOME_PATH}") || exit 2
kvaf_package_path="${kvaf_ascend_home}/opp/vendors/cust/op_impl/aicpu/kernel/${kvaf_package_name}"
for kvaf_fresh_path in \
  "${kvaf_artifact_root}" \
  "${KVAF_CROSS_NPU2_REUSE_STATE_DIR}" \
  "${kvaf_package_path}"
do
  if [[ -e "${kvaf_fresh_path}" || -L "${kvaf_fresh_path}" ]]; then
    echo "ERROR: one-shot token already owns state or an artifact" >&2
    echo "PATH=${kvaf_fresh_path}" >&2
    echo "Choose a new 16-hex token; do not reuse or clean this generation." >&2
    exit 2
  fi
done

kvaf_reuse_validator="${SCRIPT_DIR}/validate_a5_cross_npu2_reuse_risk.py"
if [[ -L "${kvaf_reuse_validator}" ||
      ! -f "${kvaf_reuse_validator}" ||
      ! -r "${kvaf_reuse_validator}" ]]; then
  echo "ERROR: the pinned reuse-evidence validator is missing or unsafe" >&2
  exit 2
fi
kvaf_evidence_reference=
kvaf_evidence_reference_set=0

kvaf_validate_reuse_evidence()
{
  local stage=$1
  local evidence evidence_sha

  evidence=$(python3 "${kvaf_reuse_validator}" \
    --new-artifact-token "${kvaf_token}") || return 2
  if (( kvaf_evidence_reference_set == 0 )); then
    kvaf_evidence_reference=${evidence}
    kvaf_evidence_reference_set=1
  elif [[ "${evidence}" != "${kvaf_evidence_reference}" ]]; then
    echo "ERROR: accepted NPU7/NPU2 evidence changed during ${stage}" >&2
    return 2
  fi
  evidence_sha=$(printf '%s\n' "${evidence}" | sha256sum | awk '{print $1}')
  if [[ -n "${KVAF_CROSS_NPU2_REUSE_VALIDATION_SHA256:-}" &&
        "${KVAF_CROSS_NPU2_REUSE_VALIDATION_SHA256}" != \
          "${evidence_sha}" ]]; then
    echo "ERROR: accepted evidence digest changed during ${stage}" >&2
    return 2
  fi
  KVAF_CROSS_NPU2_REUSE_VALIDATION_SHA256=${evidence_sha}
  echo "KVAF_CROSS_NPU2_REUSE_EVIDENCE_OK stage=${stage} sha256=${evidence_sha}"
}

kvaf_prove_npu2_idle()
{
  local stage=$1
  local inventory inventory_rc parsed parsed_rc

  set +e
  inventory=$(LC_ALL=C timeout 15s npu-smi info 2>&1)
  inventory_rc=$?
  set -e
  if (( inventory_rc != 0 )); then
    echo "ERROR: npu-smi could not prove NPU2 idle during ${stage}; rc=${inventory_rc}" >&2
    printf '%s\n' "${inventory}" >&2
    return 2
  fi

  set +e
  parsed=$(python3 "${SCRIPT_DIR}/check_a5_npu_idle.py" \
    --physical-device-id 2 <<<"${inventory}" 2>&1)
  parsed_rc=$?
  set -e
  if (( parsed_rc != 0 )) ||
     ! grep -Fxq KVAF_NPU_IDLE_STATUS=idle <<<"${parsed}"; then
    echo "ERROR: target physical NPU2 is not strictly proven idle during ${stage}" >&2
    printf '%s\n' "${parsed}" >&2
    return 2
  fi
  echo "KVAF_A5_NPU_IDLE_PROVEN physical_device_id=2 stage=${stage}"
}

kvaf_publish_one_shot_claim()
{
  local claim_parent claim temporary expected actual identity

  claim_parent="/data/.kvaf-state/KVAF/experimental/npu-2-reuse/${kvaf_source_run_id}/${kvaf_prior_run_id}"
  claim=${kvaf_attempt_claim}
  if [[ -e "${claim}" || -L "${claim}" ]]; then
    echo "ERROR: the one permitted second-order NPU2 attempt was already selected" >&2
    echo "ATTEMPT_CLAIM=${claim}" >&2
    if [[ ! -L "${claim}" && -f "${claim}" && -r "${claim}" ]]; then
      cat "${claim}" >&2
    fi
    echo "A new token and a same-token retry are both forbidden." >&2
    return 2
  fi
  if [[ -L "${claim_parent}" ||
        ( -e "${claim_parent}" && ! -d "${claim_parent}" ) ]]; then
    echo "ERROR: the fixed one-shot claim parent is unsafe" >&2
    return 2
  fi
  mkdir -p "${claim_parent}"
  chmod 0700 "${claim_parent}"
  if [[ "$(realpath -e "${claim_parent}")" != "${claim_parent}" ]]; then
    echo "ERROR: the fixed one-shot claim parent is not canonical" >&2
    return 2
  fi
  expected=$(printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-reuse-attempt-v1' \
    "source_run_id=${kvaf_source_run_id}" \
    "prior_target_run_id=${kvaf_prior_run_id}" \
    "artifact_token=${kvaf_token}" \
    "source_commit=${kvaf_source_commit}" \
    'execution_contract=risk-accepted-cross-npu2-reuse-v2')
  temporary=$(mktemp "${claim_parent}/.attempt.selected.XXXXXX")
  printf '%s\n' "${expected}" >"${temporary}"
  chmod 0400 "${temporary}"
  if ! ln "${temporary}" "${claim}"; then
    rm -f "${temporary}"
    echo "ERROR: another process selected the one permitted attempt" >&2
    echo "ATTEMPT_CLAIM=${claim}" >&2
    return 2
  fi
  rm -f "${temporary}"
  actual=$(cat "${claim}") || return 2
  identity=$(stat -c '%h:%a' "${claim}") || return 2
  if [[ "${actual}" != "${expected}" || "${identity}" != 1:400 ]]; then
    echo "ERROR: the published one-shot claim is not exact and sealed" >&2
    return 2
  fi
  KVAF_CROSS_NPU2_REUSE_ATTEMPT_CLAIM_FILE=${claim}
  export KVAF_CROSS_NPU2_REUSE_ATTEMPT_CLAIM_FILE
  echo "KVAF_CROSS_NPU2_REUSE_ATTEMPT_SELECTED=${claim}"
}

export KVAF_UMDK_ROOT=${KVAF_UMDK_ROOT:-/data/c00982601/umdk}
export KVAF_LIBUMMU_ROOT=${KVAF_LIBUMMU_ROOT:-/data/c00982601/libummu}
export KVAF_DIRECT_PROVIDER_BUILD_DIR="${kvaf_artifact_root}/provider"
export KVAF_A5_DIRECT_BUILD_DIR="${kvaf_artifact_root}/direct"
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}

echo "WARNING: starting one second-order cross-NPU2 zero-WQE experiment" >&2
echo "CONTRACT=risk-accepted-cross-npu2-reuse-v2" >&2
echo "ARTIFACT_TOKEN=${kvaf_token}"
kvaf_validate_reuse_evidence initial
kvaf_publish_one_shot_claim
kvaf_validate_reuse_evidence after-attempt-selection

echo "STAGE=build-versioned-provider"
bash "${SCRIPT_DIR}/build_a5_direct_urma_provider.sh" \
  cross-npu2-reuse-versioned-offline-v1
kvaf_validate_reuse_evidence after-provider-build

echo "STAGE=install-versioned-aicpu-package"
bash "${SCRIPT_DIR}/package_a5_cross_npu2_reuse_aicpu.sh"
kvaf_validate_reuse_evidence after-package-install

echo "STAGE=build-versioned-direct-binaries"
bash "${SCRIPT_DIR}/build_a5_direct_urma.sh" \
  cross-npu2-reuse-versioned-offline-v1
kvaf_validate_reuse_evidence after-direct-build

echo "STAGE=versioned-runtime-preflight"
kvaf_require_clean_source
kvaf_verify_cross_npu2_versioned_runtime_artifacts
kvaf_expected_provider_soname="libkvaf_ud_${kvaf_token}.so"
if [[ "${KVAF_DIRECT_PROVIDER_SONAME}" != \
      "${kvaf_expected_provider_soname}" ]]; then
  echo "ERROR: versioned preflight selected an unexpected Device SO" >&2
  exit 2
fi
export KVAF_A5_DIRECT_BINARY KVAF_A5_OWNER_PROBE_BINARY
export KVAF_A5_DIRECT_BUILD_MANIFEST
export KVAF_DIRECT_HOST_PROVIDER_LIBRARY
export KVAF_DIRECT_DEVICE_PROVIDER_LIBRARY
export KVAF_DIRECT_PROVIDER_SOURCE_MANIFEST
export KVAF_DIRECT_PROVIDER_BUILD_MANIFEST
export KVAF_DIRECT_PROVIDER_SUCCESS_STAMP
export KVAF_DIRECT_PROVIDER_SONAME
export KVAF_DIRECT_AICPU_ARCHIVE
export KVAF_DIRECT_AICPU_REGISTRATION_FILE
export KVAF_CROSS_NPU2_PROVIDER_PROFILE
export KVAF_CROSS_NPU2_PACKAGE_PROFILE
export KVAF_CROSS_NPU2_DIRECT_PROFILE
kvaf_validate_reuse_evidence before-owner-probe
kvaf_prove_npu2_idle before-owner-probe

echo "STAGE=resource-free-versioned-aicpu-owner-probe"
set +e
"${KVAF_A5_OWNER_PROBE_BINARY}" \
  --logical-device-id 2 \
  --timeout-seconds 5 \
  --provider-soname "${KVAF_DIRECT_PROVIDER_SONAME}"
kvaf_probe_rc=$?
set -e
kvaf_validate_reuse_evidence after-owner-probe
if (( kvaf_probe_rc != 0 )); then
  echo "HARD_STOP: unique-SO resource-free AICPU probe failed; no URMA/UMMU resource was requested" >&2
  echo "Do not retry this token or alter either earlier incident." >&2
  exit "${kvaf_probe_rc}"
fi
kvaf_prove_npu2_idle after-owner-probe

export KVAF_LOGICAL_DEVICE_ID=2
export KVAF_PHYSICAL_DEVICE_ID=2
export KVAF_HOST_EID=00000000007f040000100000df0a0b00
export KVAF_DEVICE_EID=000000000000020000100000df004100
export KVAF_PAIR_CHANNEL=0
export KVAF_HOST_URMA_DEVICE=udmac1d1e4
export KVAF_HOST_EID_INDEX=0
export KVAF_DEVICE_URMA_DEVICE=udmac0d0e2
export KVAF_DEVICE_EID_INDEX=0
export KVAF_DIRECT_MANAGEMENT_MODE=ub
export KVAF_DIRECT_MANAGEMENT_JETTY_ID=602
export KVAF_H2D_TRANSFER_COUNT=0
export KVAF_H2D_BYTES_PER_TRANSFER=1024
export KVAF_H2D_BATCH_ITERATIONS=1
export KVAF_H2D_DESCRIPTOR_CAPACITY=1
export KVAF_H2D_TIMEOUT_SECONDS=120
export KVAF_H2D_RUN_TIMEOUT_SECONDS=180
export KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS=30
export KVAF_NPU_SMI_TIMEOUT_SECONDS=15
export KVAF_DIRECT_ALLOW_BUSY_NPU=0
export KVAF_A5_RUN_TAG="npu2_reuse_zero_wqe_${kvaf_token}"
unset KVAF_H2D_BYTES
unset KVAF_DIRECT_CONTROL_PORT
unset KVAF_DIRECT_CONTROL_SERVER_ADDRESS
unset KVAF_DIRECT_CONTROL_BIND_ADDRESS
unset ASCEND_RT_VISIBLE_DEVICES

kvaf_validate_reuse_evidence before-zero-wqe-run
kvaf_require_clean_source
echo "STAGE=npu2-versioned-zero-wqe"
set +e
bash "${SCRIPT_DIR}/run_a5_direct_urma_h2d_smoke.sh" \
  risk-accepted-cross-npu2-reuse-v2
kvaf_run_rc=$?
set -e
echo "KVAF_NPU2_REUSE_ZERO_WQE_RC=${kvaf_run_rc}"
if (( kvaf_run_rc != 0 )); then
  echo "HARD_STOP: retain the new generation and both earlier incidents exactly as they are" >&2
  echo "No automatic retry, process signal, state deletion, or fallback is permitted." >&2
  exit "${kvaf_run_rc}"
fi
echo "KVAF_NPU2_REUSE_ZERO_WQE_ACCEPTANCE=PASS"
