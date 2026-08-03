#!/usr/bin/env bash

# Operator-facing coordinator for the two deliberately tiny NPU2 stages.
# The lower-level wrapper and runner still own all safety checks; this script
# only derives the exact quarantine identity, creates a fresh offline binary,
# and carries a proven zero-WQE result into the one-KiB stage.

set -Eeuo pipefail
umask 077

kvaf_stage=${1:-}
if (( $# != 1 )) ||
   [[ "${kvaf_stage}" != zero-wqe && "${kvaf_stage}" != one-kib ]]; then
  echo "Usage: $0 {zero-wqe|one-kib}" >&2
  exit 2
fi
if [[ "${KVAF_CROSS_NPU_RISK_ACCEPTANCE:-}" != \
      I_ACCEPT_UNKNOWN_CROSS_NPU_DMA_RISK ]]; then
  echo "ERROR: export the exact cross-NPU DMA risk acceptance phrase" >&2
  exit 2
fi
: "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
kvaf_source_quarantine=/data/.kvaf-state/KVAF/direct-urma.quarantine
if [[ -L "${kvaf_source_quarantine}" ||
      ! -f "${kvaf_source_quarantine}" ||
      ! -r "${kvaf_source_quarantine}" ]]; then
  echo "ERROR: the expected NPU7 quarantine is unavailable" >&2
  exit 2
fi

single_value()
{
  local key=$1
  local path=$2
  awk -F= -v key="${key}" '
    $1 == key {
      value = substr($0, length(key) + 2)
      count++
    }
    END {
      if (count != 1 || value == "") exit 1
      print value
    }
  ' "${path}"
}

kvaf_source_run_id=$(single_value run_id "${kvaf_source_quarantine}") || {
  echo "ERROR: source quarantine does not contain one run_id" >&2
  exit 2
}
kvaf_source_quarantine_sha=$(
  sha256sum "${kvaf_source_quarantine}" | awk '{print $1}'
)
if [[ ! "${kvaf_source_run_id}" =~ ^[0-9a-f]{32,128}$ ||
      ! "${kvaf_source_quarantine_sha}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "ERROR: source quarantine identity is not canonical" >&2
  exit 2
fi
kvaf_source_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD) || exit 2
kvaf_lane="/data/.kvaf-state/KVAF/experimental/npu-2/${kvaf_source_run_id}"
kvaf_handoff="${kvaf_lane}/zero-wqe-pass.handoff"

export KVAF_CROSS_NPU_SOURCE_RUN_ID=${kvaf_source_run_id}
export KVAF_CROSS_NPU_SOURCE_QUARANTINE_SHA256=${kvaf_source_quarantine_sha}
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:/usr/lib64:${LD_LIBRARY_PATH:-}

if [[ "${kvaf_stage}" == zero-wqe ]]; then
  if [[ -e "${kvaf_handoff}" || -L "${kvaf_handoff}" ]]; then
    echo "ERROR: a proven zero-WQE handoff already exists" >&2
    echo "HANDOFF=${kvaf_handoff}" >&2
    echo "Run the one-kib stage instead of repeating resource creation." >&2
    exit 2
  fi
  if [[ ! -r /proc/sys/kernel/random/uuid ]]; then
    echo "ERROR: cannot generate a fresh experimental artifact ID" >&2
    exit 2
  fi
  KVAF_CROSS_NPU_ARTIFACT_ID=$(tr -d '-' </proc/sys/kernel/random/uuid)
  export KVAF_CROSS_NPU_ARTIFACT_ID
  export KVAF_A5_DIRECT_BUILD_DIR="/data/.kvaf-state/KVAF/experimental/artifacts/${kvaf_source_commit}/${KVAF_CROSS_NPU_ARTIFACT_ID}"

  echo "STAGE=offline-build"
  bash "${SCRIPT_DIR}/build_a5_direct_urma.sh" \
    cross-npu2-risk-offline-v1

  kvaf_log="/tmp/kvaf_npu2_zero_wqe_${KVAF_CROSS_NPU_ARTIFACT_ID}.log"
  set +e
  bash "${SCRIPT_DIR}/run_a5_direct_urma_cross_npu2_risk_accepted.sh" \
    zero-wqe 2>&1 | tee "${kvaf_log}"
  kvaf_rc=${PIPESTATUS[0]}
  set -e
  echo "KVAF_NPU2_ZERO_WQE_RC=${kvaf_rc}"
  if (( kvaf_rc != 0 )); then
    echo "HARD_STOP: retain every active/quarantine/worker and inspect ${kvaf_log}" >&2
    exit "${kvaf_rc}"
  fi
  kvaf_zero_result=$(
    awk '
      /^RESULT_DIR=/ {
        value = substr($0, length("RESULT_DIR=") + 1)
      }
      END {
        if (value == "") exit 1
        print value
      }
    ' "${kvaf_log}"
  ) || {
    echo "ERROR: zero-WQE PASS did not expose its result directory" >&2
    exit 5
  }
  python3 "${SCRIPT_DIR}/validate_a5_cross_npu_zero_wqe_pass.py" \
    --result-dir "${kvaf_zero_result}" \
    --source-run-id "${kvaf_source_run_id}" \
    --source-quarantine-sha256 "${kvaf_source_quarantine_sha}" \
    --artifact-id "${KVAF_CROSS_NPU_ARTIFACT_ID}" \
    --source-commit "${kvaf_source_commit}"

  mkdir -p "${kvaf_lane}"
  if [[ -L "${kvaf_lane}" || ! -d "${kvaf_lane}" ]]; then
    echo "ERROR: NPU2 experiment lane is not a real directory" >&2
    exit 2
  fi
  kvaf_handoff_temporary="${kvaf_handoff}.tmp.$$"
  if [[ -e "${kvaf_handoff_temporary}" ||
        -L "${kvaf_handoff_temporary}" ]]; then
    echo "ERROR: unsafe handoff temporary path" >&2
    exit 2
  fi
  printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-zero-wqe-handoff-v1' \
    "source_run_id=${kvaf_source_run_id}" \
    "source_quarantine_sha256=${kvaf_source_quarantine_sha}" \
    "source_commit=${kvaf_source_commit}" \
    "artifact_id=${KVAF_CROSS_NPU_ARTIFACT_ID}" \
    "build_dir=${KVAF_A5_DIRECT_BUILD_DIR}" \
    "zero_wqe_result_dir=${kvaf_zero_result}" \
    >"${kvaf_handoff_temporary}"
  chmod 0400 "${kvaf_handoff_temporary}"
  if ! ln "${kvaf_handoff_temporary}" "${kvaf_handoff}"; then
    echo "ERROR: handoff appeared concurrently; refusing to overwrite it" >&2
    exit 5
  fi
  rm -f "${kvaf_handoff_temporary}"
  echo "KVAF_NPU2_ZERO_WQE_ACCEPTANCE=PASS"
  echo "HANDOFF=${kvaf_handoff}"
  echo "NEXT_STAGE=one-kib"
  exit 0
fi

if [[ -L "${kvaf_handoff}" ||
      ! -f "${kvaf_handoff}" ||
      ! -r "${kvaf_handoff}" ]]; then
  echo "ERROR: run and pass the zero-wqe stage first" >&2
  echo "EXPECTED_HANDOFF=${kvaf_handoff}" >&2
  exit 2
fi
if [[ "$(single_value schema "${kvaf_handoff}")" != \
      kvaf-a5-cross-npu2-zero-wqe-handoff-v1 ]]; then
  echo "ERROR: zero-WQE handoff schema is invalid" >&2
  exit 2
fi
kvaf_handoff_source_run=$(single_value source_run_id "${kvaf_handoff}")
kvaf_handoff_source_sha=$(
  single_value source_quarantine_sha256 "${kvaf_handoff}"
)
kvaf_handoff_commit=$(single_value source_commit "${kvaf_handoff}")
kvaf_artifact_id=$(single_value artifact_id "${kvaf_handoff}")
kvaf_build_dir=$(single_value build_dir "${kvaf_handoff}")
kvaf_zero_result=$(single_value zero_wqe_result_dir "${kvaf_handoff}")
if [[ "${kvaf_handoff_source_run}" != "${kvaf_source_run_id}" ||
      "${kvaf_handoff_source_sha}" != "${kvaf_source_quarantine_sha}" ||
      "${kvaf_handoff_commit}" != "${kvaf_source_commit}" ||
      ! "${kvaf_artifact_id}" =~ ^[0-9a-f]{32}$ ]]; then
  echo "ERROR: zero-WQE handoff no longer matches the source incident/checkout" >&2
  exit 2
fi
kvaf_expected_build_dir="/data/.kvaf-state/KVAF/experimental/artifacts/${kvaf_source_commit}/${kvaf_artifact_id}"
if [[ "${kvaf_build_dir}" != "${kvaf_expected_build_dir}" ]]; then
  echo "ERROR: zero-WQE handoff build path is inconsistent" >&2
  exit 2
fi
export KVAF_CROSS_NPU_ARTIFACT_ID=${kvaf_artifact_id}
export KVAF_A5_DIRECT_BUILD_DIR=${kvaf_build_dir}
export KVAF_CROSS_NPU_ZERO_WQE_RESULT_DIR=${kvaf_zero_result}

kvaf_log="/tmp/kvaf_npu2_one_kib_${kvaf_artifact_id}.log"
set +e
bash "${SCRIPT_DIR}/run_a5_direct_urma_cross_npu2_risk_accepted.sh" \
  one-kib 2>&1 | tee "${kvaf_log}"
kvaf_rc=${PIPESTATUS[0]}
set -e
echo "KVAF_NPU2_ONE_KIB_RC=${kvaf_rc}"
if (( kvaf_rc != 0 )); then
  echo "HARD_STOP: retain every active/quarantine/worker and inspect ${kvaf_log}" >&2
  exit "${kvaf_rc}"
fi
echo "KVAF_NPU2_ONE_KIB_H2D_ACCEPTANCE=PASS"
echo "LOG=${kvaf_log}"
