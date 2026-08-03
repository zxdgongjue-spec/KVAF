#!/usr/bin/env bash

# Fixed, fail-closed NPU2 validation for a clean machine state.
#
# This is deliberately not a cross-NPU recovery bypass.  It uses the normal
# machine-wide direct-URMA locks and therefore refuses to run while an older
# owner, active record, or quarantine exists.  One invocation validates three
# fresh epochs in order: resource-only, one 1-KiB WQE, then the requested
# decode-style batch (4096 x 1 KiB by default).

set -Eeuo pipefail
umask 077

if (( $# != 0 )); then
  echo "Usage: $0" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
: "${ASCEND_HOME_PATH:?source the container CANN set_env.sh first}"

kvaf_reject_risk_variable()
{
  local variable=$1
  if [[ -n "${!variable:-}" ]]; then
    echo "ERROR: ${variable} is forbidden by the clean NPU2 entry" >&2
    echo "Use the dedicated cross-NPU experiment only for its pinned incident." >&2
    exit 2
  fi
}

for kvaf_variable in \
  KVAF_CROSS_NPU_RISK_ACCEPTANCE \
  KVAF_CROSS_NPU_SOURCE_RUN_ID \
  KVAF_CROSS_NPU_SOURCE_QUARANTINE_SHA256 \
  KVAF_CROSS_NPU_ARTIFACT_ID \
  KVAF_CROSS_NPU_ZERO_WQE_RESULT_DIR
do
  kvaf_reject_risk_variable "${kvaf_variable}"
done

# Reject every current or future cross-NPU experiment variable.  The explicit
# list above documents the first-generation contract; the prefix scan closes
# the same boundary for the versioned/reuse experiment family.
kvaf_cross_npu_variables=$(
  env | sed -n 's/^\(KVAF_CROSS_NPU[A-Za-z0-9_]*\)=.*/\1/p'
)
while IFS= read -r kvaf_variable; do
  [[ -n "${kvaf_variable}" ]] || continue
  kvaf_reject_risk_variable "${kvaf_variable}"
done <<<"${kvaf_cross_npu_variables}"

kvaf_pin_value()
{
  local variable=$1
  local expected=$2
  local current=${!variable:-}

  if [[ -n "${current}" && "${current}" != "${expected}" ]]; then
    echo "ERROR: ${variable} is source-owned by the clean NPU2 entry" >&2
    echo "EXPECTED=${expected}" >&2
    echo "ACTUAL=${current}" >&2
    exit 2
  fi
  printf -v "${variable}" '%s' "${expected}"
  export "${variable}"
}

kvaf_pin_value KVAF_LOGICAL_DEVICE_ID 2
kvaf_pin_value KVAF_PHYSICAL_DEVICE_ID 2
kvaf_pin_value KVAF_HOST_EID 00000000007f040000100000df0a0b00
kvaf_pin_value KVAF_DEVICE_EID 000000000000020000100000df004100
kvaf_pin_value KVAF_PAIR_CHANNEL 0
kvaf_pin_value KVAF_DIRECT_MANAGEMENT_MODE ub
kvaf_pin_value KVAF_DIRECT_MANAGEMENT_JETTY_ID 502
kvaf_pin_value KVAF_DIRECT_ALLOW_BUSY_NPU 0
kvaf_pin_value KVAF_DIRECT_PROVIDER_BUILD_DIR \
  "${ROOT_DIR}/build-a5-direct-urma-provider"
kvaf_pin_value KVAF_A5_DIRECT_BUILD_DIR \
  "${ROOT_DIR}/build-a5-direct-urma"
kvaf_pin_value KVAF_DIRECT_PROVIDER_SONAME \
  libkvaf_urma_provider_device.so
kvaf_pin_value KVAF_DIRECT_AICPU_ARCHIVE \
  "${ASCEND_HOME_PATH}/opp/vendors/cust/op_impl/aicpu/kernel/cann-kvaf-direct-urma-compat.tar.gz"
kvaf_pin_value KVAF_DIRECT_AICPU_REGISTRATION_FILE \
  "${ASCEND_HOME_PATH}/conf/ascend_package_load.ini"

# Resolve the fixed EIDs against the live Host/AICPU inventory.  Stale caller
# guesses such as udmac0d1e4 must not select an endpoint by name.
unset KVAF_HOST_URMA_DEVICE
unset KVAF_HOST_EID_INDEX
unset KVAF_DEVICE_URMA_DEVICE
unset KVAF_DEVICE_EID_INDEX
export KVAF_EID_PROC_ROOT=/proc/asdrv_ub
export KVAF_UBCORE_SYSFS_ROOT=/sys/class/ubcore
export KVAF_HCCN_TOOL=/usr/local/Ascend/driver/tools/hccn_tool

# UB management does not use the old TCP control channel.
unset KVAF_DIRECT_CONTROL_PORT
unset KVAF_DIRECT_CONTROL_SERVER_ADDRESS
unset KVAF_DIRECT_CONTROL_BIND_ADDRESS
unset KVAF_DIRECT_SERVER_START_DELAY_SECONDS
unset KVAF_H2D_BYTES
unset KVAF_A5_RESULT_DIR
unset KVAF_A5_RUN_TAG
unset ASCEND_RT_VISIBLE_DEVICES

kvaf_require_positive_uint()
{
  local variable=$1
  local value=$2
  local maximum=$3

  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ${variable} must be an integer in [1, ${maximum}]" >&2
    exit 2
  fi
  if (( ${#value} > ${#maximum} )) ||
     { (( ${#value} == ${#maximum} )) &&
       [[ "${value}" > "${maximum}" ]]; }; then
    echo "ERROR: ${variable} must be an integer in [1, ${maximum}]" >&2
    exit 2
  fi
}

KVAF_NPU2_TRANSFER_COUNT=${KVAF_NPU2_TRANSFER_COUNT:-4096}
KVAF_NPU2_BYTES_PER_TRANSFER=${KVAF_NPU2_BYTES_PER_TRANSFER:-1024}
KVAF_NPU2_BATCH_ITERATIONS=${KVAF_NPU2_BATCH_ITERATIONS:-1}
KVAF_NPU2_H2D_TIMEOUT_SECONDS=${KVAF_NPU2_H2D_TIMEOUT_SECONDS:-60}
KVAF_NPU2_RUN_TIMEOUT_SECONDS=${KVAF_NPU2_RUN_TIMEOUT_SECONDS:-180}
KVAF_NPU2_SERVER_READY_TIMEOUT_SECONDS=${KVAF_NPU2_SERVER_READY_TIMEOUT_SECONDS:-30}
KVAF_NPU2_VALIDATION_LIMIT=${KVAF_NPU2_VALIDATION_LIMIT:-full}

if [[ "${KVAF_NPU2_VALIDATION_LIMIT}" != full &&
      "${KVAF_NPU2_VALIDATION_LIMIT}" != zero-wqe ]]; then
  echo "ERROR: KVAF_NPU2_VALIDATION_LIMIT must be full or zero-wqe" >&2
  exit 2
fi

kvaf_require_positive_uint KVAF_NPU2_TRANSFER_COUNT \
  "${KVAF_NPU2_TRANSFER_COUNT}" 4096
kvaf_require_positive_uint KVAF_NPU2_BYTES_PER_TRANSFER \
  "${KVAF_NPU2_BYTES_PER_TRANSFER}" 67108864
kvaf_require_positive_uint KVAF_NPU2_BATCH_ITERATIONS \
  "${KVAF_NPU2_BATCH_ITERATIONS}" 67108864
kvaf_require_positive_uint KVAF_NPU2_H2D_TIMEOUT_SECONDS \
  "${KVAF_NPU2_H2D_TIMEOUT_SECONDS}" 3600
kvaf_require_positive_uint KVAF_NPU2_RUN_TIMEOUT_SECONDS \
  "${KVAF_NPU2_RUN_TIMEOUT_SECONDS}" 3600
kvaf_require_positive_uint KVAF_NPU2_SERVER_READY_TIMEOUT_SECONDS \
  "${KVAF_NPU2_SERVER_READY_TIMEOUT_SECONDS}" 600
if (( KVAF_NPU2_BYTES_PER_TRANSFER >
      67108864 / KVAF_NPU2_TRANSFER_COUNT )); then
  echo "ERROR: the NPU2 descriptor pool exceeds the 64 MiB registration limit" >&2
  exit 2
fi
if (( KVAF_NPU2_BATCH_ITERATIONS >
      4294967295 / KVAF_NPU2_TRANSFER_COUNT )); then
  echo "ERROR: the NPU2 epoch exceeds the 32-bit queue-counter limit" >&2
  exit 2
fi

export KVAF_H2D_TIMEOUT_SECONDS=${KVAF_NPU2_H2D_TIMEOUT_SECONDS}
export KVAF_H2D_RUN_TIMEOUT_SECONDS=${KVAF_NPU2_RUN_TIMEOUT_SECONDS}
export KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS=${KVAF_NPU2_SERVER_READY_TIMEOUT_SECONDS}

kvaf_run_stage()
{
  local stage=$1
  local transfer_count=$2
  local bytes_per_transfer=$3
  local batch_iterations=$4
  local descriptor_capacity=$5
  local log_file run_rc result_dir

  log_file=$(mktemp "/tmp/kvaf_npu2_clean_${stage}.XXXXXX.log") || {
    echo "ERROR: could not create the ${stage} log" >&2
    return 2
  }

  export KVAF_H2D_TRANSFER_COUNT=${transfer_count}
  export KVAF_H2D_BYTES_PER_TRANSFER=${bytes_per_transfer}
  export KVAF_H2D_BATCH_ITERATIONS=${batch_iterations}
  export KVAF_H2D_DESCRIPTOR_CAPACITY=${descriptor_capacity}
  export KVAF_A5_RUN_TAG="npu2_clean_${stage}_$(date +%Y%m%d_%H%M%S)_$$"

  echo "KVAF_NPU2_STAGE=${stage}"
  echo "KVAF_NPU2_SHAPE=${transfer_count}x${bytes_per_transfer}x${batch_iterations}"
  set +e
  bash "${SCRIPT_DIR}/run_a5_direct_urma_epoch_owner_validation.sh" \
    2>&1 | tee "${log_file}"
  run_rc=${PIPESTATUS[0]}
  set -e
  echo "KVAF_NPU2_STAGE_RC=${run_rc}"
  echo "KVAF_NPU2_STAGE_LOG=${log_file}"
  if (( run_rc != 0 )); then
    echo "HARD_STOP: ${stage} failed; retain all state and do not continue." >&2
    return "${run_rc}"
  fi

  result_dir=$(
    awk '
      /^RESULT_DIR=/ {
        value = substr($0, length("RESULT_DIR=") + 1)
      }
      END {
        if (value == "") exit 1
        print value
      }
    ' "${log_file}"
  ) || {
    echo "ERROR: ${stage} PASS did not expose RESULT_DIR" >&2
    return 5
  }
  echo "KVAF_NPU2_STAGE_RESULT=${result_dir}"
  echo "KVAF_NPU2_STAGE_PASS=${stage}"
}

# Every real transfer is gated by a resource-only epoch and a minimal WQE.
# A successful final stage performs one Open/register/connect lifecycle for
# the entire descriptor batch; it does not register/connect 4096 times.
kvaf_run_stage zero-wqe 0 1024 1 1
if [[ "${KVAF_NPU2_VALIDATION_LIMIT}" == zero-wqe ]]; then
  echo "KVAF_NPU2_CLEAN_VALIDATION=PASS_LIMITED_ZERO_WQE"
  exit 0
fi
kvaf_run_stage one-kib 1 1024 1 1
kvaf_run_stage batch \
  "${KVAF_NPU2_TRANSFER_COUNT}" \
  "${KVAF_NPU2_BYTES_PER_TRANSFER}" \
  "${KVAF_NPU2_BATCH_ITERATIONS}" \
  "${KVAF_NPU2_TRANSFER_COUNT}"

echo "KVAF_NPU2_CLEAN_VALIDATION=PASS"
echo "KVAF_NPU2_FINAL_SHAPE=${KVAF_NPU2_TRANSFER_COUNT}x${KVAF_NPU2_BYTES_PER_TRANSFER}x${KVAF_NPU2_BATCH_ITERATIONS}"
