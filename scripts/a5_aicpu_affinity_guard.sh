#!/usr/bin/env bash

# The source now contains a candidate single-invocation epoch owner, but that
# code has not yet passed the resource-bearing hardware/fault matrix.  Keep
# the validation state source-owned and non-configurable: an environment
# variable or a resource-free probe PASS is not evidence that checked Open,
# AIV transfer and checked Close completed in one live owner invocation.
readonly KVAF_AICPU_AFFINITY_CONTRACT=candidate-epoch-owner-unvalidated-v1

kvaf_require_aicpu_affinity_contract()
{
  local requested_mode=${1:-production}

  case "${KVAF_AICPU_AFFINITY_CONTRACT}" in
    documented-same-stream-process-affinity-v1|validated-single-invocation-epoch-owner-v1)
      return 0
      ;;
    candidate-epoch-owner-unvalidated-v1)
      if [[ "${requested_mode}" == \
            candidate-epoch-owner-validation-v1 ]]; then
        echo "WARNING: running staged resource-bearing epoch-owner validation" >&2
        echo "CONTRACT=${KVAF_AICPU_AFFINITY_CONTRACT}" >&2
        return 0
      fi
      ;;
  esac

  case "${KVAF_AICPU_AFFINITY_CONTRACT}" in
    *)
      echo "ERROR: AICPU KFC worker/process affinity is not proven" >&2
      echo "CONTRACT=${KVAF_AICPU_AFFINITY_CONTRACT}" >&2
      echo "The candidate Device path keeps Open, transfer ownership and checked" >&2
      echo "Close in one epoch-owner invocation, but its resource-bearing hardware" >&2
      echo "and fault-injection matrix has not been validated yet." >&2
      echo "A bounded, resource-free AICPU owner probe may validate one sampled" >&2
      echo "AICPU/AIV control-buffer handshake, but probe PASS does NOT prove the" >&2
      echo "Open-transfer-Close provider lifetime and does NOT unlock direct URMA." >&2
      echo "Hardware execution stays disabled until KVAF validates the complete" >&2
      echo "Open-transfer-SHUTDOWN_QUIESCED-Close lifecycle on the pinned stack." >&2
      echo "There is no environment-variable override for this ownership gate." >&2
      echo "No topology/EID/NPU/URMA operation was attempted." >&2
      return 2
      ;;
  esac
}
