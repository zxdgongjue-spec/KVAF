#!/usr/bin/env bash

# A KFC launch accepts an associated stream, but the public CANN ABI exposed
# to this project has no worker/process/SO-instance affinity token.  Sampling
# getpid(), a process cookie, the SO base and an instance generation over a
# few completed launches would describe only those past launches; it cannot
# bind the later Open and Close launches that own process-local UDMA/UMMU
# handles.  Keep this source-owned value non-configurable: an environment
# variable is not evidence of a runtime ownership contract.
readonly KVAF_AICPU_AFFINITY_CONTRACT=unproven-kfc-multilaunch

kvaf_require_aicpu_affinity_contract()
{
  case "${KVAF_AICPU_AFFINITY_CONTRACT}" in
    documented-same-stream-process-affinity-v1|single-invocation-owner-v1)
      return 0
      ;;
    *)
      echo "ERROR: AICPU KFC worker/process affinity is not proven" >&2
      echo "CONTRACT=${KVAF_AICPU_AFFINITY_CONTRACT}" >&2
      echo "The current Device provider keeps UDMA/UMMU handles in process-local state," >&2
      echo "but Open and Close are separate KFC launches." >&2
      echo "A same-stream pid/cookie/SO-base/generation sample is diagnostic only;" >&2
      echo "the public launch ABI cannot bind either future launch to that worker." >&2
      echo "Hardware execution stays disabled until KVAF validates a documented" >&2
      echo "same-stream same-process/SO runtime contract, or moves the complete" >&2
      echo "Open-transfer-Close lifetime into one invocation/long-lived owner." >&2
      echo "There is no environment-variable override for this ownership gate." >&2
      echo "No topology/EID/NPU/URMA operation was attempted." >&2
      return 2
      ;;
  esac
}
