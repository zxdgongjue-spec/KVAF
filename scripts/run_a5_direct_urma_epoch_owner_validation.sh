#!/usr/bin/env bash

# Dedicated staged entry point for the resource-bearing epoch-owner matrix.
# The normal direct runner invokes the affinity guard in production mode and
# continues to reject the candidate contract.  This wrapper changes only the
# source-defined positional mode; registration, artifact preflight, durable
# locks, idle-NPU proof and fail-closed recovery all remain in the normal
# runner and execute unchanged.

unset BASH_ENV ENV
unset -f exit builtin 2>/dev/null || true

echo "KVAF_A5_EPOCH_OWNER_VALIDATION_COORDINATOR_DISABLED" >&2
echo "CAPABILITY_LEDGER=not-implemented" >&2
echo "No argument/environment/source/state/CANN/docker/NPU operation was attempted." >&2
builtin exit 2
if [[ 0 == 1 ]]; then

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "${SCRIPT_DIR}/run_a5_direct_urma_h2d_smoke.sh" \
  candidate-epoch-owner-validation-v1
fi
/bin/sh -c 'exit 2'
