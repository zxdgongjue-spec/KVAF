#!/usr/bin/env bash

# Dedicated staged entry point for the resource-bearing epoch-owner matrix.
# The normal direct runner invokes the affinity guard in production mode and
# continues to reject the candidate contract.  This wrapper changes only the
# source-defined positional mode; registration, artifact preflight, durable
# locks, idle-NPU proof and fail-closed recovery all remain in the normal
# runner and execute unchanged.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "${SCRIPT_DIR}/run_a5_direct_urma_h2d_smoke.sh" \
  candidate-epoch-owner-validation-v1
