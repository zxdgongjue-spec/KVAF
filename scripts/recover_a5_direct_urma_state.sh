#!/usr/bin/env bash

# Recover only a stale direct-URMA state whose immutable result bundle proves
# PASS or FAIL_SAFE.  The normal runner exits immediately after the recovery
# transaction, before provider verification or any hardware operation.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

unset KVAF_H2D_BYTES
unset KVAF_DIRECT_SERVER_START_DELAY_SECONDS
unset KVAF_PHYSICAL_DEVICE_ID
unset KVAF_PAIR_CHANNEL
export KVAF_LOGICAL_DEVICE_ID=0
export KVAF_DIRECT_MANAGEMENT_MODE=ub
export KVAF_DIRECT_PROVIDER_SONAME=libkvaf_urma_provider_device.so
export KVAF_DIRECT_ALLOW_BUSY_NPU=0
export KVAF_EID_LOCK_TIMEOUT_SECONDS=10
export KVAF_NPU_SMI_TIMEOUT_SECONDS=1
export KVAF_HCCN_TIMEOUT_SECONDS=1
export KVAF_H2D_TRANSFER_COUNT=1
export KVAF_H2D_BYTES_PER_TRANSFER=1
export KVAF_H2D_BATCH_ITERATIONS=1
export KVAF_H2D_DESCRIPTOR_CAPACITY=1
export KVAF_H2D_TIMEOUT_SECONDS=1
export KVAF_H2D_RUN_TIMEOUT_SECONDS=60
export KVAF_DIRECT_SERVER_READY_TIMEOUT_SECONDS=1
export KVAF_HOST_CONTROL_TIMEOUT_SECONDS=1
export KVAF_A5_RUN_TAG=recovery_only

exec bash "${SCRIPT_DIR}/run_a5_direct_urma_h2d_smoke.sh" \
  recovery-only-v1
