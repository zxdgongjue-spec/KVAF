#!/usr/bin/env bash
# Run the configurable BatchGet benchmark through the remote CPU DDR service.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/kvaf/build-npu [batch-get-perf-options...]" >&2
  exit 2
fi

KVAF_BUILD_DIR="$1"
shift
KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KVAF_GET_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_execute_batch_get_kernels.o"
KVAF_PUT_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_execute_batch_put_kernels.o"

[[ -s "$KVAF_GET_KERNEL" && -s "$KVAF_PUT_KERNEL" && -d "$KVAF_BUILD_DIR/python/kvaf" ]] || {
  echo "ERROR: expected AIV kernels and Python binding under $KVAF_BUILD_DIR" >&2
  exit 1
}

: "${KVAF_CPU_SERVICE_ADDR:=127.0.0.1:28999}"
: "${DEVICE_ID:=0}"
: "${KVAF_REMOTE_CPU_PERF_TIMEOUT:=600}"

[[ "$KVAF_REMOTE_CPU_PERF_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: KVAF_REMOTE_CPU_PERF_TIMEOUT must be a positive integer in seconds" >&2
  exit 2
}

export KVAF_CPU_SERVICE_ADDR
export KVAF_RUN_NPU_TESTS=1
export KVAF_BATCH_GET_KERNEL="$KVAF_GET_KERNEL"
export KVAF_BATCH_PUT_KERNEL="$KVAF_PUT_KERNEL"
export DEVICE_ID
export PYTHONPATH="$KVAF_BUILD_DIR/python${PYTHONPATH:+:$PYTHONPATH}"

# Keep Python result lines ahead of native teardown messages when stdout and
# stderr are redirected to the same file; buffered writes can split a record.
# A Python process inside a blocking native call may not handle TERM promptly.
# Escalate after ten seconds rather than waiting indefinitely after the timeout.
exec timeout --kill-after=10s "${KVAF_REMOTE_CPU_PERF_TIMEOUT}s" \
  python3 -u "$KVAF_ROOT/tests/python/test_npu_batch_get_perf.py" "$@"
