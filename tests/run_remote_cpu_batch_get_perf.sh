#!/usr/bin/env bash
# Run the configurable BatchGet benchmark through a CPU DDR or NPU HBM service.
# NPU: KVAF_PEER_DEVICE=npu KVAF_PEER_SERVICE_ADDR=host:port (or matching CLI flags).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/kvaf/build-npu [--batch-get-mode lookup|no-pipeline|raw] [batch-get-perf-options...]" >&2
  exit 2
fi

KVAF_BUILD_DIR="$1"
shift
KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$KVAF_ROOT/tests/batch_get_perf_mode.sh"
kvaf_batch_get_parse_mode "$@"
KVAF_GET_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_execute_batch_get_kernels.o"
if [[ "$batch_get_mode" == no-pipeline ]]; then
  KVAF_GET_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_execute_batch_get_no_pipeline_kernels.o"
fi
KVAF_PUT_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_execute_batch_put_kernels.o"

[[ -s "$KVAF_GET_KERNEL" && -s "$KVAF_PUT_KERNEL" && -d "$KVAF_BUILD_DIR/python/kvaf" ]] || {
  echo "ERROR: expected $KVAF_GET_KERNEL, $KVAF_PUT_KERNEL and Python binding under $KVAF_BUILD_DIR; rebuild with ./build.sh npu" >&2
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

exec timeout "${KVAF_REMOTE_CPU_PERF_TIMEOUT}s" \
  "${PYTHON:-python3}" "$KVAF_ROOT/tests/python/test_npu_batch_get_perf.py" "$@"
