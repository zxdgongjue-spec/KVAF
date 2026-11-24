#!/usr/bin/env bash
# Run the configurable BatchGet benchmark through the remote CPU DDR service.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 build-npu --backend aiv|ubmem|gather_read|cpu_push_scatter [options...]" >&2
  exit 2
fi

KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$1" == --help || "$1" == -h ]]; then
  exec "${PYTHON:-python3}" "$KVAF_ROOT/tests/python/test_npu_batch_get_perf.py" --help
fi
KVAF_BUILD_DIR="$1"
shift
for argument in "$@"; do
  if [[ "$argument" == --help || "$argument" == -h ]]; then
    exec "${PYTHON:-python3}" "$KVAF_ROOT/tests/python/test_npu_batch_get_perf.py" --build-dir "$KVAF_BUILD_DIR" "$@"
  fi
done
KVAF_GET_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_execute_batch_get_kernels.o"
KVAF_PUT_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_execute_batch_put_kernels.o"

[[ -d "$KVAF_BUILD_DIR/python/kvaf" ]] || {
  echo "ERROR: expected Python binding under $KVAF_BUILD_DIR" >&2
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
  "${PYTHON:-python3}" "$KVAF_ROOT/tests/python/test_npu_batch_get_perf.py" --build-dir "$KVAF_BUILD_DIR" "$@"
