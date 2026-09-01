#!/usr/bin/env bash
# Run the NPU half of the first CPU-service / NPU-process integration test.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/kvaf/build-npu" >&2
  exit 2
fi

KVAF_BUILD_DIR="$1"
KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KVAF_GET_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_execute_batch_get_kernels.o"
KVAF_PUT_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_execute_batch_put_kernels.o"

[[ -s "$KVAF_GET_KERNEL" && -s "$KVAF_PUT_KERNEL" && -d "$KVAF_BUILD_DIR/python/kvaf" ]] || {
  echo "ERROR: expected AIV kernels and Python binding under $KVAF_BUILD_DIR" >&2
  exit 1
}
: "${KVAF_CPU_SERVICE_ADDR:=127.0.0.1:28999}"

export KVAF_CPU_SERVICE_ADDR
export KVAF_RUN_NPU_TESTS=1
export KVAF_BATCH_GET_KERNEL="$KVAF_GET_KERNEL"
export KVAF_BATCH_PUT_KERNEL="$KVAF_PUT_KERNEL"
export DEVICE_ID="${DEVICE_ID:=0}"
export PYTHONPATH="$KVAF_BUILD_DIR/python${PYTHONPATH:+:$PYTHONPATH}"

exec python3 "$KVAF_ROOT/tests/python/test_npu_batch_get.py"
