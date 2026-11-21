#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: KVAF_CPU_SERVICE_ADDR=HOST:28999 bash $0 /absolute/build-npu [Python test options]" >&2
  exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$(cd "$1" && pwd)"
shift
: "${KVAF_CPU_SERVICE_ADDR:?Set the separately running URMA CPU service HOST:PORT}"
export KVAF_BATCH_GET_KERNEL="$build/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$build/bin/kvaf_execute_batch_put_kernels.o"
[[ -s "$KVAF_BATCH_GET_KERNEL" && -s "$KVAF_BATCH_PUT_KERNEL" && -d "$build/python/kvaf" ]] || {
  echo "ERROR: rebuild AIV Get/Put kernels and Python binding under $build" >&2
  exit 2
}
export KVAF_RUN_NPU_TESTS=1
export PYTHONPATH="$build/python${PYTHONPATH:+:$PYTHONPATH}"
if command -v timeout >/dev/null 2>&1; then
  exec timeout "${KVAF_RANGE_TEST_TIMEOUT:-300}" "${PYTHON:-python3}" \
    "$root/tests/python/test_npu_batch_get_ranges.py" "$@"
fi
exec "${PYTHON:-python3}" "$root/tests/python/test_npu_batch_get_ranges.py" "$@"
