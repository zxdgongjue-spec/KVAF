#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: KVAF_CPU_SERVICE_ADDR=HOST:PORT $0 /absolute/build-npu [--device ID]" >&2
  exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$(cd "$1" && pwd)"
shift
: "${KVAF_CPU_SERVICE_ADDR:?Set the separately running CPU service HOST:PORT}"
export KVAF_BATCH_GET_KERNEL="$build/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$build/bin/kvaf_execute_batch_put_kernels.o"
[[ -s "$KVAF_BATCH_GET_KERNEL" && -s "$KVAF_BATCH_PUT_KERNEL" && -d "$build/python/kvaf" ]] || {
  echo "ERROR: matching AIV Get/Put objects and rebuilt Python binding required" >&2
  exit 2
}
export PYTHONPATH="$build/python${PYTHONPATH:+:$PYTHONPATH}"
export KVAF_RUN_NPU_TESTS=1
if command -v timeout >/dev/null 2>&1; then
  exec timeout "${KVAF_APPEND_TEST_TIMEOUT:-300}" "${PYTHON:-python3}" "$root/tests/python/test_npu_append_only.py" "$@"
fi
exec "${PYTHON:-python3}" "$root/tests/python/test_npu_append_only.py" "$@"
