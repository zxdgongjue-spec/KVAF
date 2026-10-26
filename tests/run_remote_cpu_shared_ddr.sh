#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/build-npu [--devices 0 1 --rounds 100 ...]" >&2
  exit 2
fi
KVAF_BUILD_DIR="$1"
shift
KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KVAF_BATCH_GET_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_execute_batch_put_kernels.o"
export PYTHONPATH="$KVAF_BUILD_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$KVAF_ROOT/tests/python/test_npu_shared_ddr.py" "$@"
