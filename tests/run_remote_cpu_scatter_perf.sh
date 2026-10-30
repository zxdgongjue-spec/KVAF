#!/usr/bin/env bash
# Baseline + optional CPU/AIV timings for the real CPU-push scatter path.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/build-npu [scatter-perf-options...]" >&2
  exit 2
fi
KVAF_BUILD_DIR="$(cd "$1" && pwd)"
shift
KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KVAF_BATCH_GET_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_scatter_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_execute_batch_put_kernels.o"
[[ -s "$KVAF_BATCH_GET_KERNEL" && -s "$KVAF_BATCH_PUT_KERNEL" && -d "$KVAF_BUILD_DIR/python/kvaf" ]] || {
  echo "ERROR: rebuild the scatter/BatchPut kernels and Python binding in $KVAF_BUILD_DIR" >&2
  exit 1
}
export KVAF_CPU_SERVICE_ADDR="${KVAF_CPU_SERVICE_ADDR:-127.0.0.1:28990}"
export DEVICE_ID="${DEVICE_ID:-7}"
export KVAF_RUN_NPU_TESTS=1
export PYTHONPATH="$KVAF_BUILD_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
KVAF_SCATTER_PERF_TIMEOUT="${KVAF_SCATTER_PERF_TIMEOUT:-600}"
[[ "$KVAF_SCATTER_PERF_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: KVAF_SCATTER_PERF_TIMEOUT must be positive seconds" >&2
  exit 2
}
exec timeout "${KVAF_SCATTER_PERF_TIMEOUT}s" \
  python3 "$KVAF_ROOT/tests/python/test_aiv_scatter_perf.py" "$@"
