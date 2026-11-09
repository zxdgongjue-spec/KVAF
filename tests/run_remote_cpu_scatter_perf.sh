#!/usr/bin/env bash
# One complete BatchGet per sample; default is 1024 blocks x 4 x 656 B.
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
KVAF_PERF_ARGS=("$@")
KVAF_PERF_HAS_JSON=0
KVAF_PERF_HAS_CSV=0
for KVAF_PERF_ARG in "$@"; do
  case "$KVAF_PERF_ARG" in
    --json|--json=*) KVAF_PERF_HAS_JSON=1 ;;
    --csv|--csv=*) KVAF_PERF_HAS_CSV=1 ;;
  esac
done
KVAF_PERF_PREFIX="${KVAF_SCATTER_PERF_RESULT_PREFIX:-$KVAF_ROOT/results/aiv-scatter-$(date +%Y%m%d-%H%M%S)-$$}"
if [[ "$KVAF_PERF_HAS_JSON" == 0 ]]; then
  KVAF_PERF_ARGS+=(--json "${KVAF_PERF_PREFIX}.json")
fi
if [[ "$KVAF_PERF_HAS_CSV" == 0 ]]; then
  KVAF_PERF_ARGS+=(--csv "${KVAF_PERF_PREFIX}.csv")
fi
exec timeout "${KVAF_SCATTER_PERF_TIMEOUT}s" \
  python3 "$KVAF_ROOT/tests/python/test_aiv_scatter_perf.py" "${KVAF_PERF_ARGS[@]}"
