#!/usr/bin/env bash
# Reference-aligned CPU SGL WRITE + AIV scatter: sweep by default, --single for one case.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 /absolute/path/to/build-npu [--single] [--dry-run] [perf-options...]" >&2
  exit 2
fi
KVAF_SCATTER_BUILD="$1"
shift
KVAF_SCATTER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KVAF_CPU_SERVICE_ADDR="${KVAF_CPU_SERVICE_ADDR:-127.0.0.1:28999}"
export DEVICE_ID="${DEVICE_ID:-0}"
export KVAF_RUN_NPU_TESTS=1
# Reference miss sweep always passes this clock explicitly, overriding the
# single-case PERF clock environment. Mirror that precedence here.
export KVAF_BATCH_GET_PERF_KERNEL_PROFILE_CLOCK_MHZ="${KVAF_BATCH_GET_KERNEL_CLOCK_MHZ:-1000}"
KVAF_SCATTER_TIMEOUT="${KVAF_REMOTE_CPU_PERF_TIMEOUT:-600}"
[[ "$KVAF_SCATTER_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: KVAF_REMOTE_CPU_PERF_TIMEOUT must be positive seconds" >&2
  exit 2
}
exec "${KVAF_SCATTER_PERF_PYTHON:-python3}" \
  "$KVAF_SCATTER_ROOT/tests/python/test_aiv_scatter_miss_count_perf.py" \
  --build "$KVAF_SCATTER_BUILD" --case-timeout "$KVAF_SCATTER_TIMEOUT" "$@" \
  --kernel-profile-clock-mhz "$KVAF_BATCH_GET_PERF_KERNEL_PROFILE_CLOCK_MHZ"
