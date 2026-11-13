#!/usr/bin/env bash
# Compare synchronous BatchPut calls through the remote CPU DDR service.
set -euo pipefail

usage() {
  cat <<EOF
Usage: bash $0 /absolute/path/to/build-npu [options]

Options:
  --transport urma|ubmem          Default: urma
  --executor aiv|aicpu           Default: aiv; aicpu requires urma
  --execution-mode legacy|per-call  Default: legacy
  -h, --help                     Show this help

Shape and sampling use KVAF_NPU_REQUESTS, KVAF_NPU_TOKEN_BYTES,
KVAF_NPU_TOKENS_PER_BLOCK, KVAF_NPU_CORE_COUNT, KVAF_NPU_JETTYS_PER_CORE,
KVAF_NPU_WARMUP and KVAF_NPU_ITERATIONS. The CPU service must be running.
KVAF_CPU_SERVICE_ADDR defaults to 127.0.0.1:28999; DEVICE_ID defaults to 0.
KVAF_REMOTE_CPU_PERF_TIMEOUT defaults to 600 seconds.
Per-call URMA AIV/AICPU reuse an input binding by default; set
KVAF_PERF_REUSE_PUT_INPUT=0 to measure transient registration per call.
EOF
}

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi
for argument in "$@"; do
  if [[ "$argument" == --help || "$argument" == -h ]]; then
    usage
    exit 0
  fi
done

KVAF_BUILD_DIR="$1"
shift
KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -d "$KVAF_BUILD_DIR/python/kvaf" ]] || {
  echo "ERROR: expected Python binding under $KVAF_BUILD_DIR/python/kvaf" >&2
  exit 1
}
# The Python runner checks kernel requirements for the selected provider.
# AICPU-only builds load the installed device package and need no AIV objects.
export KVAF_BATCH_GET_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$KVAF_BUILD_DIR/bin/kvaf_execute_batch_put_kernels.o"

: "${KVAF_CPU_SERVICE_ADDR:=127.0.0.1:28999}"
: "${DEVICE_ID:=0}"
: "${KVAF_REMOTE_CPU_PERF_TIMEOUT:=600}"
[[ "$KVAF_REMOTE_CPU_PERF_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: KVAF_REMOTE_CPU_PERF_TIMEOUT must be a positive integer in seconds" >&2
  exit 2
}

export KVAF_CPU_SERVICE_ADDR DEVICE_ID
export KVAF_RUN_NPU_TESTS=1
export PYTHONPATH="$KVAF_BUILD_DIR/python${PYTHONPATH:+:$PYTHONPATH}"

exec timeout --signal=TERM --kill-after=10s "${KVAF_REMOTE_CPU_PERF_TIMEOUT}s" \
  "${PYTHON:-python3}" "$KVAF_ROOT/tests/python/test_npu_batch_put.py" "$@"
