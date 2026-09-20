#!/usr/bin/env bash
# Run CPU DRAM -> NPU HBM BatchPut through the production CPU AccessOperator.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/kvaf/build-cpu [cpu-batch-put-perf-options...]" >&2
  exit 2
fi

KVAF_BUILD_DIR="$1"
shift
KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KVAF_TEST_PROGRAM="$KVAF_ROOT/tests/python/test_cpu_batch_put_perf.py"

[[ -d "$KVAF_BUILD_DIR/python/kvaf" && -f "$KVAF_TEST_PROGRAM" ]] || {
  echo "ERROR: expected CPU Python binding under $KVAF_BUILD_DIR/python/kvaf" >&2
  exit 1
}

: "${KVAF_NPU_SERVICE_ADDR:=${KVAF_CPU_SERVICE_ADDR:-127.0.0.1:28999}}"
: "${DEVICE_ID:=0}"
: "${KVAF_CPU_BATCH_PUT_PERF_TIMEOUT:=600}"
# libgomp reads this at process startup; set it before launching Python.
OMP_WAIT_POLICY=ACTIVE

[[ "$KVAF_CPU_BATCH_PUT_PERF_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: KVAF_CPU_BATCH_PUT_PERF_TIMEOUT must be a positive integer" >&2
  exit 2
}

export KVAF_NPU_SERVICE_ADDR DEVICE_ID OMP_WAIT_POLICY
export PYTHONPATH="$KVAF_BUILD_DIR/python${PYTHONPATH:+:$PYTHONPATH}"

echo "CPU_BATCH_PUT_OMP wait_policy=$OMP_WAIT_POLICY proc_bind=${OMP_PROC_BIND:-unset} places=${OMP_PLACES:-unset}"

exec timeout "${KVAF_CPU_BATCH_PUT_PERF_TIMEOUT}s" \
  python3 "$KVAF_TEST_PROGRAM" "$@"
