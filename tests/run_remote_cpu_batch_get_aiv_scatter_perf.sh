#!/usr/bin/env bash
# Compare isolated and concurrent AIV Direct / CPU Push Scatter on one NPU.
set -euo pipefail

KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 build-npu [--device-id 0] [--peer HOST:PORT] [options...]" >&2
  exit 2
fi
if [[ "$1" == --help || "$1" == -h ]]; then
  exec "${PYTHON:-python3}" "$KVAF_ROOT/tests/python/batch_get_aiv_scatter_perf.py" --help
fi
KVAF_BUILD_DIR="$1"
shift
for argument in "$@"; do
  if [[ "$argument" == --help || "$argument" == -h ]]; then
    exec "${PYTHON:-python3}" "$KVAF_ROOT/tests/python/batch_get_aiv_scatter_perf.py" --build-dir "$KVAF_BUILD_DIR" "$@"
  fi
done
: "${KVAF_REMOTE_CPU_PERF_TIMEOUT:=900}"
[[ "$KVAF_REMOTE_CPU_PERF_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: KVAF_REMOTE_CPU_PERF_TIMEOUT must be a positive integer in seconds" >&2
  exit 2
}
export PYTHONPATH="$KVAF_BUILD_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
exec timeout "${KVAF_REMOTE_CPU_PERF_TIMEOUT}s" \
  "${PYTHON:-python3}" "$KVAF_ROOT/tests/python/batch_get_aiv_scatter_perf.py" --build-dir "$KVAF_BUILD_DIR" "$@"
