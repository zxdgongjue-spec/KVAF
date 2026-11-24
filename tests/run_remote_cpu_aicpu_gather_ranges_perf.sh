#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: KVAF_CPU_SERVICE_ADDR=HOST:PORT $0 BUILD_DIR [perf options]" >&2
  exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$(cd "$1" && pwd)"
shift
[[ -d "$build/python/kvaf" ]] || {
  echo "ERROR: expected the KVAF Python binding under $build" >&2
  exit 2
}
export PYTHONPATH="$build/python${PYTHONPATH:+:$PYTHONPATH}"
runner="$root/tests/python/aicpu_gather_ranges_perf.py"
if command -v timeout >/dev/null 2>&1; then
  exec timeout "${KVAF_AICPU_RANGE_PERF_TIMEOUT:-1200}" "${PYTHON:-python3}" "$runner" "$@"
fi
exec "${PYTHON:-python3}" "$runner" "$@"
