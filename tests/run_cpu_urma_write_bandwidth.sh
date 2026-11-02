#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 /absolute/path/to/build-npu [benchmark-options...]" >&2
  exit 2
fi
KVAF_WRITE_BUILD="$1"
shift
KVAF_WRITE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KVAF_RUN_NPU_TESTS=1
exec "${KVAF_SCATTER_PERF_PYTHON:-python3}" \
  "$KVAF_WRITE_ROOT/tests/python/test_cpu_urma_write_bandwidth.py" --build "$KVAF_WRITE_BUILD" "$@"
