#!/usr/bin/env bash
# Hardware-only acceptance. Missing dependencies/devices are failures.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 /absolute/path/to/build-npu [test options]" >&2
  exit 2
fi
KVAF_TEST_BUILD_DIR="$1"
shift
KVAF_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python3}" "$KVAF_TEST_ROOT/tests/python/test_cache_execution_e2e.py" \
  --build-dir "$KVAF_TEST_BUILD_DIR" "$@"
