#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [build-cpu-directory]" >&2
  exit 2
fi

readonly root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly build_dir="${1:-${KVAF_CPU_RUNTIME_BUILD_DIR:-$root_dir/build-cpu}}"
readonly test_program="$root_dir/tests/python/test_cpu_runtime_npu_hbm.py"

[[ -d "$build_dir/python/kvaf" ]] || {
  echo "ERROR: CPU Python binding not found under $build_dir/python/kvaf; run ./build.sh cpu first" >&2
  exit 1
}

: "${KVAF_NPU_SERVICE_ADDR:=127.0.0.1:28999}"
: "${DEVICE_ID:=0}"
export KVAF_NPU_SERVICE_ADDR DEVICE_ID
export PYTHONPATH="$build_dir/python${PYTHONPATH:+:$PYTHONPATH}"

exec python3 "$test_program"
