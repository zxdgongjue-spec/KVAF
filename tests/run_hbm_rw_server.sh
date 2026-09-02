#!/usr/bin/env bash
set -euo pipefail
readonly KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${KVAF_NPU_BUILD_DIR:-${KVAF_HBM_RW_BUILD_DIR:-$KVAF_ROOT/build-hbm-rw}}"
if [[ -n "${1:-}" && ( -x "$1/runtime/kvaf_npu_service" || -x "$1/tests/kvaf_npu_hbm_server" ) ]]; then
  build_dir="$1"; shift
fi
if [[ $# -eq 0 ]]; then
  echo "Usage: $0 [build-dir] [--port PORT] [--device-id N] [--eid-index N]" >&2
  exit 2
fi
if [[ -x "$build_dir/runtime/kvaf_npu_service" ]]; then
  exec "$build_dir/runtime/kvaf_npu_service" "$@"
fi
echo "NPU service binary not found: $build_dir/runtime/kvaf_npu_service" >&2
echo "Build it with: ./build.sh npu --build-dir '$build_dir'" >&2
exit 1
