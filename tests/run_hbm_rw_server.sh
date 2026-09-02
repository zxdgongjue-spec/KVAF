#!/usr/bin/env bash
set -euo pipefail
readonly KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${KVAF_HBM_RW_BUILD_DIR:-$KVAF_ROOT/build-hbm-rw}"
if [[ -n "${1:-}" && -x "$1/tests/kvaf_npu_hbm_server" ]]; then
  build_dir="$1"; shift
fi
if [[ $# -eq 0 ]]; then
  echo "Usage: $0 [build-dir] [--eid HEX|--npu-eid-list-index N] [server options]" >&2
  exit 2
fi
if [[ ! -x "$build_dir/tests/kvaf_npu_hbm_server" ]]; then
  echo "NPU HBM server binary not found: $build_dir/tests/kvaf_npu_hbm_server" >&2
  echo "Build it with: ./build.sh hbm-rw --build-dir '$build_dir'" >&2
  exit 1
fi
exec "$build_dir/tests/kvaf_npu_hbm_server" "$@"
