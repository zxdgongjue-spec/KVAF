#!/usr/bin/env bash
set -euo pipefail
readonly KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${KVAF_HBM_RW_BUILD_DIR:-$KVAF_ROOT/build-hbm-rw}"
if [[ -n "${1:-}" && -x "$1/tests/kvaf_cpu_hbm_client" ]]; then
  build_dir="$1"; shift
fi
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [build-dir] <npu-host> [port] [--host-eid HEX] [client options]" >&2
  exit 2
fi
host="$1"; shift
if [[ ! -x "$build_dir/tests/kvaf_cpu_hbm_client" ]]; then
  echo "CPU HBM client binary not found: $build_dir/tests/kvaf_cpu_hbm_client" >&2
  echo "Build it with: ./build.sh hbm-rw --build-dir '$build_dir'" >&2
  exit 1
fi
exec "$build_dir/tests/kvaf_cpu_hbm_client" "$host" "$@"
