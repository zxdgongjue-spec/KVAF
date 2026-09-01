#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <build-dir> [--eid HEX|--npu-eid-list-index N] [server options]" >&2
  exit 2
fi
build_dir="$1"; shift
exec "$build_dir/tests/kvaf_npu_hbm_server" "$@"
