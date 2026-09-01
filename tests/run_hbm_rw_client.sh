#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <build-dir> <npu-host> [port] [--host-eid HEX] [client options]" >&2
  exit 2
fi
build_dir="$1"; host="$2"; shift 2
exec "$build_dir/tests/kvaf_cpu_hbm_client" "$host" "$@"
