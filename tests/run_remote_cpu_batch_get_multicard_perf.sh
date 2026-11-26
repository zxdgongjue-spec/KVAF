#!/usr/bin/env bash
# Compare isolated and concurrent bandwidth on the selected NPU devices.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 build-npu --backend aiv|ubmem|gather_read|cpu_push_scatter [--devices 1 2] [--peer HOST:PORT] [--device-peer DEVICE=HOST:PORT ...] [--graph-calls 256] [--kernel-profile] [options...]" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$1" == --help || "$1" == -h ]]; then
  exec "${PYTHON:-python3}" "$root/tests/python/batch_get_multicard_perf.py" --help
fi
build_dir="$1"
shift
exec "${PYTHON:-python3}" "$root/tests/python/batch_get_multicard_perf.py" "$build_dir" "$@"
