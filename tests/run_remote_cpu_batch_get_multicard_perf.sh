#!/usr/bin/env bash
# Compare isolated and concurrent bandwidth on the selected NPU devices.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/build-npu [--transport ubmem|urma] [--devices 6 7] [--aiv-cores 1 2 4] [options...]" >&2
  exit 2
fi

build_dir="$1"
shift
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$root/tests/python/batch_get_multicard_perf.py" "$build_dir" "$@"
