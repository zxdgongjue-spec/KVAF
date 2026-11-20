#!/usr/bin/env bash
# Compare with the queued ACL Graph mode in ascend-copy-bench.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/build-npu [options]" >&2
  exit 2
fi

build_dir="$1"
shift
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$root/tests/python/batch_get_multicard_perf.py" "$build_dir" \
  --queued-replays --devices 2 3 4 5 --aiv-cores 32 --graph-replays 10 \
  --requests 16384 --table-capacity 64 "$@"
