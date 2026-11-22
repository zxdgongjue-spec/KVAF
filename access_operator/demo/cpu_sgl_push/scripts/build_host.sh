#!/usr/bin/env bash
# Build the CPU-side SGL pusher on the host (liburma + vendored UMDK headers).
# Usage: build_host.sh /path/to/kvaf /path/to/out_dir
set -euo pipefail
KVAF_ROOT="$1"
OUT="$2"
mkdir -p "$OUT"

UMDK_INCLUDE="${UMDK_INCLUDE:-$KVAF_ROOT/third_party/umdk/include}"

g++ -std=c++17 -O2 -Wall \
  "$KVAF_ROOT/access_operator/demo/cpu_sgl_push/cpu/cpu_sgl_push.cpp" \
  -I"$UMDK_INCLUDE" \
  -L/usr/lib64 -Wl,-rpath,/usr/lib64 -lurma -ldl \
  -o "$OUT/cpu_sgl_push"

echo "built $OUT/cpu_sgl_push"
