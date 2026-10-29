#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 /absolute/path/to/build-ubmem-npu CPU_IP:29000 [test options]" >&2
  exit 2
fi
ubmem_build="$1"
ubmem_peer="$2"
shift 2
if [[ ! -x "$ubmem_build/bin/kvaf_ubmem_test" || ! -s "$ubmem_build/bin/kvaf_ubmem_kernels.o" ]]; then
  echo "ERROR: build the UBmem NPU test with ./build.sh ubmem-npu first" >&2
  exit 1
fi
exec "$ubmem_build/bin/kvaf_ubmem_test" --peer "$ubmem_peer" \
  --kernel "$ubmem_build/bin/kvaf_ubmem_kernels.o" "$@"
