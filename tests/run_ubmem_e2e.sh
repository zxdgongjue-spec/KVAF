#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 /absolute/path/to/build-npu CPU_IP:28888 [test options]" >&2
  exit 2
fi
ubmem_build="$1"
ubmem_peer="$2"
shift 2
if [[ ! -x "$ubmem_build/bin/kvaf_ubmem_test" || ! -s "$ubmem_build/bin/kvaf_execute_batch_get_kernels.o" || ! -s "$ubmem_build/bin/kvaf_execute_batch_put_kernels.o" ]]; then
  echo "ERROR: build the UBmem NPU test with ./build.sh npu --without-python first" >&2
  exit 1
fi
exec "$ubmem_build/bin/kvaf_ubmem_test" --peer "$ubmem_peer" \
  --get-kernel "$ubmem_build/bin/kvaf_execute_batch_get_kernels.o" \
  --put-kernel "$ubmem_build/bin/kvaf_execute_batch_put_kernels.o" "$@"
