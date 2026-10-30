#!/usr/bin/env bash
# Pure C++ raw-address GET, 4096 x 656 bytes, across the reference AIV core sweep.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 BUILD_DIR --peer CPU:PORT [--device N] [bandwidth options]" >&2
  echo "Sweeps MTE over 1,2,4,8,16,32 cores; use kvaf_ubmem_bandwidth for a single configuration." >&2
  exit 2
fi
ubmem_build="$1"
shift
for option in "$@"; do
  if [[ "$option" == --cores ]]; then
    echo "ERROR: this sweep selects --cores; use the bandwidth executable to select it." >&2
    exit 2
  fi
done
for cores in 1 2 4 8 16 32; do
  "$ubmem_build/bin/kvaf_ubmem_bandwidth" \
    --kernel "$ubmem_build/bin/kvaf_execute_batch_get_kernels.o" \
    --rows 4096 --token-bytes 656 "$@" --cores "$cores"
done
