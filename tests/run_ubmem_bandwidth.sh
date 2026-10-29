#!/usr/bin/env bash
# Pure C++ raw-address GET, 4096 x 656 bytes, across the reference AIV core sweep.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 BUILD_DIR --peer CPU:PORT [--device N] [bandwidth options]" >&2
  echo "Sweeps simt/mte and 1,2,4,8,16,32 cores; use kvaf_ubmem_bandwidth for a single configuration." >&2
  exit 2
fi
ubmem_build="$1"
shift
for option in "$@"; do
  if [[ "$option" == --engine || "$option" == --cores ]]; then
    echo "ERROR: this sweep selects --engine and --cores; use the bandwidth executable to select them." >&2
    exit 2
  fi
done
for engine in simt mte; do
  for cores in 1 2 4 8 16 32; do
    "$ubmem_build/bin/kvaf_ubmem_bandwidth" \
      --kernel "$ubmem_build/bin/kvaf_ubmem_kernels.o" \
      --rows 4096 --token-bytes 656 "$@" --engine "$engine" --cores "$cores"
  done
done
