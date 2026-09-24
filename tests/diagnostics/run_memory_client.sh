#!/usr/bin/env bash
set -euo pipefail
# Run against the dedicated memory_service.sh endpoint supplied in the environment.
if (( $# != 3 )); then
  echo 'Usage: run_memory_client.sh BUILD OUTPUT_DIR CLOCK_MHZ' >&2; exit 2
fi
build=$1; output=$2; clock=$3
[[ ! -e $output ]] || { echo 'Use a new output directory' >&2; exit 2; }
mkdir -p "$output"
runner=$(dirname "$0")/bandwidth_scopes.py
for stride in 656 704 768 1024; do
  python3 "$runner" --build "$build" --output "$output/stride-$stride.json" \
    --clock-mhz "$clock" --record-stride "$stride" \
    --token-bytes 656 --tokens-per-block 128 --request-length 32768 \
    --table-capacity 16384 --miss-count 5000 --aiv-cores 2 --jettys-per-core 1 \
    --source-access-pattern random --output-access-pattern contiguous \
    --seed 20260903 --warmup 3 --iterations 30 2>&1 | tee "$output/stride-$stride.log"
done
