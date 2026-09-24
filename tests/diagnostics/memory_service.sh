#!/usr/bin/env bash
# Launch a dedicated service. Do not stop or alter an existing service.
set -euo pipefail
if (( $# < 5 )); then
  echo 'Usage: memory_service.sh CPU_SERVICE_BINARY CPU_NODE MEMORY_NODE base|thp OUTPUT_DIR [service arguments...]' >&2
  exit 2
fi
binary=$1; cpu_node=$2; memory_node=$3; mode=$4; output=$5; shift 5
[[ $mode == base || $mode == thp ]] || exit 2
command -v numactl >/dev/null
[[ ! -e $output ]] || { echo 'Use a new output directory' >&2; exit 2; }
mkdir -p "$output"
numactl --hardware > "$output/numa-hardware.txt"
cat /sys/kernel/mm/transparent_hugepage/enabled > "$output/thp-enabled.txt"
printf '%s\n' "cpu_node=$cpu_node memory_node=$memory_node mode=$mode" > "$output/config.txt"
numactl --cpunodebind="$cpu_node" --membind="$memory_node" "$binary" --memory-page-mode "$mode" "$@" > "$output/service.log" 2>&1 &
service_pid=$!
echo "$service_pid" > "$output/pid"
trap 'kill "$service_pid" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM
while kill -0 "$service_pid" 2>/dev/null; do
  # Preserve full snapshots: THP advice alone is NOT evidence of huge pages.
  stamp=$(date +%s)
  cat "/proc/$service_pid/smaps" > "$output/smaps-$stamp.txt" 2>/dev/null || true
  cat "/proc/$service_pid/numa_maps" > "$output/numa_maps-$stamp.txt" 2>/dev/null || true
  sleep 5
done
wait "$service_pid"
