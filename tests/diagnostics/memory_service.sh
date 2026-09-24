#!/usr/bin/env bash
# Launch a dedicated service. Do not stop or alter an existing service.
set -euo pipefail
if (( $# < 5 )); then
  echo 'Usage: memory_service.sh CPU_SERVICE_BINARY CPU_NODE MEMORY_NODE base|thp OUTPUT_DIR [service arguments...]' >&2
  exit 2
fi
binary=$1; cpu_node=$2; memory_node=$3; mode=$4; output=$5; shift 5
[[ $mode == base || $mode == thp ]] || { echo "Invalid page mode: $mode" >&2; exit 2; }
command -v numactl >/dev/null || { echo "numactl is missing; install it on the CPU service host" >&2; exit 2; }
[[ -x $binary ]] || { echo "Service binary does not exist or is not executable: $binary" >&2; echo "Locate it with: find build-cpu build-npu -type f -name kvaf_cpu_service" >&2; exit 2; }
[[ ! -e $output ]] || { echo 'Use a new output directory' >&2; exit 2; }
mkdir -p "$output"
numactl --hardware > "$output/numa-hardware.txt"
if [[ -r /sys/kernel/mm/transparent_hugepage/enabled ]]; then
  cat /sys/kernel/mm/transparent_hugepage/enabled > "$output/thp-enabled.txt"
else
  echo "THP sysfs unavailable" > "$output/thp-enabled.txt"
fi
# Fail visibly if container permissions or node selection reject the policy.
if ! numactl --cpunodebind="$cpu_node" --membind="$memory_node" true > "$output/numa-policy-check.log" 2>&1; then
  cat "$output/numa-policy-check.log" >&2
  echo "NUMA policy rejected; no service started and no fallback applied." >&2
  exit 2
fi
printf '%s\n' "cpu_node=$cpu_node memory_node=$memory_node mode=$mode" > "$output/config.txt"
numactl --cpunodebind="$cpu_node" --membind="$memory_node" "$binary" --memory-page-mode "$mode" "$@" > "$output/service.log" 2>&1 &
service_pid=$!
echo "$service_pid" > "$output/pid"
tail -n +1 -f "$output/service.log" &
log_pid=$!
cleanup() {
  kill "$log_pid" 2>/dev/null || true
  if [[ -n ${service_pid:-} ]]; then
    kill "$service_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
echo "Service process launched pid=$service_pid; startup output follows (not a readiness confirmation)."
trap 'exit 130' INT TERM
while kill -0 "$service_pid" 2>/dev/null; do
  # Preserve full snapshots: THP advice alone is NOT evidence of huge pages.
  stamp=$(date +%s)
  cat "/proc/$service_pid/smaps" > "$output/smaps-$stamp.txt" 2>/dev/null || true
  cat "/proc/$service_pid/numa_maps" > "$output/numa_maps-$stamp.txt" 2>/dev/null || true
  sleep 5
done
status=0
wait "$service_pid" || status=$?
service_pid=
kill "$log_pid" 2>/dev/null || true
echo "Service exited with status=$status. Full log: $output/service.log" >&2
cat "$output/service.log" >&2
exit "$status"
