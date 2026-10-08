#!/usr/bin/env bash
# Sweep BatchPut entry sizes and row counts, then summarize AIV kernel cycles.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/kvaf/build-npu [batch-put-perf-options...]" >&2
  exit 2
fi

KVAF_BUILD_DIR="$1"
shift
KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KVAF_BATCH_PUT_PERF_RUNNER="$KVAF_ROOT/tests/run_remote_cpu_batch_put_perf.sh"
KVAF_BATCH_PUT_ENTRY_BYTES=(656 1024)
KVAF_BATCH_PUT_ROW_COUNTS=(1024 2048 4096 8192)

[[ -f "$KVAF_BATCH_PUT_PERF_RUNNER" ]] || {
  echo "ERROR: BatchPut performance runner is missing: $KVAF_BATCH_PUT_PERF_RUNNER" >&2
  exit 1
}

# GetSystemCycle frequency is platform-specific. Override this with the
# verified AIV clock before using the converted time and bandwidth columns.
: "${KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ:=1000}"
if [[ ! "$KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] ||
  ! awk -v value="$KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ" \
    'BEGIN { exit !(value > 0) }'; then
  echo "ERROR: KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ must be a positive number" >&2
  exit 2
fi

: "${KVAF_BATCH_PUT_SWEEP_LOG_DIR:=}"
if [[ -z "$KVAF_BATCH_PUT_SWEEP_LOG_DIR" ]]; then
  KVAF_BATCH_PUT_SWEEP_LOG_DIR="$(
    mktemp -d "${TMPDIR:-/tmp}/kvaf_batch_put_kernel_sweep.XXXXXX"
  )"
else
  mkdir -p "$KVAF_BATCH_PUT_SWEEP_LOG_DIR"
fi
echo "BATCH_PUT_SWEEP_LOG_DIR path=$KVAF_BATCH_PUT_SWEEP_LOG_DIR"

extract_field() {
  local record="$1"
  local name="$2"
  local field
  for field in $record; do
    if [[ "$field" == "$name="* ]]; then
      printf '%s' "${field#*=}"
      return 0
    fi
  done
  return 1
}

entry_bytes_values=()
row_count_values=()
mean_cycles_values=()
p50_cycles_values=()
p90_cycles_values=()
p99_cycles_values=()
mean_us_values=()
p50_us_values=()
p90_us_values=()
p99_us_values=()
mean_gbps_values=()
p50_gbps_values=()

for entry_bytes in "${KVAF_BATCH_PUT_ENTRY_BYTES[@]}"; do
  echo "BATCH_PUT_ENTRY_SWEEP_BEGIN entry_bytes=$entry_bytes"
  for row_count in "${KVAF_BATCH_PUT_ROW_COUNTS[@]}"; do
    echo "BATCH_PUT_ROW_SWEEP_BEGIN entry_bytes=$entry_bytes row_count=$row_count"
    log_file="$KVAF_BATCH_PUT_SWEEP_LOG_DIR/entry_bytes_${entry_bytes}_row_count_${row_count}.log"
    if bash "$KVAF_BATCH_PUT_PERF_RUNNER" "$KVAF_BUILD_DIR" "$@" \
      --entry-bytes "$entry_bytes" --row-count "$row_count" \
      --kernel-profile 2>&1 | tee "$log_file"; then
      :
    else
      status=$?
      echo "BATCH_PUT_ROW_SWEEP_FAILED entry_bytes=$entry_bytes row_count=$row_count status=$status" >&2
      exit "$status"
    fi

    profile_result="$(
      awk '$1 == "PERF_RESULT" && $2 == "operation=batch_put_kernel" { result = $0 } END { print result }' \
        "$log_file"
    )"
    if [[ -z "$profile_result" ]]; then
      echo "ERROR: no batch_put_kernel PERF_RESULT for entry_bytes=$entry_bytes row_count=$row_count" >&2
      exit 1
    fi

    mean_execute_cycles="$(extract_field "$profile_result" mean_execute_cycles)"
    p50_execute_cycles="$(extract_field "$profile_result" p50_execute_cycles)"
    p90_execute_cycles="$(extract_field "$profile_result" p90_execute_cycles)"
    p99_execute_cycles="$(extract_field "$profile_result" p99_execute_cycles)"
    mean_non_execute_cycles="$(extract_field "$profile_result" mean_non_execute_cycles)"
    p50_non_execute_cycles="$(extract_field "$profile_result" p50_non_execute_cycles)"
    p90_non_execute_cycles="$(extract_field "$profile_result" p90_non_execute_cycles)"
    p99_non_execute_cycles="$(extract_field "$profile_result" p99_non_execute_cycles)"
    mean_cycles="$(awk -v execute="$mean_execute_cycles" -v non_execute="$mean_non_execute_cycles" 'BEGIN { printf "%.3f", execute + non_execute }')"
    p50_cycles="$(awk -v execute="$p50_execute_cycles" -v non_execute="$p50_non_execute_cycles" 'BEGIN { printf "%.3f", execute + non_execute }')"
    p90_cycles="$(awk -v execute="$p90_execute_cycles" -v non_execute="$p90_non_execute_cycles" 'BEGIN { printf "%.3f", execute + non_execute }')"
    p99_cycles="$(awk -v execute="$p99_execute_cycles" -v non_execute="$p99_non_execute_cycles" 'BEGIN { printf "%.3f", execute + non_execute }')"
    mean_us="$(awk -v cycles="$mean_cycles" -v mhz="$KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ" 'BEGIN { printf "%.3f", cycles / mhz }')"
    p50_us="$(awk -v cycles="$p50_cycles" -v mhz="$KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ" 'BEGIN { printf "%.3f", cycles / mhz }')"
    p90_us="$(awk -v cycles="$p90_cycles" -v mhz="$KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ" 'BEGIN { printf "%.3f", cycles / mhz }')"
    p99_us="$(awk -v cycles="$p99_cycles" -v mhz="$KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ" 'BEGIN { printf "%.3f", cycles / mhz }')"
    payload_bytes=$((row_count * entry_bytes))
    mean_gbps="$(awk -v byte_count="$payload_bytes" -v cycle_count="$mean_cycles" \
      -v mhz="$KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ" \
      'BEGIN { printf "%.3f", (cycle_count > 0 ? byte_count * mhz / cycle_count / 1000 : 0) }')"
    p50_gbps="$(awk -v byte_count="$payload_bytes" -v cycle_count="$p50_cycles" \
      -v mhz="$KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ" \
      'BEGIN { printf "%.3f", (cycle_count > 0 ? byte_count * mhz / cycle_count / 1000 : 0) }')"

    entry_bytes_values+=("$entry_bytes")
    row_count_values+=("$row_count")
    mean_cycles_values+=("$mean_cycles")
    p50_cycles_values+=("$p50_cycles")
    p90_cycles_values+=("$p90_cycles")
    p99_cycles_values+=("$p99_cycles")
    mean_us_values+=("$mean_us")
    p50_us_values+=("$p50_us")
    p90_us_values+=("$p90_us")
    p99_us_values+=("$p99_us")
    mean_gbps_values+=("$mean_gbps")
    p50_gbps_values+=("$p50_gbps")

    printf '%s\n' "BATCH_PUT_SWEEP_RESULT entry_bytes=$entry_bytes row_count=$row_count mean_cycles=$mean_cycles p50_cycles=$p50_cycles p90_cycles=$p90_cycles p99_cycles=$p99_cycles clock_mhz=$KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ mean_us=$mean_us p50_us=$p50_us p90_us=$p90_us p99_us=$p99_us payload_bytes=$payload_bytes mean_GBps=$mean_gbps p50_GBps=$p50_gbps"
    echo "BATCH_PUT_ROW_SWEEP_DONE entry_bytes=$entry_bytes row_count=$row_count"
  done
  echo "BATCH_PUT_ENTRY_SWEEP_DONE entry_bytes=$entry_bytes"
done

printf '\nBATCH_PUT_SWEEP_SUMMARY combinations=%s units=us clock_mhz=%s aggregation=max_elapsed_across_cores\n' \
  "${#mean_us_values[@]}" "$KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ"
printf '%-12s %-12s %15s %15s %15s %15s %15s %15s\n' \
  entry_bytes row_count mean_us p50_us p90_us p99_us mean_GB/s p50_GB/s
for index in "${!mean_us_values[@]}"; do
  printf '%-12s %-12s %15s %15s %15s %15s %15s %15s\n' \
    "${entry_bytes_values[$index]}" \
    "${row_count_values[$index]}" \
    "${mean_us_values[$index]}" \
    "${p50_us_values[$index]}" \
    "${p90_us_values[$index]}" \
    "${p99_us_values[$index]}" \
    "${mean_gbps_values[$index]}" \
    "${p50_gbps_values[$index]}"
done

echo "BATCH_PUT_SWEEP_CLOCK clock_mhz=$KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ conversion=cycles_per_mhz_to_us"

: "${KVAF_BATCH_PUT_SWEEP_CSV:=$KVAF_BATCH_PUT_SWEEP_LOG_DIR/batch_put_kernel_sweep_summary.csv}"
mkdir -p "$(dirname "$KVAF_BATCH_PUT_SWEEP_CSV")"
{
  printf 'entry_bytes,row_count,mean_cycles,p50_cycles,p90_cycles,p99_cycles,clock_mhz,mean_us,p50_us,p90_us,p99_us,mean_GBps,p50_GBps\n'
  for index in "${!mean_us_values[@]}"; do
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${entry_bytes_values[$index]}" \
      "${row_count_values[$index]}" \
      "${mean_cycles_values[$index]}" \
      "${p50_cycles_values[$index]}" \
      "${p90_cycles_values[$index]}" \
      "${p99_cycles_values[$index]}" \
      "$KVAF_BATCH_PUT_KERNEL_CLOCK_MHZ" \
      "${mean_us_values[$index]}" \
      "${p50_us_values[$index]}" \
      "${p90_us_values[$index]}" \
      "${p99_us_values[$index]}" \
      "${mean_gbps_values[$index]}" \
      "${p50_gbps_values[$index]}"
  done
} >"$KVAF_BATCH_PUT_SWEEP_CSV"
echo "BATCH_PUT_SWEEP_CSV_WRITTEN path=$KVAF_BATCH_PUT_SWEEP_CSV rows=${#mean_us_values[@]}"
