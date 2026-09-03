#!/usr/bin/env bash
# Sweep BatchGet token sizes and exact miss counts, then summarize AIV cycles.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/kvaf/build-npu [batch-get-perf-options...]" >&2
  exit 2
fi

KVAF_BUILD_DIR="$1"
shift
KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KVAF_BATCH_GET_PERF_RUNNER="$KVAF_ROOT/tests/run_remote_cpu_batch_get_perf.sh"
KVAF_BATCH_GET_TOKEN_BYTES=(128 256 512 656 1024 2048 4096)
KVAF_BATCH_GET_MISS_COUNTS=(32 64 128 256 512 1024 2048 4096 8192)

[[ -x "$KVAF_BATCH_GET_PERF_RUNNER" ]] || {
  echo "ERROR: BatchGet performance runner is not executable: $KVAF_BATCH_GET_PERF_RUNNER" >&2
  exit 1
}

# GetSystemCycle frequency is platform-specific. Use 1000 MHz by default and
# allow the target environment to override it for cycle-to-microsecond output.
: "${KVAF_BATCH_GET_KERNEL_CLOCK_MHZ:=1000}"
if [[ ! "$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] ||
  ! awk -v value="$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" \
    'BEGIN { exit !(value > 0) }'; then
  echo "ERROR: KVAF_BATCH_GET_KERNEL_CLOCK_MHZ must be a positive number" >&2
  exit 2
fi

: "${KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR:=}"
if [[ -z "$KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR" ]]; then
  KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR="$(
    mktemp -d "${TMPDIR:-/tmp}/kvaf_batch_get_miss_sweep.XXXXXX"
  )"
else
  mkdir -p "$KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR"
fi
echo "BATCH_GET_MISS_SWEEP_LOG_DIR path=$KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR"

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

token_bytes_values=()
miss_count_values=()
mean_us_values=()
p50_us_values=()
p90_us_values=()
p99_us_values=()
p50_gbps_values=()

for token_bytes in "${KVAF_BATCH_GET_TOKEN_BYTES[@]}"; do
  echo "BATCH_GET_TOKEN_SWEEP_BEGIN token_bytes=$token_bytes"
  for miss_count in "${KVAF_BATCH_GET_MISS_COUNTS[@]}"; do
    echo "BATCH_GET_MISS_SWEEP_BEGIN token_bytes=$token_bytes miss_count=$miss_count"
    log_file="$KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR/token_bytes_${token_bytes}_miss_count_${miss_count}.log"
    if "$KVAF_BATCH_GET_PERF_RUNNER" "$KVAF_BUILD_DIR" "$@" \
      --token-bytes "$token_bytes" --miss-count "$miss_count" \
      --kernel-profile 2>&1 | tee "$log_file"; then
      :
    else
      status=$?
      echo "BATCH_GET_MISS_SWEEP_FAILED token_bytes=$token_bytes miss_count=$miss_count status=$status" >&2
      exit "$status"
    fi

    profile_result="$(
      awk '$1 == "PERF_RESULT" && $2 == "operation=batch_get_kernel" { result = $0 } END { print result }' \
        "$log_file"
    )"
    if [[ -z "$profile_result" ]]; then
      echo "ERROR: no batch_get_kernel PERF_RESULT for token_bytes=$token_bytes miss_count=$miss_count" >&2
      exit 1
    fi

    mean_cycles="$(extract_field "$profile_result" mean_cycles)"
    p50_cycles="$(extract_field "$profile_result" p50_cycles)"
    p90_cycles="$(extract_field "$profile_result" p90_cycles)"
    p99_cycles="$(extract_field "$profile_result" p99_cycles)"
    token_bytes_values+=("$token_bytes")
    miss_count_values+=("$miss_count")
    mean_us="$(awk -v cycles="$mean_cycles" -v mhz="$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" 'BEGIN { printf "%.3f", cycles / mhz }')"
    p50_us="$(awk -v cycles="$p50_cycles" -v mhz="$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" 'BEGIN { printf "%.3f", cycles / mhz }')"
    p90_us="$(awk -v cycles="$p90_cycles" -v mhz="$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" 'BEGIN { printf "%.3f", cycles / mhz }')"
    p99_us="$(awk -v cycles="$p99_cycles" -v mhz="$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" 'BEGIN { printf "%.3f", cycles / mhz }')"
    payload_bytes=$((miss_count * token_bytes))
    p50_gbps="$(awk -v byte_count="$payload_bytes" -v cycle_count="$p50_cycles" \
      -v mhz="$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" \
      'BEGIN { printf "%.3f", (cycle_count > 0 ? byte_count * mhz / cycle_count / 1000 : 0) }')"
    mean_us_values+=("$mean_us")
    p50_us_values+=("$p50_us")
    p90_us_values+=("$p90_us")
    p99_us_values+=("$p99_us")
    p50_gbps_values+=("$p50_gbps")

    result=(
      "BATCH_GET_MISS_SWEEP_RESULT"
      "token_bytes=$token_bytes"
      "miss_count=$miss_count"
      "mean_cycles=$mean_cycles"
      "p50_cycles=$p50_cycles"
      "p90_cycles=$p90_cycles"
      "p99_cycles=$p99_cycles"
      "clock_mhz=$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ"
      "mean_us=$mean_us"
      "p50_us=$p50_us"
      "p90_us=$p90_us"
      "p99_us=$p99_us"
      "payload_bytes=$payload_bytes"
      "p50_GBps=$p50_gbps"
    )
    printf '%s\n' "${result[*]}"
    echo "BATCH_GET_MISS_SWEEP_DONE token_bytes=$token_bytes miss_count=$miss_count"
  done
  echo "BATCH_GET_TOKEN_SWEEP_DONE token_bytes=$token_bytes"
done

printf '\nBATCH_GET_MISS_SWEEP_SUMMARY combinations=%s units=us clock_mhz=%s aggregation=max_elapsed_across_cores\n' \
  "${#mean_us_values[@]}" "$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ"
printf '%-12s %-12s %15s %15s %15s %15s %15s\n' \
  token_bytes miss_count mean_us p50_us p90_us p99_us p50_GB/s
for index in "${!mean_us_values[@]}"; do
  printf '%-12s %-12s %15s %15s %15s %15s %15s\n' \
    "${token_bytes_values[$index]}" \
    "${miss_count_values[$index]}" \
    "${mean_us_values[$index]}" \
    "${p50_us_values[$index]}" \
    "${p90_us_values[$index]}" \
    "${p99_us_values[$index]}" \
    "${p50_gbps_values[$index]}"
done

echo "BATCH_GET_MISS_SWEEP_CLOCK clock_mhz=$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ conversion=cycles_per_mhz_to_us"
