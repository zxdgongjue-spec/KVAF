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

# GetSystemCycle frequency is platform-specific. When this optional value is
# supplied, cycles / MHz is also reported as microseconds.
: "${KVAF_BATCH_GET_KERNEL_CLOCK_MHZ:=}"
if [[ -n "$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" ]]; then
  if [[ ! "$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] ||
    ! awk -v value="$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" \
      'BEGIN { exit !(value > 0) }'; then
    echo "ERROR: KVAF_BATCH_GET_KERNEL_CLOCK_MHZ must be a positive number" >&2
    exit 2
  fi
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
mean_cycles_values=()
p50_cycles_values=()
p90_cycles_values=()
p99_cycles_values=()

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
    mean_cycles_values+=("$mean_cycles")
    p50_cycles_values+=("$p50_cycles")
    p90_cycles_values+=("$p90_cycles")
    p99_cycles_values+=("$p99_cycles")

    result=(
      "BATCH_GET_MISS_SWEEP_RESULT"
      "token_bytes=$token_bytes"
      "miss_count=$miss_count"
      "mean_cycles=$mean_cycles"
      "p50_cycles=$p50_cycles"
      "p90_cycles=$p90_cycles"
      "p99_cycles=$p99_cycles"
    )
    if [[ -n "$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" ]]; then
      mean_us="$(awk -v cycles="$mean_cycles" -v mhz="$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" 'BEGIN { printf "%.3f", cycles / mhz }')"
      p50_us="$(awk -v cycles="$p50_cycles" -v mhz="$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" 'BEGIN { printf "%.3f", cycles / mhz }')"
      p90_us="$(awk -v cycles="$p90_cycles" -v mhz="$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" 'BEGIN { printf "%.3f", cycles / mhz }')"
      p99_us="$(awk -v cycles="$p99_cycles" -v mhz="$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" 'BEGIN { printf "%.3f", cycles / mhz }')"
      result+=(
        "clock_mhz=$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ"
        "mean_us=$mean_us"
        "p50_us=$p50_us"
        "p90_us=$p90_us"
        "p99_us=$p99_us"
      )
    fi
    printf '%s\n' "${result[*]}"
    echo "BATCH_GET_MISS_SWEEP_DONE token_bytes=$token_bytes miss_count=$miss_count"
  done
  echo "BATCH_GET_TOKEN_SWEEP_DONE token_bytes=$token_bytes"
done

printf '\nBATCH_GET_MISS_SWEEP_SUMMARY combinations=%s units=cycles aggregation=max_elapsed_across_cores\n' \
  "${#mean_cycles_values[@]}"
printf '%-12s %-12s %15s %15s %15s %15s\n' \
  token_bytes miss_count mean_cycles p50_cycles p90_cycles p99_cycles
for index in "${!mean_cycles_values[@]}"; do
  printf '%-12s %-12s %15s %15s %15s %15s\n' \
    "${token_bytes_values[$index]}" \
    "${miss_count_values[$index]}" \
    "${mean_cycles_values[$index]}" \
    "${p50_cycles_values[$index]}" \
    "${p90_cycles_values[$index]}" \
    "${p99_cycles_values[$index]}"
done

if [[ -n "$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" ]]; then
  echo "BATCH_GET_MISS_SWEEP_CLOCK clock_mhz=$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ conversion=cycles_per_mhz_to_us"
fi
