#!/usr/bin/env bash
# Sweep BatchGet token sizes and exact miss counts, then summarize AIV time.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/kvaf/build-npu [batch-get-perf-options...]" >&2
  exit 2
fi

KVAF_BUILD_DIR="$1"
shift
KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${KVAF_BATCH_GET_PERF_RUNNER:=$KVAF_ROOT/tests/run_remote_cpu_batch_get_perf.sh}"
KVAF_BATCH_GET_TOKEN_BYTES=(128 256 512 656 1024 2048 4096)
KVAF_BATCH_GET_MISS_COUNTS=(32 64 128 256 512 1024 2048 4096 8192)

[[ -x "$KVAF_BATCH_GET_PERF_RUNNER" ]] || {
  echo "ERROR: BatchGet performance runner is not executable: $KVAF_BATCH_GET_PERF_RUNNER" >&2
  exit 1
}

# GetSystemCycle frequency is platform-specific. Use 1000 MHz by default and
# pass the same value to the Python profiler for cycle-to-millisecond output.
: "${KVAF_BATCH_GET_KERNEL_CLOCK_MHZ:=1000}"
if [[ ! "$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: KVAF_BATCH_GET_KERNEL_CLOCK_MHZ must be a positive integer" >&2
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

extract_profile_result() {
  local log_file="$1"
  local operation="$2"
  awk -v target="operation=$operation" \
    '$1 == "PERF_RESULT" && $2 == target { result = $0 } END { print result }' \
    "$log_file"
}

cycles_to_us() {
  local cycles="$1"
  awk -v cycles="$cycles" -v mhz="$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" \
    'BEGIN { printf "%.3f", cycles / mhz }'
}

adaptive_requested=false
base_options=()
for option in "$@"; do
  if [[ "$option" == "--adaptive-ub-device" ]]; then
    adaptive_requested=true
  else
    base_options+=("$option")
  fi
done

selected_ub_device=""
sweep_options=()
if ((${#base_options[@]} > 0)); then
  sweep_options=("${base_options[@]}")
fi
if [[ "$adaptive_requested" == true ]]; then
  calibration_token_bytes=656
  calibration_miss_count=2048
  calibration_log="$KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR/ub_device_calibration.log"
  echo "BATCH_GET_UB_DEVICE_CALIBRATION_BEGIN token_bytes=$calibration_token_bytes miss_count=$calibration_miss_count"
  if "$KVAF_BATCH_GET_PERF_RUNNER" "$KVAF_BUILD_DIR" \
    ${base_options[@]+"${base_options[@]}"} \
    --adaptive-ub-device \
    --adaptive-calibration-only \
    --request-length "$calibration_miss_count" \
    --token-bytes "$calibration_token_bytes" \
    --miss-count "$calibration_miss_count" \
    --no-kernel-profile \
    2>&1 | tee "$calibration_log"; then
    :
  else
    status=$?
    echo "BATCH_GET_UB_DEVICE_CALIBRATION_FAILED status=$status" >&2
    exit "$status"
  fi
  selected_record="$(
    awk '$1 == "[KVAF_AIV]" && $2 == "UB_DEVICE_SELECTED" { result = $0 }
         END { print result }' "$calibration_log"
  )"
  if [[ -z "$selected_record" ]]; then
    echo "ERROR: adaptive calibration did not select a UB device; ensure K uses an explicit --ub-device list" >&2
    exit 1
  fi
  selected_ub_device="$(extract_field "$selected_record" ub_device)"
  if [[ -z "$selected_ub_device" ]]; then
    echo "ERROR: malformed UB_DEVICE_SELECTED record: $selected_record" >&2
    exit 1
  fi
  sweep_options+=(--peer-ub-device "$selected_ub_device")
  echo "BATCH_GET_UB_DEVICE_CALIBRATION_DONE selected_ub_device=$selected_ub_device reuse=subsequent_shapes"
fi

token_bytes_values=()
miss_count_values=()
mean_us_values=()
p50_us_values=()
p90_us_values=()
p99_us_values=()
p50_gbps_values=()
post_mean_us_values=()
post_p50_us_values=()
post_p90_us_values=()
post_p99_us_values=()
completion_wait_mean_us_values=()
completion_wait_p50_us_values=()
completion_wait_p90_us_values=()
completion_wait_p99_us_values=()
selected_ub_device_values=()

for token_bytes in "${KVAF_BATCH_GET_TOKEN_BYTES[@]}"; do
  echo "BATCH_GET_TOKEN_SWEEP_BEGIN token_bytes=$token_bytes"
  for miss_count in "${KVAF_BATCH_GET_MISS_COUNTS[@]}"; do
    echo "BATCH_GET_MISS_SWEEP_BEGIN token_bytes=$token_bytes miss_count=$miss_count"
    log_file="$KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR/token_bytes_${token_bytes}_miss_count_${miss_count}.log"
    if "$KVAF_BATCH_GET_PERF_RUNNER" "$KVAF_BUILD_DIR" \
      ${sweep_options[@]+"${sweep_options[@]}"} \
      --token-bytes "$token_bytes" --miss-count "$miss_count" \
      --kernel-profile \
      --kernel-profile-clock-mhz "$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" \
      2>&1 | tee "$log_file"; then
      :
    else
      status=$?
      echo "BATCH_GET_MISS_SWEEP_FAILED token_bytes=$token_bytes miss_count=$miss_count status=$status" >&2
      exit "$status"
    fi

    total_profile_result="$(
      extract_profile_result "$log_file" batch_get_kernel_total
    )"
    post_profile_result="$(
      extract_profile_result "$log_file" batch_get_kernel_post
    )"
    completion_wait_profile_result="$(
      extract_profile_result "$log_file" batch_get_kernel_completion_wait
    )"
    if [[ -z "$total_profile_result" || -z "$post_profile_result" ||
      -z "$completion_wait_profile_result" ]]; then
      echo "ERROR: incomplete kernel profile results for token_bytes=$token_bytes miss_count=$miss_count" >&2
      exit 1
    fi

    active_record="$(
      awk '$1 == "[KVAF_AIV]" && ($2 == "UB_DEVICE_SELECTED" || $2 == "UB_DEVICE_REUSED") { result = $0 }
           END { print result }' "$log_file"
    )"
    shape_ub_device="$selected_ub_device"
    if [[ -n "$active_record" ]]; then
      active_ub_device="$(extract_field "$active_record" ub_device)"
      if [[ -z "$active_ub_device" ]]; then
        echo "ERROR: malformed UB device record: $active_record" >&2
        exit 1
      fi
      if [[ -n "$shape_ub_device" && "$active_ub_device" != "$shape_ub_device" ]]; then
        echo "ERROR: fixed UB device changed from $shape_ub_device to $active_ub_device" >&2
        exit 1
      fi
      shape_ub_device="$active_ub_device"
    fi
    if [[ "$adaptive_requested" == true && -z "$shape_ub_device" ]]; then
      echo "ERROR: fixed UB device was not reported for token_bytes=$token_bytes miss_count=$miss_count" >&2
      exit 1
    fi

    mean_ms="$(extract_field "$total_profile_result" mean_ms)"
    p50_ms="$(extract_field "$total_profile_result" p50_ms)"
    p90_ms="$(extract_field "$total_profile_result" p90_ms)"
    p99_ms="$(extract_field "$total_profile_result" p99_ms)"
    mean_cycles="$(extract_field "$total_profile_result" mean_cycles)"
    p50_cycles="$(extract_field "$total_profile_result" p50_cycles)"
    p90_cycles="$(extract_field "$total_profile_result" p90_cycles)"
    p99_cycles="$(extract_field "$total_profile_result" p99_cycles)"
    post_mean_cycles="$(extract_field "$post_profile_result" mean_cycles)"
    post_p50_cycles="$(extract_field "$post_profile_result" p50_cycles)"
    post_p90_cycles="$(extract_field "$post_profile_result" p90_cycles)"
    post_p99_cycles="$(extract_field "$post_profile_result" p99_cycles)"
    completion_wait_mean_cycles="$(extract_field "$completion_wait_profile_result" mean_cycles)"
    completion_wait_p50_cycles="$(extract_field "$completion_wait_profile_result" p50_cycles)"
    completion_wait_p90_cycles="$(extract_field "$completion_wait_profile_result" p90_cycles)"
    completion_wait_p99_cycles="$(extract_field "$completion_wait_profile_result" p99_cycles)"
    token_bytes_values+=("$token_bytes")
    miss_count_values+=("$miss_count")
    mean_us="$(cycles_to_us "$mean_cycles")"
    p50_us="$(cycles_to_us "$p50_cycles")"
    p90_us="$(cycles_to_us "$p90_cycles")"
    p99_us="$(cycles_to_us "$p99_cycles")"
    post_mean_us="$(cycles_to_us "$post_mean_cycles")"
    post_p50_us="$(cycles_to_us "$post_p50_cycles")"
    post_p90_us="$(cycles_to_us "$post_p90_cycles")"
    post_p99_us="$(cycles_to_us "$post_p99_cycles")"
    completion_wait_mean_us="$(cycles_to_us "$completion_wait_mean_cycles")"
    completion_wait_p50_us="$(cycles_to_us "$completion_wait_p50_cycles")"
    completion_wait_p90_us="$(cycles_to_us "$completion_wait_p90_cycles")"
    completion_wait_p99_us="$(cycles_to_us "$completion_wait_p99_cycles")"
    payload_bytes=$((miss_count * token_bytes))
    p50_gbps="$(awk -v byte_count="$payload_bytes" -v cycle_count="$p50_cycles" \
      -v mhz="$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" \
      'BEGIN { printf "%.3f", (cycle_count > 0 ? byte_count * mhz / cycle_count / 1000 : 0) }')"
    mean_us_values+=("$mean_us")
    p50_us_values+=("$p50_us")
    p90_us_values+=("$p90_us")
    p99_us_values+=("$p99_us")
    p50_gbps_values+=("$p50_gbps")
    post_mean_us_values+=("$post_mean_us")
    post_p50_us_values+=("$post_p50_us")
    post_p90_us_values+=("$post_p90_us")
    post_p99_us_values+=("$post_p99_us")
    completion_wait_mean_us_values+=("$completion_wait_mean_us")
    completion_wait_p50_us_values+=("$completion_wait_p50_us")
    completion_wait_p90_us_values+=("$completion_wait_p90_us")
    completion_wait_p99_us_values+=("$completion_wait_p99_us")
    selected_ub_device_values+=("$shape_ub_device")

    result=(
      "BATCH_GET_MISS_SWEEP_RESULT"
      "token_bytes=$token_bytes"
      "miss_count=$miss_count"
      "selected_ub_device=$shape_ub_device"
      "mean_ms=$mean_ms"
      "p50_ms=$p50_ms"
      "p90_ms=$p90_ms"
      "p99_ms=$p99_ms"
      "mean_cycles=$mean_cycles"
      "p50_cycles=$p50_cycles"
      "p90_cycles=$p90_cycles"
      "p99_cycles=$p99_cycles"
      "clock_mhz=$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ"
      "mean_us=$mean_us"
      "p50_us=$p50_us"
      "p90_us=$p90_us"
      "p99_us=$p99_us"
      "post_mean_us=$post_mean_us"
      "post_p50_us=$post_p50_us"
      "post_p90_us=$post_p90_us"
      "post_p99_us=$post_p99_us"
      "completion_wait_mean_us=$completion_wait_mean_us"
      "completion_wait_p50_us=$completion_wait_p50_us"
      "completion_wait_p90_us=$completion_wait_p90_us"
      "completion_wait_p99_us=$completion_wait_p99_us"
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
printf 'BATCH_GET_MISS_SWEEP_STAGE_SUMMARY stage=kernel_total\n'
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

printf '\nBATCH_GET_MISS_SWEEP_STAGE_SUMMARY stage=kernel_post\n'
printf '%-12s %-12s %15s %15s %15s %15s\n' \
  token_bytes miss_count mean_us p50_us p90_us p99_us
for index in "${!post_mean_us_values[@]}"; do
  printf '%-12s %-12s %15s %15s %15s %15s\n' \
    "${token_bytes_values[$index]}" \
    "${miss_count_values[$index]}" \
    "${post_mean_us_values[$index]}" \
    "${post_p50_us_values[$index]}" \
    "${post_p90_us_values[$index]}" \
    "${post_p99_us_values[$index]}"
done

printf '\nBATCH_GET_MISS_SWEEP_STAGE_SUMMARY stage=kernel_completion_wait\n'
printf '%-12s %-12s %15s %15s %15s %15s\n' \
  token_bytes miss_count mean_us p50_us p90_us p99_us
for index in "${!completion_wait_mean_us_values[@]}"; do
  printf '%-12s %-12s %15s %15s %15s %15s\n' \
    "${token_bytes_values[$index]}" \
    "${miss_count_values[$index]}" \
    "${completion_wait_mean_us_values[$index]}" \
    "${completion_wait_p50_us_values[$index]}" \
    "${completion_wait_p90_us_values[$index]}" \
    "${completion_wait_p99_us_values[$index]}"
done

echo "BATCH_GET_MISS_SWEEP_CLOCK clock_mhz=$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ conversion=cycles_per_mhz_to_us"

: "${KVAF_BATCH_GET_MISS_SWEEP_CSV:=$KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR/batch_get_miss_sweep_summary.csv}"
mkdir -p "$(dirname "$KVAF_BATCH_GET_MISS_SWEEP_CSV")"
{
  printf '%s\n' \
    'token_bytes,miss_count,mean_us,p50_us,p90_us,p99_us,p50_GBps,post_mean_us,post_p50_us,post_p90_us,post_p99_us,completion_wait_mean_us,completion_wait_p50_us,completion_wait_p90_us,completion_wait_p99_us,selected_ub_device'
  for index in "${!mean_us_values[@]}"; do
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${token_bytes_values[$index]}" \
      "${miss_count_values[$index]}" \
      "${mean_us_values[$index]}" \
      "${p50_us_values[$index]}" \
      "${p90_us_values[$index]}" \
      "${p99_us_values[$index]}" \
      "${p50_gbps_values[$index]}" \
      "${post_mean_us_values[$index]}" \
      "${post_p50_us_values[$index]}" \
      "${post_p90_us_values[$index]}" \
      "${post_p99_us_values[$index]}" \
      "${completion_wait_mean_us_values[$index]}" \
      "${completion_wait_p50_us_values[$index]}" \
      "${completion_wait_p90_us_values[$index]}" \
      "${completion_wait_p99_us_values[$index]}" \
      "${selected_ub_device_values[$index]}"
  done
} >"$KVAF_BATCH_GET_MISS_SWEEP_CSV"
echo "BATCH_GET_MISS_SWEEP_CSV_WRITTEN path=$KVAF_BATCH_GET_MISS_SWEEP_CSV rows=${#mean_us_values[@]}"
