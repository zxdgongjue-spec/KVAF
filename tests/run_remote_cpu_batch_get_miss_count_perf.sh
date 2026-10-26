#!/usr/bin/env bash
# Sweep BatchGet token sizes, exact miss counts, AIV cores and Jettys per core,
# then summarize AIV time.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/kvaf/build-npu [batch-get-perf-options...]" >&2
  exit 2
fi

KVAF_BUILD_DIR="$1"
shift
KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${KVAF_BATCH_GET_PERF_RUNNER:=$KVAF_ROOT/tests/run_remote_cpu_batch_get_perf.sh}"
KVAF_BATCH_GET_TOKEN_BYTES=(656)
KVAF_BATCH_GET_MISS_COUNTS=(2048 4096 8192)
KVAF_BATCH_GET_AIV_CORES=(2)
KVAF_BATCH_GET_JETTYS_PER_CORE=(2)

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
: "${KVAF_ADAPTIVE_UB_MISS_COUNT:=2048}"
if [[ ! "$KVAF_ADAPTIVE_UB_MISS_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: KVAF_ADAPTIVE_UB_MISS_COUNT must be a positive integer" >&2
  exit 2
fi
export KVAF_ADAPTIVE_UB_MISS_COUNT
: "${KVAF_ADAPTIVE_UB_DEVICES:=udmac0d1e2,udmac0d1e3,udmac1d1e2,udmac1d1e3}"
export KVAF_ADAPTIVE_UB_DEVICES

: "${KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR:=/home/w50065599/batchGetN2NLog/kvaf_tests_priority_sl/$(date +%Y%m%d_%H%M%S)_$$}"
if [[ -z "$KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR" ]]; then
  KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR="$(
    mktemp -d "${TMPDIR:-/tmp}/kvaf_batch_get_miss_sweep.XXXXXX"
  )"
else
  mkdir -p "$KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR"
fi
echo "BATCH_GET_MISS_SWEEP_LOG_DIR path=$KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR"
KVAF_BATCH_GET_SWEEP_COMBINATIONS=$((${#KVAF_BATCH_GET_TOKEN_BYTES[@]} *
  ${#KVAF_BATCH_GET_MISS_COUNTS[@]} * ${#KVAF_BATCH_GET_AIV_CORES[@]} *
  ${#KVAF_BATCH_GET_JETTYS_PER_CORE[@]}))
echo "BATCH_GET_MISS_SWEEP_PLAN token_sizes=${#KVAF_BATCH_GET_TOKEN_BYTES[@]} miss_counts=${#KVAF_BATCH_GET_MISS_COUNTS[@]} aiv_core_counts=${#KVAF_BATCH_GET_AIV_CORES[@]} jetty_counts=${#KVAF_BATCH_GET_JETTYS_PER_CORE[@]} combinations=$KVAF_BATCH_GET_SWEEP_COMBINATIONS"

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
adaptive_selection_skipped=false
selection_log="$KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR/ub_device_selection.log"
sweep_options=()
if ((${#base_options[@]} > 0)); then
  sweep_options=("${base_options[@]}")
fi
if [[ "$adaptive_requested" == true ]]; then
  IFS=',' read -r -a adaptive_ub_devices <<< "$KVAF_ADAPTIVE_UB_DEVICES"
  for index in "${!adaptive_ub_devices[@]}"; do
    device="${adaptive_ub_devices[$index]}"
    device="${device#"${device%%[![:space:]]*}"}"
    device="${device%"${device##*[![:space:]]}"}"
    adaptive_ub_devices[$index]="$device"
  done
  calibration_token_bytes=656
  calibration_miss_count="$KVAF_ADAPTIVE_UB_MISS_COUNT"
  calibration_options=(
    --request-length "$calibration_miss_count"
    --token-bytes "$calibration_token_bytes" --tokens-per-block 128
    --miss-count "$calibration_miss_count"
    --aiv-cores 2 --jettys-per-core 2
    --warmup 1 --iterations 5 --no-kernel-profile
  )
  echo "BATCH_GET_UB_DEVICE_FILTER_CHECK_BEGIN" |
    tee "$selection_log"
  filter_check_status=0
  if "$KVAF_BATCH_GET_PERF_RUNNER" "$KVAF_BUILD_DIR" \
    ${base_options[@]+"${base_options[@]}"} \
    --adaptive-ub-device --adaptive-calibration-only \
    "${calibration_options[@]}" \
    2>&1 | tee -a "$selection_log"; then
    :
  else
    filter_check_status=$?
  fi
  if ((filter_check_status != 0)); then
    echo "BATCH_GET_UB_DEVICE_FILTER_CHECK_FAILED status=$filter_check_status" >&2
    exit "$filter_check_status"
  fi
  if awk '$1 == "[KVAF_AIV]" && $2 == "UB_DEVICE_ADAPTIVE_SKIPPED" &&
           $0 ~ /reason=k_side_device_filter/ { found = 1 }
           END { exit !found }' "$selection_log"; then
    adaptive_selection_skipped=true
    echo "BATCH_GET_UB_DEVICE_CALIBRATION_SKIPPED reason=k_side_device_filter" |
      tee -a "$selection_log"
  else
    best_gibps=""
    echo "BATCH_GET_UB_DEVICE_CALIBRATION_BEGIN token_bytes=$calibration_token_bytes miss_count=$calibration_miss_count aiv_cores=2 jettys_per_core=2 warmup=1 samples=5 metric=batch_get_useful_GiBps mode=full_perf_path" |
      tee -a "$selection_log"
    for candidate_ub_device in "${adaptive_ub_devices[@]}"; do
      echo "BATCH_GET_UB_DEVICE_CANDIDATE_BEGIN ub_device=$candidate_ub_device" |
        tee -a "$selection_log"
      candidate_status=0
      if "$KVAF_BATCH_GET_PERF_RUNNER" "$KVAF_BUILD_DIR" \
        ${base_options[@]+"${base_options[@]}"} \
        --peer-ub-device "$candidate_ub_device" \
        "${calibration_options[@]}" \
        2>&1 | tee -a "$selection_log"; then
        :
      else
        candidate_status=$?
      fi
      if ((candidate_status != 0)); then
        echo "BATCH_GET_UB_DEVICE_CANDIDATE_FAILED ub_device=$candidate_ub_device status=$candidate_status" |
          tee -a "$selection_log"
        continue
      fi
      candidate_result="$(extract_profile_result "$selection_log" batch_get)"
      candidate_gibps="$(extract_field "$candidate_result" useful_GiBps || true)"
      if [[ ! "$candidate_gibps" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "ERROR: invalid BatchGet result for UB device $candidate_ub_device: $candidate_result" >&2
        exit 1
      fi
      echo "BATCH_GET_UB_DEVICE_CANDIDATE_RESULT ub_device=$candidate_ub_device useful_GiBps=$candidate_gibps" |
        tee -a "$selection_log"
      if [[ -z "$best_gibps" ]] ||
        awk -v candidate="$candidate_gibps" -v best="$best_gibps" \
          'BEGIN { exit !(candidate > best) }'; then
        selected_ub_device="$candidate_ub_device"
        best_gibps="$candidate_gibps"
      fi
    done
    if [[ -z "$selected_ub_device" ]]; then
      echo "ERROR: no configured UB device completed full-path calibration" >&2
      exit 1
    fi
    sweep_options=()
    if ((${#base_options[@]} > 0)); then
      sweep_options=("${base_options[@]}")
    fi
    sweep_options+=(--peer-ub-device "$selected_ub_device")
    echo "BATCH_GET_UB_DEVICE_CALIBRATION_DONE selected_ub_device=$selected_ub_device useful_GiBps=$best_gibps reuse=all_formal_shapes" |
      tee -a "$selection_log"
  fi
  echo "BATCH_GET_UB_DEVICE_SELECTION_LOG_WRITTEN path=$selection_log"
fi

token_bytes_values=()
miss_count_values=()
aiv_core_values=()
jettys_per_core_values=()
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
jetty_csv_files=()

for token_bytes in "${KVAF_BATCH_GET_TOKEN_BYTES[@]}"; do
  echo "BATCH_GET_TOKEN_SWEEP_BEGIN token_bytes=$token_bytes"
  for miss_count in "${KVAF_BATCH_GET_MISS_COUNTS[@]}"; do
    echo "BATCH_GET_MISS_SWEEP_BEGIN token_bytes=$token_bytes miss_count=$miss_count"
    for aiv_cores in "${KVAF_BATCH_GET_AIV_CORES[@]}"; do
      echo "BATCH_GET_CORE_SWEEP_BEGIN token_bytes=$token_bytes miss_count=$miss_count aiv_cores=$aiv_cores"
      for jettys_per_core in "${KVAF_BATCH_GET_JETTYS_PER_CORE[@]}"; do
        echo "BATCH_GET_CONFIG_SWEEP_BEGIN token_bytes=$token_bytes miss_count=$miss_count aiv_cores=$aiv_cores jettys_per_core=$jettys_per_core"
        log_file="$KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR/token_bytes_${token_bytes}_miss_count_${miss_count}_aiv_cores_${aiv_cores}_jettys_per_core_${jettys_per_core}.log"
        jetty_csv_file="${log_file%.log}.jetty.csv"
        run_status=0
        if "$KVAF_BATCH_GET_PERF_RUNNER" "$KVAF_BUILD_DIR" \
          ${sweep_options[@]+"${sweep_options[@]}"} \
          --token-bytes "$token_bytes" --miss-count "$miss_count" \
          --aiv-cores "$aiv_cores" --jettys-per-core "$jettys_per_core" \
          --kernel-profile \
          --jetty-csv "$jetty_csv_file" \
          --kernel-profile-clock-mhz "$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ" \
          2>&1 | tee "$log_file"; then
          :
        else
          run_status=$?
        fi
        if ((run_status != 0)); then
          echo "BATCH_GET_MISS_SWEEP_FAILED token_bytes=$token_bytes miss_count=$miss_count aiv_cores=$aiv_cores jettys_per_core=$jettys_per_core status=$run_status" >&2
          exit "$run_status"
        fi
        if [[ ! -s "$jetty_csv_file" ]] ||
          ! grep -q '^BATCH_GET_JETTY_CSV_WRITTEN ' "$log_file"; then
          echo "ERROR: missing fresh per-Jetty CSV: $jetty_csv_file" >&2
          exit 1
        fi
        jetty_csv_files+=("$jetty_csv_file")

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
          echo "ERROR: incomplete kernel profile results for token_bytes=$token_bytes miss_count=$miss_count aiv_cores=$aiv_cores jettys_per_core=$jettys_per_core" >&2
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
        if [[ "$adaptive_requested" == true && "$adaptive_selection_skipped" == false &&
          -z "$shape_ub_device" ]]; then
          echo "ERROR: fixed UB device was not reported for token_bytes=$token_bytes miss_count=$miss_count aiv_cores=$aiv_cores jettys_per_core=$jettys_per_core" >&2
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
        aiv_core_values+=("$aiv_cores")
        jettys_per_core_values+=("$jettys_per_core")
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
      "aiv_cores=$aiv_cores"
      "jettys_per_core=$jettys_per_core"
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
      "selected_ub_device=$shape_ub_device"
    )
    printf '%s\n' "${result[*]}"
        echo "BATCH_GET_CONFIG_SWEEP_DONE token_bytes=$token_bytes miss_count=$miss_count aiv_cores=$aiv_cores jettys_per_core=$jettys_per_core"
      done
      echo "BATCH_GET_CORE_SWEEP_DONE token_bytes=$token_bytes miss_count=$miss_count aiv_cores=$aiv_cores"
    done
    echo "BATCH_GET_MISS_SWEEP_DONE token_bytes=$token_bytes miss_count=$miss_count"
  done
  echo "BATCH_GET_TOKEN_SWEEP_DONE token_bytes=$token_bytes"
done

printf '\nBATCH_GET_MISS_SWEEP_SUMMARY combinations=%s units=us clock_mhz=%s aggregation=max_elapsed_across_cores\n' \
  "${#mean_us_values[@]}" "$KVAF_BATCH_GET_KERNEL_CLOCK_MHZ"
printf 'BATCH_GET_MISS_SWEEP_STAGE_SUMMARY stage=kernel_total\n'
printf '%-12s %-12s %-12s %-18s %15s %15s %15s %15s %15s\n' \
  token_bytes miss_count aiv_cores jettys_per_core mean_us p50_us p90_us p99_us p50_GB/s
for index in "${!mean_us_values[@]}"; do
  printf '%-12s %-12s %-12s %-18s %15s %15s %15s %15s %15s\n' \
    "${token_bytes_values[$index]}" \
    "${miss_count_values[$index]}" \
    "${aiv_core_values[$index]}" \
    "${jettys_per_core_values[$index]}" \
    "${mean_us_values[$index]}" \
    "${p50_us_values[$index]}" \
    "${p90_us_values[$index]}" \
    "${p99_us_values[$index]}" \
    "${p50_gbps_values[$index]}"
done

printf '\nBATCH_GET_MISS_SWEEP_STAGE_SUMMARY stage=kernel_post\n'
printf '%-12s %-12s %-12s %-18s %15s %15s %15s %15s\n' \
  token_bytes miss_count aiv_cores jettys_per_core mean_us p50_us p90_us p99_us
for index in "${!post_mean_us_values[@]}"; do
  printf '%-12s %-12s %-12s %-18s %15s %15s %15s %15s\n' \
    "${token_bytes_values[$index]}" \
    "${miss_count_values[$index]}" \
    "${aiv_core_values[$index]}" \
    "${jettys_per_core_values[$index]}" \
    "${post_mean_us_values[$index]}" \
    "${post_p50_us_values[$index]}" \
    "${post_p90_us_values[$index]}" \
    "${post_p99_us_values[$index]}"
done

printf '\nBATCH_GET_MISS_SWEEP_STAGE_SUMMARY stage=kernel_completion_wait\n'
printf '%-12s %-12s %-12s %-18s %15s %15s %15s %15s\n' \
  token_bytes miss_count aiv_cores jettys_per_core mean_us p50_us p90_us p99_us
for index in "${!completion_wait_mean_us_values[@]}"; do
  printf '%-12s %-12s %-12s %-18s %15s %15s %15s %15s\n' \
    "${token_bytes_values[$index]}" \
    "${miss_count_values[$index]}" \
    "${aiv_core_values[$index]}" \
    "${jettys_per_core_values[$index]}" \
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
    'token_bytes,miss_count,aiv_cores,jettys_per_core,mean_us,p50_us,p90_us,p99_us,p50_GBps,post_mean_us,post_p50_us,post_p90_us,post_p99_us,completion_wait_mean_us,completion_wait_p50_us,completion_wait_p90_us,completion_wait_p99_us,selected_ub_device'
  for index in "${!mean_us_values[@]}"; do
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${token_bytes_values[$index]}" \
      "${miss_count_values[$index]}" \
      "${aiv_core_values[$index]}" \
      "${jettys_per_core_values[$index]}" \
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

: "${KVAF_BATCH_GET_JETTY_SWEEP_CSV:=$KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR/batch_get_jetty_summary.csv}"
python3 "$KVAF_ROOT/tests/python/batch_get_jetty_profile.py" \
  "$KVAF_BATCH_GET_JETTY_SWEEP_CSV" "${jetty_csv_files[@]}"
