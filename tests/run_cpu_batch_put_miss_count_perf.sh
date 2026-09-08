#!/usr/bin/env bash
# Sweep CPU BatchPut token sizes and compact-prefix row counts.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/kvaf/build-cpu [cpu-batch-put-perf-options...]" >&2
  exit 2
fi

KVAF_BUILD_DIR="$1"
shift
KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KVAF_PERF_RUNNER="$KVAF_ROOT/tests/run_cpu_batch_put_perf.sh"
read -r -a KVAF_TOKEN_BYTES <<< \
  "${KVAF_CPU_BATCH_PUT_TOKEN_SWEEP:-128 256 512 656 1024 2048 4096}"
read -r -a KVAF_MISS_COUNTS <<< \
  "${KVAF_CPU_BATCH_PUT_MISS_COUNT_SWEEP:-32 64 128 256 512 1024 2048 4096 8192}"

[[ -x "$KVAF_PERF_RUNNER" ]] || {
  echo "ERROR: CPU BatchPut performance runner is not executable: $KVAF_PERF_RUNNER" >&2
  exit 1
}
for value in "${KVAF_TOKEN_BYTES[@]}" "${KVAF_MISS_COUNTS[@]}"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: CPU BatchPut sweep values must be positive integers" >&2
    exit 2
  }
done

: "${KVAF_CPU_BATCH_PUT_SWEEP_LOG_DIR:=}"
if [[ -z "$KVAF_CPU_BATCH_PUT_SWEEP_LOG_DIR" ]]; then
  KVAF_CPU_BATCH_PUT_SWEEP_LOG_DIR="$(
    mktemp -d "${TMPDIR:-/tmp}/kvaf_cpu_batch_put_sweep.XXXXXX"
  )"
else
  mkdir -p "$KVAF_CPU_BATCH_PUT_SWEEP_LOG_DIR"
fi
: "${KVAF_CPU_BATCH_PUT_SWEEP_CSV:=$KVAF_CPU_BATCH_PUT_SWEEP_LOG_DIR/cpu_batch_put_sweep_summary.csv}"
mkdir -p "$(dirname "$KVAF_CPU_BATCH_PUT_SWEEP_CSV")"
echo "CPU_BATCH_PUT_SWEEP_LOG_DIR path=$KVAF_CPU_BATCH_PUT_SWEEP_LOG_DIR"

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

extract_result() {
  local log_file="$1"
  local operation="$2"
  awk -v target="operation=$operation" \
    '$1 == "PERF_RESULT" && $2 == target { result = $0 } END { print result }' \
    "$log_file"
}

printf '%s\n' \
  'token_bytes,miss_count,payload_bytes,call_mean_us,call_p50_us,call_p90_us,call_p99_us,total_mean_us,total_p50_us,total_p90_us,total_p99_us,prepare_mean_us,prepare_p50_us,prepare_p90_us,prepare_p99_us,post_mean_us,post_p50_us,post_p90_us,post_p99_us,completion_wait_mean_us,completion_wait_p50_us,completion_wait_p90_us,completion_wait_p99_us,p50_GBps' \
  >"$KVAF_CPU_BATCH_PUT_SWEEP_CSV"

for token_bytes in "${KVAF_TOKEN_BYTES[@]}"; do
  echo "CPU_BATCH_PUT_TOKEN_SWEEP_BEGIN token_bytes=$token_bytes"
  for miss_count in "${KVAF_MISS_COUNTS[@]}"; do
    echo "CPU_BATCH_PUT_MISS_SWEEP_BEGIN token_bytes=$token_bytes miss_count=$miss_count"
    log_file="$KVAF_CPU_BATCH_PUT_SWEEP_LOG_DIR/token_bytes_${token_bytes}_miss_count_${miss_count}.log"
    if "$KVAF_PERF_RUNNER" "$KVAF_BUILD_DIR" "$@" \
      --token-bytes "$token_bytes" --miss-count "$miss_count" \
      2>&1 | tee "$log_file"; then
      :
    else
      status=$?
      echo "CPU_BATCH_PUT_MISS_SWEEP_FAILED token_bytes=$token_bytes miss_count=$miss_count status=$status" >&2
      exit "$status"
    fi

    call_result="$(extract_result "$log_file" batch_put)"
    total_result="$(extract_result "$log_file" batch_put_transport_total)"
    prepare_result="$(extract_result "$log_file" batch_put_transport_prepare)"
    post_result="$(extract_result "$log_file" batch_put_transport_post)"
    completion_result="$(extract_result "$log_file" batch_put_transport_completion_wait)"
    if [[ -z "$call_result" || -z "$total_result" || -z "$prepare_result" ||
      -z "$post_result" || -z "$completion_result" ]]; then
      echo "ERROR: incomplete CPU BatchPut results for token_bytes=$token_bytes miss_count=$miss_count" >&2
      exit 1
    fi

    payload_bytes=$((token_bytes * miss_count))
    call_mean_us="$(extract_field "$call_result" mean_us)"
    call_p50_us="$(extract_field "$call_result" p50_us)"
    call_p90_us="$(extract_field "$call_result" p90_us)"
    call_p99_us="$(extract_field "$call_result" p99_us)"
    total_mean_us="$(extract_field "$total_result" mean_us)"
    total_p50_us="$(extract_field "$total_result" p50_us)"
    total_p90_us="$(extract_field "$total_result" p90_us)"
    total_p99_us="$(extract_field "$total_result" p99_us)"
    prepare_mean_us="$(extract_field "$prepare_result" mean_us)"
    prepare_p50_us="$(extract_field "$prepare_result" p50_us)"
    prepare_p90_us="$(extract_field "$prepare_result" p90_us)"
    prepare_p99_us="$(extract_field "$prepare_result" p99_us)"
    post_mean_us="$(extract_field "$post_result" mean_us)"
    post_p50_us="$(extract_field "$post_result" p50_us)"
    post_p90_us="$(extract_field "$post_result" p90_us)"
    post_p99_us="$(extract_field "$post_result" p99_us)"
    completion_mean_us="$(extract_field "$completion_result" mean_us)"
    completion_p50_us="$(extract_field "$completion_result" p50_us)"
    completion_p90_us="$(extract_field "$completion_result" p90_us)"
    completion_p99_us="$(extract_field "$completion_result" p99_us)"
    p50_gbps="$(awk -v bytes="$payload_bytes" -v us="$total_p50_us" \
      'BEGIN { printf "%.3f", (us > 0 ? bytes / us / 1000 : 0) }')"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$token_bytes" "$miss_count" "$payload_bytes" \
      "$call_mean_us" "$call_p50_us" "$call_p90_us" "$call_p99_us" \
      "$total_mean_us" "$total_p50_us" "$total_p90_us" "$total_p99_us" \
      "$prepare_mean_us" "$prepare_p50_us" "$prepare_p90_us" "$prepare_p99_us" \
      "$post_mean_us" "$post_p50_us" "$post_p90_us" "$post_p99_us" \
      "$completion_mean_us" "$completion_p50_us" "$completion_p90_us" \
      "$completion_p99_us" "$p50_gbps" >>"$KVAF_CPU_BATCH_PUT_SWEEP_CSV"
    echo "CPU_BATCH_PUT_MISS_SWEEP_RESULT token_bytes=$token_bytes miss_count=$miss_count total_mean_us=$total_mean_us total_p50_us=$total_p50_us prepare_mean_us=$prepare_mean_us post_mean_us=$post_mean_us completion_wait_mean_us=$completion_mean_us p50_GBps=$p50_gbps"
    echo "CPU_BATCH_PUT_MISS_SWEEP_DONE token_bytes=$token_bytes miss_count=$miss_count"
  done
  echo "CPU_BATCH_PUT_TOKEN_SWEEP_DONE token_bytes=$token_bytes"
done

print_summary() {
  local stage="$1"
  local first_column="$2"
  printf '\nCPU_BATCH_PUT_SWEEP_STAGE_SUMMARY stage=%s\n' "$stage"
  printf '%-12s %-12s %15s %15s %15s %15s\n' \
    token_bytes miss_count mean_us p50_us p90_us p99_us
  awk -F, -v first="$first_column" 'NR > 1 {
    printf "%-12s %-12s %15s %15s %15s %15s\n", $1, $2,
           $(first), $(first + 1), $(first + 2), $(first + 3)
  }' "$KVAF_CPU_BATCH_PUT_SWEEP_CSV"
}

printf '\nCPU_BATCH_PUT_SWEEP_SUMMARY combinations=%s units=us\n' \
  "$(( ${#KVAF_TOKEN_BYTES[@]} * ${#KVAF_MISS_COUNTS[@]} ))"
print_summary runtime_call 4
print_summary transport_total 8
print_summary transport_prepare 12
print_summary transport_post 16
print_summary transport_completion_wait 20
echo "CPU_BATCH_PUT_SWEEP_CSV_WRITTEN path=$KVAF_CPU_BATCH_PUT_SWEEP_CSV"
