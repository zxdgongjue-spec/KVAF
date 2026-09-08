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
: "${KVAF_CPU_BATCH_PUT_COMPARISON_CSV:=$KVAF_CPU_BATCH_PUT_SWEEP_LOG_DIR/cpu_batch_put_transport_total.csv}"
mkdir -p "$(dirname "$KVAF_CPU_BATCH_PUT_SWEEP_CSV")" \
  "$(dirname "$KVAF_CPU_BATCH_PUT_COMPARISON_CSV")"
summary_path="$(cd "$(dirname "$KVAF_CPU_BATCH_PUT_SWEEP_CSV")" && pwd -P)/$(basename "$KVAF_CPU_BATCH_PUT_SWEEP_CSV")"
transport_path="$(cd "$(dirname "$KVAF_CPU_BATCH_PUT_COMPARISON_CSV")" && pwd -P)/$(basename "$KVAF_CPU_BATCH_PUT_COMPARISON_CSV")"
if [[ "$summary_path" == "$transport_path" ||
      "$KVAF_CPU_BATCH_PUT_SWEEP_CSV" -ef "$KVAF_CPU_BATCH_PUT_COMPARISON_CSV" ]]; then
  echo "ERROR: runtime and transport CSV paths must be different" >&2
  exit 2
fi
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

result_to_csv() {
  local token="$1" miss="$2" record="$3"
  local mean p50 p90 p99 gbps
  mean="$(extract_field "$record" mean_us)" || return 1
  p50="$(extract_field "$record" p50_us)" || return 1
  p90="$(extract_field "$record" p90_us)" || return 1
  p99="$(extract_field "$record" p99_us)" || return 1
  # Keep the bandwidth computed from this operation's unrounded ns samples.
  gbps="$(extract_field "$record" p50_GBps)" || return 1
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$token" "$miss" "$mean" "$p50" "$p90" "$p99" "$gbps"
}

for csv_file in "$KVAF_CPU_BATCH_PUT_SWEEP_CSV" "$KVAF_CPU_BATCH_PUT_COMPARISON_CSV"; do
  printf '%s\n' \
    'token_bytes,miss_count,mean_us,p50_us,p90_us,p99_us,p50_GBps' >"$csv_file"
done

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
    if [[ -z "$call_result" || -z "$total_result" ]]; then
      echo "ERROR: incomplete CPU BatchPut results for token_bytes=$token_bytes miss_count=$miss_count" >&2
      exit 1
    fi

    if ! call_row="$(result_to_csv "$token_bytes" "$miss_count" "$call_result")" ||
       ! total_row="$(result_to_csv "$token_bytes" "$miss_count" "$total_result")"; then
      echo "ERROR: incomplete CPU BatchPut metrics for token_bytes=$token_bytes miss_count=$miss_count" >&2
      exit 1
    fi
    printf '%s\n' "$call_row" >>"$KVAF_CPU_BATCH_PUT_SWEEP_CSV"
    printf '%s\n' "$total_row" >>"$KVAF_CPU_BATCH_PUT_COMPARISON_CSV"
    echo "CPU_BATCH_PUT_MISS_SWEEP_RESULT scope=runtime_call csv=$call_row"
    echo "CPU_BATCH_PUT_MISS_SWEEP_RESULT scope=transport_total csv=$total_row"
    echo "CPU_BATCH_PUT_MISS_SWEEP_DONE token_bytes=$token_bytes miss_count=$miss_count"
  done
  echo "CPU_BATCH_PUT_TOKEN_SWEEP_DONE token_bytes=$token_bytes"
done

printf '\nCPU_BATCH_PUT_SWEEP_SUMMARY scope=runtime_call combinations=%s units=us\n' \
  "$(( ${#KVAF_TOKEN_BYTES[@]} * ${#KVAF_MISS_COUNTS[@]} ))"
cat "$KVAF_CPU_BATCH_PUT_SWEEP_CSV"
echo "CPU_BATCH_PUT_SWEEP_CSV_WRITTEN path=$KVAF_CPU_BATCH_PUT_SWEEP_CSV rows=$(( ${#KVAF_TOKEN_BYTES[@]} * ${#KVAF_MISS_COUNTS[@]} ))"
printf '\nCPU_BATCH_PUT_SWEEP_SUMMARY scope=transport_total combinations=%s units=us\n' \
  "$(( ${#KVAF_TOKEN_BYTES[@]} * ${#KVAF_MISS_COUNTS[@]} ))"
cat "$KVAF_CPU_BATCH_PUT_COMPARISON_CSV"
echo "CPU_BATCH_PUT_COMPARISON_CSV_WRITTEN path=$KVAF_CPU_BATCH_PUT_COMPARISON_CSV rows=$(( ${#KVAF_TOKEN_BYTES[@]} * ${#KVAF_MISS_COUNTS[@]} ))"
