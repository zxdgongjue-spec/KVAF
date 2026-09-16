#!/usr/bin/env bash
set -euo pipefail

KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/kvaf_sweep_reuse_test.XXXXXX")"
trap 'rm -rf "$temp_dir"' EXIT

mock_runner="$temp_dir/mock_runner.sh"
cat >"$mock_runner" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
shift # build directory
adaptive=false
calibration_only=false
peer_device=""
token_bytes=""
miss_count=""
request_length=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --adaptive-ub-device) adaptive=true; shift ;;
    --adaptive-calibration-only) calibration_only=true; shift ;;
    --peer-ub-device) peer_device="$2"; shift 2 ;;
    --token-bytes) token_bytes="$2"; shift 2 ;;
    --miss-count) miss_count="$2"; shift 2 ;;
    --request-length) request_length="$2"; shift 2 ;;
    *) shift ;;
  esac
done

count=0
[[ ! -f "$KVAF_MOCK_COUNT_FILE" ]] || count="$(<"$KVAF_MOCK_COUNT_FILE")"
count=$((count + 1))
printf '%s\n' "$count" >"$KVAF_MOCK_COUNT_FILE"
if ((count == 1)); then
  [[ "$adaptive" == true && "$calibration_only" == true ]]
  [[ -z "$peer_device" && "$token_bytes" == 656 ]]
  [[ "$miss_count" == 2048 && "$request_length" == 2048 ]]
  echo '[KVAF_AIV] UB_DEVICE_SELECTED device_id=7 peer_candidate=2 ub_device=udmac1d1e2 npu_group=0 p50_kernel_cycles=100'
  exit 0
fi

[[ "$adaptive" == false && "$calibration_only" == false ]]
[[ "$peer_device" == udmac1d1e2 ]]
echo '[KVAF_AIV] UB_DEVICE_REUSED device_id=7 peer_candidate=2 ub_device=udmac1d1e2 npu_group=0'
for operation in batch_get_kernel_total batch_get_kernel_post batch_get_kernel_completion_wait; do
  echo "PERF_RESULT operation=$operation mean_ms=0.001 p50_ms=0.001 p90_ms=0.001 p99_ms=0.001 mean_cycles=1 p50_cycles=1 p90_cycles=1 p99_cycles=1"
done
EOF
chmod +x "$mock_runner"

export KVAF_BATCH_GET_PERF_RUNNER="$mock_runner"
export KVAF_MOCK_COUNT_FILE="$temp_dir/count"
export KVAF_BATCH_GET_MISS_SWEEP_LOG_DIR="$temp_dir/logs"
export KVAF_BATCH_GET_MISS_SWEEP_CSV="$temp_dir/summary.csv"
"$KVAF_ROOT/tests/run_remote_cpu_batch_get_miss_count_perf.sh" \
  "$temp_dir/fake-build" --adaptive-ub-device >"$temp_dir/output.log"

[[ "$(<"$KVAF_MOCK_COUNT_FILE")" == 64 ]]
grep -q '^BATCH_GET_UB_DEVICE_CALIBRATION_DONE selected_ub_device=udmac1d1e2 reuse=subsequent_shapes$' \
  "$temp_dir/output.log"
[[ "$(grep -c '^\[KVAF_AIV\] UB_DEVICE_SELECTED ' "$temp_dir/output.log")" == 1 ]]
[[ "$(grep -c '^\[KVAF_AIV\] UB_DEVICE_REUSED ' "$temp_dir/output.log")" == 63 ]]
awk -F, 'NR == 1 { if ($NF != "selected_ub_device") exit 1; next }
           $NF != "udmac1d1e2" { exit 1 }
           END { if (NR != 64) exit 1 }' "$temp_dir/summary.csv"
echo "batch get sweep UB device reuse test passed"
