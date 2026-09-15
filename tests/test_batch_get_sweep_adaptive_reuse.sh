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
peer=""
eids=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --adaptive-ub-device) adaptive=true; shift ;;
    --peer-candidate-id) peer="$2"; shift 2 ;;
    --npu-eids) eids="$2"; shift 2 ;;
    *) shift ;;
  esac
done
count=0
[[ ! -f "$KVAF_MOCK_COUNT_FILE" ]] || count="$(<"$KVAF_MOCK_COUNT_FILE")"
count=$((count + 1))
printf '%s\n' "$count" >"$KVAF_MOCK_COUNT_FILE"
if ((count == 1)); then
  [[ "$adaptive" == true && -z "$peer" ]]
  echo '[KVAF_AIV] UB_DEVICE_SELECTED device_id=7 peer_candidate=2 ub_device=udmac1d1e2 peer_eid_index=3 npu_group=0 npu_eids=00112233445566778899aabbccddeeff p50_kernel_cycles=100'
else
  [[ "$adaptive" == false ]]
  [[ "$peer" == 2 ]]
  [[ "$eids" == 00112233445566778899aabbccddeeff ]]
fi
echo '[KVAF_AIV] UB_DEVICE_ACTIVE device_id=7 peer_candidate=2 ub_device=udmac1d1e2 peer_eid_index=3 npu_eids=00112233445566778899aabbccddeeff selection=adaptive'
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

[[ "$(<"$KVAF_MOCK_COUNT_FILE")" == 63 ]]
grep -q '^BATCH_GET_SWEEP_UB_DEVICE_SELECTED peer_candidate=2 .*reuse=subsequent_shapes$' \
  "$temp_dir/output.log"
[[ "$(grep -c '^\[KVAF_AIV\] UB_DEVICE_SELECTED ' "$temp_dir/output.log")" == 1 ]]
head -n 1 "$temp_dir/summary.csv" | grep -q '^ub_device,peer_candidate,peer_eid_index,npu_eids,'
tail -n +2 "$temp_dir/summary.csv" | grep -q '^"udmac1d1e2",2,3,"00112233445566778899aabbccddeeff",'
echo "batch get sweep adaptive reuse test passed"
