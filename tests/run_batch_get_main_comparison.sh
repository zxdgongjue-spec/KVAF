#!/usr/bin/env bash
# Run main and unified Get benchmarks with the same eager device-wait timer.
# Each branch builds and installs its own artifacts and owns one CPU service.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "存在未提交的代码修改，请先保存。" >&2
  exit 1
fi
export PYTHON="${PYTHON:-python3}"
export DEVICE_ID=1 KVAF_CPU_SERVICE_ADDR=127.0.0.1:21111
export KVAF_PERF_REUSE_GET_OUTPUT=1
unset KVAF_GET_DISPATCH_TRACE KVAF_PUT_DISPATCH_TRACE

"$PYTHON" - <<'PY'
import socket
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(('0.0.0.0', 21111))
    except OSError as error:
        raise SystemExit(f'21111 端口不可用，请先停止旧测试服务：{error}')
PY

original_ref="$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)"
git fetch --no-tags origin \
  +refs/heads/main:refs/remotes/origin/main \
  +refs/heads/codex/cache-transport-execution:refs/remotes/origin/codex/cache-transport-execution
report_root="reports/main-vs-unified-async-wait-d1-p21111-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$report_root"
cpu_pid=
stop_cpu() {
  if [[ -n "$cpu_pid" ]]; then
    if kill -0 "$cpu_pid" 2>/dev/null; then kill "$cpu_pid"; fi
    wait "$cpu_pid" 2>/dev/null || true
    cpu_pid=
  fi
}
trap stop_cpu EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for label in main execution; do
  ref=origin/main
  if [[ "$label" == execution ]]; then ref=origin/codex/cache-transport-execution; fi
  git switch --detach "$ref"
  log_dir="$report_root/$label"
  cpu_build="build-${label}-cpu"
  npu_build="build-${label}-npu"
  mkdir -p "$log_dir"
  git log -1 --format='%H %s' | tee "$log_dir/commit.txt"
  bash build.sh cpu --build-dir "$cpu_build" --jobs 16 2>&1 | tee "$log_dir/build-cpu.log"
  bash build.sh npu --build-dir "$npu_build" --jobs 16 2>&1 | tee "$log_dir/build-npu.log"
  bash build.sh npu --build-dir "$npu_build" --check-kernel 2>&1 | tee "$log_dir/check-kernel.log"

  "./$cpu_build/runtime/kvaf_cpu_service" \
    --listen 0.0.0.0 --port 21111 --gather-threads 8 --gather-mode active \
    >"$log_dir/cpu-service.log" 2>&1 &
  cpu_pid=$!
  for ((attempt=0; attempt<100; ++attempt)); do
    if ! kill -0 "$cpu_pid" 2>/dev/null; then
      cat "$log_dir/cpu-service.log"
      exit 1
    fi
    if grep -q 'LISTEN.*port=21111' "$log_dir/cpu-service.log"; then break; fi
    sleep 0.2
  done
  if ! grep -q 'LISTEN.*port=21111' "$log_dir/cpu-service.log"; then
    cat "$log_dir/cpu-service.log"
    exit 1
  fi

  extra=()
  if [[ "$label" == execution ]]; then extra=(--get-timing async-wait); fi
  for misses in 2048 4096 8192; do
    for spec in aiv:2 ubmem:16 gather_read:5 cpu_push_scatter:5; do
      backend="${spec%:*}"
      cores="${spec#*:}"
      bash tests/run_remote_cpu_batch_get_perf.sh "$npu_build" \
        ${extra[@]+"${extra[@]}"} --backend "$backend" --execution both \
        --graph-api torch --graph-replays 5 \
        --request-length 32768 --table-capacity 4096 --miss-count "$misses" \
        --token-bytes 656 --tokens-per-block 128 \
        --aiv-cores "$cores" --jettys-per-core 2 --max-inflight-total 2048 \
        --kernel-profile-clock-mhz 1000 --seed 20260903 --no-sort-input-indices \
        --source-access-pattern random --output-access-pattern contiguous \
        --warmup 10 --iterations 100 --kernel-profile \
        2>&1 | tee "$log_dir/get-${backend}-${misses}.log"
    done
  done
  grep -hE '^BATCH_GET_RESULT |^BATCH_GET_GRAPH_PASS |^PERF_RESULT ' \
    "$log_dir"/get-*.log >"$log_dir/summary.txt"
  stop_cpu
done
git switch "$original_ref"
printf '\nmain 结果：%s/main/summary.txt\nexecution 结果：%s/execution/summary.txt\n' \
  "$report_root" "$report_root"
