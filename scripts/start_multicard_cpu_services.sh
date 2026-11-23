#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cpu_service="$root/build-cpu/runtime/kvaf_cpu_service"
driver_lib=/usr/local/Ascend/driver/lib64
hal_library="$driver_lib/driver/libascend_hal.so"
listen_address=127.0.0.1
startup_timeout=10

if [[ ! -x "$cpu_service" ]]; then
  printf 'FAIL CPU service binary is missing or not executable: %s\n' "$cpu_service" >&2
  exit 1
fi
if [[ ! -r "$hal_library" ]]; then
  printf 'FAIL HAL library is not readable: %s\n' "$hal_library" >&2
  exit 1
fi

if ! kvaf_syslib="$(env -u LD_LIBRARY_PATH -u LD_PRELOAD -u LD_AUDIT ldd /bin/true |
  awk '$1 == "libc.so.6" && $2 == "=>" && $3 ~ /^\// {sub(/\/[^/]+$/, "", $3); print $3}')" ||
  [[ ! -d "$kvaf_syslib" ]]; then
  printf 'FAIL could not locate the system libc directory\n' >&2
  exit 1
fi
library_path="$kvaf_syslib:$driver_lib/common:$driver_lib/driver"
log_dir="$(mktemp -d "${TMPDIR:-/tmp}/kvaf-cpu-services.XXXXXX")"
printf 'CPU_SERVICE_LOG_DIR %s\n' "$log_dir"

start_service() {
  local npu="$1" port="$2" ub_device="$3"
  local log="$log_dir/npu_${npu}.log"
  local pid status=0 deadline

  nohup numactl --cpunodebind=3 --membind=3 \
    env -u LD_PRELOAD -u LD_AUDIT LD_LIBRARY_PATH="$library_path" \
    "$cpu_service" --listen "$listen_address" --port "$port" --memory-page-mode hugetlb-1g \
    --hal-library "$hal_library" --ub-device "$ub_device" --eid-index 0 \
    </dev/null >"$log" 2>&1 &
  pid=$!
  deadline=$((SECONDS + startup_timeout))

  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || status=$?
      printf 'FAIL npu=%s peer=%s:%s ub_device=%s pid=%s exit=%s log=%s\n' \
        "$npu" "$listen_address" "$port" "$ub_device" "$pid" "$status" "$log" >&2
      tail -n 12 "$log" >&2
      return 1
    fi
    if grep -Fxsq "[KVAF_CPU] LISTEN address=$listen_address port=$port" "$log"; then
      printf 'OK npu=%s peer=%s:%s ub_device=%s pid=%s log=%s\n' \
        "$npu" "$listen_address" "$port" "$ub_device" "$pid" "$log"
      return 0
    fi
    sleep 0.1
  done

  kill "$pid" 2>/dev/null || true
  printf 'FAIL npu=%s peer=%s:%s ub_device=%s pid=%s timeout=%ss log=%s\n' \
    "$npu" "$listen_address" "$port" "$ub_device" "$pid" "$startup_timeout" "$log" >&2
  tail -n 12 "$log" >&2
  return 1
}

failed=0
for route in 0:28990:udmac1d1e2 1:28991:udmac1d1e3 6:28996:udmac1d1e4 7:28997:udmac1d1e5; do
  IFS=: read -r npu port ub_device <<< "$route"
  if ! start_service "$npu" "$port" "$ub_device"; then
    failed=1
  fi
done
exit "$failed"
