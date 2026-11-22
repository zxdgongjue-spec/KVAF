#!/usr/bin/env bash
# End-to-end SGL push demo on one host.
# Usage: run_sgl_demo.sh OUT_DIR DEVICE_ID MISS GROUP ITERS [WARMUP]
# OUT_DIR must contain npu_push_target and cpu_sgl_push (see build_*.sh).
set -euo pipefail
OUT="$1"
DEV="$2"
MISS="$3"
GROUP="$4"
ITERS="$5"
WARMUP="${6:-20}"

WORK="$OUT/run_dev${DEV}_m${MISS}_g${GROUP}"
mkdir -p "$WORK"
KEYFILE="$WORK/keyfile.txt"
CPU_CSV="$WORK/cpu.csv"
NPU_CSV="$WORK/npu.csv"
rm -f "$KEYFILE" "$CPU_CSV" "$NPU_CSV"

# 1) NPU EID from hccn_tool.
NPU_EID="$(hccn_tool -i "$DEV" -eid -g 2>/dev/null | grep -ioE '([0-9a-f]{2}[: ]){15}[0-9a-f]{2}|[0-9a-f]{32}' | head -1 | tr -d ': ' | tr 'A-F' 'a-f')"
if [[ ${#NPU_EID} -ne 32 ]]; then
  echo "cannot parse NPU EID for device $DEV: '$NPU_EID'" >&2
  exit 1
fi
echo "npu_eid=$NPU_EID"

# 2) CPU EID probe.
CPU_EID="$("$OUT/cpu_sgl_push" --print-eid 2>/dev/null | grep -oE '[0-9a-f]{32}' | head -1)"
if [[ ${#CPU_EID} -ne 32 ]]; then
  echo "cannot get CPU EID: '$CPU_EID'" >&2
  exit 1
fi
echo "cpu_eid=$CPU_EID"

# 3) Start the NPU target; wait for its keyfile.
SGL_DEMO_TOKEN_BYTES=656 "$OUT/npu_push_target" \
  --device-id "$DEV" --local-eid "$NPU_EID" --peer-eid "$CPU_EID" \
  --keyfile "$KEYFILE" --csv "$NPU_CSV" >"$WORK/npu.log" 2>&1 &
NPU_PID=$!
for _ in $(seq 1 120); do
  [[ -s "$KEYFILE" ]] && break
  kill -0 "$NPU_PID" 2>/dev/null || { echo "npu target died:"; cat "$WORK/npu.log"; exit 1; }
  sleep 1
done
[[ -s "$KEYFILE" ]] || { echo "keyfile timeout"; cat "$WORK/npu.log"; exit 1; }

# 4) Run the CPU pusher.
"$OUT/cpu_sgl_push" \
  --keyfile "$KEYFILE" --npu-eid "$NPU_EID" \
  --miss "$MISS" --group "$GROUP" --iters "$ITERS" --warmup "$WARMUP" \
  --csv "$CPU_CSV" >"$WORK/cpu.log" 2>&1 || { echo "cpu pusher failed:"; cat "$WORK/cpu.log"; kill "$NPU_PID" 2>/dev/null; exit 1; }

# 5) NPU verifies and exits.
wait "$NPU_PID" || true
echo "--- cpu.log tail ---"; tail -3 "$WORK/cpu.log"
echo "--- npu.log tail ---"; tail -5 "$WORK/npu.log"
echo "csv: $CPU_CSV $NPU_CSV"
