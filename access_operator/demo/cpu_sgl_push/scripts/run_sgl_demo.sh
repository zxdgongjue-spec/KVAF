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
DOCKER_NAME="${7:-}"

# The NPU target needs the CANN runtime + device nodes: run it inside the
# container when DOCKER_NAME is given (paths are shared via /home/fjj).
run_target() {
  if [[ -n "$DOCKER_NAME" ]]; then
    docker exec "$DOCKER_NAME" env SGL_DEMO_TOKEN_BYTES=656 /home/fjj/sgl-out/npu_push_target "$@"
  else
    SGL_DEMO_TOKEN_BYTES=656 "$OUT/npu_push_target" "$@"
  fi
}

WORK="$OUT/run_dev${DEV}_m${MISS}_g${GROUP}"
mkdir -p "$WORK"
KEYFILE="$WORK/keyfile.txt"
CPU_CSV="$WORK/cpu.csv"
NPU_CSV="$WORK/npu.csv"
rm -f "$KEYFILE" "$CPU_CSV" "$NPU_CSV"

# 1) NPU EID candidates from /etc/hccl_rootinfo.json: ALL EID addrs listed for
# this device_id (any net_layer/net_type), deduped. ctx_init rejects EIDs that
# do not belong to the device-side RA, so the launcher tries them in order
# (same candidate-fallback design as KVAF's npu_runtime EID discovery).
mapfile -t NPU_EIDS < <(python3 - "$DEV" <<'PY'
import json, re, sys
dev = int(sys.argv[1])
try:
    root = json.load(open("/etc/hccl_rootinfo.json"))
except Exception:
    sys.exit(1)
seen = set()
out = []
for rank in root.get("rank_list", []):
    if rank.get("device_id") != dev:
        continue
    for level in rank.get("level_list", []):
        for addr in level.get("rank_addr_list", []):
            if addr.get("addr_type") != "EID":
                continue
            hexed = re.sub(r"[^0-9a-fA-F]", "", str(addr.get("addr", ""))).lower()
            if len(hexed) == 32 and hexed != "0" * 32 and hexed not in seen:
                seen.add(hexed)
                out.append(hexed)
# CLOS layer-1 first, then the rest.
def clos_first(h):
    return 0 if h.startswith(("000000000041", "000000000047")) else 1
print("\n".join(out))
PY
)
if [[ ${#NPU_EIDS[@]} -eq 0 ]]; then
  echo "no NPU EID found in rootinfo for device $DEV" >&2
  exit 1
fi
echo "npu_eid_candidates: ${#NPU_EIDS[@]}"

# 2) CPU EID probe (the probe prints to stderr).
CPU_EID="$("$OUT/cpu_sgl_push" --print-eid 2>&1 | grep -oE '[0-9a-f]{32}' | head -1)"
if [[ ${#CPU_EID} -ne 32 ]]; then
  echo "cannot get CPU EID: '$CPU_EID'" >&2
  exit 1
fi
echo "cpu_eid=$CPU_EID"

# 3) Start the NPU target; wait for its keyfile. Try each EID candidate.
NPU_PID=""
NPU_EID=""
for cand in "${NPU_EIDS[@]}"; do
  echo "trying npu_eid=$cand"
  run_target \
    --device-id "$DEV" --local-eid "$cand" --peer-eid "$CPU_EID" \
    --keyfile "$KEYFILE" --csv "$NPU_CSV" >"$WORK/npu.log" 2>&1 &
  NPU_PID=$!
  ok=0
  for _ in $(seq 1 30); do
    [[ -s "$KEYFILE" ]] && { ok=1; break; }
    kill -0 "$NPU_PID" 2>/dev/null || break
    sleep 1
  done
  if [[ $ok -eq 1 ]]; then
    NPU_EID="$cand"
    break
  fi
  echo "candidate $cand failed; log:"; tail -5 "$WORK/npu.log" || true
  wait "$NPU_PID" 2>/dev/null || true
  NPU_PID=""
  rm -f "$KEYFILE"
done
[[ -n "$NPU_PID" ]] || { echo "no NPU EID candidate worked"; exit 1; }
echo "npu_eid=$NPU_EID"

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
