#!/usr/bin/env bash
set -euo pipefail
readonly KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${KVAF_AICPU_CPU_BUILD_DIR:-$KVAF_ROOT/build-aicpu-cpu}"
output=""
smoke=false
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      cat <<'EOF'
Usage: tests/run_aicpu_get_cpu.sh --peer NPU_HOST:29991 --device-id N [options]

Wrapper options:
  --build-dir DIR    Default: build-aicpu-cpu (or KVAF_AICPU_CPU_BUILD_DIR)
  --smoke            16 rows x 64 bytes, 1 warmup, 3 measured iterations
  --output FILE      Save CSV and also print it to stdout

Fixed path (V22): ordered WRITE chain, 4 workers/Jettys,
chunk 1024, doorbell 32, securec copy, indexed CQ batch 8/deadline 32,
final output readback. Defaults: 4096 x 656 B, warmup 50/iterations 1000.

  --rows N --row-bytes N
  --warmup N --iterations N --timeout-ms N
  --npu-candidate N  Pin the remote EID candidate; default automatic selection.

CSV includes all three bandwidth scopes:
  e2e_GBps       Request -> all scatter and producer done, excluding preparation/checks.
  cpu_data_GBps  First data Submit -> last observed chain tail, including markers.
  cpu_cycle_GBps Full CPU round, including request wait, source update and scheduled checks.
Times are in ns; effective GB/s = rows * row_bytes / corresponding mean_ns.
Communication time is cpu_data_span_ns, not first_ready_ns or CQ poll time.

Start the NPU script first. Restart it for every test invocation.
device-id selects the remote NPU (default 0); CPU UB/EID selection is automatic.
Optional diagnostic overrides: --ub-device NAME --eid-index N.
Build: bash build.sh aicpu-cpu
EOF
      exit 0 ;;
    --build-dir|--output)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "$1 requires a value" >&2; exit 2; }
      if [[ "$1" == --build-dir ]]; then build_dir="$2"; else output="$2"; fi
      shift 2 ;;
    --smoke) smoke=true; shift ;;
    *) args+=("$1"); shift ;;
  esac
done
binary="$build_dir/tests/aicpu_batch_get/kvaf_aicpu_get_cpu"
[[ -x "$binary" ]] || {
  echo "Missing $binary; run: bash build.sh aicpu-cpu --build-dir '$build_dir'" >&2
  exit 1
}
defaults=()
if "$smoke"; then defaults=(--rows 16 --row-bytes 64 --warmup 1 --iterations 3); fi
if [[ -n "$output" ]]; then
  mkdir -p "$(dirname "$output")"
  "$binary" ${defaults[@]+"${defaults[@]}"} ${args[@]+"${args[@]}"} | tee "$output"
else
  exec "$binary" ${defaults[@]+"${defaults[@]}"} ${args[@]+"${args[@]}"}
fi
