#!/usr/bin/env bash
set -euo pipefail
readonly KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${KVAF_AICPU_NPU_BUILD_DIR:-$KVAF_ROOT/build-aicpu-npu}"
cann_root="${ASCEND_CANN_PACKAGE_PATH:-${ASCEND_HOME_PATH:-}}"
install_kernel=false
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      cat <<'EOF'
Usage: tests/run_aicpu_get_npu.sh [--install-kernel] [options]

Wrapper options:
  --build-dir DIR     Default: build-aicpu-npu (or KVAF_AICPU_NPU_BUILD_DIR)
  --install-kernel    Explicitly install the built kernel before starting
                     (requires write permission to CANN directories)
  --cann-root DIR     Installation root; defaults to CANN environment/cache

Pass-through options: --port 29991 --device-id 0 --eid HEX --kernel-json PATH
  --profile-scatter   Report AICPU CPU placement and scatter phase timing after the run
  --probe-aicpu-blocks N [--probe-hold-ms 20]
                      Standalone blockDim-to-core probe; requires --device-id and exits
  --probe-scatter-blocks N
                      Standalone local-copy diagnostics; requires --device-id; no CPU peer
    --probe-rows-per-worker 1024 --probe-row-bytes 656
    --probe-iterations 200 --probe-warmup 5 --probe-trace-rounds 3
    --probe-scatter-case all|pipeline|phased|serial_copy|copy_fenced|packed|packed_serial|
                         read|read_serial|write|write_serial|compute|compute_serial
    --probe-timeout-ms 1000   Per-barrier timeout, fails if blocks cannot run together
    --probe-pmu             Optional per-thread Linux hardware counters; reports unsupported events
    --probe-pmu-mask N      Select a smaller event group: 3=cycles/instructions, 11=+misses, 19=+stalls
  --aicpu-blocks N    Use N AICPU blocks for scatter; N must equal active scatter ranges
                     direct_output always launches one coordinator, even if N > 1
Without --device-id, use the CPU request's NPU device ID. With it, reject a mismatch.
The built kernel JSON is selected automatically. No installation by default.
Set --rows, --row-bytes, --scatter-threads and --transfer-mode on the CPU script.
direct_output skips --profile-scatter because there is no scatter phase.
This server handles one test run; restart it for the next run.
Build: bash build.sh aicpu-npu
EOF
      exit 0 ;;
    --build-dir|--cann-root)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo "$1 requires a value" >&2; exit 2; }
      if [[ "$1" == --build-dir ]]; then build_dir="$2"; else cann_root="$2"; fi
      shift 2 ;;
    --install-kernel) install_kernel=true; shift ;;
    *) args+=("$1"); shift ;;
  esac
done
artifact_dir="$build_dir/tests/aicpu_batch_get"
binary="$artifact_dir/kvaf_aicpu_get_npu"
[[ -x "$binary" ]] || {
  echo "Missing $binary; run: bash build.sh aicpu-npu --build-dir '$build_dir'" >&2
  exit 1
}
if "$install_kernel"; then
  if [[ -z "$cann_root" && -f "$build_dir/CMakeCache.txt" ]]; then
    cann_root="$(sed -n 's/^ASCEND_CANN_PACKAGE_PATH:[^=]*=//p' "$build_dir/CMakeCache.txt")"
  fi
  [[ -n "$cann_root" ]] || { echo "Installation needs --cann-root or CANN environment" >&2; exit 1; }
  bash "$KVAF_ROOT/tests/aicpu_batch_get/install_kernel.sh" "$artifact_dir" "$cann_root"
fi
exec "$binary" --kernel-json "$artifact_dir/libkvaf_aicpu_test.json" ${args[@]+"${args[@]}"}
