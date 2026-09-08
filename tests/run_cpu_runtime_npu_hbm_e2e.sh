#!/usr/bin/env bash
# Run the CPU Runtime -> NPU Service -> HBM integration test.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <npu-service-host:port> [--bytes N] [--slot-size N] [--ub-device NAME] [--eid-index N] [--device-id N] [--jettys N] [--omp-threads N] [--doorbell-wqes N]" >&2
  exit 2
fi

readonly KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly peer="$1"
shift
if [[ "${1:-}" == "--" ]]; then
  shift
fi
readonly build_dir="${KVAF_CPU_RUNTIME_BUILD_DIR:-$KVAF_ROOT/build-cpu}"

"$KVAF_ROOT/build.sh" cpu --build-dir "$build_dir"
exec "$build_dir/runtime/kvaf_cpu_runtime_npu_hbm_e2e" --peer "$peer" "$@"
