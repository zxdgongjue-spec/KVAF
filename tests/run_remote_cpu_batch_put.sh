#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then echo "Usage: $0 /absolute/path/to/kvaf/build-npu" >&2; exit 2; fi
build_dir="$1"; root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
get_kernel="$build_dir/bin/kvaf_execute_batch_get_kernels.o"; put_kernel="$build_dir/bin/kvaf_execute_batch_put_kernels.o"
[[ -s "$get_kernel" && -s "$put_kernel" && -d "$build_dir/python/kvaf" ]] || { echo "ERROR: missing NPU build artifacts" >&2; exit 1; }
: "${KVAF_CPU_SERVICE_ADDR:=127.0.0.1:28999}"; : "${DEVICE_ID:=0}"; : "${KVAF_REMOTE_BATCH_PUT_TIMEOUT:=600}"
export KVAF_CPU_SERVICE_ADDR DEVICE_ID KVAF_RUN_NPU_TESTS=1 KVAF_BATCH_GET_KERNEL="$get_kernel" KVAF_BATCH_PUT_KERNEL="$put_kernel" PYTHONPATH="$build_dir/python${PYTHONPATH:+:$PYTHONPATH}"
exec timeout "${KVAF_REMOTE_BATCH_PUT_TIMEOUT}s" python3 "$root_dir/tests/python/test_npu_batch_put.py"
