#!/usr/bin/env bash
# Build the NPU-side push target inside the CANN container.
# Usage: build_device.sh /path/to/kvaf /path/to/out_dir
set -euo pipefail
KVAF_ROOT="$1"
OUT="$2"
mkdir -p "$OUT"

ASCEND_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"

g++ -std=c++17 -O2 -Wall -ffunction-sections -fdata-sections \
  "$KVAF_ROOT/access_operator/demo/cpu_sgl_push/device/npu_push_target.cpp" \
  "$KVAF_ROOT/access_operator/src/direct_hccp/direct_hccp_session.cc" \
  "$KVAF_ROOT/access_operator/src/direct_hccp/direct_hccp_remote_transport.cc" \
  -I"$KVAF_ROOT/access_operator/include" \
  -I"$KVAF_ROOT/runtime/include" \
  -I"$ASCEND_HOME/include" \
  -L"$ASCEND_HOME/lib64" -lascendcl -ldl -lpthread \
  -Wl,--gc-sections \
  -o "$OUT/npu_push_target"

echo "built $OUT/npu_push_target"
