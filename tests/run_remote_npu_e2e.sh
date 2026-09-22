#!/usr/bin/env bash
# NPU0 client; start kvaf_npu_service on NPU1 first (see docs/npu_peer.md).
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: KVAF_NPU_SERVICE_ADDR=host:port DEVICE_ID=0 bash $0 /absolute/path/build-npu" >&2
  exit 2
fi
KVAF_PEER_BUILD_DIR="$1"
KVAF_PEER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${KVAF_NPU_SERVICE_ADDR:?set the address of the NPU1 HBM service}"
export KVAF_PEER_DEVICE=npu
export KVAF_PEER_SERVICE_ADDR="$KVAF_NPU_SERVICE_ADDR"
export KVAF_RUN_NPU_TESTS=1
export DEVICE_ID="${DEVICE_ID:-0}"
export KVAF_BATCH_GET_KERNEL="$KVAF_PEER_BUILD_DIR/bin/kvaf_execute_batch_get_kernels.o"
export KVAF_BATCH_PUT_KERNEL="$KVAF_PEER_BUILD_DIR/bin/kvaf_execute_batch_put_kernels.o"
export KVAF_NPU_GRAPH_REPLAYS="${KVAF_NPU_GRAPH_REPLAYS:-3}"
export PYTHONPATH="$KVAF_PEER_BUILD_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
[[ -s "$KVAF_BATCH_GET_KERNEL" && -s "$KVAF_BATCH_PUT_KERNEL" && -d "$KVAF_PEER_BUILD_DIR/python/kvaf" ]] || {
  echo "Missing AIV kernels or Python binding under $KVAF_PEER_BUILD_DIR" >&2; exit 1;
}
# An explicitly requested hardware run must fail if its prerequisites are
# missing, rather than returning success through the generic test's SKIP path.
"${PYTHON:-python3}" -c 'import kvaf, torch, torch_npu; assert hasattr(kvaf, "NpuRuntime"), "NpuRuntime binding missing"; assert torch.npu.is_available(), "NPU unavailable"'
exec "${PYTHON:-python3}" "$KVAF_PEER_ROOT/tests/python/test_npu_batch_get.py"
