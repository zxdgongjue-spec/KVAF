#!/usr/bin/env bash
# Portable Pull regression runner. Real ACL/URMA/device kernels remain separate
# hardware acceptance; these tests use real control/Core code with fault injection.
set -euo pipefail
readonly KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly KVAF_CHECK_BUILD="${1:-$(mktemp -d "${TMPDIR:-/tmp}/kvaf-pull-host.XXXXXX")}"
readonly KVAF_CMAKE="${CMAKE:-cmake}"
readonly KVAF_CXX="${CXX:-c++}"
readonly KVAF_PYTHON="${PYTHON:-python3}"
cd "$KVAF_ROOT"
mkdir -p "$KVAF_CHECK_BUILD"
"$KVAF_CMAKE" -S . -B "$KVAF_CHECK_BUILD/core" -DBUILD_TESTING=OFF -DKVAF_BUILD_PYTHON=OFF \
  -DKVAF_BUILD_REMOTE_CPU_SERVICE=OFF -DKVAF_ENABLE_AIV_BATCH_GET=OFF -DKVAF_ENABLE_AIV_BATCH_PUT=OFF \
  -DKVAF_ENABLE_AICPU_URMA=OFF -DKVAF_ENABLE_AICPU_CPU_PUSH=OFF \
  -DKVAF_ENABLE_CPU_OFFLOAD=OFF -DKVAF_ENABLE_NPU_PULL=OFF
"$KVAF_CMAKE" --build "$KVAF_CHECK_BUILD/core" --parallel "${KVAF_BUILD_JOBS:-4}"
includes=(-Icommon/include -Iruntime/include -Iruntime/src -Imeta_manager/include -Iaccess_operator/include
  -Itests -Itests/offload_fake_acl -Itests/aiv_model
  -Iaccess_operator/algorithms/get/direct/urma_aiv/include -Iaccess_operator/algorithms/get/raw_addr/urma_aiv/include
  "-I$KVAF_CHECK_BUILD/core/runtime/generated")
libraries=("$KVAF_CHECK_BUILD/core/runtime/libkvaf_runtime.a"
  "$KVAF_CHECK_BUILD/core/meta_manager/libkvaf_meta_manager.a"
  "$KVAF_CHECK_BUILD/core/runtime/libkvaf_offload_control.a"
  "$KVAF_CHECK_BUILD/core/runtime/libkvaf_peer_link.a")
system_libraries=()
if [[ "$(uname -s)" != Darwin ]]; then system_libraries+=(-ldl); fi
compile() { local name="$1"; shift; "$KVAF_CXX" -std=c++17 -O1 -pthread "${includes[@]}" "$@" "${libraries[@]}" ${system_libraries[@]+"${system_libraries[@]}"} -o "$KVAF_CHECK_BUILD/$name"; }
run() {
  "$KVAF_PYTHON" - "$@" <<'PY'
import subprocess, sys
print("[PULL_HOST_CHECK] " + " ".join(sys.argv[1:]), flush=True)
subprocess.run(sys.argv[1:], check=True, timeout=60)
PY
}
compile get_recovery tests/diagnostics/get_recovery.cc
compile profile tests/cpu_pull_profile_test.cc
compile eid tests/npu_eid_discovery_test.cc
compile wire tests/hbm_read_control_test.cc
compile region tests/diagnostics/cpu_pull_region_lease.cc runtime/src/remote_pull_service.cc
compile recovery tests/diagnostics/cpu_control_recovery_capacity.cc
compile transaction tests/diagnostics/cpu_pull_write_transaction.cc
compile raw_model tests/batch_get_raw_addr_kernel_model_test.cc
compile raw_transport tests/hbm_read_transport_test.cc \
  access_operator/transports/direct_hccp/direct_hccp_remote_transport.cc \
  access_operator/transports/direct_hccp/direct_hccp_session.cc
compile raw_runtime -DKVAF_ENABLE_NPU_PULL=1 tests/hbm_read_test.cc runtime/src/npu_runtime.cc \
  runtime/src/hbm_read_source.cc runtime/src/hbm_read_client.cc
compile remote -DKVAF_ENABLE_CPU_OFFLOAD=1 tests/remote_offload_test.cc runtime/src/npu_runtime.cc \
  runtime/src/remote_offload.cc runtime/src/remote_pull_operator.cc runtime/src/remote_pull_service.cc
for check in get_recovery profile eid wire region recovery transaction raw_model raw_transport raw_runtime remote; do
  run "$KVAF_CHECK_BUILD/$check"
done
run "$KVAF_CHECK_BUILD/raw_runtime" launch-failure
run "$KVAF_CHECK_BUILD/raw_runtime" sync-failure
if [[ -n "${KVAF_TEST_URMA_INCLUDE_DIR:-}" ]]; then
  compile urma -I"$KVAF_TEST_URMA_INCLUDE_DIR" -Wl,-export_dynamic \
    tests/cpu_pull_urma_test.cc runtime/src/cpu_pull_urma.cc
  run "$KVAF_CHECK_BUILD/urma"
else
  echo '[PULL_HOST_CHECK] SKIP fake URMA provider: set KVAF_TEST_URMA_INCLUDE_DIR to UMDK 24.12 headers'
fi
export PYTHONPATH="$KVAF_ROOT/tests/python:$KVAF_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
run "$KVAF_PYTHON" -m unittest test_pull_benchmarks test_tcp_control
# Optional binding metadata checks: no hardware or native KVAF module required.
pybind_include="${PYBIND11_INCLUDE:-}"
if [[ -z "$pybind_include" ]]; then
  pybind_include="$("$KVAF_PYTHON" -c 'import pybind11; print(pybind11.get_include())' 2>/dev/null || true)"
fi
if [[ -n "$pybind_include" ]]; then
  python_include="$("$KVAF_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
  extension_suffix="$("$KVAF_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
  module_flags=(-shared -fPIC)
  if [[ "$(uname -s)" == Darwin ]]; then module_flags+=(-undefined dynamic_lookup); fi
  "$KVAF_CXX" -std=c++17 "${module_flags[@]}" -Ibindings/include -Itests/offload_fake_acl -Icommon/include \
    -I"$pybind_include" -I"$python_include" tests/npu_binding_metadata_module.cc bindings/src/acl_python_utils.cc \
    -o "$KVAF_CHECK_BUILD/_kvaf_binding_metadata_test$extension_suffix"
  PYTHONPATH="$KVAF_CHECK_BUILD:$PYTHONPATH" run "$KVAF_PYTHON" -m unittest test_npu_binding_metadata
else
  echo '[PULL_HOST_CHECK] SKIP binding metadata: install pybind11 or set PYBIND11_INCLUDE'
fi
echo "[PULL_HOST_CHECK] PASS (executed checks only); artifacts: $KVAF_CHECK_BUILD"
