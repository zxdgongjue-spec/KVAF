#!/usr/bin/env bash
# Explicit, manual installation on the NPU test host. Never run by CMake.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: bash $0 BUILD/tests/aicpu_batch_get CANN_ROOT" >&2
  exit 2
fi
artifact_dir=$1
cann_root=$2
test_package=kvaf-aicpu-test.tar.gz
test_json=libkvaf_aicpu_test.json
test_kernel_dir="$cann_root/opp/vendors/cust/op_impl/aicpu/kernel"
test_config_dir="$cann_root/opp/vendors/cust/op_impl/aicpu/config"
test_load_config="$cann_root/conf/ascend_package_load.ini"
[[ -f "$artifact_dir/$test_package" && -f "$artifact_dir/$test_json" ]]
[[ -d "$cann_root/conf" ]]
tar tzf "$artifact_dir/$test_package" > /dev/null
mkdir -p "$test_kernel_dir" "$test_config_dir"
install -m 644 "$artifact_dir/$test_package" "$test_kernel_dir/$test_package"
install -m 644 "$artifact_dir/$test_json" "$test_config_dir/$test_json"
if [[ -f "$test_load_config" && ! -f "$test_load_config.kvaf-test.bak" ]]; then
  cp -p "$test_load_config" "$test_load_config.kvaf-test.bak"
fi
if ! grep -Fqx "name:$test_package" "$test_load_config" 2>/dev/null; then
  cat >> "$test_load_config" <<'EOF'

name:kvaf-aicpu-test.tar.gz
install_path:2
optional:true
package_path:opp/vendors/cust/op_impl/aicpu/kernel
load_as_per_soc:false
EOF
fi
echo "Installed $test_package. Restart the NPU test process before loading the kernel."
