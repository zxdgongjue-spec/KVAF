#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "Usage: bash $0 BUILD/access_operator/src/aicpu_urma CANN_ROOT" >&2
  exit 2
fi
artifact_dir=$1
cann_root=$2
package=kvaf-aicpu-urma.tar.gz
json=kernel.json
kernel_dir="$cann_root/opp/vendors/cust/op_impl/aicpu/kernel"
config_dir="$cann_root/opp/vendors/cust/op_impl/aicpu/config"
load_config="$cann_root/conf/ascend_package_load.ini"

[[ -f "$artifact_dir/$package" && -f "$artifact_dir/$json" && -d "$cann_root/conf" ]]
tar tzf "$artifact_dir/$package" > /dev/null
mkdir -p "$kernel_dir" "$config_dir"
install -m 644 "$artifact_dir/$package" "$kernel_dir/$package"
install -m 644 "$artifact_dir/$json" "$config_dir/kvaf_aicpu_urma.json"
if [[ -f "$load_config" && ! -f "$load_config.kvaf-aicpu-urma.bak" ]]; then
  cp -p "$load_config" "$load_config.kvaf-aicpu-urma.bak"
fi
if ! grep -Fqx "name:$package" "$load_config" 2>/dev/null; then
  printf '\nname:%s\ninstall_path:2\noptional:true\npackage_path:opp/vendors/cust/op_impl/aicpu/kernel\nload_as_per_soc:false\n' \
    "$package" >> "$load_config"
fi
echo "Installed $package; restart the NPU process before loading $config_dir/kvaf_aicpu_urma.json"
