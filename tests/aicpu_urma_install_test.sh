#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
installer="$repo_root/access_operator/src/aicpu_urma/install_kernel.sh"
temporary="$(mktemp -d)"
trap 'rm -rf "$temporary"' EXIT
root="$temporary/cann"
artifacts="$temporary/artifacts"
staging="$temporary/staging"
ini="$root/conf/ascend_package_load.ini"
mkdir -p "$root/conf" "$root/include/acl" "$artifacts" "$staging/aicpu_kernels_device"
touch "$root/include/acl/acl.h"
printf '%s\n' 'name:another-package.tar.gz' 'install_path:2' 'optional:true' \
  'package_path:opp/built-in/op_impl/aicpu/kernel' 'load_as_per_soc:false' > "$ini"
printf 'test kernel\n' > "$staging/aicpu_kernels_device/libkvaf_aicpu_urma.so"
tar -czf "$artifacts/kvaf-aicpu-urma.tar.gz" -C "$staging" aicpu_kernels_device
cp "$repo_root/access_operator/src/aicpu_urma/kernel.json" "$artifacts/kernel.json"

bash "$installer" --install "$artifacts" "$root"
bash "$installer" --check "$root" "$artifacts"
ASCEND_CANN_PACKAGE_PATH="$root" ASCEND_HOME_PATH="$root" \
  bash "$repo_root/build.sh" aicpu-urma --check-kernel
( unset ASCEND_HOME_PATH; ASCEND_CANN_PACKAGE_PATH="$root" \
  bash "$repo_root/build.sh" aicpu-urma --check-kernel )
bash "$installer" --install "$artifacts" "$root"
[[ "$(grep -Fxc 'name:kvaf-aicpu-urma.tar.gz' "$ini")" == 1 ]]
grep -Fxq 'name:another-package.tar.gz' "$ini"

cp "$ini.kvaf-aicpu-urma.bak" "$ini"
if ASCEND_CANN_PACKAGE_PATH="$root" ASCEND_HOME_PATH="$root" \
  bash "$repo_root/build.sh" aicpu-urma --check-kernel >/dev/null 2>&1; then
  echo 'missing package-load entry was accepted' >&2
  exit 1
fi
bash "$installer" --install "$artifacts" "$root"
ASCEND_CANN_PACKAGE_PATH="$root" ASCEND_HOME_PATH="$root" \
  bash "$repo_root/build.sh" aicpu-urma --uninstall-kernel
cp "$ini" "$temporary/after-uninstall.ini"
bash "$installer" --uninstall "$root"
cmp -s "$ini" "$temporary/after-uninstall.ini"
grep -Fxq 'name:another-package.tar.gz' "$ini"
if grep -Fxq 'name:kvaf-aicpu-urma.tar.gz' "$ini"; then
  echo 'package-load entry survived uninstall' >&2
  exit 1
fi
[[ ! -e "$root/opp/vendors/cust/op_impl/aicpu/kernel/kvaf-aicpu-urma.tar.gz" ]]
[[ ! -e "$root/opp/vendors/cust/op_impl/aicpu/config/kvaf_aicpu_urma.json" ]]
mkdir -p "$temporary/other-cann"
if ASCEND_CANN_PACKAGE_PATH="$root" ASCEND_HOME_PATH="$temporary/other-cann" \
  bash "$repo_root/build.sh" aicpu-urma --check-kernel >/dev/null 2>&1; then
  echo 'mismatched build and runtime CANN roots were accepted' >&2
  exit 1
fi
echo 'AICPU URMA installer: PASS'
