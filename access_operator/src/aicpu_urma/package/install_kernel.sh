#!/usr/bin/env bash
set -euo pipefail

package=kvaf-aicpu-urma.tar.gz
json=kvaf_aicpu_urma.json
kernel_so=aicpu_kernels_device/libkvaf_aicpu_urma.so
header="name:$package"
expected_block="$(printf '%s\n' "$header" 'install_path:2' 'optional:true' \
  'package_path:opp/vendors/cust/op_impl/aicpu/kernel' 'load_as_per_soc:false')"

usage() {
  echo "Usage: bash $0 --install ARTIFACT_DIR CANN_ROOT | --check CANN_ROOT [ARTIFACT_DIR] | --uninstall CANN_ROOT" >&2
}

die() {
  echo "[KVAF_AICPU_URMA] $*" >&2
  exit 1
}

check_archive() {
  local members
  members="$(tar -tzf "$1")" || die "invalid AICPU package: $1"
  grep -Fxq "$kernel_so" <<< "$members" || die "package lacks $kernel_so: $1"
}

check_artifacts() {
  local dir="$1"
  [[ -s "$dir/$package" && -s "$dir/kernel.json" ]] || die "missing build artifacts in $dir"
  check_archive "$dir/$package"
  grep -Fq '"KvafAicpuBatchPut"' "$dir/kernel.json" || die "BatchPut missing from $dir/kernel.json"
  grep -Fq '"KvafAicpuGatherRead"' "$dir/kernel.json" || die "GatherRead missing from $dir/kernel.json"
  grep -Fq '"KvafAicpuUmdkControl"' "$dir/kernel.json" || die "UMDK control missing from $dir/kernel.json"
}

check_installed() {
  local artifact_dir="${1:-}"
  local count block
  [[ -s "$kernel_dir/$package" && -s "$config_dir/$json" ]] || die "package or JSON missing under $cann_root"
  check_archive "$kernel_dir/$package"
  grep -Fq '"KvafAicpuBatchPut"' "$config_dir/$json" || die "installed JSON lacks BatchPut"
  grep -Fq '"KvafAicpuGatherRead"' "$config_dir/$json" || die "installed JSON lacks GatherRead"
  grep -Fq '"KvafAicpuUmdkControl"' "$config_dir/$json" || die "installed JSON lacks UMDK control"
  count="$(grep -Fxc "$header" "$load_config" || true)"
  [[ "$count" == 1 ]] || die "expected one $header entry in $load_config, found $count"
  block="$(grep -A4 -Fx "$header" "$load_config")"
  [[ "$block" == "$expected_block" ]] || die "invalid $header entry in $load_config"
  if [[ -n "$artifact_dir" ]]; then
    check_artifacts "$artifact_dir"
    cmp -s "$artifact_dir/$package" "$kernel_dir/$package" || die "installed package differs from build artifact"
    cmp -s "$artifact_dir/kernel.json" "$config_dir/$json" || die "installed JSON differs from build artifact"
  fi
  echo "[KVAF_AICPU_URMA] CHECK_OK root=$cann_root package=$kernel_dir/$package json=$config_dir/$json"
}

rewrite_load_config() {
  local add_block="$1"
  local temporary
  temporary="$(mktemp "${load_config}.kvaf.XXXXXX")"
  cp -p "$load_config" "$temporary"
  if ! awk -v target="$header" '
    BEGIN {
      expected[1] = "install_path:2"
      expected[2] = "optional:true"
      expected[3] = "package_path:opp/vendors/cust/op_impl/aicpu/kernel"
      expected[4] = "load_as_per_soc:false"
    }
    remaining > 0 {
      if ($0 != expected[5 - remaining]) exit 2
      --remaining
      next
    }
    $0 == target { remaining = 4; next }
    { print }
    END { if (remaining > 0) exit 2 }
  ' "$load_config" > "$temporary"; then
    rm -f "$temporary"
    die "malformed $header block in $load_config; left configuration unchanged"
  fi
  if [[ "$add_block" == true ]]; then
    printf '%s\n' "$expected_block" >> "$temporary"
  fi
  mv -f "$temporary" "$load_config"
}

action="${1:-}"
case "$action" in
  --install)
    [[ $# == 3 ]] || { usage; exit 2; }
    artifact_dir="$2"
    cann_root="$3"
    ;;
  --check)
    [[ $# == 2 || $# == 3 ]] || { usage; exit 2; }
    cann_root="$2"
    artifact_dir="${3:-}"
    ;;
  --uninstall)
    [[ $# == 2 ]] || { usage; exit 2; }
    cann_root="$2"
    ;;
  --help|-h)
    usage
    exit 0
    ;;
  *)
    # Keep the original two-argument install invocation working.
    [[ $# == 2 ]] || { usage; exit 2; }
    action=--install
    artifact_dir="$1"
    cann_root="$2"
    ;;
esac

[[ -d "$cann_root" ]] || die "CANN root does not exist: $cann_root"
cann_root="$(cd "$cann_root" && pwd -P)"
kernel_dir="$cann_root/opp/vendors/cust/op_impl/aicpu/kernel"
config_dir="$cann_root/opp/vendors/cust/op_impl/aicpu/config"
load_config="$cann_root/conf/ascend_package_load.ini"
[[ -f "$load_config" ]] || die "CANN package loader config missing: $load_config"

case "$action" in
  --install)
    check_artifacts "$artifact_dir"
    mkdir -p "$kernel_dir" "$config_dir"
    if [[ ! -f "$load_config.kvaf-aicpu-urma.bak" ]]; then
      cp -p "$load_config" "$load_config.kvaf-aicpu-urma.bak"
    fi
    install -m 644 "$artifact_dir/$package" "$kernel_dir/$package"
    install -m 644 "$artifact_dir/kernel.json" "$config_dir/$json"
    rewrite_load_config true
    check_installed "$artifact_dir"
    echo "[KVAF_AICPU_URMA] installed; restart the NPU client process"
    ;;
  --check)
    check_installed "$artifact_dir"
    ;;
  --uninstall)
    if grep -Fxq "$header" "$load_config"; then
      rewrite_load_config false
    fi
    rm -f "$kernel_dir/$package" "$config_dir/$json"
    [[ ! -e "$kernel_dir/$package" && ! -e "$config_dir/$json" ]] || die "failed to remove AICPU artifacts"
    if grep -Fxq "$header" "$load_config"; then
      die "failed to remove $header from $load_config"
    fi
    echo "[KVAF_AICPU_URMA] UNINSTALLED root=$cann_root; restart the NPU client process"
    ;;
esac
