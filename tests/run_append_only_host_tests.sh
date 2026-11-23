#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="${KVAF_APPEND_HOST_BUILD_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/kvaf-append-host.XXXXXX")}"
mkdir -p "$build"
cxx="${CXX:-c++}"
echo "APPEND_ONLY_HOST_BUILD=$build CXX=$cxx"
includes=(-I"$root/common/include" -I"$root/meta_manager/include" -I"$root/access_operator/include" -I"$root/runtime/include")
meta=("$root/meta_manager/src/in_memory_meta_manager.cc" "$root/meta_manager/src/append_only_meta_manager.cc")
"$cxx" -std=c++17 -O2 -Wall -Wextra "${includes[@]}" \
  "$root/tests/append_only_meta_manager_test.cc" "${meta[@]}" -o "$build/meta"
"$cxx" -std=c++17 -O2 -Wall -Wextra "${includes[@]}" \
  "$root/tests/append_only_runtime_test.cc" "$root/runtime/src/runtime.cc" "${meta[@]}" -o "$build/runtime"
"$cxx" -std=c++17 -O2 "${includes[@]}" \
  "$root/meta_manager/tests/in_memory_meta_manager_test.cc" "${meta[@]}" -o "$build/legacy_meta"
"$cxx" -std=c++17 -O2 "${includes[@]}" \
  "$root/tests/npu_runtime_config_test.cc" "$root/runtime/src/npu_runtime_config.cc" -o "$build/config"
"$build/meta"
"$build/runtime"
"$build/legacy_meta"
"$build/config"
echo "APPEND_ONLY_HOST_PASS"
