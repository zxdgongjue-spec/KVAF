#!/usr/bin/env bash
# Bootstrap and build one KVAF deployment role without installing CANN or UB
# drivers. Those platform packages remain an administrator responsibility.
set -euo pipefail

readonly KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly UMDK_REPOSITORY="https://gitcode.com/openeuler/umdk.git"
readonly UMDK_BRANCH="br_UMDK_24.12.0_LTS"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: ./build.sh <cpu|npu|aicpu-urma> [--build-dir DIR] [--jobs N]

cpu  Builds the liburma-only CPU Runtime service. If liburma is not installed,
     it clones and builds the pinned UMDK source beside this repository.
aicpu-urma Builds the UMDK pipeline BatchGet Python binding and AICPU kernel.
     --install-kernel installs it; --check-kernel and --uninstall-kernel manage the package.
npu  Builds the CANN AIV BatchGet/BatchPut Runtime and Python binding. It
     installs pybind11 for the selected Python interpreter when necessary.

Environment overrides:
  KVAF_UMDK_ROOT       UMDK source/header root for the CPU build
  KVAF_URMA_LIBRARY    Existing liburma.so path for the CPU build
  KVAF_UMDK_BUILD_DIR  Build directory used when building bundled UMDK
  ASCEND_CANN_PACKAGE_PATH
                        CANN toolkit root for the NPU build
  PYTHON               Python interpreter for the NPU binding (default: python3)
  KVAF_AICPU_CXX       AICPU cross compiler override
  KVAF_BUILD_JOBS      Parallel build jobs
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

default_jobs() {
  if [[ -n "${KVAF_BUILD_JOBS:-}" ]]; then
    printf '%s\n' "$KVAF_BUILD_JOBS"
  elif command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
  else
    printf '4\n'
  fi
}

find_urma_library() {
  local source_root="$1"
  local build_root="$2"
  local candidate
  if [[ -n "${KVAF_URMA_LIBRARY:-}" ]]; then
    [[ -f "$KVAF_URMA_LIBRARY" ]] || die "KVAF_URMA_LIBRARY does not exist: $KVAF_URMA_LIBRARY"
    printf '%s\n' "$KVAF_URMA_LIBRARY"
    return
  fi
  for candidate in \
    "$source_root/lib64/liburma.so" \
    "$source_root/lib/liburma.so" \
    /lib64/liburma.so \
    /usr/lib64/liburma.so \
    /usr/local/lib64/liburma.so; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  if command -v ldconfig >/dev/null 2>&1; then
    candidate="$(ldconfig -p 2>/dev/null | awk '/liburma\.so \(/{print $NF; exit}')"
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  fi
  if [[ -d "$build_root" ]]; then
    candidate="$(find -L "$build_root" -type f -name liburma.so -print -quit 2>/dev/null)"
    if [[ -n "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  fi
}

ensure_umdk_source() {
  local source_root="$1"
  if [[ -f "$source_root/include/urma_api.h" ]]; then
    return
  fi
  if [[ -e "$source_root" ]]; then
    die "UMDK root exists but lacks include/urma_api.h: $source_root; set KVAF_UMDK_ROOT to a valid checkout"
  fi
  require_command git
  mkdir -p "$(dirname "$source_root")"
  echo "[KVAF_BUILD] cloning UMDK branch $UMDK_BRANCH into $source_root"
  git clone --depth 1 --branch "$UMDK_BRANCH" "$UMDK_REPOSITORY" "$source_root"
  [[ -f "$source_root/include/urma_api.h" ]] || die "cloned UMDK is missing include/urma_api.h"
}

build_cpu() {
  local build_dir="$1"
  local jobs="$2"
  local umdk_root="${KVAF_UMDK_ROOT:-$KVAF_ROOT/../third_party/umdk}"
  local umdk_build_dir="${KVAF_UMDK_BUILD_DIR:-$umdk_root/build-kvaf}"
  local urma_library

  require_command cmake
  ensure_umdk_source "$umdk_root"
  urma_library="$(find_urma_library "$umdk_root" "$umdk_build_dir")"
  if [[ -z "$urma_library" ]]; then
    echo "[KVAF_BUILD] liburma.so not found; building bundled UMDK without TPS/HW drivers"
    cmake -S "$umdk_root" -B "$umdk_build_dir" \
      -DTPS=disable -DHW=disable -DURMA_INSTALL=disable
    cmake --build "$umdk_build_dir" --target urma --parallel "$jobs"
    urma_library="$(find_urma_library "$umdk_root" "$umdk_build_dir")"
  fi
  [[ -n "$urma_library" ]] || die "unable to find liburma.so after UMDK setup"

  cmake -S "$KVAF_ROOT" -B "$build_dir" \
    -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON \
    -DKVAF_ENABLE_AICPU_URMA=OFF \
    -DKVAF_URMA_ROOT="$umdk_root" \
    -DKVAF_URMA_LIBRARY="$urma_library"
  cmake --build "$build_dir" --target kvaf_cpu_service --parallel "$jobs"
  echo "[KVAF_BUILD] CPU service: $build_dir/runtime/kvaf_cpu_service"
}

find_cann_root() {
  local candidate
  for candidate in \
    "${ASCEND_CANN_PACKAGE_PATH:-}" \
    "${ASCEND_HOME_PATH:-}" \
    /usr/local/Ascend/ascend-toolkit/latest \
    /usr/local/Ascend/cann/aarch64-linux/asc; do
    if [[ -n "$candidate" && -f "$candidate/include/acl/acl.h" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
}

build_npu() {
  local build_dir="$1"
  local jobs="$2"
  local python_bin="${PYTHON:-python3}"
  local cann_root
  local pybind11_dir

  require_command cmake
  require_command "$python_bin"
  cann_root="$(find_cann_root)"
  [[ -n "$cann_root" ]] || die "CANN toolkit was not found; source CANN set_env.sh or set ASCEND_CANN_PACKAGE_PATH"
  if ! "$python_bin" -m pybind11 --cmakedir >/dev/null 2>&1; then
    echo "[KVAF_BUILD] installing pybind11 for $python_bin"
    "$python_bin" -m pip install --user pybind11
  fi
  pybind11_dir="$($python_bin -m pybind11 --cmakedir)"

  cmake -S "$KVAF_ROOT" -B "$build_dir" \
    -DBUILD_TESTING=ON \
    -DKVAF_BUILD_PYTHON=ON \
    -DKVAF_ENABLE_AIV_BATCH_GET=ON \
    -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
    -DASCEND_CANN_PACKAGE_PATH="$cann_root" \
    -Dpybind11_DIR="$pybind11_dir"
  cmake --build "$build_dir" --parallel "$jobs"
  echo "[KVAF_BUILD] NPU Python package: $build_dir/python/kvaf"
  echo "[KVAF_BUILD] BatchGet kernel: $build_dir/bin/kvaf_execute_batch_get_kernels.o"
  echo "[KVAF_BUILD] BatchPut kernel: $build_dir/bin/kvaf_execute_batch_put_kernels.o"
}

check_aicpu_cann_root() {
  local selected_root="$1"
  local runtime_root
  [[ -n "${ASCEND_HOME_PATH:-}" ]] || return 0
  [[ -d "$ASCEND_HOME_PATH" ]] || die "ASCEND_HOME_PATH does not exist: $ASCEND_HOME_PATH"
  runtime_root="$(cd "$ASCEND_HOME_PATH" && pwd -P)"
  selected_root="$(cd "$selected_root" && pwd -P)"
  [[ "$runtime_root" == "$selected_root" ]] ||
    die "AICPU build CANN root ($selected_root) differs from runtime ASCEND_HOME_PATH ($runtime_root)"
}

build_aicpu_urma() {
  local build_dir="$1" jobs="$2" kernel_action="$3"
  local cann_root umdk_root="${KVAF_UMDK_ROOT:-$KVAF_ROOT/../third_party/umdk}"
  local -a compiler_option=()
  cann_root="$(find_cann_root)"
  [[ -n "$cann_root" ]] || die "CANN toolkit not found; source CANN set_env.sh first"
  check_aicpu_cann_root "$cann_root"
  if [[ "$kernel_action" == check || "$kernel_action" == uninstall ]]; then
    bash "$KVAF_ROOT/access_operator/src/aicpu_urma/package/install_kernel.sh" "--$kernel_action" "$cann_root"
    return
  fi
  require_command cmake
  local python_bin="${PYTHON:-python3}"
  require_command "$python_bin"
  if ! "$python_bin" -m pybind11 --cmakedir >/dev/null 2>&1; then
    "$python_bin" -m pip install --user pybind11
  fi
  ensure_umdk_source "$umdk_root"
  if [[ -n "${KVAF_AICPU_CXX:-}" ]]; then
    compiler_option+=("-DKVAF_AICPU_CXX=$KVAF_AICPU_CXX")
  fi
  cmake -S "$KVAF_ROOT" -B "$build_dir" \
    -DBUILD_TESTING=OFF -DKVAF_BUILD_PYTHON=ON -DPython_EXECUTABLE="$python_bin" \
    -DKVAF_ENABLE_AIV_BATCH_GET=OFF -DKVAF_ENABLE_AIV_BATCH_PUT=OFF \
    -DKVAF_BUILD_REMOTE_CPU_SERVICE=OFF -DKVAF_ENABLE_AICPU_URMA=ON \
    -DKVAF_URMA_ROOT="$umdk_root" -DASCEND_CANN_PACKAGE_PATH="$cann_root" \
    "${compiler_option[@]}"
  cmake --build "$build_dir" --target _kvaf kvaf_aicpu_urma_kernel --parallel "$jobs"
  if [[ "$kernel_action" == install ]]; then
    bash "$KVAF_ROOT/access_operator/src/aicpu_urma/package/install_kernel.sh" --install \
      "$build_dir/access_operator/src/aicpu_urma" "$cann_root"
  fi
  echo "[KVAF_BUILD] NPU Python package: $build_dir/python/kvaf"
}

mode="${1:-}"
case "$mode" in
  cpu|npu|aicpu-urma) shift ;;
  -h|--help|help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

build_dir="$KVAF_ROOT/build-$mode"
jobs="$(default_jobs)"
kernel_action=build
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      [[ $# -ge 2 ]] || die "--build-dir requires a value"
      build_dir="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || die "--jobs requires a positive integer"
      jobs="$2"
      shift 2
      ;;
    --install-kernel|--check-kernel|--uninstall-kernel)
      [[ "$mode" == aicpu-urma ]] || die "$1 is only valid for aicpu-urma"
      case "$1" in
        --install-kernel) kernel_action=install ;;
        --check-kernel) kernel_action=check ;;
        --uninstall-kernel) kernel_action=uninstall ;;
      esac
      shift
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "KVAF_BUILD_JOBS must be a positive integer"
if [[ "$mode" == aicpu-urma ]]; then
  build_aicpu_urma "$build_dir" "$jobs" "$kernel_action"
elif [[ "$mode" == cpu ]]; then
  build_cpu "$build_dir" "$jobs"
else
  build_npu "$build_dir" "$jobs"
fi
