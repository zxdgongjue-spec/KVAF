#!/usr/bin/env bash
# Bootstrap and build one KVAF deployment role without installing CANN or UB
# drivers. Those platform packages remain an administrator responsibility.
set -euo pipefail

readonly KVAF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly UMDK_REPOSITORY="https://gitcode.com/openeuler/umdk.git"
readonly UMDK_BRANCH="br_UMDK_24.12.0_LTS"
source "$KVAF_ROOT/scripts/cann_root.sh"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage: ./build.sh <cpu|npu|aicpu-urma> [--build-dir DIR] [--jobs N] [mode-specific options]

cpu  Builds the CPU Runtime service with liburma and OpenMP. If liburma is not installed,
     it clones and builds the pinned UMDK source beside this repository.
     --without-urma includes only the UBmem backend in the same service.
aicpu-urma Builds and installs AICPU gather_read, BatchPut and Python bindings.
     --cpu-push adds Host direct HCCP and the CPU push/scatter kernel (needs device securec).
npu --transport ubmem builds only local Host DDR Put/Get (no UMDK or CPU service),
     with THP/base registration and build-ubmem-npu output. Requires SOC_VERSION.
npu (default --transport urma) builds all service backends (AIV URMA, UBmem, AICPU Gather/Scatter and BatchPut),
     outputs one Python package and installs the AICPU device package into CANN.
     --without-aicpu builds only AIV/UBmem, without AICPU dependencies or installation.
     --without-python builds NPU C++ libraries and tests without the binding.
Both NPU modes install pybind11 for the selected Python interpreter when necessary.
     --check-kernel / --uninstall-kernel manage the AICPU package without building.
     --install-kernel remains accepted for compatibility; installation is automatic.
     Installing requires write access to the selected CANN package/config directories.

Environment overrides:
  KVAF_UMDK_ROOT       UMDK source/header root for CPU and AICPU builds
  KVAF_URMA_LIBRARY    Existing liburma.so path for the CPU build
  KVAF_UMDK_BUILD_DIR  Build directory used when building bundled UMDK
  ASCEND_HOME_PATH / ASCEND_CANN_PACKAGE_PATH
                        CANN root; both must resolve to the same directory if set
  PYTHON               Python interpreter for the NPU binding (default: python3)
  KVAF_AICPU_CXX       AICPU cross compiler override
  KVAF_AICPU_SECUREC_INCLUDE_DIR
  KVAF_AICPU_SECUREC_LIBRARY
                        Device securec header and AArch64 library for memcpy_sp
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
  if [[ "$without_urma" == ON ]]; then
    cmake -S "$KVAF_ROOT" -B "$build_dir" -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON \
      -DKVAF_CPU_ENABLE_URMA=OFF -DBUILD_TESTING=ON
    cmake --build "$build_dir" --target kvaf_cpu_service --parallel "$jobs"
    echo "[KVAF_BUILD] CPU service: $build_dir/runtime/kvaf_cpu_service"
    return
  fi
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
    -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON -DKVAF_CPU_ENABLE_URMA=ON -DBUILD_TESTING=ON \
    -DKVAF_ENABLE_AICPU_URMA=OFF \
    -DKVAF_URMA_ROOT="$umdk_root" \
    -DKVAF_URMA_LIBRARY="$urma_library"
  cmake --build "$build_dir" --target kvaf_cpu_service --parallel "$jobs"
  echo "[KVAF_BUILD] CPU service: $build_dir/runtime/kvaf_cpu_service"
}

find_cann_root() {
  kvaf_find_cann_root
}

build_npu() {
  local build_dir="$1"
  local jobs="$2"
  local transport="$3"
  local python_bin="${PYTHON:-python3}"
  local cann_root
  local build_python=OFF
  local build_aiv=ON build_aicpu=ON build_testing=ON
  local umdk_root="${KVAF_UMDK_ROOT:-$KVAF_ROOT/../third_party/umdk}"
  local -a cmake_options=()
  local -a package_options=()
  if [[ "$mode" == aicpu-urma ]]; then build_aiv=OFF; build_testing=OFF; fi
  if [[ "$without_aicpu" == ON ]]; then build_aicpu=OFF; fi
  if [[ "$cpu_push" == ON ]]; then package_options+=(--require-scatter); fi
  if [[ "$transport" == ubmem ]]; then
    [[ "$mode" == npu && "$kernel_action" == build ]] || die "local UBmem supports npu without AICPU package actions"
    [[ -n "${SOC_VERSION:-}" ]] || die "SOC_VERSION must name the actual device SoC for UBmem kernels"
    build_aiv=OFF
    build_aicpu=OFF
    cpu_push=OFF
    cmake_options+=("-DSOC_VERSION=$SOC_VERSION")
  fi
  cmake_options+=("-DKVAF_TRANSPORT=$transport" "-DKVAF_ENABLE_NPU=ON")
  cann_root="$(find_cann_root)"
  [[ -n "$cann_root" ]] || die "CANN toolkit not found; source CANN set_env.sh first"
  if [[ "$kernel_action" == check || "$kernel_action" == uninstall ]]; then
    local -a artifact_options=()
    local artifact_dir="$build_dir/access_operator/src/aicpu_urma"
    if [[ "$kernel_action" == check && -s "$artifact_dir/kernel.json" && -s "$artifact_dir/kvaf-aicpu-urma.tar.gz" ]]; then
      artifact_options+=("$artifact_dir")
    fi
    bash "$KVAF_ROOT/access_operator/src/aicpu_urma/package/install_kernel.sh" \
      ${package_options[@]+"${package_options[@]}"} "--$kernel_action" "$cann_root" \
      ${artifact_options[@]+"${artifact_options[@]}"}
    return
  fi
  [[ -f "$cann_root/include/acl/acl.h" ]] || die "ACL headers missing under $cann_root"
  require_command cmake
  if [[ "$without_python" != ON ]]; then
    build_python=ON
    require_command "$python_bin"
    python_bin="$(command -v "$python_bin")"
    if ! "$python_bin" -m pybind11 --cmakedir >/dev/null 2>&1; then
      echo "[KVAF_BUILD] installing pybind11 for $python_bin"
      "$python_bin" -m pip install --user pybind11
    fi
    cmake_options+=("-DPython_EXECUTABLE=$python_bin" "-DPython3_EXECUTABLE=$python_bin"
      "-Dpybind11_DIR=$("$python_bin" -m pybind11 --cmakedir)")
  fi
  if [[ "$build_aicpu" == ON ]]; then
    ensure_umdk_source "$umdk_root"
    cmake_options+=("-DKVAF_URMA_ROOT=$umdk_root")
    if [[ -n "${KVAF_AICPU_CXX:-}" ]]; then
      cmake_options+=("-DKVAF_AICPU_CXX=$KVAF_AICPU_CXX")
    fi
    if [[ -n "${KVAF_AICPU_SECUREC_INCLUDE_DIR:-}" ]]; then
      cmake_options+=("-DKVAF_AICPU_SECUREC_INCLUDE_DIR=$KVAF_AICPU_SECUREC_INCLUDE_DIR")
    fi
    if [[ -n "${KVAF_AICPU_SECUREC_LIBRARY:-}" ]]; then
      cmake_options+=("-DKVAF_AICPU_SECUREC_LIBRARY=$KVAF_AICPU_SECUREC_LIBRARY")
    fi
  fi
  # Set every backend switch explicitly so reusing a build directory cannot
  # retain AICPU/Scatter (or a CPU service) after selecting a trimmed build.
  cmake -S "$KVAF_ROOT" -B "$build_dir" \
    -DBUILD_TESTING="$build_testing" -DKVAF_BUILD_PYTHON="$build_python" \
    -DKVAF_ENABLE_AIV_BATCH_GET="$build_aiv" -DKVAF_ENABLE_AIV_BATCH_PUT="$build_aiv" \
    -DKVAF_BUILD_REMOTE_CPU_SERVICE=OFF -DKVAF_ENABLE_AICPU_URMA="$build_aicpu" \
    -DKVAF_ENABLE_AICPU_CPU_PUSH="$cpu_push" \
    -DASCEND_CANN_PACKAGE_PATH="$cann_root" \
    ${cmake_options[@]+"${cmake_options[@]}"}
  cmake --build "$build_dir" --parallel "$jobs"
  if [[ "$build_aicpu" == ON ]]; then
    bash "$KVAF_ROOT/access_operator/src/aicpu_urma/package/install_kernel.sh" \
      ${package_options[@]+"${package_options[@]}"} --install \
      "$build_dir/access_operator/src/aicpu_urma" "$cann_root"
  fi
  if [[ "$build_python" == ON ]]; then echo "[KVAF_BUILD] NPU Python package: $build_dir/python/kvaf"; fi
  if [[ "$build_aiv" == ON ]]; then
    echo "[KVAF_BUILD] BatchGet kernel: $build_dir/bin/kvaf_execute_batch_get_kernels.o"
    echo "[KVAF_BUILD] BatchPut kernel: $build_dir/bin/kvaf_execute_batch_put_kernels.o"
  fi
}

mode="${1:-}"
case "$mode" in
  cpu|npu|aicpu-urma) shift ;;
  -h|--help|help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

build_dir=""
transport=urma
jobs="$(default_jobs)"
kernel_action=build
cpu_push=OFF
if [[ "$mode" == npu ]]; then cpu_push=ON; fi
without_urma=OFF
without_aicpu=OFF
without_python=OFF
while [[ $# -gt 0 ]]; do
  case "$1" in
    --transport)
      [[ $# -ge 2 && ( "$2" == urma || "$2" == ubmem ) ]] || die "--transport requires urma or ubmem"
      transport="$2"; shift 2 ;;
    --without-urma)
      [[ "$mode" == cpu ]] || die "--without-urma is only valid for cpu"
      without_urma=ON; shift ;;
    --without-python)
      [[ "$mode" == npu ]] || die "--without-python is only valid for npu"
      without_python=ON; shift ;;
    --without-aicpu)
      [[ "$mode" == npu ]] || die "--without-aicpu is only valid for npu"
      without_aicpu=ON; cpu_push=OFF; shift ;;
    --cpu-push)
      [[ "$mode" == aicpu-urma ]] || die "--cpu-push is only valid for aicpu-urma"
      cpu_push=ON
      shift
      ;;
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
      [[ "$mode" != cpu ]] || die "$1 is only valid for npu or aicpu-urma"
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
if [[ "$without_aicpu" == ON && "$kernel_action" != build ]]; then
  die "AICPU package actions cannot be combined with --without-aicpu"
fi
if [[ -z "$build_dir" ]]; then
  if [[ "$transport" == ubmem ]]; then build_dir="$KVAF_ROOT/build-ubmem-$mode"; else build_dir="$KVAF_ROOT/build-$mode"; fi
fi
if [[ "$mode" == cpu ]]; then
  [[ "$transport" == urma ]] || die "UBmem uses a worker-local Host DDR arena; build with 'npu --transport ubmem', not a CPU service"
  build_cpu "$build_dir" "$jobs"
else
  build_npu "$build_dir" "$jobs" "$transport"
fi
