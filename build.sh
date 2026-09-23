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
Usage: ./build.sh <cpu|npu|aicpu-urma|hbm-rw|aicpu-cpu|aicpu-npu> [--build-dir DIR] [--jobs N] [--install-kernel] [--compare-aiv]

cpu  Builds the liburma-only CPU Runtime service. If liburma is not installed,
     it clones and builds the pinned UMDK source beside this repository.
npu  Builds the CANN AIV BatchGet/BatchPut Runtime and Python binding. It
     installs pybind11 for the selected Python interpreter when necessary.
aicpu-urma Builds the native NPU Runtime client and the AICPU URMA BatchPut/
           BatchGet kernel package; no Python binding. --install-kernel installs
           the package into the detected CANN root (write access required).
           --compare-aiv also builds AIV kernels for a same-peer diagnostic run.
hbm-rw Builds the standalone benchmark together with the production CPU URMA
       transport and NPU HBM service.
aicpu-cpu Builds only the CPU side of the AICPU CPU-push bandwidth test.
aicpu-npu Builds its NPU executable and AICPU kernel package (no installation).
          These two benchmark modes do not build Python bindings or AIV kernels.

Environment overrides:
  KVAF_UMDK_ROOT       UMDK source/header root for the CPU build
  KVAF_URMA_LIBRARY    Existing liburma.so path for the CPU build
  KVAF_UMDK_BUILD_DIR  Build directory used when building bundled UMDK
  ASCEND_CANN_PACKAGE_PATH
                        CANN toolkit root for the NPU build
  PYTHON               Python interpreter for the NPU binding (default: python3)
  KVAF_BUILD_JOBS      Parallel build jobs
  KVAF_AICPU_CXX      CANN AICPU cross compiler override (aicpu-npu, aicpu-urma)
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
  local python_bin="${PYTHON:-python3}"
  local pybind11_dir
  local umdk_root="${KVAF_UMDK_ROOT:-$KVAF_ROOT/../third_party/umdk}"
  local umdk_build_dir="${KVAF_UMDK_BUILD_DIR:-$umdk_root/build-kvaf}"
  local urma_library

  require_command cmake
  require_command "$python_bin"
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
  if ! "$python_bin" -m pybind11 --cmakedir >/dev/null 2>&1; then
    echo "[KVAF_BUILD] installing pybind11 for $python_bin"
    "$python_bin" -m pip install --user pybind11
  fi
  pybind11_dir="$($python_bin -m pybind11 --cmakedir)"

  # Release optimization triggers SIGBUS in the deployed CPU URMA
  # initialization ABI. Keep the known-good build mode until that provider
  # boundary is isolated from the performance-sensitive transfer code.
  cmake -S "$KVAF_ROOT" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE= \
    -DBUILD_TESTING=ON \
    -DKVAF_BUILD_PYTHON=ON \
    -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON \
    -Dpybind11_DIR="$pybind11_dir" \
    -DKVAF_URMA_ROOT="$umdk_root" \
    -DKVAF_URMA_LIBRARY="$urma_library"
  cmake --build "$build_dir" --target kvaf_cpu_service kvaf_cpu_runtime_npu_hbm_e2e kvaf_cpu_hbm_client kvaf_cpu_urma_eid_perf kvaf_cpu_urma_cpu_bw _kvaf --parallel "$jobs"
  echo "[KVAF_BUILD] CPU service: $build_dir/runtime/kvaf_cpu_service"
  echo "[KVAF_BUILD] CPU NPU-HBM client: $build_dir/runtime/kvaf_cpu_runtime_npu_hbm_e2e"
  echo "[KVAF_BUILD] CPU EID bandwidth test: $build_dir/tests/kvaf_cpu_urma_eid_perf"
  echo "[KVAF_BUILD] CPU-to-CPU bandwidth test: $build_dir/tests/kvaf_cpu_urma_cpu_bw"
  echo "[KVAF_BUILD] CPU Python binding: $build_dir/python/kvaf"
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
  local umdk_root="${KVAF_UMDK_ROOT:-$KVAF_ROOT/../third_party/umdk}"

  require_command cmake
  require_command "$python_bin"
  ensure_umdk_source "$umdk_root"
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
    -DKVAF_ENABLE_AICPU_URMA=OFF \
    -DASCEND_CANN_PACKAGE_PATH="$cann_root" \
    -Dpybind11_DIR="$pybind11_dir" \
    -DKVAF_BUILD_NPU_SERVICE=ON \
    -DKVAF_URMA_ROOT="$umdk_root"
  cmake --build "$build_dir" --parallel "$jobs"
  echo "[KVAF_BUILD] NPU Python package: $build_dir/python/kvaf"
  echo "[KVAF_BUILD] BatchGet kernel: $build_dir/bin/kvaf_execute_batch_get_kernels.o"
  echo "[KVAF_BUILD] BatchPut kernel: $build_dir/bin/kvaf_execute_batch_put_kernels.o"
}

build_aicpu_urma() {
  local build_dir="$1"
  local jobs="$2"
  local install_kernel="$3"
  local compare_aiv="$4"
  local cann_root
  local -a compiler_option=()
  local -a targets=(kvaf_aicpu_urma_client kvaf_aicpu_urma_kernel
    kvaf_aicpu_urma_wqe_test kvaf_aicpu_urma_kernel_test
    kvaf_aicpu_urma_client_options_test)
  require_command cmake
  cann_root="$(find_cann_root)"
  [[ -n "$cann_root" ]] || die "CANN toolkit not found; source CANN set_env.sh first"
  if [[ -n "${KVAF_AICPU_CXX:-}" ]]; then
    compiler_option+=("-DKVAF_AICPU_CXX=$KVAF_AICPU_CXX")
  fi
  cmake -S "$KVAF_ROOT" -B "$build_dir" \
    -DBUILD_TESTING=ON \
    -DKVAF_BUILD_PYTHON=OFF \
    -DKVAF_BUILD_NPU_SERVICE=OFF \
    -DKVAF_ENABLE_AIV_BATCH_GET=ON \
    -DKVAF_ENABLE_AIV_BATCH_PUT=ON \
    -DKVAF_ENABLE_AICPU_URMA=ON \
    -DASCEND_CANN_PACKAGE_PATH="$cann_root" \
    "${compiler_option[@]}"
  if [[ "$compare_aiv" == true ]]; then
    targets+=(kvaf_execute_batch_get_kernel kvaf_execute_batch_put_kernel)
  fi
  cmake --build "$build_dir" --target "${targets[@]}" --parallel "$jobs"
  if [[ "$install_kernel" == true ]]; then
    bash "$KVAF_ROOT/access_operator/src/aicpu_urma/install_kernel.sh" \
      "$build_dir/access_operator/src/aicpu_urma" "$cann_root"
  fi
  echo "[KVAF_BUILD] Native test: $build_dir/tests/kvaf_aicpu_urma_client"
  echo "[KVAF_BUILD] AICPU kernel package: $build_dir/access_operator/src/aicpu_urma/kvaf-aicpu-urma.tar.gz"
  echo "[KVAF_BUILD] Host tests: ctest --test-dir '$build_dir' -R kvaf_aicpu_urma_ --output-on-failure"
  echo "[KVAF_BUILD] Run: '$build_dir/tests/kvaf_aicpu_urma_client' --help"
  if [[ "$compare_aiv" == true ]]; then
    echo "[KVAF_BUILD] AIV comparison: '$build_dir/tests/kvaf_aicpu_urma_client' --backend aiv --peer HOST:PORT"
  fi
}

build_hbm_rw() {
  local build_dir="$1"
  local jobs="$2"
  local cann_root
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
  cann_root="$(find_cann_root)"
  [[ -n "$cann_root" ]] || die "CANN toolkit was not found; set ASCEND_CANN_PACKAGE_PATH"
  cmake -S "$KVAF_ROOT" -B "$build_dir" \
    -DBUILD_TESTING=ON \
    -DKVAF_BUILD_REMOTE_CPU_SERVICE=ON \
    -DKVAF_ENABLE_AIV_BATCH_GET=ON \
    -DKVAF_ENABLE_AIV_BATCH_PUT=OFF \
    -DKVAF_BUILD_NPU_SERVICE=ON \
    -DASCEND_CANN_PACKAGE_PATH="$cann_root" \
    -DKVAF_URMA_ROOT="$umdk_root" \
    -DKVAF_URMA_LIBRARY="$urma_library"
  cmake --build "$build_dir" --target kvaf_npu_service kvaf_cpu_hbm_client --parallel "$jobs"
  echo "[KVAF_BUILD] HBM test binaries: $build_dir/runtime/kvaf_npu_service and $build_dir/tests/kvaf_cpu_hbm_client"
}

build_aicpu_test() {
  local role="$1" build_dir="$2" jobs="$3"
  local umdk_root="${KVAF_UMDK_ROOT:-$KVAF_ROOT/../third_party/umdk}"
  local umdk_build_dir="${KVAF_UMDK_BUILD_DIR:-$umdk_root/build-kvaf}"
  local urma_library cann_root
  local -a options=(
    -DCMAKE_BUILD_TYPE=
    -DBUILD_TESTING=ON
    -DKVAF_BUILD_PYTHON=OFF
    -DKVAF_ENABLE_AIV_BATCH_GET=OFF
    -DKVAF_ENABLE_AIV_BATCH_PUT=OFF
    -DKVAF_BUILD_NPU_SERVICE=OFF
    -DKVAF_BUILD_HBM_RW_TEST=OFF
    -DKVAF_BUILD_CPU_URMA_PROBE=OFF
  )
  require_command cmake
  require_command "${PYTHON:-python3}"
  ensure_umdk_source "$umdk_root"
  options+=("-DKVAF_URMA_ROOT=$umdk_root"
            "-DPython3_EXECUTABLE=$(command -v "${PYTHON:-python3}")")
  if [[ "$role" == cpu ]]; then
    urma_library="$(find_urma_library "$umdk_root" "$umdk_build_dir")"
    if [[ -z "$urma_library" ]]; then
      cmake -S "$umdk_root" -B "$umdk_build_dir" \
        -DTPS=disable -DHW=disable -DURMA_INSTALL=disable
      cmake --build "$umdk_build_dir" --target urma --parallel "$jobs"
      urma_library="$(find_urma_library "$umdk_root" "$umdk_build_dir")"
    fi
    [[ -n "$urma_library" ]] || die "unable to find liburma.so after UMDK setup"
    options+=(-DKVAF_BUILD_REMOTE_CPU_SERVICE=ON -DKVAF_BUILD_AICPU_GET_TEST=OFF
              "-DKVAF_URMA_LIBRARY=$urma_library")
  else
    cann_root="$(find_cann_root)"
    [[ -n "$cann_root" ]] || die "CANN toolkit not found; source CANN set_env.sh first"
    options+=(-DKVAF_BUILD_REMOTE_CPU_SERVICE=OFF -DKVAF_BUILD_AICPU_GET_TEST=ON
              "-DASCEND_CANN_PACKAGE_PATH=$cann_root")
    if [[ -n "${KVAF_AICPU_CXX:-}" ]]; then
      options+=("-DKVAF_AICPU_CXX=$KVAF_AICPU_CXX")
    fi
  fi
  cmake -S "$KVAF_ROOT" -B "$build_dir" "${options[@]}"
  cmake --build "$build_dir" --target "kvaf_aicpu_get_$role" --parallel "$jobs"
  echo "[KVAF_BUILD] AICPU test: $build_dir/tests/aicpu_batch_get/kvaf_aicpu_get_$role"
  echo "[KVAF_BUILD] Run: bash tests/run_aicpu_get_$role.sh --build-dir '$build_dir' --help"
}

mode="${1:-}"
case "$mode" in
  cpu|npu|aicpu-urma|hbm-rw|aicpu-cpu|aicpu-npu) shift ;;
  -h|--help|help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

build_dir="$KVAF_ROOT/build-$mode"
jobs="$(default_jobs)"
install_kernel=false
compare_aiv=false
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
    --install-kernel)
      [[ "$mode" == aicpu-urma ]] || die "--install-kernel is only valid for aicpu-urma"
      install_kernel=true
      shift
      ;;
    --compare-aiv)
      [[ "$mode" == aicpu-urma ]] || die "--compare-aiv is only valid for aicpu-urma"
      compare_aiv=true
      shift
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "KVAF_BUILD_JOBS must be a positive integer"
case "$mode" in
  cpu) build_cpu "$build_dir" "$jobs" ;;
  npu) build_npu "$build_dir" "$jobs" ;;
  aicpu-urma) build_aicpu_urma "$build_dir" "$jobs" "$install_kernel" "$compare_aiv" ;;
  hbm-rw) build_hbm_rw "$build_dir" "$jobs" ;;
  aicpu-cpu) build_aicpu_test cpu "$build_dir" "$jobs" ;;
  aicpu-npu) build_aicpu_test npu "$build_dir" "$jobs" ;;
esac
