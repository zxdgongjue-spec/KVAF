#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

: "${ASCEND_HOME_PATH:?source the CANN set_env.sh before building}"

KVAF_A5_BUILD_DIR=${KVAF_A5_BUILD_DIR:-"${ROOT_DIR}/build-a5-aiv-urma"}
KVAF_A5_BUILD_JOBS=${KVAF_A5_BUILD_JOBS:-8}
KVAF_A5_COMPILE_MODE=${KVAF_A5_COMPILE_MODE:-auto}
KVAF_BISHENG=${KVAF_BISHENG:-$(command -v bisheng || true)}
KVAF_ASC_DEVKIT_PATH=${KVAF_ASC_DEVKIT_PATH:-}

if [[ -z "${KVAF_BISHENG}" || ! -x "${KVAF_BISHENG}" ]]; then
  echo "ERROR: bisheng was not found; check the CANN compiler environment" >&2
  exit 2
fi
if ! [[ "${KVAF_A5_BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: KVAF_A5_BUILD_JOBS must be a positive integer" >&2
  exit 2
fi
case "${KVAF_A5_COMPILE_MODE}" in
  auto|xasc|xcce) ;;
  *)
    echo "ERROR: KVAF_A5_COMPILE_MODE must be auto, xasc, or xcce" >&2
    exit 2
    ;;
esac

if [[ -z "${KVAF_ASC_DEVKIT_PATH}" &&
      ! -r "${ASCEND_HOME_PATH}/include/adv_api/hcomm/hcomm.h" &&
      ! -r "${ASCEND_HOME_PATH}/$(uname -m)-linux/include/adv_api/hcomm/hcomm.h" &&
      -r "${ROOT_DIR}/../asc-devkit/include/adv_api/hcomm/hcomm.h" ]]; then
  KVAF_ASC_DEVKIT_PATH="${ROOT_DIR}/../asc-devkit"
fi

cmake_args=(
  -S "${ROOT_DIR}"
  -B "${KVAF_A5_BUILD_DIR}"
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DCMAKE_CXX_COMPILER="${KVAF_BISHENG}"
  -DKVAF_BUILD_A5_AIV_URMA=ON
  -DKVAF_A5_COMPILE_MODE="${KVAF_A5_COMPILE_MODE}"
)
if [[ -n "${KVAF_ASC_DEVKIT_PATH}" ]]; then
  cmake_args+=("-DKVAF_ASC_DEVKIT_PATH=${KVAF_ASC_DEVKIT_PATH}")
fi

cmake "${cmake_args[@]}"
cmake --build "${KVAF_A5_BUILD_DIR}" \
  --target kvaf_a5_aiv_urma_h2d_smoke \
  --parallel "${KVAF_A5_BUILD_JOBS}"

binary="${KVAF_A5_BUILD_DIR}/access_operator/a5/kvaf_a5_aiv_urma_h2d_smoke"
if [[ ! -x "${binary}" ]]; then
  echo "ERROR: expected smoke binary is missing: ${binary}" >&2
  exit 3
fi

echo "KVAF_A5_AIV_URMA_BUILD_OK"
echo "BINARY=${binary}"
