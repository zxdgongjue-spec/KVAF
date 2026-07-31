#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

: "${ASCEND_HOME_PATH:?source the CANN set_env.sh before building}"

KVAF_A5_DIRECT_BUILD_DIR=${KVAF_A5_DIRECT_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma"}
KVAF_A5_BUILD_JOBS=${KVAF_A5_BUILD_JOBS:-8}
KVAF_A5_COMPILE_MODE=${KVAF_A5_COMPILE_MODE:-auto}
KVAF_BISHENG=${KVAF_BISHENG:-$(command -v bisheng || true)}
KVAF_ASC_DEVKIT_PATH=${KVAF_ASC_DEVKIT_PATH:-}
KVAF_DIRECT_DEPLOY_RUN_LOCK=${KVAF_DIRECT_DEPLOY_RUN_LOCK:-/run/lock/kvaf-a5-direct-urma.lock}

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
if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock is required to serialize direct URMA deploy/run" >&2
  exit 2
fi
mkdir -p "$(dirname "${KVAF_DIRECT_DEPLOY_RUN_LOCK}")"
exec 9>"${KVAF_DIRECT_DEPLOY_RUN_LOCK}"
if ! flock -w 30 9; then
  echo "ERROR: another direct URMA deploy/run owns ${KVAF_DIRECT_DEPLOY_RUN_LOCK}" >&2
  exit 2
fi

if [[ -z "${KVAF_ASC_DEVKIT_PATH}" &&
      ( ! -r "${ASCEND_HOME_PATH}/include/kernel_operator.h" ||
        ! -r "${ASCEND_HOME_PATH}/include/simt_api/asc_simt.h" ) &&
      -r "${ROOT_DIR}/../asc-devkit/include/kernel_operator.h" &&
      -r "${ROOT_DIR}/../asc-devkit/include/simt_api/asc_simt.h" ]]; then
  KVAF_ASC_DEVKIT_PATH="${ROOT_DIR}/../asc-devkit"
fi

cmake \
  -S "${ROOT_DIR}" \
  -B "${KVAF_A5_DIRECT_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER="${KVAF_BISHENG}" \
  -DKVAF_BUILD_A5_AIV_URMA=ON \
  -DKVAF_A5_BUILD_HCOMM_BACKEND=OFF \
  -DKVAF_A5_BUILD_DIRECT_URMA_BACKEND=ON \
  -DKVAF_A5_COMPILE_MODE="${KVAF_A5_COMPILE_MODE}" \
  -DKVAF_ASC_DEVKIT_PATH="${KVAF_ASC_DEVKIT_PATH}"
cmake --build "${KVAF_A5_DIRECT_BUILD_DIR}" \
  --target kvaf_a5_aiv_direct_urma_h2d_smoke \
  --parallel "${KVAF_A5_BUILD_JOBS}"

binary="${KVAF_A5_DIRECT_BUILD_DIR}/access_operator/a5/kvaf_a5_aiv_direct_urma_h2d_smoke"
if [[ ! -x "${binary}" ]]; then
  echo "ERROR: expected direct smoke binary is missing: ${binary}" >&2
  exit 3
fi

readelf -d "${binary}" \
  >"${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke.dynamic.txt"
if grep -Fq 'libhcomm' \
    "${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke.dynamic.txt"; then
  echo "ERROR: direct smoke unexpectedly has a DT_NEEDED on HCOMM" >&2
  exit 3
fi

if ! ldd -r "${binary}" \
    >"${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke.ldd.txt" 2>&1; then
  cat "${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke.ldd.txt" >&2
  echo "ERROR: ldd could not load the direct smoke" >&2
  exit 3
fi
if grep -Eq 'not found|undefined symbol' \
    "${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke.ldd.txt"; then
  cat "${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke.ldd.txt" >&2
  echo "ERROR: direct smoke has unresolved runtime dependencies" >&2
  exit 3
fi

echo "KVAF_A5_DIRECT_URMA_BUILD_OK"
echo "BINARY=${binary}"
echo "HCOMM_DT_NEEDED=absent"
