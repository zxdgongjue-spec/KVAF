#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
source "${SCRIPT_DIR}/a5_direct_urma_legacy_guard.sh"
kvaf_configure_direct_urma_state_paths "${ROOT_DIR}" || exit 2

: "${ASCEND_HOME_PATH:?source the CANN set_env.sh before building}"

KVAF_A5_DIRECT_BUILD_DIR=${KVAF_A5_DIRECT_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma"}
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
if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock is required to serialize direct URMA deploy/run" >&2
  exit 2
fi
if ! kvaf_check_legacy_direct_urma_state "${ROOT_DIR}"; then
  exit 4
fi
if [[ -L "${KVAF_GLOBAL_STATE_DIR}" ||
      ( -e "${KVAF_GLOBAL_STATE_DIR}" &&
        ! -d "${KVAF_GLOBAL_STATE_DIR}" ) ]]; then
  echo "ERROR: global KVAF state path must be a real directory" >&2
  exit 4
fi
mkdir -p "${KVAF_GLOBAL_STATE_DIR}"
chmod 0700 "${KVAF_GLOBAL_STATE_DIR}"
if [[ -L "${KVAF_DIRECT_STATE_DIR}" ||
      ( -e "${KVAF_DIRECT_STATE_DIR}" &&
        ! -d "${KVAF_DIRECT_STATE_DIR}" ) ]]; then
  echo "ERROR: direct URMA state path must be a real directory" >&2
  exit 4
fi
mkdir -p "${KVAF_DIRECT_STATE_DIR}" \
  "$(dirname "${KVAF_DIRECT_DEPLOY_RUN_LOCK}")" \
  "$(dirname "${KVAF_DIRECT_ACTIVE_SESSION_LOCK}")"
chmod 0700 "${KVAF_DIRECT_STATE_DIR}"
for guarded_path in \
  "${KVAF_DIRECT_DEPLOY_RUN_LOCK}" \
  "${KVAF_DIRECT_ACTIVE_SESSION_LOCK}" \
  "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" \
  "${KVAF_DIRECT_QUARANTINE_FILE}"
do
  if [[ -L "${guarded_path}" ]]; then
    echo "ERROR: persistent state path must not be a symlink: ${guarded_path}" >&2
    exit 4
  fi
done
exec 9>"${KVAF_DIRECT_DEPLOY_RUN_LOCK}"
if ! flock -w 30 9; then
  echo "ERROR: another direct URMA deploy/run owns ${KVAF_DIRECT_DEPLOY_RUN_LOCK}" >&2
  exit 2
fi
if [[ -e "${KVAF_DIRECT_QUARANTINE_FILE}" ||
      -L "${KVAF_DIRECT_QUARANTINE_FILE}" ||
      -e "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ||
      -L "${KVAF_DIRECT_ACTIVE_SESSION_FILE}" ]]; then
  echo "ERROR: direct URMA active/quarantine state blocks artifact mutation" >&2
  exit 4
fi
if [[ -L "${KVAF_DIRECT_ACTIVE_SESSION_LOCK}" ]]; then
  echo "ERROR: active-session lock path must not be a symlink" >&2
  exit 4
fi
exec 8>"${KVAF_DIRECT_ACTIVE_SESSION_LOCK}"
if ! flock -n 8; then
  echo "ERROR: a live direct URMA worker blocks artifact mutation" >&2
  exit 4
fi

for command in awk cmp find realpath sha256sum sort xargs; do
  command -v "${command}" >/dev/null 2>&1 || {
    echo "ERROR: ${command} is required" >&2
    exit 2
  }
done

mkdir -p "${KVAF_A5_DIRECT_BUILD_DIR}"
build_manifest="${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke.manifest"
source_manifest="${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke-source.sha256"
# Invalidate the runnable proof before configuration. A failed rebuild must
# never leave an older binary+manifest pair eligible for hardware execution.
rm -f "${build_manifest}" "${source_manifest}"

write_direct_source_manifest()
{
  local output=$1
  {
    printf '%s\0' "${ROOT_DIR}/CMakeLists.txt"
    find "${ROOT_DIR}/include" \
      -type f -name '*.h' -print0
    find "${ROOT_DIR}/access_operator/include" \
      -type f -name '*.h' -print0
    find "${ROOT_DIR}/access_operator/a5" \
      -type f \( -name '*.h' -o -name '*.c' -o -name '*.cpp' \
      -o -name 'CMakeLists.txt' \) -print0
  } | LC_ALL=C sort -zu | xargs -0 sha256sum >"${output}"
}

write_direct_source_manifest \
  "${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke-source.before.sha256"

if [[ -z "${KVAF_ASC_DEVKIT_PATH}" &&
      ( ! -r "${ASCEND_HOME_PATH}/include/kernel_operator.h" ||
        ! -r "${ASCEND_HOME_PATH}/include/simt_api/asc_simt.h" ) &&
      -r "${ROOT_DIR}/../asc-devkit/include/kernel_operator.h" &&
      -r "${ROOT_DIR}/../asc-devkit/include/simt_api/asc_simt.h" ]]; then
  KVAF_ASC_DEVKIT_PATH="${ROOT_DIR}/../asc-devkit"
fi

cmake \
  -U KVAF_ACL_INCLUDE_DIR \
  -U KVAF_RUNTIME_INCLUDE_DIR \
  -U KVAF_ASCENDCL_LIBRARY \
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

write_direct_source_manifest \
  "${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke-source.after.sha256"
if ! cmp -s \
    "${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke-source.before.sha256" \
    "${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke-source.after.sha256"; then
  echo "ERROR: direct smoke source changed during the build" >&2
  exit 3
fi
mv -f "${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke-source.after.sha256" \
  "${source_manifest}"
rm -f "${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke-source.before.sha256"

binary="${KVAF_A5_DIRECT_BUILD_DIR}/access_operator/a5/kvaf_a5_aiv_direct_urma_h2d_smoke"
if [[ ! -x "${binary}" ]]; then
  echo "ERROR: expected direct smoke binary is missing: ${binary}" >&2
  exit 3
fi

cmake_cache="${KVAF_A5_DIRECT_BUILD_DIR}/CMakeCache.txt"
cmake_cache_value()
{
  local key=$1
  awk -F= -v key="${key}" '
    $1 ~ ("^" key ":[A-Z]+$") { value=substr($0, index($0, "=") + 1); count++ }
    END { if (count != 1 || value == "") exit 1; print value }
  ' "${cmake_cache}"
}
runtime_include=$(cmake_cache_value KVAF_RUNTIME_INCLUDE_DIR) || {
  echo "ERROR: CMake did not record the runtime include it compiled" >&2
  exit 3
}
acl_include=$(cmake_cache_value KVAF_ACL_INCLUDE_DIR) || {
  echo "ERROR: CMake did not record the ACL include it compiled" >&2
  exit 3
}
ascendcl_library=$(cmake_cache_value KVAF_ASCENDCL_LIBRARY) || {
  echo "ERROR: CMake did not record the ascendcl library it linked" >&2
  exit 3
}
runtime_header=$(realpath -e \
  "${runtime_include}/runtime/rt_external_kernel.h")
acl_header=$(realpath -e "${acl_include}/acl/acl.h")
ascendcl_library=$(realpath -e "${ascendcl_library}")

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
resolved_ascendcl=$(awk '
  $1 ~ /^libascendcl\.so/ && $2 == "=>" { path=$3; count++ }
  END { if (count != 1 || path == "") exit 1; print path }
' "${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke.ldd.txt") || {
  echo "ERROR: ldd did not resolve exactly one libascendcl" >&2
  exit 3
}
if [[ "$(realpath -e "${resolved_ascendcl}")" != \
      "${ascendcl_library}" ]]; then
  echo "ERROR: CMake link-time and runtime libascendcl differ" >&2
  exit 3
fi

binary_sha=$(sha256sum "${binary}" | awk '{print $1}')
runtime_header_sha=$(sha256sum "${runtime_header}" | awk '{print $1}')
acl_header_sha=$(sha256sum "${acl_header}" | awk '{print $1}')
ascendcl_library_sha=$(sha256sum "${ascendcl_library}" | awk '{print $1}')
source_manifest_sha=$(sha256sum "${source_manifest}" | awk '{print $1}')
compiler_sha=$(sha256sum "${KVAF_BISHENG}" | awk '{print $1}')
cat >"${build_manifest}" <<EOF
schema=kvaf-a5-direct-smoke-build-v2
binary_path=${binary}
binary_sha256=${binary_sha}
runtime_header_path=${runtime_header}
runtime_header_sha256=${runtime_header_sha}
acl_header_path=${acl_header}
acl_header_sha256=${acl_header_sha}
ascendcl_library_path=${ascendcl_library}
ascendcl_library_sha256=${ascendcl_library_sha}
source_manifest_path=${source_manifest}
source_manifest_sha256=${source_manifest_sha}
compiler_path=${KVAF_BISHENG}
compiler_sha256=${compiler_sha}
ascend_home_path=${ASCEND_HOME_PATH}
EOF

echo "KVAF_A5_DIRECT_URMA_BUILD_OK"
echo "BINARY=${binary}"
echo "BUILD_MANIFEST=${build_manifest}"
echo "HCOMM_DT_NEEDED=absent"
