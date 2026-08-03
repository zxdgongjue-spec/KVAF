#!/usr/bin/env bash

set -Eeuo pipefail

kvaf_direct_build_mode=${1:-production}
if (( $# > 1 )) ||
   [[ "${kvaf_direct_build_mode}" != production &&
      "${kvaf_direct_build_mode}" != \
        cross-npu2-risk-offline-v1 &&
      "${kvaf_direct_build_mode}" != \
        cross-npu2-reuse-versioned-offline-v1 ]]; then
  echo "ERROR: invalid direct URMA build mode" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
source "${SCRIPT_DIR}/a5_direct_urma_legacy_guard.sh"
source "${SCRIPT_DIR}/a5_cross_npu2_reuse_state.sh"

kvaf_cross_npu_offline_build=0
kvaf_cross_npu_reuse_offline_build=0
if [[ "${kvaf_direct_build_mode}" == cross-npu2-risk-offline-v1 ]]; then
  kvaf_cross_npu_offline_build=1
  : "${KVAF_CROSS_NPU_SOURCE_RUN_ID:?set the pinned NPU7 run ID}"
  : "${KVAF_CROSS_NPU_SOURCE_QUARANTINE_SHA256:?set the pinned NPU7 quarantine SHA256}"
  if [[ "${KVAF_CROSS_NPU_RISK_ACCEPTANCE:-}" != \
        I_ACCEPT_UNKNOWN_CROSS_NPU_DMA_RISK ]]; then
    echo "ERROR: the offline risk build requires the exact acceptance phrase" >&2
    exit 2
  fi
  kvaf_configure_cross_npu2_risk_state_paths \
    "${ROOT_DIR}" "${KVAF_CROSS_NPU_SOURCE_RUN_ID}" || exit 2
elif [[ "${kvaf_direct_build_mode}" == \
        cross-npu2-reuse-versioned-offline-v1 ]]; then
  kvaf_cross_npu_reuse_offline_build=1
  if [[ "${KVAF_CROSS_NPU2_REUSE_RISK_ACCEPTANCE:-}" != \
        I_ACCEPT_UNKNOWN_NPU2_REUSE_AND_SHARED_UMMU_RISK ]]; then
    echo "ERROR: the versioned NPU2 reuse build requires the exact second-order risk acceptance phrase" >&2
    exit 2
  fi
  KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN=${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN:-}
  if [[ ! "${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}" =~ \
        ^[0-9a-f]{16}$ ]]; then
    echo "ERROR: KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN must be 16 lowercase hexadecimal digits" >&2
    exit 2
  fi
  readonly kvaf_reuse_source_run_id=530f5998a0b646d48db858170a9996ff
  readonly kvaf_reuse_prior_run_id=30ef7336d212459a9317fd6647f6fa15
  kvaf_configure_cross_npu2_reuse_state_paths \
    "${ROOT_DIR}" "${kvaf_reuse_source_run_id}" \
    "${kvaf_reuse_prior_run_id}" \
    "${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}" || exit 2
else
  kvaf_configure_direct_urma_state_paths "${ROOT_DIR}" || exit 2
fi

: "${ASCEND_HOME_PATH:?source the CANN set_env.sh before building}"

if (( kvaf_cross_npu_offline_build != 0 )); then
  : "${KVAF_CROSS_NPU_ARTIFACT_ID:?set a fresh 32-hex artifact ID}"
  if [[ ! "${KVAF_CROSS_NPU_ARTIFACT_ID}" =~ ^[0-9a-f]{32}$ ]]; then
    echo "ERROR: KVAF_CROSS_NPU_ARTIFACT_ID must be 32 lowercase hexadecimal digits" >&2
    exit 2
  fi
  kvaf_source_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD) || exit 2
  if [[ ! "${kvaf_source_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: could not pin the KVAF source commit" >&2
    exit 2
  fi
  kvaf_expected_build_dir="${KVAF_GLOBAL_STATE_DIR}/KVAF/experimental/artifacts/${kvaf_source_commit}/${KVAF_CROSS_NPU_ARTIFACT_ID}"
  if [[ -n "${KVAF_A5_DIRECT_BUILD_DIR+x}" &&
        "${KVAF_A5_DIRECT_BUILD_DIR}" != \
          "${kvaf_expected_build_dir}" ]]; then
    echo "ERROR: the risk build directory is source-owned" >&2
    echo "EXPECTED=${kvaf_expected_build_dir}" >&2
    echo "ACTUAL=${KVAF_A5_DIRECT_BUILD_DIR}" >&2
    exit 2
  fi
  KVAF_A5_DIRECT_BUILD_DIR=${kvaf_expected_build_dir}
elif (( kvaf_cross_npu_reuse_offline_build != 0 )); then
  kvaf_source_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD) || exit 2
  if [[ ! "${kvaf_source_commit}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: could not pin the KVAF source commit" >&2
    exit 2
  fi
  kvaf_reuse_artifact_root="${KVAF_GLOBAL_STATE_DIR}/KVAF/experimental/reuse-artifacts/${kvaf_source_commit}/${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}"
  kvaf_expected_build_dir="${kvaf_reuse_artifact_root}/direct"
  if [[ -n "${KVAF_A5_DIRECT_BUILD_DIR+x}" &&
        "${KVAF_A5_DIRECT_BUILD_DIR}" != \
          "${kvaf_expected_build_dir}" ]]; then
    echo "ERROR: the versioned NPU2 reuse direct-build directory is source-owned" >&2
    echo "EXPECTED=${kvaf_expected_build_dir}" >&2
    echo "ACTUAL=${KVAF_A5_DIRECT_BUILD_DIR}" >&2
    exit 2
  fi
  KVAF_A5_DIRECT_BUILD_DIR=${kvaf_expected_build_dir}
else
  KVAF_A5_DIRECT_BUILD_DIR=${KVAF_A5_DIRECT_BUILD_DIR:-"${ROOT_DIR}/build-a5-direct-urma"}
fi
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
if (( kvaf_cross_npu_offline_build == 0 &&
      kvaf_cross_npu_reuse_offline_build == 0 )); then
  if ! kvaf_check_legacy_direct_urma_state "${ROOT_DIR}"; then
    exit 4
  fi
else
  for kvaf_required_command in git python3 realpath sha256sum; do
    command -v "${kvaf_required_command}" >/dev/null 2>&1 || {
      echo "ERROR: ${kvaf_required_command} is required for the risk build" >&2
      exit 2
    }
  done
  if ! git -C "${ROOT_DIR}" diff --quiet --ignore-submodules -- ||
     ! git -C "${ROOT_DIR}" diff --cached --quiet --ignore-submodules --; then
    echo "ERROR: the cross-NPU risk build requires a clean tracked checkout" >&2
    exit 2
  fi
  if (( kvaf_cross_npu_offline_build != 0 )); then
    kvaf_cross_npu_validator="${SCRIPT_DIR}/validate_a5_cross_npu_risk_acceptance.py"
    if [[ -L "${kvaf_cross_npu_validator}" ||
          ! -f "${kvaf_cross_npu_validator}" ||
          ! -r "${kvaf_cross_npu_validator}" ]]; then
      echo "ERROR: cross-NPU acceptance validator is missing or unsafe" >&2
      exit 2
    fi
    python3 "${kvaf_cross_npu_validator}" \
      --quarantine /data/.kvaf-state/KVAF/direct-urma.quarantine \
      --quarantine-sha256 \
        "${KVAF_CROSS_NPU_SOURCE_QUARANTINE_SHA256}" \
      --run-id "${KVAF_CROSS_NPU_SOURCE_RUN_ID}" \
      --source-logical-device-id 7 \
      --result-root /data/.kvaf-state/KVAF/results
  else
    kvaf_cross_npu_validator="${SCRIPT_DIR}/validate_a5_cross_npu2_reuse_risk.py"
    if [[ -L "${kvaf_cross_npu_validator}" ||
          ! -f "${kvaf_cross_npu_validator}" ||
          ! -r "${kvaf_cross_npu_validator}" ]]; then
      echo "ERROR: NPU2 reuse acceptance validator is missing or unsafe" >&2
      exit 2
    fi
    python3 "${kvaf_cross_npu_validator}" \
      --new-artifact-token \
        "${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}"
  fi

  kvaf_cross_npu_experiment_root="${KVAF_GLOBAL_STATE_DIR}/KVAF/experimental"
  if [[ -L "${kvaf_cross_npu_experiment_root}" ||
        ( -e "${kvaf_cross_npu_experiment_root}" &&
          ! -d "${kvaf_cross_npu_experiment_root}" ) ]]; then
    echo "ERROR: cross-NPU experiment root must be a real directory" >&2
    exit 2
  fi
  mkdir -p "${kvaf_cross_npu_experiment_root}"
  chmod 0700 "${kvaf_cross_npu_experiment_root}"
  if (( kvaf_cross_npu_reuse_offline_build != 0 )); then
    kvaf_build_experiment_lock="${kvaf_cross_npu_experiment_root}/cross-npu2-reuse-build.lock"
  else
    kvaf_build_experiment_lock="${kvaf_cross_npu_experiment_root}/cross-npu-risk.lock"
  fi
  exec 10>"${kvaf_build_experiment_lock}"
  if ! flock -n 10; then
    echo "ERROR: a cross-NPU experiment is already active" >&2
    exit 5
  fi
  kvaf_artifact_parent=$(dirname "${KVAF_A5_DIRECT_BUILD_DIR}")
  if [[ -L "${kvaf_artifact_parent}" ||
        ( -e "${kvaf_artifact_parent}" &&
          ! -d "${kvaf_artifact_parent}" ) ]]; then
    echo "ERROR: experimental artifact parent must be a real directory" >&2
    exit 2
  fi
  mkdir -p "${kvaf_artifact_parent}"
  chmod 0700 "${kvaf_artifact_parent}"
  kvaf_artifact_parent=$(realpath -e "${kvaf_artifact_parent}") || exit 2
  if (( kvaf_cross_npu_reuse_offline_build != 0 )); then
    kvaf_expected_artifact_parent="${kvaf_reuse_artifact_root}"
  else
    kvaf_expected_artifact_parent="${KVAF_GLOBAL_STATE_DIR}/KVAF/experimental/artifacts/${kvaf_source_commit}"
  fi
  if [[ "${kvaf_artifact_parent}" != \
        "${kvaf_expected_artifact_parent}" ]]; then
    echo "ERROR: experimental artifact directory escaped its fixed root" >&2
    exit 2
  fi
  if [[ -e "${KVAF_A5_DIRECT_BUILD_DIR}" ||
        -L "${KVAF_A5_DIRECT_BUILD_DIR}" ]]; then
    if (( kvaf_cross_npu_reuse_offline_build != 0 )); then
      echo "ERROR: risk build requires a fresh direct artifact directory" >&2
    else
      echo "ERROR: risk build requires a fresh artifact ID" >&2
    fi
    echo "PATH=${KVAF_A5_DIRECT_BUILD_DIR}" >&2
    exit 2
  fi
  exec 11>"${KVAF_A5_DIRECT_BUILD_DIR}.lock"
  if ! flock -n 11; then
    echo "ERROR: another process owns this experimental artifact ID" >&2
    exit 5
  fi
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
  --target \
    kvaf_a5_aiv_direct_urma_h2d_smoke \
    kvaf_a5_aicpu_owner_probe \
    kvaf_a5_aicpu_runtime_urma_probe \
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
owner_probe="${KVAF_A5_DIRECT_BUILD_DIR}/access_operator/a5/kvaf_a5_aicpu_owner_probe"
runtime_urma_probe="${KVAF_A5_DIRECT_BUILD_DIR}/access_operator/a5/kvaf_a5_aicpu_runtime_urma_probe"
if [[ ! -x "${binary}" ]]; then
  echo "ERROR: expected direct smoke binary is missing: ${binary}" >&2
  exit 3
fi
if [[ ! -x "${owner_probe}" ]]; then
  echo "ERROR: expected AICPU owner probe is missing: ${owner_probe}" >&2
  exit 3
fi
if [[ ! -x "${runtime_urma_probe}" ]]; then
  echo "ERROR: expected AICPU runtime URMA probe is missing: ${runtime_urma_probe}" >&2
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
readelf -d "${owner_probe}" \
  >"${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-owner-probe.dynamic.txt"
readelf -d "${runtime_urma_probe}" \
  >"${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-runtime-urma-probe.dynamic.txt"
if grep -Fq 'libhcomm' \
    "${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke.dynamic.txt" ||
   grep -Fq 'libhcomm' \
    "${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-owner-probe.dynamic.txt" ||
   grep -Fq 'libhcomm' \
    "${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-runtime-urma-probe.dynamic.txt"; then
  echo "ERROR: direct A5 artifact unexpectedly has a DT_NEEDED on HCOMM" >&2
  exit 3
fi
if grep -Eq '\(NEEDED\).*(liburma|liburma-udma|libummu)' \
    "${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-runtime-urma-probe.dynamic.txt"; then
  echo "ERROR: Host runtime probe must not link URMA/UDMA/UMMU; the Device AICPU DSO dlopens its runtime by handle" >&2
  exit 3
fi
if ! ldd -r "${runtime_urma_probe}" \
    >"${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-runtime-urma-probe.ldd.txt" 2>&1; then
  cat "${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-runtime-urma-probe.ldd.txt" >&2
  echo "ERROR: ldd could not load the AICPU runtime URMA probe" >&2
  exit 3
fi
if grep -Eq 'not found|undefined symbol' \
    "${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-runtime-urma-probe.ldd.txt"; then
  cat "${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-runtime-urma-probe.ldd.txt" >&2
  echo "ERROR: AICPU runtime URMA probe has unresolved dependencies" >&2
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
if ! ldd -r "${owner_probe}" \
    >"${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-owner-probe.ldd.txt" 2>&1; then
  cat "${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-owner-probe.ldd.txt" >&2
  echo "ERROR: ldd could not load the AICPU owner probe" >&2
  exit 3
fi
if grep -Eq 'not found|undefined symbol' \
    "${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-owner-probe.ldd.txt"; then
  cat "${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-owner-probe.ldd.txt" >&2
  echo "ERROR: AICPU owner probe has unresolved runtime dependencies" >&2
  exit 3
fi
resolved_ascendcl=$(awk '
  $1 ~ /^libascendcl\.so/ && $2 == "=>" { path=$3; count++ }
  END { if (count != 1 || path == "") exit 1; print path }
' "${KVAF_A5_DIRECT_BUILD_DIR}/direct-smoke.ldd.txt") || {
  echo "ERROR: ldd did not resolve exactly one libascendcl" >&2
  exit 3
}
resolved_probe_ascendcl=$(awk '
  $1 ~ /^libascendcl\.so/ && $2 == "=>" { path=$3; count++ }
  END { if (count != 1 || path == "") exit 1; print path }
' "${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-owner-probe.ldd.txt") || {
  echo "ERROR: ldd did not resolve exactly one owner-probe libascendcl" >&2
  exit 3
}
resolved_runtime_probe_ascendcl=$(awk '
  $1 ~ /^libascendcl\.so/ && $2 == "=>" { path=$3; count++ }
  END { if (count != 1 || path == "") exit 1; print path }
' "${KVAF_A5_DIRECT_BUILD_DIR}/aicpu-runtime-urma-probe.ldd.txt") || {
  echo "ERROR: ldd did not resolve exactly one runtime-probe libascendcl" >&2
  exit 3
}
if [[ "$(realpath -e "${resolved_ascendcl}")" != \
      "${ascendcl_library}" ]]; then
  echo "ERROR: CMake link-time and runtime libascendcl differ" >&2
  exit 3
fi
if [[ "$(realpath -e "${resolved_probe_ascendcl}")" != \
      "${ascendcl_library}" ]]; then
  echo "ERROR: owner-probe link-time and runtime libascendcl differ" >&2
  exit 3
fi
if [[ "$(realpath -e "${resolved_runtime_probe_ascendcl}")" != \
      "${ascendcl_library}" ]]; then
  echo "ERROR: runtime-probe link-time and runtime libascendcl differ" >&2
  exit 3
fi

binary_sha=$(sha256sum "${binary}" | awk '{print $1}')
owner_probe_sha=$(sha256sum "${owner_probe}" | awk '{print $1}')
runtime_urma_probe_sha=$(sha256sum "${runtime_urma_probe}" | awk '{print $1}')
runtime_header_sha=$(sha256sum "${runtime_header}" | awk '{print $1}')
acl_header_sha=$(sha256sum "${acl_header}" | awk '{print $1}')
ascendcl_library_sha=$(sha256sum "${ascendcl_library}" | awk '{print $1}')
source_manifest_sha=$(sha256sum "${source_manifest}" | awk '{print $1}')
compiler_sha=$(sha256sum "${KVAF_BISHENG}" | awk '{print $1}')
cat >"${build_manifest}" <<EOF
schema=kvaf-a5-direct-smoke-build-v2
binary_path=${binary}
binary_sha256=${binary_sha}
owner_probe_path=${owner_probe}
owner_probe_sha256=${owner_probe_sha}
runtime_urma_probe_path=${runtime_urma_probe}
runtime_urma_probe_sha256=${runtime_urma_probe_sha}
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

if (( kvaf_cross_npu_offline_build != 0 )); then
  # Recheck the accepted source incident after the potentially long compile.
  # The build is CPU-only; a changed source worker/quarantine invalidates the
  # artifact before it can be selected by the hardware runner.
  python3 "${kvaf_cross_npu_validator}" \
    --quarantine /data/.kvaf-state/KVAF/direct-urma.quarantine \
    --quarantine-sha256 \
      "${KVAF_CROSS_NPU_SOURCE_QUARANTINE_SHA256}" \
    --run-id "${KVAF_CROSS_NPU_SOURCE_RUN_ID}" \
    --source-logical-device-id 7 \
    --result-root /data/.kvaf-state/KVAF/results
  printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-offline-artifact-v1' \
    "source_commit=${kvaf_source_commit}" \
    "artifact_id=${KVAF_CROSS_NPU_ARTIFACT_ID}" \
    "source_run_id=${KVAF_CROSS_NPU_SOURCE_RUN_ID}" \
    "source_quarantine_sha256=${KVAF_CROSS_NPU_SOURCE_QUARANTINE_SHA256}" \
    'artifact_mutation_policy=forbidden-during-hardware-run' \
    >"${KVAF_A5_DIRECT_BUILD_DIR}/cross-npu2-artifact.manifest"
  chmod 0400 "${KVAF_A5_DIRECT_BUILD_DIR}/cross-npu2-artifact.manifest"
  echo "KVAF_CROSS_NPU2_OFFLINE_ARTIFACT=${KVAF_A5_DIRECT_BUILD_DIR}"
elif (( kvaf_cross_npu_reuse_offline_build != 0 )); then
  kvaf_reuse_evidence=$(
    python3 "${kvaf_cross_npu_validator}" \
      --new-artifact-token \
        "${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}"
  ) || {
    echo "ERROR: NPU2 reuse evidence changed during the direct build" >&2
    exit 2
  }
  kvaf_reuse_evidence_sha=$(
    printf '%s\n' "${kvaf_reuse_evidence}" | sha256sum | awk '{print $1}'
  )
  kvaf_direct_profile="${kvaf_reuse_artifact_root}/experiment-direct.profile"
  if [[ -e "${kvaf_direct_profile}" || -L "${kvaf_direct_profile}" ]]; then
    echo "ERROR: versioned direct profile appeared concurrently" >&2
    exit 2
  fi
  kvaf_direct_profile_temporary="${kvaf_direct_profile}.tmp.$$"
  printf '%s\n' \
    'schema=kvaf-a5-cross-npu2-versioned-direct-v1' \
    "artifact_token=${KVAF_CROSS_NPU2_REUSE_ARTIFACT_TOKEN}" \
    "source_commit=${kvaf_source_commit}" \
    "direct_build_dir=${KVAF_A5_DIRECT_BUILD_DIR}" \
    "direct_binary=${binary}" \
    "direct_binary_sha256=${binary_sha}" \
    "owner_probe=${owner_probe}" \
    "owner_probe_sha256=${owner_probe_sha}" \
    "runtime_urma_probe=${runtime_urma_probe}" \
    "runtime_urma_probe_sha256=${runtime_urma_probe_sha}" \
    "direct_build_manifest=${build_manifest}" \
    "direct_build_manifest_sha256=$(sha256sum "${build_manifest}" | awk '{print $1}')" \
    "direct_source_manifest=${source_manifest}" \
    "direct_source_manifest_sha256=${source_manifest_sha}" \
    "reuse_evidence_sha256=${kvaf_reuse_evidence_sha}" \
    'artifact_mutation_policy=forbidden-during-hardware-run' \
    >"${kvaf_direct_profile_temporary}"
  chmod 0400 "${kvaf_direct_profile_temporary}"
  if ! ln "${kvaf_direct_profile_temporary}" "${kvaf_direct_profile}"; then
    echo "ERROR: could not publish the immutable versioned direct profile" >&2
    exit 2
  fi
  rm -f "${kvaf_direct_profile_temporary}"
  echo "KVAF_CROSS_NPU2_REUSE_DIRECT_ARTIFACT=${KVAF_A5_DIRECT_BUILD_DIR}"
  echo "KVAF_CROSS_NPU2_REUSE_DIRECT_PROFILE=${kvaf_direct_profile}"
fi

echo "KVAF_A5_DIRECT_URMA_BUILD_OK"
echo "BINARY=${binary}"
echo "OWNER_PROBE=${owner_probe}"
echo "RUNTIME_URMA_PROBE=${runtime_urma_probe}"
echo "BUILD_MANIFEST=${build_manifest}"
echo "HCOMM_DT_NEEDED=absent"
