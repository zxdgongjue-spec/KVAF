set(package_dir "${CMAKE_CURRENT_BINARY_DIR}/..")
set(kernel_dir "${package_dir}/aicpu_kernels_device")
set(kernel_so "${kernel_dir}/libkvaf_aicpu_urma.so")
find_path(KVAF_AICPU_UMDK_RUNTIME_DIR NAMES liburma.so.0 liburma.so HINTS "${KVAF_URMA_ROOT}/lib64" PATHS /usr/lib64 /usr/lib)
set(umdk_runtime_files)
if(KVAF_AICPU_UMDK_RUNTIME_DIR)
  file(GLOB umdk_runtime_files CONFIGURE_DEPENDS
    "${KVAF_AICPU_UMDK_RUNTIME_DIR}/liburma.so*"
    "${KVAF_AICPU_UMDK_RUNTIME_DIR}/liburma_common.so*"
    "${KVAF_AICPU_UMDK_RUNTIME_DIR}/urma/*.so*")
endif()
configure_file(../package/kernel.json "${package_dir}/kernel.json" COPYONLY)
# One SO keeps the transport/session state shared by both demos.
set(device_sources "${CMAKE_CURRENT_SOURCE_DIR}/session.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/transfer.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/control_kernel.cc"
  ${KVAF_AICPU_BATCH_GET_DEVICE_SOURCES} ${KVAF_AICPU_BATCH_PUT_DEVICE_SOURCES})
set(device_headers "${CMAKE_CURRENT_SOURCE_DIR}/transport_internal.h"
  ${KVAF_AICPU_BATCH_GET_DEVICE_HEADERS} ${KVAF_AICPU_BATCH_PUT_DEVICE_HEADERS}
  "${CMAKE_SOURCE_DIR}/access_operator/include/aicpu_urma/control_abi.h"
  "${CMAKE_SOURCE_DIR}/access_operator/include/aicpu_urma/transport.h"
  "${CMAKE_SOURCE_DIR}/access_operator/include/aicpu_urma/cache.h"
  "${CMAKE_SOURCE_DIR}/common/include/urma_transport.h"
  "${CMAKE_SOURCE_DIR}/runtime/include/cpu_gather.h"
  "${CMAKE_SOURCE_DIR}/common/include/gather_span.h"
  "${CMAKE_SOURCE_DIR}/common/include/gather_workspace.h"
  "${CMAKE_SOURCE_DIR}/common/include/batch_get_profile.h"
  "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include/batch_get_abi.h")
add_custom_command(OUTPUT "${kernel_so}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${kernel_dir}"
  COMMAND "${KVAF_AICPU_CXX}" -std=c++17 -O2 -shared -fPIC
    -I "${CMAKE_CURRENT_SOURCE_DIR}"
    -I "${CMAKE_SOURCE_DIR}/access_operator/include"
    -I "${CMAKE_SOURCE_DIR}/access_operator/demo"
    -I "${CMAKE_SOURCE_DIR}/runtime/include"
    -I "${CMAKE_SOURCE_DIR}/common/include" -I "${KVAF_AICPU_UMDK_INCLUDE_DIR}"
    ${device_sources} -ldl -pthread "-Wl,-rpath,$ORIGIN" -Wl,-z,defs -o "${kernel_so}"
  COMMAND "${CMAKE_COMMAND}" "-DUMDK_RUNTIME_DIR=${KVAF_AICPU_UMDK_RUNTIME_DIR}" "-DOUTPUT_DIR=${kernel_dir}/umdk_runtime"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/../package/bundle_umdk.cmake"
  DEPENDS ${device_sources} ${device_headers} "${CMAKE_CURRENT_SOURCE_DIR}/../package/bundle_umdk.cmake" ${umdk_runtime_files}
  VERBATIM)
add_custom_command(OUTPUT "${package_dir}/kvaf-aicpu-urma.tar.gz"
  COMMAND "${CMAKE_COMMAND}" -E tar czf "${package_dir}/kvaf-aicpu-urma.tar.gz" aicpu_kernels_device
  WORKING_DIRECTORY "${package_dir}" DEPENDS "${kernel_so}" VERBATIM)
add_custom_target(kvaf_aicpu_urma_kernel DEPENDS "${package_dir}/kvaf-aicpu-urma.tar.gz")
