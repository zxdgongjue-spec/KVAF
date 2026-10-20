if(NOT ASCEND_CANN_PACKAGE_PATH)
  if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit")
  else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "CANN toolkit")
  endif()
endif()
set(KVAF_AICPU_CXX "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++" CACHE FILEPATH "AICPU cross compiler")
if(NOT EXISTS "${KVAF_AICPU_CXX}")
  message(FATAL_ERROR "AICPU cross compiler not found: ${KVAF_AICPU_CXX}")
endif()
find_path(KVAF_AICPU_UMDK_INCLUDE_DIR urma_api.h HINTS "${KVAF_URMA_ROOT}/include" "${CMAKE_SOURCE_DIR}/../third_party/umdk/include")
if(NOT KVAF_AICPU_UMDK_INCLUDE_DIR)
  message(FATAL_ERROR "Set KVAF_AICPU_UMDK_INCLUDE_DIR to the UMDK headers")
endif()
add_library(kvaf_aicpu_gather_host STATIC host.cc umdk_session.cc access_operator.cc)
target_include_directories(kvaf_aicpu_gather_host PRIVATE
  "${CMAKE_SOURCE_DIR}/access_operator/src" "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include"
  "${CMAKE_SOURCE_DIR}/runtime/include" "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_libraries(kvaf_aicpu_gather_host PUBLIC kvaf_access_operator PRIVATE kvaf_meta_manager kvaf_peer_link kvaf_gather_host_plan ascendcl)
target_link_directories(kvaf_aicpu_gather_host PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
set(kernel_dir "${CMAKE_CURRENT_BINARY_DIR}/aicpu_kernels_device")
set(kernel_so "${kernel_dir}/libkvaf_aicpu_urma.so")
find_path(KVAF_AICPU_UMDK_RUNTIME_DIR NAMES liburma.so.0 liburma.so HINTS "${KVAF_URMA_ROOT}/lib64" PATHS /usr/lib64 /usr/lib)
set(umdk_runtime_files)
if(KVAF_AICPU_UMDK_RUNTIME_DIR)
  file(GLOB umdk_runtime_files CONFIGURE_DEPENDS
    "${KVAF_AICPU_UMDK_RUNTIME_DIR}/liburma.so*"
    "${KVAF_AICPU_UMDK_RUNTIME_DIR}/liburma_common.so*"
    "${KVAF_AICPU_UMDK_RUNTIME_DIR}/urma/*.so*")
endif()
configure_file(kernel.json kernel.json COPYONLY)
add_custom_command(OUTPUT "${kernel_so}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${kernel_dir}"
  COMMAND "${KVAF_AICPU_CXX}" -std=c++17 -O2 -shared -fPIC
    -I "${CMAKE_CURRENT_SOURCE_DIR}" -I "${CMAKE_SOURCE_DIR}/access_operator/include"
    -I "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include" -I "${CMAKE_SOURCE_DIR}/runtime/include"
    -I "${CMAKE_SOURCE_DIR}/common/include" -I "${KVAF_AICPU_UMDK_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cc" "${CMAKE_CURRENT_SOURCE_DIR}/umdk_transport.cc"
    -ldl -pthread "-Wl,-rpath,$ORIGIN" -Wl,-z,defs -o "${kernel_so}"
  COMMAND "${CMAKE_COMMAND}" "-DUMDK_RUNTIME_DIR=${KVAF_AICPU_UMDK_RUNTIME_DIR}" "-DOUTPUT_DIR=${kernel_dir}/umdk_runtime"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/bundle_umdk.cmake"
  DEPENDS kernel.cc umdk_transport.cc gather_read.h umdk_gather.h umdk_control.h umdk_abi.h umdk_diagnostics.h gather_cache.h
    bundle_umdk.cmake ${umdk_runtime_files}
    "${CMAKE_SOURCE_DIR}/runtime/include/cpu_gather.h"
    "${CMAKE_SOURCE_DIR}/common/include/gather_span.h"
    "${CMAKE_SOURCE_DIR}/access_operator/include/gather_output.h"
    "${CMAKE_SOURCE_DIR}/access_operator/include/gather_read_profile.h"
    "${CMAKE_SOURCE_DIR}/access_operator/include/gather_read_storage.h"
    "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include/batch_get_abi.h"
  VERBATIM)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/kvaf-aicpu-urma.tar.gz"
  COMMAND "${CMAKE_COMMAND}" -E tar czf "${CMAKE_CURRENT_BINARY_DIR}/kvaf-aicpu-urma.tar.gz" aicpu_kernels_device
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" DEPENDS "${kernel_so}" VERBATIM)
add_custom_target(kvaf_aicpu_urma_kernel DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/kvaf-aicpu-urma.tar.gz")
