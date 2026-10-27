set(package_dir "${CMAKE_CURRENT_BINARY_DIR}/..")
set(kernel_dir "${package_dir}/aicpu_kernels_device")
set(kernel_so "${kernel_dir}/libkvaf_aicpu_urma.so")
set(cpu_push_options)
set(cpu_push_libraries)
set(cpu_push_dependencies)
set(cpu_push_commands)
set(KVAF_AICPU_CPU_PUSH_ENTRY "")
if(KVAF_ENABLE_AICPU_CPU_PUSH)
find_path(KVAF_AICPU_SECUREC_INCLUDE_DIR securec.h
  PATHS "${ASCEND_CANN_PACKAGE_PATH}/include" "${ASCEND_CANN_PACKAGE_PATH}/include/experiment"
        "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include" NO_DEFAULT_PATH)
find_file(KVAF_AICPU_SECUREC_LIBRARY NAMES libc_sec.so libsecurec.so
  PATHS "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/devlib"
        "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64"
        "${ASCEND_CANN_PACKAGE_PATH}/lib64/device" "${ASCEND_CANN_PACKAGE_PATH}/lib64"
  NO_DEFAULT_PATH)
if(NOT KVAF_AICPU_SECUREC_INCLUDE_DIR OR NOT KVAF_AICPU_SECUREC_LIBRARY)
  message(FATAL_ERROR "Set KVAF_AICPU_SECUREC_INCLUDE_DIR and KVAF_AICPU_SECUREC_LIBRARY for AICPU memcpy_sp")
endif()
file(READ "${KVAF_AICPU_SECUREC_LIBRARY}" securec_elf_header OFFSET 0 LIMIT 20 HEX)
if(NOT securec_elf_header MATCHES "^7f454c46" OR NOT securec_elf_header MATCHES "(b700|00b7)$")
  message(FATAL_ERROR "KVAF_AICPU_SECUREC_LIBRARY must be an AArch64 ELF library")
endif()
get_filename_component(securec_library_name "${KVAF_AICPU_SECUREC_LIBRARY}" NAME)
set(cpu_push_options -DKVAF_ENABLE_AICPU_CPU_PUSH=1 -I "${KVAF_AICPU_SECUREC_INCLUDE_DIR}")
set(cpu_push_libraries "${KVAF_AICPU_SECUREC_LIBRARY}")
set(cpu_push_dependencies "${KVAF_AICPU_SECUREC_LIBRARY}" "${KVAF_AICPU_SECUREC_INCLUDE_DIR}/securec.h")
set(cpu_push_commands COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${KVAF_AICPU_SECUREC_LIBRARY}" "${kernel_dir}/${securec_library_name}")
set(KVAF_AICPU_CPU_PUSH_ENTRY [=[  "KvafAicpuCpuPushScatterA12V24": {
    "opInfo": {
      "functionName": "KvafAicpuCpuPushScatterA12V24",
      "kernelSo": "libkvaf_aicpu_urma.so",
      "opKernelLib": "AICPUKernel"
    }
  },]=])
endif()
find_path(KVAF_AICPU_UMDK_RUNTIME_DIR NAMES liburma.so.0 liburma.so HINTS "${KVAF_URMA_ROOT}/lib64" PATHS /usr/lib64 /usr/lib)
set(umdk_runtime_files)
if(KVAF_AICPU_UMDK_RUNTIME_DIR)
  file(GLOB umdk_runtime_files CONFIGURE_DEPENDS
    "${KVAF_AICPU_UMDK_RUNTIME_DIR}/liburma.so*"
    "${KVAF_AICPU_UMDK_RUNTIME_DIR}/liburma_common.so*"
    "${KVAF_AICPU_UMDK_RUNTIME_DIR}/urma/*.so*")
endif()
configure_file(../package/kernel.json.in "${package_dir}/kernel.json" @ONLY)
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
  "${CMAKE_SOURCE_DIR}/common/include/cpu_push_get_abi.h"
  "${CMAKE_SOURCE_DIR}/common/include/cpu_push_get_layout.h"
  "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include/batch_get_abi.h")
set(cpu_push_objects)
if(KVAF_ENABLE_AICPU_CPU_PUSH)
  # Optimize scatter separately from the UMDK/session sources.
  set(cpu_push_source "${CMAKE_SOURCE_DIR}/access_operator/demo/aicpu_batch_get/kernel/cpu_push_scatter.cc")
  set(cpu_push_object "${CMAKE_CURRENT_BINARY_DIR}/cpu_push_scatter.o")
  list(REMOVE_ITEM device_sources "${cpu_push_source}")
  add_custom_command(OUTPUT "${cpu_push_object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND "${KVAF_AICPU_CXX}" -std=c++17 -O3 -fPIC -c -DKVAF_CPU_PUSH_DEVICE_KERNEL=1
      -I "${CMAKE_CURRENT_SOURCE_DIR}"
      -I "${CMAKE_SOURCE_DIR}/access_operator/include"
      -I "${CMAKE_SOURCE_DIR}/access_operator/demo"
      -I "${CMAKE_SOURCE_DIR}/runtime/include"
      -I "${CMAKE_SOURCE_DIR}/common/include" -I "${KVAF_AICPU_UMDK_INCLUDE_DIR}"
      ${cpu_push_options} "${cpu_push_source}" -o "${cpu_push_object}"
    DEPENDS "${cpu_push_source}" ${device_headers} ${cpu_push_dependencies}
    VERBATIM)
  list(APPEND cpu_push_objects "${cpu_push_object}")
endif()
add_custom_command(OUTPUT "${kernel_so}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${kernel_dir}"
  COMMAND "${KVAF_AICPU_CXX}" -std=c++17 -O2 -shared -fPIC
    -I "${CMAKE_CURRENT_SOURCE_DIR}"
    -I "${CMAKE_SOURCE_DIR}/access_operator/include"
    -I "${CMAKE_SOURCE_DIR}/access_operator/demo"
    -I "${CMAKE_SOURCE_DIR}/runtime/include"
    -I "${CMAKE_SOURCE_DIR}/common/include" -I "${KVAF_AICPU_UMDK_INCLUDE_DIR}"
    ${cpu_push_options} ${device_sources} ${cpu_push_objects} ${cpu_push_libraries}
    -ldl -pthread "-Wl,-rpath,$ORIGIN" -Wl,-z,defs -o "${kernel_so}"
  ${cpu_push_commands}
  COMMAND "${CMAKE_COMMAND}" "-DUMDK_RUNTIME_DIR=${KVAF_AICPU_UMDK_RUNTIME_DIR}" "-DOUTPUT_DIR=${kernel_dir}/umdk_runtime"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/../package/bundle_umdk.cmake"
  DEPENDS ${device_sources} ${cpu_push_objects} ${device_headers} ${cpu_push_dependencies}
    "${CMAKE_CURRENT_SOURCE_DIR}/../package/bundle_umdk.cmake" ${umdk_runtime_files}
  VERBATIM)
add_custom_command(OUTPUT "${package_dir}/kvaf-aicpu-urma.tar.gz"
  COMMAND "${CMAKE_COMMAND}" -E tar czf "${package_dir}/kvaf-aicpu-urma.tar.gz" aicpu_kernels_device
  WORKING_DIRECTORY "${package_dir}" DEPENDS "${kernel_so}" VERBATIM)
add_custom_target(kvaf_aicpu_urma_kernel DEPENDS "${package_dir}/kvaf-aicpu-urma.tar.gz")
