get_filename_component(KVAF_ACCESS_OPERATOR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(KVAF_RAW_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(KVAF_DIRECT_INCLUDE_DIR "${KVAF_ACCESS_OPERATOR_SOURCE_DIR}/algorithms/get/direct/urma_aiv/include")
set(KVAF_RAW_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernel/execute_batch_get_raw_addr_aiv.asc")
set(KVAF_RAW_KERNEL_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/kvaf_execute_batch_get_raw_addr_kernels.o")
set(KVAF_RAW_KERNEL_TMP "${CMAKE_CURRENT_BINARY_DIR}/kvaf_execute_batch_get_raw_addr.tmp.o")
include("${KVAF_ACCESS_OPERATOR_SOURCE_DIR}/cmake/aiv_toolchain.cmake")
list(APPEND KVAF_KERNEL_INCLUDES "-I${KVAF_RAW_INCLUDE_DIR}" "-I${KVAF_DIRECT_INCLUDE_DIR}")
add_custom_command(OUTPUT "${KVAF_RAW_KERNEL_OBJECT}"
  COMMAND "${KVAF_CCEC}" -c -x cce -O2 "${KVAF_RAW_KERNEL_SOURCE}" -o "${KVAF_RAW_KERNEL_TMP}"
          ${KVAF_AIV_COMPILE_OPTIONS} ${KVAF_KERNEL_INCLUDES}
  COMMAND "${KVAF_LLD}" ${KVAF_AIV_LINK_OPTIONS} "${KVAF_RAW_KERNEL_TMP}" -o "${KVAF_RAW_KERNEL_OBJECT}"
  DEPENDS "${KVAF_RAW_KERNEL_SOURCE}" "${KVAF_RAW_INCLUDE_DIR}/batch_get_raw_addr_abi.h"
          "${KVAF_DIRECT_INCLUDE_DIR}/batch_get_abi.h" "${KVAF_DIRECT_INCLUDE_DIR}/batch_get_schedule.h"
  VERBATIM)
add_custom_target(kvaf_execute_batch_get_raw_addr_kernel ALL DEPENDS "${KVAF_RAW_KERNEL_OBJECT}")
add_custom_command(TARGET kvaf_execute_batch_get_raw_addr_kernel POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/bin"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${KVAF_RAW_KERNEL_OBJECT}" "${CMAKE_BINARY_DIR}/bin/")
add_library(kvaf_execute_batch_get_raw_addr_host SHARED host/execute_batch_get_raw_addr_loader.cpp)
target_include_directories(kvaf_execute_batch_get_raw_addr_host PUBLIC
  "${KVAF_RAW_INCLUDE_DIR}" "${KVAF_DIRECT_INCLUDE_DIR}" PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_directories(kvaf_execute_batch_get_raw_addr_host PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(kvaf_execute_batch_get_raw_addr_host PUBLIC kvaf_execute_batch_get_host PRIVATE ascendcl)
target_compile_options(kvaf_execute_batch_get_raw_addr_host PRIVATE -O2)
add_dependencies(kvaf_execute_batch_get_raw_addr_host kvaf_execute_batch_get_raw_addr_kernel)
