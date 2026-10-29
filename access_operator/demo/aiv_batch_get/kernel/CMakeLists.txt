set(KVAF_EXECUTE_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/execute_batch_get_aiv.asc")
set(KVAF_EXECUTE_KERNEL_TMP "${CMAKE_CURRENT_BINARY_DIR}/kvaf_execute_batch_get.tmp.o")
set(KVAF_EXECUTE_KERNEL_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/kvaf_execute_batch_get_kernels.o")
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/aiv_toolchain.cmake")
list(APPEND KVAF_KERNEL_INCLUDES "-I${KVAF_AIV_DEMO_INCLUDE_DIR}")

add_custom_command(
  OUTPUT "${KVAF_EXECUTE_KERNEL_OBJECT}"
  COMMAND "${KVAF_CCEC}" -c -x cce -O2 "${KVAF_EXECUTE_KERNEL_SOURCE}" -o "${KVAF_EXECUTE_KERNEL_TMP}"
          ${KVAF_AIV_COMPILE_OPTIONS} ${KVAF_KERNEL_INCLUDES}
  COMMAND "${KVAF_LLD}" ${KVAF_AIV_LINK_OPTIONS} "${KVAF_EXECUTE_KERNEL_TMP}"
          -o "${KVAF_EXECUTE_KERNEL_OBJECT}"
  DEPENDS "${KVAF_EXECUTE_KERNEL_SOURCE}"
          "${KVAF_AIV_DEMO_INCLUDE_DIR}/batch_get_abi.h"
          "${KVAF_AIV_DEMO_INCLUDE_DIR}/batch_get_schedule.h"
          "${KVAF_SIMT_API_DIR}/simt_api/device_atomic_functions.h"
  VERBATIM)
add_custom_target(kvaf_execute_batch_get_kernel ALL DEPENDS "${KVAF_EXECUTE_KERNEL_OBJECT}")
add_custom_command(TARGET kvaf_execute_batch_get_kernel POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${KVAF_AIV_DEMO_BIN_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${KVAF_EXECUTE_KERNEL_OBJECT}"
          "${KVAF_AIV_DEMO_BIN_DIR}/kvaf_execute_batch_get_kernels.o")
