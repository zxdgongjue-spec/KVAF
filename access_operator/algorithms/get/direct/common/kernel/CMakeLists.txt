set(KVAF_EXECUTE_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/execute_batch_get_kernels.asc")
set(KVAF_URMA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../urma_aiv/kernel/execute_batch_get_aiv.asc")
set(KVAF_EXECUTE_KERNEL_TMP "${CMAKE_CURRENT_BINARY_DIR}/kvaf_execute_batch_get.tmp.o")
set(KVAF_EXECUTE_KERNEL_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/kvaf_execute_batch_get_kernels.o")
include("${KVAF_ACCESS_OPERATOR_SOURCE_DIR}/cmake/aiv_toolchain.cmake")
list(APPEND KVAF_KERNEL_INCLUDES "-I${KVAF_AIV_ALGORITHM_INCLUDE_DIR}")

set(KVAF_UBMEM_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../ubmem_aiv/kernel/execute_batch_get_ubmem.asc")
set(KVAF_OPERATOR_INCLUDE "${KVAF_ACCESS_OPERATOR_SOURCE_DIR}/include")
list(APPEND KVAF_KERNEL_INCLUDES "-I${KVAF_OPERATOR_INCLUDE}")

add_custom_command(
  OUTPUT "${KVAF_EXECUTE_KERNEL_OBJECT}"
  COMMAND "${KVAF_CCEC}" -c -x cce -O2 "${KVAF_EXECUTE_KERNEL_SOURCE}" -o "${KVAF_EXECUTE_KERNEL_TMP}"
          ${KVAF_AIV_COMPILE_OPTIONS} ${KVAF_KERNEL_INCLUDES}
  COMMAND "${KVAF_LLD}" ${KVAF_AIV_LINK_OPTIONS} "${KVAF_EXECUTE_KERNEL_TMP}"
          -o "${KVAF_EXECUTE_KERNEL_OBJECT}"
  DEPENDS "${KVAF_EXECUTE_KERNEL_SOURCE}" "${KVAF_URMA_SOURCE}" "${KVAF_UBMEM_SOURCE}"
          "${KVAF_OPERATOR_INCLUDE}/ubmem/copy.h" "${KVAF_OPERATOR_INCLUDE}/ubmem/batch_abi.h"
          "${KVAF_AIV_ALGORITHM_INCLUDE_DIR}/batch_get_abi.h"
          "${KVAF_AIV_ALGORITHM_INCLUDE_DIR}/batch_get_schedule.h"
          "${KVAF_SIMT_API_DIR}/simt_api/device_atomic_functions.h"
  VERBATIM)
add_custom_target(kvaf_execute_batch_get_kernel ALL DEPENDS "${KVAF_EXECUTE_KERNEL_OBJECT}")
add_custom_command(TARGET kvaf_execute_batch_get_kernel POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${KVAF_AIV_ALGORITHM_BIN_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${KVAF_EXECUTE_KERNEL_OBJECT}"
          "${KVAF_AIV_ALGORITHM_BIN_DIR}/kvaf_execute_batch_get_kernels.o")

# Scatter and Gather each emit AscendC globals. Keep their objects separate
# from the main URMA/UBmem binary and from each other.
get_filename_component(KVAF_AIV_COMMON_INCLUDE_DIR
  "${KVAF_ACCESS_OPERATOR_SOURCE_DIR}/../common/include" ABSOLUTE)
foreach(mode scatter gather)
  set(stem kvaf_${mode}_batch_get)
  set(sequential_kernel_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../urma_aiv/kernel")
  set(source "${sequential_kernel_dir}/execute_${mode}_batch_get_aiv.asc")
  set(object "${CMAKE_CURRENT_BINARY_DIR}/${stem}_kernels.o")
  set(temporary "${CMAKE_CURRENT_BINARY_DIR}/${stem}.tmp.o")
  set(installed "${KVAF_AIV_ALGORITHM_BIN_DIR}/${stem}_kernels.o")
  set(mode_dependencies)
  if(mode STREQUAL "scatter")
    list(APPEND mode_dependencies
      "${sequential_kernel_dir}/scatter_mailbox_kernel.h"
      "${KVAF_SIMT_API_DIR}/simt_api/device_functions.h"
      "${KVAF_AIV_COMMON_INCLUDE_DIR}/scatter_batch_get_profile.h"
      "${KVAF_AIV_COMMON_INCLUDE_DIR}/cpu_push_get_abi.h")
  elseif(mode STREQUAL "gather")
    list(APPEND mode_dependencies
      "${KVAF_AIV_COMMON_INCLUDE_DIR}/gather_prefix_layout.h"
      "${KVAF_AIV_COMMON_INCLUDE_DIR}/gather_span.h")
  endif()
  add_custom_command(
    OUTPUT "${object}" "${installed}"
    COMMAND "${KVAF_CCEC}" -c -x cce -O2 "${source}" -o "${temporary}"
            ${KVAF_AIV_COMPILE_OPTIONS} ${KVAF_KERNEL_INCLUDES}
            "-I${KVAF_AIV_COMMON_INCLUDE_DIR}"
    COMMAND "${KVAF_LLD}" ${KVAF_AIV_LINK_OPTIONS} "${temporary}" -o "${object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${KVAF_AIV_ALGORITHM_BIN_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${object}" "${installed}"
    DEPENDS "${source}"
            "${KVAF_AIV_ALGORITHM_INCLUDE_DIR}/batch_get_abi.h"
            "${KVAF_AIV_ALGORITHM_INCLUDE_DIR}/${mode}_batch_get_abi.h"
            "${KVAF_AIV_ALGORITHM_INCLUDE_DIR}/sequential_batch_get_abi.h"
            "${sequential_kernel_dir}/batch_get_kernel_common.h"
            "${KVAF_SIMT_API_DIR}/simt_api/device_atomic_functions.h"
            ${mode_dependencies}
    VERBATIM)
  add_custom_target(${stem}_kernel DEPENDS "${object}" "${installed}")
endforeach()
add_custom_target(kvaf_batch_get_kernels ALL)
add_dependencies(kvaf_batch_get_kernels kvaf_execute_batch_get_kernel
  kvaf_scatter_batch_get_kernel kvaf_gather_batch_get_kernel)
