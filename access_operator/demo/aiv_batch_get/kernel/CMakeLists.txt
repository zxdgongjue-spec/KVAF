set(KVAF_CCEC_DIR "${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin")

find_program(KVAF_CCEC NAMES ccec HINTS "${KVAF_CCEC_DIR}")
find_program(KVAF_LLD NAMES ld.lld HINTS "${KVAF_CCEC_DIR}")
if(NOT KVAF_CCEC OR NOT KVAF_LLD)
  message(FATAL_ERROR "A5 ccec/ld.lld not found under ${KVAF_CCEC_DIR}")
endif()

find_path(KVAF_KERNEL_OPERATOR_DIR NAMES kernel_operator.h HINTS
  "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface"
  "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw"
  "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/interface"
  "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw"
  "${ASCEND_CANN_PACKAGE_PATH}/include/tikcpp/tikcfw/interface"
  "${ASCEND_CANN_PACKAGE_PATH}/include/tikcpp/tikcfw")
if(NOT KVAF_KERNEL_OPERATOR_DIR)
  message(FATAL_ERROR "kernel_operator.h not found under ${ASCEND_CANN_PACKAGE_PATH}")
endif()

find_path(KVAF_SIMT_API_DIR NAMES simt_api/device_atomic_functions.h HINTS
  "$ENV{ASCEND_SIMT_API_DIR}"
  "${ASCEND_CANN_PACKAGE_PATH}/include"
  "${ASCEND_CANN_PACKAGE_PATH}/include/ascendc"
  "${ASCEND_CANN_PACKAGE_PATH}/compiler/include"
  "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include"
  "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/asc/include"
  "${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include"
  "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/include"
  "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib"
  "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw"
  "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/lib"
  "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw"
  "${ASCEND_CANN_PACKAGE_PATH}/include/tikcpp/tikcfw/lib"
  "${ASCEND_CANN_PACKAGE_PATH}/include/tikcpp/tikcfw")
if(NOT KVAF_SIMT_API_DIR)
  message(FATAL_ERROR
    "simt_api/device_atomic_functions.h was not found. Install the matching "
    "CANN asc-devkit package, or set ASCEND_SIMT_API_DIR to the directory "
    "that contains simt_api/device_atomic_functions.h. "
    "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")
endif()
message(STATUS "BatchGet SIMT API root: ${KVAF_SIMT_API_DIR}")

set(KVAF_KERNEL_INCLUDES
  "-I${KVAF_KERNEL_OPERATOR_DIR}"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl"
  "-I${KVAF_SIMT_API_DIR}"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/interface"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/impl"
  "-I${KVAF_AIV_DEMO_INCLUDE_DIR}")

# AscendC emits SDK globals (including g_tilingKey) in each translation unit.
# Link each kernel separately; combining the two objects gives duplicate symbols.
function(kvaf_compile_batch_get_kernel stem source)
  set(object "${CMAKE_CURRENT_BINARY_DIR}/${stem}_kernels.o")
  set(temporary "${CMAKE_CURRENT_BINARY_DIR}/${stem}.tmp.o")
  set(installed "${KVAF_AIV_DEMO_BIN_DIR}/${stem}_kernels.o")
  add_custom_command(
    OUTPUT "${object}" "${installed}"
    COMMAND "${KVAF_CCEC}" -c -x cce -O2 "${CMAKE_CURRENT_SOURCE_DIR}/${source}" -o "${temporary}"
            --cce-aicore-arch=dav-c310-vec --cce-aicore-input-parameter-size=8192 --cce-aicore-only
            -mllvm -cce-aicore-function-stack-size=0x4000 -mllvm -cce-aicore-stack-size=0x4000
            -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform
            -mllvm -cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all ${KVAF_KERNEL_INCLUDES}
    COMMAND "${KVAF_LLD}" -m aicorelinux -Ttext=0 "${temporary}" -static -o "${object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${KVAF_AIV_DEMO_BIN_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${object}" "${installed}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
            "${KVAF_AIV_DEMO_INCLUDE_DIR}/batch_get_abi.h"
            "${KVAF_SIMT_API_DIR}/simt_api/device_atomic_functions.h"
            ${ARGN}
    VERBATIM)
  set(KVAF_BATCH_GET_KERNEL_OUTPUTS ${KVAF_BATCH_GET_KERNEL_OUTPUTS}
      "${object}" "${installed}" PARENT_SCOPE)
endfunction()

kvaf_compile_batch_get_kernel(kvaf_execute_batch_get execute_batch_get_aiv.asc
  "${KVAF_AIV_DEMO_INCLUDE_DIR}/batch_get_schedule.h")
kvaf_compile_batch_get_kernel(kvaf_sequential_batch_get execute_sequential_batch_get_aiv.asc
  "${KVAF_AIV_DEMO_INCLUDE_DIR}/sequential_batch_get_abi.h")
add_custom_target(kvaf_execute_batch_get_kernel ALL DEPENDS ${KVAF_BATCH_GET_KERNEL_OUTPUTS})
