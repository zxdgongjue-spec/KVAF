set(KVAF_EXECUTE_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/execute_batch_put_aiv.asc")
set(KVAF_EXECUTE_KERNEL_TMP "${CMAKE_CURRENT_BINARY_DIR}/kvaf_execute_batch_put.tmp.o")
set(KVAF_EXECUTE_KERNEL_OBJECT "${CMAKE_CURRENT_BINARY_DIR}/kvaf_execute_batch_put_kernels.o")
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

set(KVAF_KERNEL_INCLUDES
  "-I${KVAF_KERNEL_OPERATOR_DIR}"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/interface"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/impl"
  "-I${KVAF_AIV_DEMO_INCLUDE_DIR}")

add_custom_command(
  OUTPUT "${KVAF_EXECUTE_KERNEL_OBJECT}"
  COMMAND "${KVAF_CCEC}" -c -x cce -O2 "${KVAF_EXECUTE_KERNEL_SOURCE}" -o "${KVAF_EXECUTE_KERNEL_TMP}"
          --cce-aicore-arch=dav-c310-vec --cce-aicore-input-parameter-size=8192 --cce-aicore-only
          -mllvm -cce-aicore-function-stack-size=0x4000 -mllvm -cce-aicore-stack-size=0x4000
          -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform
          -mllvm -cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all ${KVAF_KERNEL_INCLUDES}
  COMMAND "${KVAF_LLD}" -m aicorelinux -Ttext=0 "${KVAF_EXECUTE_KERNEL_TMP}" -static
          -o "${KVAF_EXECUTE_KERNEL_OBJECT}"
  DEPENDS "${KVAF_EXECUTE_KERNEL_SOURCE}"
          "${KVAF_AIV_DEMO_INCLUDE_DIR}/batch_put_abi.h"
          "${KVAF_AIV_DEMO_INCLUDE_DIR}/batch_put_internal.h"
          VERBATIM)
add_custom_target(kvaf_execute_batch_put_kernel ALL
  DEPENDS "${KVAF_EXECUTE_KERNEL_OBJECT}")
add_custom_command(TARGET kvaf_execute_batch_put_kernel POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${KVAF_AIV_DEMO_BIN_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${KVAF_EXECUTE_KERNEL_OBJECT}"
          "${KVAF_AIV_DEMO_BIN_DIR}/kvaf_execute_batch_put_kernels.o")
