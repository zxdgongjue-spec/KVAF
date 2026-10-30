# Shared toolchain discovery and flags extracted from the existing BatchGet/BatchPut kernels.
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
message(STATUS "AIV SIMT API root: ${KVAF_SIMT_API_DIR}")

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
)

set(KVAF_AIV_COMPILE_OPTIONS
  --cce-aicore-arch=dav-c310-vec --cce-aicore-input-parameter-size=8192 --cce-aicore-only
  -mllvm -cce-aicore-function-stack-size=0x4000 -mllvm -cce-aicore-stack-size=0x4000
  -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform
  -mllvm -cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all)
set(KVAF_AIV_LINK_OPTIONS -m aicorelinux -Ttext=0 -static)
