set(aiv_copy_object "${CMAKE_CURRENT_BINARY_DIR}/kvaf_aiv_sq_copy.o")
set(aiv_copy_temporary "${CMAKE_CURRENT_BINARY_DIR}/kvaf_aiv_sq_copy.tmp.o")
set(aiv_copy_includes
  "-I${CMAKE_SOURCE_DIR}/access_operator/src"
  "-I${CMAKE_SOURCE_DIR}/access_operator/include"
  "-I${KVAF_KERNEL_OPERATOR_DIR}"
  "-I${KVAF_SIMT_API_DIR}"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/impl"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/interface"
  "-I${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/impl")
add_custom_command(OUTPUT "${aiv_copy_object}"
  COMMAND "${KVAF_CCEC}" -c -x cce -O2 "${CMAKE_CURRENT_SOURCE_DIR}/kernel.asc"
          -o "${aiv_copy_temporary}"
          --cce-aicore-arch=dav-c310-vec --cce-aicore-input-parameter-size=8192 --cce-aicore-only
          -mllvm -cce-aicore-function-stack-size=0x4000 -mllvm -cce-aicore-stack-size=0x4000
          -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform
          -mllvm -cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all ${aiv_copy_includes}
  COMMAND "${KVAF_LLD}" -m aicorelinux -Ttext=0 "${aiv_copy_temporary}" -static
          -o "${aiv_copy_object}"
  DEPENDS kernel.asc "${CMAKE_SOURCE_DIR}/access_operator/src/aicpu_urma/sq_copy_benchmark.h"
          "${CMAKE_SOURCE_DIR}/access_operator/include/gather_output.h" VERBATIM)
add_custom_target(kvaf_aiv_sq_copy_kernel DEPENDS "${aiv_copy_object}")
add_library(kvaf_aiv_sq_copy_host STATIC host.cc)
target_include_directories(kvaf_aiv_sq_copy_host PUBLIC
  "${CMAKE_SOURCE_DIR}/access_operator/src" "${CMAKE_SOURCE_DIR}/access_operator/include"
  "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_directories(kvaf_aiv_sq_copy_host PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(kvaf_aiv_sq_copy_host PRIVATE ascendcl)
target_compile_definitions(kvaf_aiv_sq_copy_host INTERFACE
  "KVAF_AIV_SQ_COPY_KERNEL_OBJECT=\"${aiv_copy_object}\"")
add_dependencies(kvaf_aiv_sq_copy_host kvaf_aiv_sq_copy_kernel)
