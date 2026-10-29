if(NOT ASCEND_CANN_PACKAGE_PATH)
  if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
  elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
  else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
endif()
find_path(KVAF_UBMEM_KERNEL_INCLUDE kernel_operator.h HINTS
  "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface"
  "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/interface"
  "${ASCEND_CANN_PACKAGE_PATH}/include/tikcpp/tikcfw/interface")
find_path(KVAF_UBMEM_SIMT_INCLUDE simt_api/device_atomic_functions.h HINTS
  "$ENV{ASCEND_SIMT_API_DIR}"
  "${ASCEND_CANN_PACKAGE_PATH}/include"
  "${ASCEND_CANN_PACKAGE_PATH}/include/ascendc"
  "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/lib"
  "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/lib"
  "${ASCEND_CANN_PACKAGE_PATH}/include/tikcpp/tikcfw/lib")
find_program(KVAF_UBMEM_CCEC ccec HINTS "${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin")
find_program(KVAF_UBMEM_LLD ld.lld HINTS "${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin")
if(NOT KVAF_UBMEM_KERNEL_INCLUDE OR NOT KVAF_UBMEM_SIMT_INCLUDE OR NOT KVAF_UBMEM_CCEC OR NOT KVAF_UBMEM_LLD)
  message(FATAL_ERROR "UBmem NPU test requires the Ascend950 CANN toolkit, asc-devkit SIMT headers, ccec, and ld.lld; set ASCEND_CANN_PACKAGE_PATH/ASCEND_SIMT_API_DIR")
endif()
set(ubmem_object "${CMAKE_BINARY_DIR}/bin/kvaf_ubmem_kernels.o")
set(ubmem_tmp "${CMAKE_CURRENT_BINARY_DIR}/kvaf_ubmem.tmp.o")
set(ubmem_includes "-I${PROJECT_SOURCE_DIR}/access_operator/include"
  "-I${KVAF_UBMEM_KERNEL_INCLUDE}" "-I${KVAF_UBMEM_SIMT_INCLUDE}")
foreach(root tools/tikcpp/tikcfw tikcpp/tikcfw include/tikcpp/tikcfw)
  foreach(sub "" /lib /interface /impl)
    list(APPEND ubmem_includes "-I${ASCEND_CANN_PACKAGE_PATH}/${root}${sub}")
  endforeach()
endforeach()
add_custom_command(OUTPUT "${ubmem_object}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/bin"
  COMMAND "${KVAF_UBMEM_CCEC}" -c -x cce -O2 "${CMAKE_CURRENT_SOURCE_DIR}/batch_kernel.asc" -o "${ubmem_tmp}"
    --cce-aicore-arch=dav-c310-vec --cce-aicore-input-parameter-size=8192 --cce-aicore-only
    -mllvm -cce-aicore-function-stack-size=0x4000 -mllvm -cce-aicore-stack-size=0x4000
    -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-jump-expand=true -std=c++17 -fstack-protector-all ${ubmem_includes}
  COMMAND "${KVAF_UBMEM_LLD}" -m aicorelinux -Ttext=0 "${ubmem_tmp}" -static -o "${ubmem_object}"
  DEPENDS batch_kernel.asc ${PROJECT_SOURCE_DIR}/access_operator/include/ubmem/batch_abi.h
  VERBATIM)
add_custom_target(kvaf_ubmem_kernel ALL DEPENDS "${ubmem_object}")
add_library(kvaf_ubmem_operator STATIC access_operator.cc)
target_include_directories(kvaf_ubmem_operator PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_directories(kvaf_ubmem_operator PUBLIC "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(kvaf_ubmem_operator PUBLIC kvaf_ubmem_hal kvaf_peer_link kvaf_meta_manager ascendcl)
add_dependencies(kvaf_ubmem_operator kvaf_ubmem_kernel)
add_executable(kvaf_ubmem_test ${PROJECT_SOURCE_DIR}/tests/ubmem/npu_test.cc)
target_include_directories(kvaf_ubmem_test PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_libraries(kvaf_ubmem_test PRIVATE kvaf_ubmem_operator kvaf_ubmem_cpu_runtime)
set_target_properties(kvaf_ubmem_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
