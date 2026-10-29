if(NOT ASCEND_CANN_PACKAGE_PATH)
  if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
  elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
  else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
endif()
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/aiv_toolchain.cmake")
list(APPEND KVAF_KERNEL_INCLUDES "-I${PROJECT_SOURCE_DIR}/access_operator/include")
set(ubmem_object "${CMAKE_BINARY_DIR}/bin/kvaf_ubmem_kernels.o")
set(ubmem_tmp "${CMAKE_CURRENT_BINARY_DIR}/kvaf_ubmem.tmp.o")
add_custom_command(OUTPUT "${ubmem_object}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/bin"
  COMMAND "${KVAF_CCEC}" -c -x cce -O2 "${CMAKE_CURRENT_SOURCE_DIR}/batch_kernel.asc" -o "${ubmem_tmp}"
    ${KVAF_AIV_COMPILE_OPTIONS} ${KVAF_KERNEL_INCLUDES}
  COMMAND "${KVAF_LLD}" ${KVAF_AIV_LINK_OPTIONS} "${ubmem_tmp}" -o "${ubmem_object}"
  DEPENDS batch_kernel.asc ${PROJECT_SOURCE_DIR}/access_operator/include/ubmem/batch_abi.h
  VERBATIM)
add_custom_target(kvaf_ubmem_kernel ALL DEPENDS "${ubmem_object}")
add_library(kvaf_ubmem_operator STATIC access_operator.cc)
target_include_directories(kvaf_ubmem_operator PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_directories(kvaf_ubmem_operator PUBLIC "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(kvaf_ubmem_operator PUBLIC kvaf_ubmem_hal kvaf_peer_link kvaf_meta_manager ascendcl runtime)
add_dependencies(kvaf_ubmem_operator kvaf_ubmem_kernel)
add_executable(kvaf_ubmem_test ${PROJECT_SOURCE_DIR}/tests/ubmem/npu_test.cc)
target_include_directories(kvaf_ubmem_test PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_libraries(kvaf_ubmem_test PRIVATE kvaf_ubmem_operator kvaf_ubmem_cpu_runtime)
set_target_properties(kvaf_ubmem_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
