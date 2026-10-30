add_library(kvaf_ubmem_operator STATIC access_operator.cc device_mapping.cc)
target_include_directories(kvaf_ubmem_operator PRIVATE
  "${ASCEND_CANN_PACKAGE_PATH}/include" "${PROJECT_SOURCE_DIR}/access_operator/demo/common/include")
target_link_directories(kvaf_ubmem_operator PUBLIC "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(kvaf_ubmem_operator PUBLIC kvaf_ubmem_hal kvaf_peer_link kvaf_meta_manager ascendcl runtime)
add_dependencies(kvaf_ubmem_operator kvaf_execute_batch_get_kernel)
if(KVAF_ENABLE_AIV_BATCH_PUT)
  add_dependencies(kvaf_ubmem_operator kvaf_execute_batch_put_kernel)
endif()

if(BUILD_TESTING)
  add_executable(kvaf_ubmem_test ${PROJECT_SOURCE_DIR}/tests/ubmem/npu_test.cc)
  target_include_directories(kvaf_ubmem_test PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
  target_link_libraries(kvaf_ubmem_test PRIVATE kvaf_npu_runtime kvaf_ubmem_operator)
  set_target_properties(kvaf_ubmem_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  add_executable(kvaf_ubmem_bandwidth ${PROJECT_SOURCE_DIR}/tests/ubmem/bandwidth.cc device_mapping.cc)
  target_include_directories(kvaf_ubmem_bandwidth PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include"
    "${PROJECT_SOURCE_DIR}/access_operator/demo/common/include")
  target_link_directories(kvaf_ubmem_bandwidth PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
  target_link_libraries(kvaf_ubmem_bandwidth PRIVATE kvaf_ubmem_hal kvaf_peer_link ascendcl runtime)
  add_dependencies(kvaf_ubmem_bandwidth kvaf_execute_batch_get_kernel)
  set_target_properties(kvaf_ubmem_bandwidth PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
