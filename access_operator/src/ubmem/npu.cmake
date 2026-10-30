add_library(kvaf_ubmem_operator STATIC access_operator.cc device_mapping.cc)
target_include_directories(kvaf_ubmem_operator PRIVATE
  "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_directories(kvaf_ubmem_operator PUBLIC "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(kvaf_ubmem_operator PUBLIC kvaf_ubmem_hal kvaf_peer_link kvaf_meta_manager ascendcl runtime)
add_dependencies(kvaf_ubmem_operator kvaf_execute_batch_get_kernel)
if(KVAF_ENABLE_AIV_BATCH_PUT)
  add_dependencies(kvaf_ubmem_operator kvaf_execute_batch_put_kernel)
endif()
