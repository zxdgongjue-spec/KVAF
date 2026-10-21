add_library(kvaf_aicpu_batch_get_host STATIC
  batch_get_host.cc execute_batch_get_loader.cc)
target_include_directories(kvaf_aicpu_batch_get_host PUBLIC
  "${CMAKE_SOURCE_DIR}/access_operator/demo"
  PRIVATE "${CMAKE_SOURCE_DIR}/access_operator/src")
target_link_libraries(kvaf_aicpu_batch_get_host PUBLIC kvaf_umdk_host PRIVATE kvaf_meta_manager)
