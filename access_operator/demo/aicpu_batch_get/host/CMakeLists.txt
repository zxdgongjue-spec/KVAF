add_library(kvaf_aicpu_batch_get_host STATIC
  batch_get_host.cc execute_batch_get_loader.cc)
target_include_directories(kvaf_aicpu_batch_get_host PUBLIC
  "${CMAKE_SOURCE_DIR}/access_operator/demo"
  PRIVATE "${CMAKE_SOURCE_DIR}/access_operator/src" "${KVAF_AICPU_UMDK_INCLUDE_DIR}")
target_link_libraries(kvaf_aicpu_batch_get_host PUBLIC kvaf_umdk_host PRIVATE kvaf_meta_manager)
if(KVAF_ENABLE_AICPU_CPU_PUSH)
  target_sources(kvaf_aicpu_batch_get_host PRIVATE cpu_push_host.cc)
  target_link_libraries(kvaf_aicpu_batch_get_host PRIVATE kvaf_direct_hccp_session)
endif()
