target_sources(kvaf_aicpu_batch_get_host PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/host/cpu_push_host.cc")
target_link_libraries(kvaf_aicpu_batch_get_host PRIVATE kvaf_direct_hccp_session)
