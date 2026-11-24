# Preserve the existing Host library and device package while organizing each
# algorithm's implementation independently. The shared entry ABI stays here.
get_filename_component(KVAF_AICPU_GET_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_subdirectory(host)
add_subdirectory("${KVAF_AICPU_GET_ROOT}/gather/urma_aicpu"
                 "${CMAKE_CURRENT_BINARY_DIR}/gather")
if(KVAF_ENABLE_AICPU_CPU_PUSH)
  add_subdirectory("${KVAF_AICPU_GET_ROOT}/scatter/urma_aicpu"
                   "${CMAKE_CURRENT_BINARY_DIR}/scatter")
endif()
add_subdirectory(kernel)
set(KVAF_AICPU_BATCH_GET_DEVICE_SOURCES ${KVAF_AICPU_BATCH_GET_DEVICE_SOURCES} PARENT_SCOPE)
set(KVAF_AICPU_BATCH_GET_DEVICE_HEADERS ${KVAF_AICPU_BATCH_GET_DEVICE_HEADERS} PARENT_SCOPE)
