set(KVAF_AICPU_BATCH_GET_DEVICE_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/batch_get.cc"
  "${KVAF_AICPU_GET_ROOT}/gather/urma_aicpu/kernel/gather_read.cc")
if(KVAF_ENABLE_AICPU_CPU_PUSH)
  list(APPEND KVAF_AICPU_BATCH_GET_DEVICE_SOURCES
    "${KVAF_AICPU_GET_ROOT}/scatter/urma_aicpu/kernel/cpu_push_scatter.cc")
endif()
set(KVAF_AICPU_BATCH_GET_DEVICE_SOURCES ${KVAF_AICPU_BATCH_GET_DEVICE_SOURCES} PARENT_SCOPE)
set(KVAF_AICPU_BATCH_GET_DEVICE_HEADERS
  "${KVAF_AICPU_GET_ROOT}/gather/urma_aicpu/kernel/gather_read.h"
  "${KVAF_AICPU_GET_ROOT}/scatter/urma_aicpu/kernel/cpu_push_scatter.h"
  "${KVAF_AICPU_GET_ROOT}/scatter/urma_aicpu/kernel/cpu_push_pipeline.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../include/batch_get_abi.h"
  PARENT_SCOPE)
