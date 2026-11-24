add_library(kvaf_access_operator INTERFACE)

if(BUILD_TESTING)
  add_executable(kvaf_aicpu_get_window_test ../tests/aicpu_get_window_test.cc)
  target_link_libraries(kvaf_aicpu_get_window_test PRIVATE kvaf_common)
  add_test(NAME aicpu_get_window COMMAND kvaf_aicpu_get_window_test)
  add_executable(kvaf_aicpu_scatter_window_test ../tests/aicpu_scatter_window_test.cc
    algorithms/get/scatter/urma_aicpu/kernel/cpu_push_scatter.cc)
  target_include_directories(kvaf_aicpu_scatter_window_test PRIVATE algorithms include)
  target_link_libraries(kvaf_aicpu_scatter_window_test PRIVATE kvaf_common pthread)
  add_test(NAME aicpu_scatter_window COMMAND kvaf_aicpu_scatter_window_test)
  add_executable(kvaf_aicpu_gather_window_test ../tests/aicpu_gather_window_test.cc
    algorithms/get/gather/urma_aicpu/kernel/gather_read.cc
    ../runtime/cpu_service/gather/cpu_gather_executor.cc)
  target_include_directories(kvaf_aicpu_gather_window_test PRIVATE
    ../tests/aicpu_gather_model algorithms include ../runtime/include)
  target_link_libraries(kvaf_aicpu_gather_window_test PRIVATE kvaf_common pthread)
  add_test(NAME aicpu_gather_window COMMAND kvaf_aicpu_gather_window_test)
  add_executable(kvaf_batch_get_window_model_test ../tests/batch_get_window_model_test.cc)
  target_include_directories(kvaf_batch_get_window_model_test PRIVATE
    ../tests/aiv_model include algorithms/get/direct/urma_aiv/include)
  add_test(NAME batch_get_window_model COMMAND kvaf_batch_get_window_model_test)
endif()

target_include_directories(kvaf_access_operator
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kvaf_access_operator INTERFACE kvaf_common)

target_compile_features(kvaf_access_operator INTERFACE cxx_std_17)

if((KVAF_ENABLE_AIV_BATCH_GET OR KVAF_ENABLE_AICPU_URMA OR KVAF_ENABLE_CPU_OFFLOAD OR KVAF_ENABLE_NPU_PULL) AND NOT ASCEND_CANN_PACKAGE_PATH)
  if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
  elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
  else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
  set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_CANN_PACKAGE_PATH}" CACHE PATH "CANN toolkit")
endif()
add_subdirectory(transports/ubmem)

if(KVAF_ENABLE_AIV_BATCH_GET OR KVAF_ENABLE_AICPU_CPU_PUSH OR KVAF_ENABLE_CPU_OFFLOAD OR KVAF_ENABLE_NPU_PULL)
  add_subdirectory(transports/direct_hccp)
endif()

if(KVAF_ENABLE_AIV_BATCH_GET OR KVAF_ENABLE_NPU_PULL)
  add_subdirectory(algorithms/get/direct)
endif()
if(KVAF_ENABLE_NPU_PULL)
  add_subdirectory(algorithms/get/raw_addr/urma_aiv)
endif()

if(KVAF_ENABLE_AIV_BATCH_GET)
  if(KVAF_ENABLE_AIV_BATCH_PUT)
    add_subdirectory(algorithms/put/urma_aiv)
  endif()

  add_library(kvaf_default_access_operator
    providers/urma_aiv/access_operator.cc
    providers/urma_aiv/execution_resources.cc
  )

  target_include_directories(kvaf_default_access_operator
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/get/direct/urma_aiv/include
      ${ASCEND_CANN_PACKAGE_PATH}/include
  )

  if(KVAF_ENABLE_AIV_BATCH_PUT)
    target_include_directories(kvaf_default_access_operator
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/algorithms/put/urma_aiv/include)
  endif()

  target_compile_definitions(kvaf_default_access_operator
    PUBLIC
      _GLIBCXX_USE_CXX11_ABI=0
  )

  target_link_libraries(kvaf_default_access_operator
    PUBLIC
      kvaf_access_operator
    PRIVATE
      kvaf_meta_manager
      kvaf_execute_batch_get_host
      ascendcl
  )

  if(KVAF_ENABLE_AIV_BATCH_PUT)
    target_link_libraries(kvaf_default_access_operator
      PRIVATE kvaf_execute_batch_put_host kvaf_direct_hccp_session)
    target_compile_definitions(kvaf_default_access_operator
      PRIVATE KVAF_ENABLE_AIV_BATCH_PUT=1)
  endif()

  target_link_directories(kvaf_default_access_operator
    PRIVATE
      ${ASCEND_CANN_PACKAGE_PATH}/lib64
  )

  target_compile_features(kvaf_default_access_operator PUBLIC cxx_std_17)
endif()

if(KVAF_ENABLE_AICPU_URMA)
  add_subdirectory(transports/aicpu_urma)
  add_subdirectory(algorithms/get/common/aicpu)
  add_subdirectory(algorithms/put/urma_aicpu)
  add_subdirectory(transports/aicpu_urma/kernel)
  add_library(kvaf_aicpu_access_operator STATIC
    providers/urma_aicpu/access_operator.cc
    providers/urma_aicpu/execution_resources.cc)
  target_include_directories(kvaf_aicpu_access_operator PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/providers/common")
  if(KVAF_ENABLE_AICPU_CPU_PUSH)
    target_compile_definitions(kvaf_aicpu_access_operator PRIVATE KVAF_ENABLE_AICPU_CPU_PUSH=1)
  endif()
  target_link_libraries(kvaf_aicpu_access_operator PUBLIC kvaf_access_operator PRIVATE
    kvaf_aicpu_batch_get_host kvaf_aicpu_batch_put_host)
endif()
