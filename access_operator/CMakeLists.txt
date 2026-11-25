add_library(kvaf_access_operator INTERFACE)

target_include_directories(kvaf_access_operator
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kvaf_access_operator INTERFACE kvaf_common)

target_compile_features(kvaf_access_operator INTERFACE cxx_std_17)

if(BUILD_TESTING AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(kvaf_multi_fe_kernel_test ../tests/aiv_model/multi_fe_batch_get.cc)
  target_link_libraries(kvaf_multi_fe_kernel_test PRIVATE kvaf_access_operator)
  target_include_directories(kvaf_multi_fe_kernel_test PRIVATE
    ../tests/aiv_model algorithms/get/direct/urma_aiv/include algorithms/get/direct/urma_aiv/kernel)
  add_test(NAME kvaf_multi_fe_kernel COMMAND kvaf_multi_fe_kernel_test)
endif()

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

# Host registration/CPU producer lifecycle is shared by AICPU Scatter/SGL and
# AIV SGL. AIV-only builds do not link/load an AICPU device package.
if(KVAF_ENABLE_AIV_BATCH_GET OR KVAF_ENABLE_AICPU_CPU_PUSH)
  add_library(kvaf_cpu_push_get_host STATIC
    algorithms/get/scatter/urma_aicpu/host/cpu_push_host.cc)
  target_include_directories(kvaf_cpu_push_get_host PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/algorithms" "${CMAKE_CURRENT_SOURCE_DIR}/providers/common"
    "${CMAKE_SOURCE_DIR}/runtime/include" "${ASCEND_CANN_PACKAGE_PATH}/include")
  target_compile_definitions(kvaf_cpu_push_get_host PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
  target_link_libraries(kvaf_cpu_push_get_host PUBLIC kvaf_access_operator PRIVATE kvaf_direct_hccp_session)
  if(KVAF_ENABLE_AICPU_CPU_PUSH)
    target_compile_definitions(kvaf_cpu_push_get_host PRIVATE KVAF_ENABLE_AICPU_CPU_PUSH=1)
    target_link_libraries(kvaf_cpu_push_get_host PRIVATE kvaf_umdk_host)
  endif()
  if(KVAF_ENABLE_AIV_BATCH_GET)
    target_sources(kvaf_cpu_push_get_host PRIVATE
      algorithms/get/cpu_sgl_write/urma_aiv/host/access_operator.cc)
    target_compile_definitions(kvaf_cpu_push_get_host PRIVATE KVAF_ENABLE_AIV_BATCH_GET=1)
    target_link_libraries(kvaf_cpu_push_get_host PRIVATE kvaf_execute_batch_get_host)
  endif()
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
