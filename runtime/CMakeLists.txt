find_package(Threads REQUIRED)

# Match the kernel outputs of build.sh npu, including a custom --build-dir.
# Record absolute paths so applications need not run from the source tree.
configure_file(cmake/npu_kernel_paths.h.in generated/npu_kernel_paths.h @ONLY)

add_library(kvaf_peer_link STATIC peer/peer_link.cc peer/execution_protocol.cc)

target_include_directories(kvaf_peer_link
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(kvaf_peer_link PUBLIC kvaf_common pthread)
target_compile_features(kvaf_peer_link PUBLIC cxx_std_17)

# Wire control has no ACL dependency; the CPU service and NPU roles share it.
add_library(kvaf_offload_control STATIC src/offload_client.cc src/remote_pull_client.cc src/hbm_read_protocol.cc)
target_include_directories(kvaf_offload_control PUBLIC include PRIVATE src)
target_link_libraries(kvaf_offload_control PUBLIC kvaf_common Threads::Threads)

add_library(kvaf_runtime
  memory/cache_region_service.cc
  execution/execution_config.cc
  execution/execution_dispatcher.cc
  execution/execution_resource_manager.cc
  src/runtime.cc
  src/connector_runtime.cc
  cpu_service/runtime_service.cc
  src/npu_runtime_config.cc
  src/npu_eid_selection.cc
)

target_include_directories(kvaf_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)

target_link_libraries(kvaf_runtime
  PUBLIC
    kvaf_common
    kvaf_peer_link
    Threads::Threads
    kvaf_meta_manager
    kvaf_access_operator
  PRIVATE
    kvaf_offload_control
)

target_compile_features(kvaf_runtime PUBLIC cxx_std_17)

if(BUILD_TESTING AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(kvaf_eid_selection_test tests/eid_selection_test.cc)
  target_include_directories(kvaf_eid_selection_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(kvaf_eid_selection_test PRIVATE kvaf_runtime)
  add_test(NAME kvaf_eid_selection COMMAND kvaf_eid_selection_test)
  add_executable(kvaf_cpu_numa_scope_test tests/cpu_numa_scope_test.cc cpu_service/urma/cpu_numa_scope.cc)
  target_include_directories(kvaf_cpu_numa_scope_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(kvaf_cpu_numa_scope_test PRIVATE kvaf_common)
  add_test(NAME kvaf_cpu_numa_scope COMMAND kvaf_cpu_numa_scope_test)
endif()

if(KVAF_ENABLE_AIV_BATCH_GET OR KVAF_ENABLE_AICPU_URMA OR KVAF_ENABLE_CPU_OFFLOAD OR KVAF_ENABLE_NPU_PULL)
  add_library(kvaf_npu_runtime STATIC src/npu_runtime.cc)
  if(KVAF_ENABLE_CPU_OFFLOAD)
    add_library(kvaf_remote_offload STATIC src/remote_offload.cc src/remote_pull_operator.cc)
    target_include_directories(kvaf_remote_offload PUBLIC include PRIVATE src "${ASCEND_CANN_PACKAGE_PATH}/include")
    target_link_directories(kvaf_remote_offload PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
    target_link_libraries(kvaf_remote_offload PUBLIC kvaf_runtime PRIVATE kvaf_offload_control kvaf_peer_link kvaf_direct_hccp_session ascendcl ${CMAKE_DL_LIBS})
    target_compile_definitions(kvaf_npu_runtime PRIVATE KVAF_ENABLE_CPU_OFFLOAD=1)
    target_link_libraries(kvaf_npu_runtime PRIVATE kvaf_remote_offload)
  endif()
  if(KVAF_ENABLE_CPU_OFFLOAD OR KVAF_ENABLE_NPU_PULL)
    add_library(kvaf_hbm_read_source STATIC src/hbm_read_source.cc)
    target_include_directories(kvaf_hbm_read_source PUBLIC include PRIVATE src "${ASCEND_CANN_PACKAGE_PATH}/include")
    target_link_directories(kvaf_hbm_read_source PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
    target_link_libraries(kvaf_hbm_read_source PUBLIC kvaf_runtime PRIVATE kvaf_offload_control kvaf_peer_link kvaf_direct_hccp_session ascendcl ${CMAKE_DL_LIBS})
  endif()
  if(KVAF_ENABLE_NPU_PULL)
    target_sources(kvaf_npu_runtime PRIVATE src/hbm_read_client.cc)
    target_compile_definitions(kvaf_npu_runtime PRIVATE KVAF_ENABLE_NPU_PULL=1)
    target_link_libraries(kvaf_npu_runtime PRIVATE kvaf_offload_control kvaf_execute_batch_get_raw_addr_host kvaf_direct_hccp_session)
    target_include_directories(kvaf_npu_runtime PRIVATE "${CMAKE_SOURCE_DIR}/access_operator/algorithms/get/raw_addr/urma_aiv/include")
  endif()
  target_include_directories(kvaf_npu_runtime
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../access_operator/algorithms
      ${CMAKE_CURRENT_SOURCE_DIR}/../access_operator/algorithms/get/direct/urma_aiv/include
      ${ASCEND_CANN_PACKAGE_PATH}/include
      ${CMAKE_CURRENT_BINARY_DIR}/generated
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(kvaf_npu_runtime
    PUBLIC kvaf_runtime
    PRIVATE kvaf_peer_link ascendcl ${CMAKE_DL_LIBS})
  if(KVAF_ENABLE_AIV_BATCH_GET)
    target_link_libraries(kvaf_npu_runtime PRIVATE kvaf_default_access_operator kvaf_cpu_push_get_host
      kvaf_ubmem_operator kvaf_execute_batch_get_host)
    target_compile_definitions(kvaf_npu_runtime PRIVATE KVAF_ENABLE_AIV_BATCH_GET=1)
  endif()
  if(KVAF_ENABLE_AICPU_URMA)
    target_link_libraries(kvaf_npu_runtime PRIVATE kvaf_aicpu_access_operator kvaf_umdk_host)
    target_compile_definitions(kvaf_npu_runtime PRIVATE KVAF_ENABLE_AICPU_URMA=1)
  endif()
  if(KVAF_ENABLE_AICPU_CPU_PUSH)
    target_compile_definitions(kvaf_npu_runtime PRIVATE KVAF_ENABLE_AICPU_CPU_PUSH=1)
  endif()
  if(KVAF_ENABLE_AIV_BATCH_PUT)
    target_link_libraries(kvaf_npu_runtime PRIVATE kvaf_execute_batch_put_host)
    target_compile_definitions(kvaf_npu_runtime PRIVATE KVAF_ENABLE_AIV_BATCH_PUT=1)
  endif()
  target_link_directories(kvaf_npu_runtime PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
  target_compile_features(kvaf_npu_runtime PUBLIC cxx_std_17)
endif()

option(KVAF_CPU_ENABLE_URMA "Include the URMA CPU backend" ON)
if(KVAF_BUILD_REMOTE_CPU_SERVICE)
  add_library(kvaf_cpu_runtime STATIC cpu_service/cpu_runtime.cc cpu_service/backends/ubmem_cpu_backend.cc
    memory/cpu_ddr_allocation.cc memory/execution_cpu_backend.cc src/remote_pull_service.cc)
  target_include_directories(kvaf_cpu_runtime
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(kvaf_cpu_runtime PUBLIC kvaf_runtime PRIVATE kvaf_ubmem_hal kvaf_peer_link kvaf_offload_control)
  if(KVAF_CPU_ENABLE_URMA)
    set(KVAF_URMA_ROOT "" CACHE PATH "UMDK source or installation root")
    find_path(KVAF_URMA_INCLUDE_DIR urma_api.h HINTS "${KVAF_URMA_ROOT}/include"
      "${CMAKE_SOURCE_DIR}/../third_party/umdk/include" /usr/include /usr/local/include)
    find_library(KVAF_URMA_LIBRARY NAMES urma HINTS "${KVAF_URMA_ROOT}/lib64"
      "${KVAF_URMA_ROOT}/lib" /lib64 /usr/lib64 /usr/local/lib64)
    if(NOT KVAF_URMA_INCLUDE_DIR OR NOT KVAF_URMA_LIBRARY)
      message(FATAL_ERROR "URMA requires urma_api.h and liburma; set KVAF_URMA_ROOT or KVAF_CPU_ENABLE_URMA=OFF")
    endif()
    find_package(Threads REQUIRED)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    add_library(kvaf_cpu_gather_executor STATIC cpu_service/gather/cpu_gather_executor.cc)
    target_include_directories(kvaf_cpu_gather_executor PUBLIC include)
    target_link_libraries(kvaf_cpu_gather_executor PUBLIC kvaf_common kvaf_access_operator Threads::Threads)
    target_compile_options(kvaf_cpu_gather_executor PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2>")
    target_sources(kvaf_cpu_runtime PRIVATE src/cpu_pull_urma.cc cpu_service/backends/urma_cpu_backend.cc cpu_service/urma/cpu_urma_discovery.cc
      cpu_service/urma/cpu_numa_scope.cc
      cpu_service/push/cpu_push_get_transport.cc cpu_service/push/cpu_push_get_worker.cc)
    target_link_libraries(kvaf_cpu_runtime PRIVATE Threads::Threads OpenMP::OpenMP_CXX kvaf_cpu_gather_executor)
    target_compile_options(kvaf_cpu_runtime PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2;-g>")
    target_include_directories(kvaf_cpu_runtime PRIVATE "${KVAF_URMA_INCLUDE_DIR}")
    target_link_libraries(kvaf_cpu_runtime PRIVATE "${KVAF_URMA_LIBRARY}" ${CMAKE_DL_LIBS})
    target_compile_definitions(kvaf_cpu_runtime PRIVATE KVAF_CPU_HAS_URMA=1)
  endif()
endif()
if(KVAF_BUILD_REMOTE_CPU_SERVICE)
  add_executable(kvaf_cpu_service cpu_service/cpu_service_main.cc)
  target_link_libraries(kvaf_cpu_service PRIVATE kvaf_cpu_runtime)
  if(KVAF_CPU_ENABLE_URMA)
    get_filename_component(KVAF_URMA_LIBRARY_DIR "${KVAF_URMA_LIBRARY}" DIRECTORY)
    set_property(TARGET kvaf_cpu_service APPEND PROPERTY BUILD_RPATH "${KVAF_URMA_LIBRARY_DIR}")
  endif()
endif()
