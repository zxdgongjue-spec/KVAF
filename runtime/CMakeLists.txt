add_library(kvaf_peer_link STATIC src/peer_link.cc)

target_include_directories(kvaf_peer_link
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(kvaf_peer_link PUBLIC kvaf_common pthread)
target_compile_features(kvaf_peer_link PUBLIC cxx_std_17)

add_library(kvaf_offload_control STATIC src/offload_client.cc
  src/remote_pull_client.cc src/remote_pull_service.cc src/hbm_read_protocol.cc)
target_include_directories(kvaf_offload_control PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(kvaf_offload_control PUBLIC kvaf_common pthread)
target_compile_features(kvaf_offload_control PUBLIC cxx_std_17)

add_library(kvaf_runtime
  src/runtime.cc
  src/connector_runtime.cc
  src/runtime_service.cc
)

target_include_directories(kvaf_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kvaf_runtime
  PUBLIC
    kvaf_common
    kvaf_meta_manager
    kvaf_access_operator
)

target_compile_features(kvaf_runtime PUBLIC cxx_std_17)

if(KVAF_ENABLE_CPU_OFFLOAD)
  add_library(kvaf_remote_offload STATIC src/remote_offload.cc src/remote_pull_operator.cc)
  target_include_directories(kvaf_remote_offload PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)
  target_link_libraries(kvaf_remote_offload PUBLIC kvaf_runtime
    PRIVATE kvaf_offload_control kvaf_peer_link kvaf_direct_hccp_session "${KVAF_ASCENDCL_LIBRARY}" ${CMAKE_DL_LIBS})
endif()

if(KVAF_ENABLE_AIV_BATCH_GET OR KVAF_ENABLE_CPU_OFFLOAD)
  add_library(kvaf_hbm_read_source STATIC src/hbm_read_source.cc)
  target_include_directories(kvaf_hbm_read_source PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/include)
  target_link_libraries(kvaf_hbm_read_source PUBLIC kvaf_runtime
    PRIVATE kvaf_offload_control kvaf_peer_link kvaf_direct_hccp_session "${KVAF_ASCENDCL_LIBRARY}" ${CMAKE_DL_LIBS})
  add_library(kvaf_npu_runtime STATIC src/npu_runtime.cc)
  target_include_directories(kvaf_npu_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../access_operator/demo/aiv_batch_get/include
            ${ASCEND_CANN_PACKAGE_PATH}/include)
  target_link_libraries(kvaf_npu_runtime PUBLIC kvaf_runtime
    PRIVATE kvaf_peer_link "${KVAF_ASCENDCL_LIBRARY}" ${CMAKE_DL_LIBS})
  if(KVAF_ENABLE_CPU_OFFLOAD)
    target_compile_definitions(kvaf_npu_runtime PRIVATE KVAF_ENABLE_CPU_OFFLOAD=1)
    target_link_libraries(kvaf_npu_runtime PRIVATE kvaf_remote_offload)
  endif()
  if(KVAF_ENABLE_AIV_BATCH_GET)
    target_sources(kvaf_npu_runtime PRIVATE src/hbm_read_client.cc)
    target_link_libraries(kvaf_npu_runtime PRIVATE kvaf_offload_control kvaf_direct_hccp_session)
    target_compile_definitions(kvaf_npu_runtime PRIVATE KVAF_ENABLE_AIV_BATCH_GET=1)
    target_link_libraries(kvaf_npu_runtime PRIVATE kvaf_default_access_operator kvaf_execute_batch_get_host)
  endif()
  if(KVAF_ENABLE_AIV_BATCH_PUT)
    target_compile_definitions(kvaf_npu_runtime PRIVATE KVAF_ENABLE_AIV_BATCH_PUT=1)
    target_link_libraries(kvaf_npu_runtime PRIVATE kvaf_execute_batch_put_host)
  endif()
  target_compile_features(kvaf_npu_runtime PUBLIC cxx_std_17)
endif()

if(KVAF_BUILD_REMOTE_CPU_SERVICE)
  find_package(Threads REQUIRED)
  set(KVAF_URMA_ROOT "" CACHE PATH "UMDK source or installation root")
  find_path(KVAF_URMA_INCLUDE_DIR urma_api.h
    HINTS
      "${KVAF_URMA_ROOT}/include"
      "${CMAKE_SOURCE_DIR}/../third_party/umdk/include"
      /usr/include /usr/local/include
  )
  find_library(KVAF_URMA_LIBRARY NAMES urma
    HINTS
      "${KVAF_URMA_ROOT}/lib64"
      "${KVAF_URMA_ROOT}/lib"
      /lib64 /usr/lib64 /usr/local/lib64
  )
  if(NOT KVAF_URMA_INCLUDE_DIR OR NOT KVAF_URMA_LIBRARY)
    message(FATAL_ERROR
      "KVAF_BUILD_REMOTE_CPU_SERVICE requires urma_api.h and liburma; set KVAF_URMA_ROOT")
  endif()

  add_library(kvaf_cpu_runtime STATIC src/cpu_runtime.cc src/cpu_pull_urma.cc)
  target_include_directories(kvaf_cpu_runtime
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
      "${KVAF_URMA_INCLUDE_DIR}")
  target_link_libraries(kvaf_cpu_runtime
    PUBLIC kvaf_runtime
    PRIVATE kvaf_offload_control "${KVAF_URMA_LIBRARY}" "${CMAKE_DL_LIBS}" Threads::Threads)
  # Match aicpu-urma's CPU transport build and deployed 32-byte SGE ABI.
  target_compile_options(kvaf_cpu_runtime PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2;-g>")
  target_compile_features(kvaf_cpu_runtime PUBLIC cxx_std_17)

  add_executable(kvaf_cpu_service src/cpu_service_main.cc)
  target_link_libraries(kvaf_cpu_service PRIVATE kvaf_cpu_runtime kvaf_runtime)
  get_filename_component(KVAF_URMA_LIBRARY_DIR "${KVAF_URMA_LIBRARY}" DIRECTORY)
  set_property(TARGET kvaf_cpu_service APPEND PROPERTY BUILD_RPATH
    "${KVAF_URMA_LIBRARY_DIR}")
  target_compile_features(kvaf_cpu_service PRIVATE cxx_std_17)
endif()
