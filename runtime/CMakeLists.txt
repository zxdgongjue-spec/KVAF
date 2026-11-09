add_library(kvaf_peer_link STATIC src/peer_link.cc)

target_include_directories(kvaf_peer_link
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(kvaf_peer_link PUBLIC kvaf_common pthread)
target_compile_features(kvaf_peer_link PUBLIC cxx_std_17)

add_library(kvaf_runtime
  src/runtime.cc
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

if(KVAF_ENABLE_AIV_BATCH_GET)
  add_library(kvaf_npu_runtime STATIC src/npu_runtime.cc)
  target_include_directories(kvaf_npu_runtime
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../access_operator/demo/aiv_batch_get/include
      ${ASCEND_CANN_PACKAGE_PATH}/include)
  target_link_libraries(kvaf_npu_runtime
    PUBLIC kvaf_runtime
    PRIVATE kvaf_peer_link kvaf_default_access_operator
            kvaf_execute_batch_get_host ascendcl ${CMAKE_DL_LIBS})
  if(KVAF_ENABLE_AIV_BATCH_PUT)
    target_link_libraries(kvaf_npu_runtime PRIVATE kvaf_execute_batch_put_host)
  endif()
  target_link_directories(kvaf_npu_runtime PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
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

  add_library(kvaf_cpu_runtime STATIC src/cpu_runtime.cc src/cpu_urma_discovery.cc
    src/cpu_gather_executor.cc src/gather_progress_transport.cc src/cpu_push_get_transport.cc src/cpu_push_get_worker.cc)
  target_include_directories(kvaf_cpu_runtime
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
      "${KVAF_URMA_INCLUDE_DIR}"
      "${PROJECT_SOURCE_DIR}/access_operator/include")
  target_link_libraries(kvaf_cpu_runtime
    PUBLIC kvaf_runtime
    PRIVATE "${KVAF_URMA_LIBRARY}" "${CMAKE_DL_LIBS}" Threads::Threads)
  target_compile_features(kvaf_cpu_runtime PUBLIC cxx_std_17)

  add_executable(kvaf_cpu_service src/cpu_service_main.cc)
  target_link_libraries(kvaf_cpu_service PRIVATE kvaf_cpu_runtime kvaf_runtime)
  get_filename_component(KVAF_URMA_LIBRARY_DIR "${KVAF_URMA_LIBRARY}" DIRECTORY)
  set_property(TARGET kvaf_cpu_service APPEND PROPERTY BUILD_RPATH
    "${KVAF_URMA_LIBRARY_DIR}")
  target_compile_features(kvaf_cpu_service PRIVATE cxx_std_17)
endif()
