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

  add_library(kvaf_cpu_urma_transport STATIC src/cpu_urma_transport.cc)
  target_include_directories(kvaf_cpu_urma_transport
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE "${KVAF_URMA_INCLUDE_DIR}")
  target_link_libraries(kvaf_cpu_urma_transport
    PUBLIC kvaf_peer_link kvaf_common
    PRIVATE "${KVAF_URMA_LIBRARY}" "${CMAKE_DL_LIBS}" Threads::Threads)
  target_compile_features(kvaf_cpu_urma_transport PUBLIC cxx_std_17)

  add_library(kvaf_cpu_urma_access_operator STATIC src/cpu_urma_access_operator.cc)
  target_include_directories(kvaf_cpu_urma_access_operator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(kvaf_cpu_urma_access_operator PUBLIC kvaf_runtime kvaf_cpu_urma_transport)
  target_compile_features(kvaf_cpu_urma_access_operator PUBLIC cxx_std_17)

  add_library(kvaf_cpu_runtime STATIC src/cpu_runtime.cc)
  target_include_directories(kvaf_cpu_runtime
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
      "${KVAF_URMA_INCLUDE_DIR}")
  target_link_libraries(kvaf_cpu_runtime
    PUBLIC kvaf_runtime
    PRIVATE "${KVAF_URMA_LIBRARY}" "${CMAKE_DL_LIBS}" Threads::Threads
            kvaf_cpu_urma_access_operator)
  target_compile_features(kvaf_cpu_runtime PUBLIC cxx_std_17)

  add_executable(kvaf_cpu_service src/cpu_service_main.cc)
  target_link_libraries(kvaf_cpu_service PRIVATE kvaf_cpu_runtime kvaf_runtime)
  get_filename_component(KVAF_URMA_LIBRARY_DIR "${KVAF_URMA_LIBRARY}" DIRECTORY)
  set_property(TARGET kvaf_cpu_service APPEND PROPERTY BUILD_RPATH
    "${KVAF_URMA_LIBRARY_DIR}")
  target_compile_features(kvaf_cpu_service PRIVATE cxx_std_17)

  add_executable(kvaf_cpu_runtime_npu_hbm_e2e src/cpu_runtime_npu_hbm_e2e_main.cc)
  target_link_libraries(kvaf_cpu_runtime_npu_hbm_e2e PRIVATE kvaf_cpu_runtime)
  target_compile_features(kvaf_cpu_runtime_npu_hbm_e2e PRIVATE cxx_std_17)

  # Standalone HBM benchmark uses the exact same CpuUrmaTransport control/data
  # path as CpuRuntime.  Keep the benchmark entry point here so a normal
  # `build.sh cpu` build produces it without pulling in the custom test wire
  # protocol or a second URMA implementation.
  add_executable(kvaf_cpu_hbm_client ../tests/cpu_hbm_client.cc)
  target_include_directories(kvaf_cpu_hbm_client PRIVATE
    "${KVAF_URMA_INCLUDE_DIR}")
  target_link_libraries(kvaf_cpu_hbm_client PRIVATE kvaf_cpu_urma_transport)
  target_compile_features(kvaf_cpu_hbm_client PRIVATE cxx_std_17)
  set_property(TARGET kvaf_cpu_hbm_client APPEND PROPERTY BUILD_RPATH
    "${KVAF_URMA_LIBRARY_DIR}")

  if(KVAF_BUILD_CPU_URMA_PROBE)
    if(NOT ASCEND_CANN_PACKAGE_PATH OR
       NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}/include/acl/acl.h")
      message(FATAL_ERROR
        "KVAF_BUILD_CPU_URMA_PROBE requires ASCEND_CANN_PACKAGE_PATH with include/acl/acl.h")
    endif()
    add_executable(kvaf_cpu_urma_probe src/cpu_urma_probe_main.cc)
    target_include_directories(kvaf_cpu_urma_probe PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(kvaf_cpu_urma_probe PRIVATE kvaf_cpu_urma_transport
      "${KVAF_URMA_LIBRARY}" "${CMAKE_DL_LIBS}" Threads::Threads)
    get_filename_component(KVAF_URMA_LIBRARY_DIR "${KVAF_URMA_LIBRARY}" DIRECTORY)
    set_property(TARGET kvaf_cpu_urma_probe APPEND PROPERTY BUILD_RPATH
      "${KVAF_URMA_LIBRARY_DIR}")
    target_compile_features(kvaf_cpu_urma_probe PRIVATE cxx_std_17)

    add_executable(kvaf_cpu_urma_hbm_probe src/cpu_urma_hbm_probe_main.cc)
    target_include_directories(kvaf_cpu_urma_hbm_probe
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        "${CMAKE_CURRENT_SOURCE_DIR}/../access_operator/include"
        "${KVAF_URMA_INCLUDE_DIR}"
        "${ASCEND_CANN_PACKAGE_PATH}/include")
    target_link_libraries(kvaf_cpu_urma_hbm_probe
      PRIVATE kvaf_cpu_urma_transport ascendcl ${CMAKE_DL_LIBS})
    target_link_directories(kvaf_cpu_urma_hbm_probe
      PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
    get_filename_component(KVAF_URMA_LIBRARY_DIR "${KVAF_URMA_LIBRARY}" DIRECTORY)
    set_property(TARGET kvaf_cpu_urma_hbm_probe APPEND PROPERTY BUILD_RPATH
      "${KVAF_URMA_LIBRARY_DIR};${ASCEND_CANN_PACKAGE_PATH}/lib64")
    target_compile_features(kvaf_cpu_urma_hbm_probe PRIVATE cxx_std_17)
  endif()
endif()

if(KVAF_BUILD_NPU_SERVICE)
  if(NOT KVAF_ENABLE_AIV_BATCH_GET)
    message(FATAL_ERROR "KVAF_BUILD_NPU_SERVICE requires KVAF_ENABLE_AIV_BATCH_GET=ON")
  endif()
  find_path(KVAF_NPU_URMA_INCLUDE_DIR urma_api.h
    HINTS "${KVAF_URMA_ROOT}/include" /usr/include /usr/local/include)
  if(NOT KVAF_NPU_URMA_INCLUDE_DIR)
    message(FATAL_ERROR "KVAF_BUILD_NPU_SERVICE requires urma_api.h")
  endif()
  target_sources(kvaf_npu_runtime PRIVATE src/npu_service.cc)
  target_compile_definitions(kvaf_npu_runtime PRIVATE KVAF_BUILD_NPU_SERVICE=1)
  target_include_directories(kvaf_npu_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../access_operator/include
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${KVAF_NPU_URMA_INCLUDE_DIR})
  target_link_libraries(kvaf_npu_runtime PRIVATE kvaf_direct_hccp_session
    ascendcl ${CMAKE_DL_LIBS} pthread)
  target_link_directories(kvaf_npu_runtime PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)

  add_executable(kvaf_npu_service src/npu_service_main.cc)
  target_link_libraries(kvaf_npu_service PRIVATE kvaf_npu_runtime kvaf_runtime)
  target_link_directories(kvaf_npu_service PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
  target_compile_features(kvaf_npu_service PRIVATE cxx_std_17)
endif()
