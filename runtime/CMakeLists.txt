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

if(KVAF_ENABLE_AIV_BATCH_GET OR KVAF_ENABLE_AICPU_URMA)
  add_library(kvaf_npu_runtime STATIC src/npu_runtime.cc)
  target_include_directories(kvaf_npu_runtime
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../access_operator/demo/aiv_batch_get/include
      ${ASCEND_CANN_PACKAGE_PATH}/include)
  target_link_libraries(kvaf_npu_runtime
    PUBLIC kvaf_runtime
    PRIVATE kvaf_peer_link ascendcl ${CMAKE_DL_LIBS})
  if(KVAF_ENABLE_AIV_BATCH_GET)
    target_sources(kvaf_npu_runtime PRIVATE src/ubmem_allocator.cc)
    target_link_libraries(kvaf_npu_runtime PRIVATE kvaf_default_access_operator
      kvaf_ubmem_operator kvaf_execute_batch_get_host)
    target_compile_definitions(kvaf_npu_runtime PRIVATE KVAF_ENABLE_AIV_BATCH_GET=1)
  endif()
  if(KVAF_ENABLE_AICPU_URMA)
    target_link_libraries(kvaf_npu_runtime PRIVATE kvaf_aicpu_access_operator)
    target_compile_definitions(kvaf_npu_runtime PRIVATE KVAF_ENABLE_AICPU_URMA=1)
  endif()
  if(KVAF_ENABLE_AICPU_CPU_PUSH)
    target_compile_definitions(kvaf_npu_runtime PRIVATE KVAF_ENABLE_AICPU_CPU_PUSH=1)
  endif()
  if(KVAF_ENABLE_AIV_BATCH_PUT)
    target_link_libraries(kvaf_npu_runtime PRIVATE kvaf_execute_batch_put_host)
  endif()
  target_link_directories(kvaf_npu_runtime PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
  target_compile_features(kvaf_npu_runtime PUBLIC cxx_std_17)
endif()

option(KVAF_CPU_ENABLE_URMA "Include the URMA CPU backend" ON)
if(KVAF_BUILD_REMOTE_CPU_SERVICE)
  add_library(kvaf_cpu_runtime STATIC src/cpu_runtime.cc src/ubmem_cpu_backend.cc)
  target_include_directories(kvaf_cpu_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(kvaf_cpu_runtime PUBLIC kvaf_runtime PRIVATE kvaf_ubmem_hal kvaf_peer_link)
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
    add_library(kvaf_cpu_gather_executor STATIC src/cpu_gather_executor.cc)
    target_include_directories(kvaf_cpu_gather_executor PUBLIC include)
    target_link_libraries(kvaf_cpu_gather_executor PUBLIC kvaf_common kvaf_access_operator Threads::Threads)
    target_compile_options(kvaf_cpu_gather_executor PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2>")
    target_sources(kvaf_cpu_runtime PRIVATE src/urma_cpu_backend.cc src/cpu_urma_discovery.cc
      src/cpu_push_get_transport.cc src/cpu_push_get_worker.cc)
    target_link_libraries(kvaf_cpu_runtime PRIVATE Threads::Threads OpenMP::OpenMP_CXX kvaf_cpu_gather_executor)
    target_compile_options(kvaf_cpu_runtime PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2;-g>")
    target_include_directories(kvaf_cpu_runtime PRIVATE "${KVAF_URMA_INCLUDE_DIR}")
    target_link_libraries(kvaf_cpu_runtime PRIVATE "${KVAF_URMA_LIBRARY}" ${CMAKE_DL_LIBS})
    target_compile_definitions(kvaf_cpu_runtime PRIVATE KVAF_CPU_HAS_URMA=1)
  endif()
endif()
if(KVAF_BUILD_REMOTE_CPU_SERVICE)
  add_executable(kvaf_cpu_service src/cpu_service_main.cc)
  target_link_libraries(kvaf_cpu_service PRIVATE kvaf_cpu_runtime)
  if(KVAF_CPU_ENABLE_URMA)
    get_filename_component(KVAF_URMA_LIBRARY_DIR "${KVAF_URMA_LIBRARY}" DIRECTORY)
    set_property(TARGET kvaf_cpu_service APPEND PROPERTY BUILD_RPATH "${KVAF_URMA_LIBRARY_DIR}")
  endif()
endif()
