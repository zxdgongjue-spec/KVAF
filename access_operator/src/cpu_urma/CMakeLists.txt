find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

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

add_library(kvaf_cpu_urma_transport STATIC cpu_urma_transport.cc cpu_urma_discovery.cc)
target_include_directories(kvaf_cpu_urma_transport
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  PRIVATE "${KVAF_URMA_INCLUDE_DIR}")
target_link_libraries(kvaf_cpu_urma_transport
  PUBLIC kvaf_peer_link kvaf_common
  PRIVATE "${KVAF_URMA_LIBRARY}" "${CMAKE_DL_LIBS}" Threads::Threads
          OpenMP::OpenMP_CXX)
target_compile_features(kvaf_cpu_urma_transport PUBLIC cxx_std_17)

add_library(kvaf_cpu_urma_access_operator STATIC cpu_urma_access_operator.cc)
target_include_directories(kvaf_cpu_urma_access_operator
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_link_libraries(kvaf_cpu_urma_access_operator
  PUBLIC kvaf_access_operator kvaf_cpu_urma_transport
  PRIVATE kvaf_meta_manager)
target_compile_features(kvaf_cpu_urma_access_operator PUBLIC cxx_std_17)
