cmake_minimum_required(VERSION 3.18)
project(kvaf_host_roce_demo LANGUAGES CXX)

option(BUILD_HOST_ROCE_CLIENT "Build the A5 HBM client" ON)
find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h REQUIRED)
find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
find_package(Threads REQUIRED)

set(repo "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(roce_runtime "${repo}/runtime/src")
add_library(roce_demo_connection STATIC "${roce_runtime}/roce_connection.cc")
target_compile_features(roce_demo_connection PUBLIC cxx_std_17)
target_include_directories(roce_demo_connection PUBLIC
  "${roce_runtime}" "${IBVERBS_INCLUDE_DIR}" "${repo}/common/include")
target_link_libraries(roce_demo_connection PUBLIC "${IBVERBS_LIBRARY}")

add_executable(kvaf_roce_cpu_service cpu_service.cc)
target_link_libraries(kvaf_roce_cpu_service PRIVATE roce_demo_connection)

if(BUILD_HOST_ROCE_CLIENT)
  set(CANN_ROOT "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN toolkit directory")
  set(ASCEND_DRIVER_PATH "/usr/local/Ascend/driver" CACHE PATH "Ascend driver directory")
  find_path(ACL_INCLUDE_DIR acl/acl.h HINTS "${CANN_ROOT}/include" REQUIRED)
  find_library(ASCENDCL_LIBRARY ascendcl HINTS "${CANN_ROOT}/lib64" REQUIRED)
  find_library(ASCEND_HAL_LIBRARY ascend_hal
    HINTS "${ASCEND_DRIVER_PATH}/lib64/driver" "${ASCEND_DRIVER_PATH}/lib64" "${CANN_ROOT}/lib64" REQUIRED)

  add_executable(kvaf_host_roce main.cc)
  target_include_directories(kvaf_host_roce PRIVATE "${ACL_INCLUDE_DIR}")
  target_link_libraries(kvaf_host_roce PRIVATE
    roce_demo_connection "${ASCENDCL_LIBRARY}" "${ASCEND_HAL_LIBRARY}" Threads::Threads)
endif()
