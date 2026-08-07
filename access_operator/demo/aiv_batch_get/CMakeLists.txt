cmake_minimum_required(VERSION 3.16)
project(kvaf_execute_batch_get LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
  if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
  elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
  else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
endif()
if(NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}")
  message(FATAL_ERROR "CANN toolkit not found: ${ASCEND_CANN_PACKAGE_PATH}")
endif()

set(KVAF_AIV_DEMO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(KVAF_AIV_DEMO_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${KVAF_AIV_DEMO_BIN_DIR}")

if(NOT TARGET kvaf_direct_hccp_session)
  add_subdirectory(
    "${CMAKE_CURRENT_SOURCE_DIR}/../direct_hccp_session"
    "${CMAKE_CURRENT_BINARY_DIR}/direct_hccp_session")
endif()

add_subdirectory(kernel)
add_subdirectory(host)
add_subdirectory(test)
