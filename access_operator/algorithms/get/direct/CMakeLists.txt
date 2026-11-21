cmake_minimum_required(VERSION 3.16)
project(kvaf_execute_batch_get LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  option(KVAF_ENABLE_AIV_BATCH_GET "Build the standalone ordinary Get kernel" ON)
endif()

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

get_filename_component(KVAF_ACCESS_OPERATOR_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(KVAF_AIV_ALGORITHM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/urma_aiv/include")
set(KVAF_AIV_ALGORITHM_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${KVAF_AIV_ALGORITHM_BIN_DIR}")

if(NOT TARGET kvaf_direct_hccp_session)
  add_subdirectory(
    "${KVAF_ACCESS_OPERATOR_SOURCE_DIR}/transports/direct_hccp"
    "${CMAKE_CURRENT_BINARY_DIR}/direct_hccp_session")
endif()

if(KVAF_ENABLE_AIV_BATCH_GET)
  add_subdirectory(common/kernel)
endif()
add_subdirectory(urma_aiv/host)
