cmake_minimum_required(VERSION 3.18)

project(kvaf_urma_perftest_provider LANGUAGES C)

set(KVAF_URMA_PERFTEST_ROOT "" CACHE PATH
  "Path to the urma_perftest repository root")
set(KVAF_URMA_LIBRARY "" CACHE FILEPATH
  "Optional exact liburma shared library")
set(KVAF_URMA_COMMON_LIBRARY "" CACHE FILEPATH
  "Optional exact liburma_common shared library")

if(NOT KVAF_URMA_PERFTEST_ROOT)
  message(FATAL_ERROR "KVAF_URMA_PERFTEST_ROOT is required")
endif()
if(NOT EXISTS
    "${KVAF_URMA_PERFTEST_ROOT}/urma/code/tools/urma_perftest/perftest_resources.h")
  message(FATAL_ERROR
    "KVAF_URMA_PERFTEST_ROOT does not contain the expected source tree")
endif()
set(KVAF_UBENGINE_ROOT
  "${KVAF_URMA_PERFTEST_ROOT}/ubengine/ssapi")
set(KVAF_UDMA_ABI_HEADER
  "${KVAF_UBENGINE_ROOT}/kernelspace/include/udma/udma_abi.h")
set(KVAF_UDMA_PRIVATE_HEADER
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/include/urma_types.h")
if(NOT EXISTS "${KVAF_UDMA_ABI_HEADER}")
  message(FATAL_ERROR
    "The direct A5 provider requires the source-matched ubengine UDMA ABI")
endif()
if(NOT EXISTS "${KVAF_UDMA_PRIVATE_HEADER}")
  message(FATAL_ERROR
    "The direct A5 provider requires the urma_perftest AIV queue layout")
endif()
file(STRINGS "${KVAF_UDMA_ABI_HEADER}" KVAF_UDMA_TID_SHIFT_LINE
  REGEX "^[ \t]*#define[ \t]+UDMA_TID_SHIFT[ \t]+8(U)?[ \t]*$")
if(NOT KVAF_UDMA_TID_SHIFT_LINE)
  message(FATAL_ERROR
    "The selected UDMA source does not use the required TID_SHIFT=8 ABI")
endif()
set(KVAF_PERFTEST_SOURCE_DIR
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/tools/urma_perftest")
foreach(KVAF_REQUIRED_PERFTEST_SOURCE
    perftest_parameters.c
    perftest_communication.c
    perftest_resources.c)
  if(NOT EXISTS
      "${KVAF_PERFTEST_SOURCE_DIR}/${KVAF_REQUIRED_PERFTEST_SOURCE}")
    message(FATAL_ERROR
      "Missing source-matched ${KVAF_REQUIRED_PERFTEST_SOURCE}")
  endif()
endforeach()

if(NOT KVAF_URMA_LIBRARY)
  find_library(KVAF_URMA_LIBRARY_DISCOVERED
    NAMES urma liburma.so.0)
  set(KVAF_URMA_LIBRARY "${KVAF_URMA_LIBRARY_DISCOVERED}")
endif()
if(NOT KVAF_URMA_COMMON_LIBRARY)
  find_library(KVAF_URMA_COMMON_LIBRARY_DISCOVERED
    NAMES urma_common liburma_common.so.0)
  set(KVAF_URMA_COMMON_LIBRARY
    "${KVAF_URMA_COMMON_LIBRARY_DISCOVERED}")
endif()
if(NOT KVAF_URMA_LIBRARY OR
   NOT EXISTS "${KVAF_URMA_LIBRARY}" OR
   NOT KVAF_URMA_COMMON_LIBRARY OR
   NOT EXISTS "${KVAF_URMA_COMMON_LIBRARY}")
  message(FATAL_ERROR
    "liburma and liburma_common are required")
endif()

add_library(kvaf_urma_provider_common OBJECT
  urma_perftest_provider.c
  source_matched_perftest.c
  source_matched_communication.c
  "${KVAF_PERFTEST_SOURCE_DIR}/perftest_parameters.c"
)
set_target_properties(kvaf_urma_provider_common PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  C_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE YES
)
target_compile_options(kvaf_urma_provider_common PRIVATE
  -Wall
  -Wextra
)
target_include_directories(kvaf_urma_provider_common PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/include"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/include/common"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/tools/urma_perftest"
)
target_compile_definitions(kvaf_urma_provider_common PRIVATE
  "KVAF_PERFTEST_RESOURCES_SOURCE=\"${KVAF_PERFTEST_SOURCE_DIR}/perftest_resources.c\""
  "KVAF_PERFTEST_COMMUNICATION_SOURCE=\"${KVAF_PERFTEST_SOURCE_DIR}/perftest_communication.c\""
)

set(KVAF_UDMA_PRIVATE_INCLUDE_DIRS
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/include"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/include/common"
)

add_library(kvaf_udma_accessor_host OBJECT private_udma_accessor.c)
set_target_properties(kvaf_udma_accessor_host PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  C_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE YES
)
target_compile_options(kvaf_udma_accessor_host PRIVATE -Wall -Wextra)
target_compile_definitions(kvaf_udma_accessor_host PRIVATE
  KVAF_UDMA_ACCESSOR_HOST
)
target_include_directories(kvaf_udma_accessor_host PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  ${KVAF_UDMA_PRIVATE_INCLUDE_DIRS}
)

add_library(kvaf_udma_accessor_device OBJECT private_udma_accessor.c)
set_target_properties(kvaf_udma_accessor_device PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  C_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE YES
)
target_compile_options(kvaf_udma_accessor_device PRIVATE -Wall -Wextra)
target_compile_definitions(kvaf_udma_accessor_device PRIVATE
  KVAF_UDMA_ACCESSOR_DEVICE
)
target_include_directories(kvaf_udma_accessor_device PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  ${KVAF_UDMA_PRIVATE_INCLUDE_DIRS}
)

function(kvaf_add_urma_provider TARGET_NAME ACCESSOR_TARGET OUTPUT_BASENAME)
  add_library(${TARGET_NAME} SHARED
    $<TARGET_OBJECTS:kvaf_urma_provider_common>
    $<TARGET_OBJECTS:${ACCESSOR_TARGET}>
  )
  target_link_libraries(${TARGET_NAME} PRIVATE
    "${KVAF_URMA_LIBRARY}"
    "${KVAF_URMA_COMMON_LIBRARY}"
    pthread
    m
    dl
  )
  target_link_options(${TARGET_NAME} PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/provider_exports.map"
    "-Wl,-Bsymbolic-functions"
    "-Wl,-z,defs"
  )
  set_target_properties(${TARGET_NAME} PROPERTIES
    OUTPUT_NAME "${OUTPUT_BASENAME}"
    C_VISIBILITY_PRESET hidden
    SKIP_BUILD_RPATH YES
    BUILD_WITH_INSTALL_RPATH NO
    INSTALL_RPATH ""
  )
endfunction()

kvaf_add_urma_provider(
  kvaf_urma_provider_host
  kvaf_udma_accessor_host
  kvaf_urma_provider_host)
kvaf_add_urma_provider(
  kvaf_urma_provider_device
  kvaf_udma_accessor_device
  kvaf_urma_provider_device)

add_executable(kvaf_urma_provider_abi_probe provider_abi_probe.c)
target_link_libraries(kvaf_urma_provider_abi_probe PRIVATE dl)
set_target_properties(kvaf_urma_provider_abi_probe PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  SKIP_BUILD_RPATH YES
  BUILD_WITH_INSTALL_RPATH NO
  INSTALL_RPATH ""
)
