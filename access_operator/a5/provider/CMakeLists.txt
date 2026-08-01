cmake_minimum_required(VERSION 3.18)

project(kvaf_urma_perftest_provider LANGUAGES C)

set(KVAF_URMA_PERFTEST_ROOT "" CACHE PATH
  "Path to the urma_perftest repository root")
set(KVAF_URMA_LIBRARY "" CACHE FILEPATH
  "Optional exact liburma shared library")
set(KVAF_URMA_COMMON_LIBRARY "" CACHE FILEPATH
  "Optional exact liburma_common shared library")
set(KVAF_UMMU_UAPI_ROOT "" CACHE PATH
  "Root containing the full A5 include/uapi/linux/ummu_core.h")
set(KVAF_UMMU_UAPI_INCLUDE_DIR "" CACHE PATH
  "Alias for KVAF_UMMU_UAPI_ROOT; no legacy-header fallback is allowed")

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
set(KVAF_UDMA_SOURCE_DIR
  "${KVAF_UBENGINE_ROOT}/userspace/udma")
set(KVAF_UMMU_SOURCE_DIR
  "${KVAF_UBENGINE_ROOT}/userspace/ummu")
set(KVAF_URMA_PUBLIC_ROOT
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code")
set(KVAF_UDMA_CORE_ROOT "${KVAF_URMA_PUBLIC_ROOT}")
set(KVAF_URMA_CORE_SOURCE_DIR
  "${KVAF_URMA_PUBLIC_ROOT}/lib/urma/core")
set(KVAF_UDMA_PRIVATE_HEADER
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_common.h")
if(NOT EXISTS "${KVAF_UDMA_ABI_HEADER}")
  message(FATAL_ERROR
    "The direct A5 provider requires the source-matched ubengine UDMA ABI")
endif()
if(NOT EXISTS "${KVAF_UDMA_PRIVATE_HEADER}")
  message(FATAL_ERROR
    "The direct A5 provider requires the urma_perftest AIV queue layout")
endif()

if(KVAF_UMMU_UAPI_ROOT AND KVAF_UMMU_UAPI_INCLUDE_DIR)
  get_filename_component(KVAF_UMMU_UAPI_ROOT_REAL
    "${KVAF_UMMU_UAPI_ROOT}" REALPATH)
  get_filename_component(KVAF_UMMU_UAPI_INCLUDE_DIR_REAL
    "${KVAF_UMMU_UAPI_INCLUDE_DIR}" REALPATH)
  if(NOT KVAF_UMMU_UAPI_ROOT_REAL STREQUAL
      KVAF_UMMU_UAPI_INCLUDE_DIR_REAL)
    message(FATAL_ERROR
      "KVAF_UMMU_UAPI_ROOT and KVAF_UMMU_UAPI_INCLUDE_DIR disagree")
  endif()
elseif(KVAF_UMMU_UAPI_ROOT)
  set(KVAF_UMMU_UAPI_INCLUDE_DIR "${KVAF_UMMU_UAPI_ROOT}")
elseif(KVAF_UMMU_UAPI_INCLUDE_DIR)
  set(KVAF_UMMU_UAPI_ROOT "${KVAF_UMMU_UAPI_INCLUDE_DIR}")
else()
  message(FATAL_ERROR
    "Missing full A5 UMMU UAPI: set KVAF_UMMU_UAPI_ROOT to a root "
    "containing include/uapi/linux/ummu_core.h with hw_cap, "
    "HW_CAP_EXPAN, HW_CAP_DOUBLE_PLBI, HW_CAP_KCMD_PLBI, "
    "UMMU_IOCPLBI_VA and UMMU_IOCPLBI_ALL; legacy drivers/ubus "
    "headers and system libummu fallback are forbidden")
endif()
get_filename_component(KVAF_UMMU_UAPI_INCLUDE_DIR
  "${KVAF_UMMU_UAPI_INCLUDE_DIR}" REALPATH)
set(KVAF_UMMU_UAPI_ROOT "${KVAF_UMMU_UAPI_INCLUDE_DIR}")
set(KVAF_UMMU_UAPI_HEADER
  "${KVAF_UMMU_UAPI_INCLUDE_DIR}/include/uapi/linux/ummu_core.h")
if(NOT EXISTS "${KVAF_UMMU_UAPI_HEADER}")
  message(FATAL_ERROR
    "KVAF_UMMU_UAPI_ROOT=${KVAF_UMMU_UAPI_INCLUDE_DIR} does not "
    "contain include/uapi/linux/ummu_core.h")
endif()
get_filename_component(KVAF_LEGACY_UMMU_UAPI_ROOT
  "${KVAF_URMA_PERFTEST_ROOT}/drivers/ubus/kernel" REALPATH)
string(FIND "${KVAF_UMMU_UAPI_INCLUDE_DIR}/"
  "${KVAF_LEGACY_UMMU_UAPI_ROOT}/" KVAF_LEGACY_UMMU_PREFIX)
if(KVAF_LEGACY_UMMU_PREFIX EQUAL 0)
  message(FATAL_ERROR
    "The legacy drivers/ubus UMMU UAPI is incomplete and forbidden: "
    "${KVAF_UMMU_UAPI_INCLUDE_DIR}")
endif()

include(CheckCSourceCompiles)
set(KVAF_SAVED_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
set(KVAF_SAVED_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
set(CMAKE_REQUIRED_INCLUDES
  "${KVAF_UMMU_UAPI_INCLUDE_DIR};${KVAF_UBENGINE_ROOT}/userspace/include/ummu")
set(CMAKE_REQUIRED_FLAGS "-Werror")
unset(KVAF_UMMU_FULL_UAPI_COMPILES CACHE)
check_c_source_compiles([[
#include <stddef.h>
#include <stdint.h>
#include "ummu_api.h"
int main(void) {
  struct ummu_tid_info info = {0};
  uint64_t caps = info.hw_cap;
  info.va = 0;
  info.size = 0;
  caps |= HW_CAP_EXPAN;
  caps |= HW_CAP_DOUBLE_PLBI;
  caps |= HW_CAP_KCMD_PLBI;
  (void)info.va;
  (void)info.size;
  return (int)(caps ^ (uint64_t)UMMU_IOCPLBI_VA ^
      (uint64_t)UMMU_IOCPLBI_ALL);
}
]] KVAF_UMMU_FULL_UAPI_COMPILES)
set(CMAKE_REQUIRED_INCLUDES "${KVAF_SAVED_REQUIRED_INCLUDES}")
set(CMAKE_REQUIRED_FLAGS "${KVAF_SAVED_REQUIRED_FLAGS}")
if(NOT KVAF_UMMU_FULL_UAPI_COMPILES)
  message(FATAL_ERROR
    "KVAF_UMMU_UAPI_ROOT does not provide the full A5 UMMU ABI. "
    "Required: struct ummu_tid_info.{hw_cap,va,size}, "
    "HW_CAP_EXPAN, HW_CAP_DOUBLE_PLBI, HW_CAP_KCMD_PLBI, "
    "UMMU_IOCPLBI_VA and UMMU_IOCPLBI_ALL. No legacy-header or "
    "system-libummu fallback will be used. Probe log: "
    "${CMAKE_BINARY_DIR}/CMakeFiles/CMakeError.log")
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
    "a reference liburma and runtime liburma_common are required")
endif()

set(KVAF_URMA_CORE_SOURCES
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_cmd.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_cmd_tlv.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_cp_api.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_device.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_dp_api.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_format_convert.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_log.c"
)
foreach(KVAF_URMA_CORE_SOURCE IN LISTS KVAF_URMA_CORE_SOURCES)
  if(NOT EXISTS "${KVAF_URMA_CORE_SOURCE}")
    message(FATAL_ERROR
      "Missing source-linked liburma core input ${KVAF_URMA_CORE_SOURCE}")
  endif()
endforeach()

add_library(kvaf_source_urma_core OBJECT
  source_linked_urma_main.c
  ${KVAF_URMA_CORE_SOURCES}
)
set_target_properties(kvaf_source_urma_core PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  C_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE YES
)
target_compile_options(kvaf_source_urma_core PRIVATE -Wall -Wextra)
target_compile_definitions(kvaf_source_urma_core PRIVATE
  "KVAF_URMA_MAIN_SOURCE=\"${KVAF_URMA_CORE_SOURCE_DIR}/urma_main.c\""
)
target_include_directories(kvaf_source_urma_core PRIVATE
  "${KVAF_URMA_PUBLIC_ROOT}/include"
  "${KVAF_URMA_PUBLIC_ROOT}/include/common"
  "${KVAF_URMA_PUBLIC_ROOT}/lib/urma/include"
  "${KVAF_URMA_CORE_SOURCE_DIR}"
)

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
  "${KVAF_UDMA_CORE_ROOT}/lib/urma/include"
  "${KVAF_UDMA_CORE_ROOT}/lib/urma/core"
  "${KVAF_UDMA_CORE_ROOT}/lib/urma/bond/include"
  "${KVAF_UDMA_CORE_ROOT}/include"
  "${KVAF_UDMA_CORE_ROOT}/include/common"
  "${KVAF_URMA_PERFTEST_ROOT}/urma/code/tools/urma_perftest"
)
target_compile_definitions(kvaf_urma_provider_common PRIVATE
  "KVAF_PERFTEST_RESOURCES_SOURCE=\"${KVAF_PERFTEST_SOURCE_DIR}/perftest_resources.c\""
  "KVAF_PERFTEST_COMMUNICATION_SOURCE=\"${KVAF_PERFTEST_SOURCE_DIR}/perftest_communication.c\""
)

set(KVAF_UDMA_PRIVATE_INCLUDE_DIRS
  "${KVAF_UMMU_UAPI_INCLUDE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${KVAF_UDMA_CORE_ROOT}/lib/urma/include"
  "${KVAF_UDMA_CORE_ROOT}/lib/urma/core"
  "${KVAF_UDMA_CORE_ROOT}/include"
  "${KVAF_UDMA_CORE_ROOT}/include/common"
  "${KVAF_UBENGINE_ROOT}/kernelspace/include/udma"
  "${KVAF_UBENGINE_ROOT}/userspace/include/external/udma"
  "${KVAF_UBENGINE_ROOT}/userspace/include/external/ummu"
  "${KVAF_UBENGINE_ROOT}/userspace/include/ummu"
  "${KVAF_UBENGINE_ROOT}/userspace/ummu"
  "${KVAF_UDMA_SOURCE_DIR}"
  "${KVAF_URMA_PERFTEST_ROOT}/drivers/ubus/kernel"
)

set(KVAF_UDMA_COMMON_SOURCES
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_buf.c"
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_ctl.c"
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_ctrlq_tp.c"
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_db.c"
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_hmap.c"
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_jfr.c"
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_jfs.c"
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_log.c"
)
foreach(KVAF_UDMA_SOURCE IN LISTS KVAF_UDMA_COMMON_SOURCES)
  if(NOT EXISTS "${KVAF_UDMA_SOURCE}")
    message(FATAL_ERROR "Missing source-built UDMA input ${KVAF_UDMA_SOURCE}")
  endif()
endforeach()

function(kvaf_add_source_udma_object TARGET_NAME ROLE_DEFINITION)
  add_library(${TARGET_NAME} OBJECT
    source_matched_udma_main.c
    source_matched_udma_ops.c
    source_matched_udma_segment.c
    source_matched_udma_tid.c
    source_matched_udma_jetty.c
    source_matched_udma_jfc.c
    ${KVAF_UDMA_COMMON_SOURCES}
  )
  set_target_properties(${TARGET_NAME} PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS YES
    C_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE YES
  )
  target_compile_options(${TARGET_NAME} PRIVATE
    -Wall
    -Wextra
    -include
    "${CMAKE_CURRENT_SOURCE_DIR}/include/udma_public_abi_preinclude.h"
  )
  target_compile_definitions(${TARGET_NAME} PRIVATE
    ${ROLE_DEFINITION}
    "KVAF_UDMA_OPS_SOURCE=\"${KVAF_UDMA_SOURCE_DIR}/udma_u_ops.c\""
    "KVAF_UDMA_SEGMENT_SOURCE=\"${KVAF_UDMA_SOURCE_DIR}/udma_u_segment.c\""
    "KVAF_UDMA_TID_SOURCE=\"${KVAF_UDMA_SOURCE_DIR}/udma_u_tid.c\""
    "KVAF_UDMA_JETTY_SOURCE=\"${KVAF_UDMA_SOURCE_DIR}/udma_u_jetty.c\""
    "KVAF_UDMA_JFC_SOURCE=\"${KVAF_UDMA_SOURCE_DIR}/udma_u_jfc.c\""
  )
  target_include_directories(${TARGET_NAME} PRIVATE
    ${KVAF_UDMA_PRIVATE_INCLUDE_DIRS}
  )
endfunction()

kvaf_add_source_udma_object(kvaf_source_udma_host ST64B)
kvaf_add_source_udma_object(kvaf_source_udma_device TGID_VA)

set(KVAF_UMMU_COMMON_SOURCE
  "${KVAF_UMMU_SOURCE_DIR}/ummu_common.c")
set(KVAF_UMMU_LEVEL_BLOCK_SOURCE
  "${KVAF_UMMU_SOURCE_DIR}/ummu_level_block.c")
set(KVAF_UMMU_SEG_MNG_SOURCE
  "${KVAF_UMMU_SOURCE_DIR}/ummu_seg_mng.c")
set(KVAF_UMMU_MAPT_SOURCE
  "${KVAF_UMMU_SOURCE_DIR}/ummu_mapt.c")
foreach(KVAF_UMMU_SOURCE
    "${KVAF_UMMU_COMMON_SOURCE}"
    "${KVAF_UMMU_LEVEL_BLOCK_SOURCE}"
    "${KVAF_UMMU_SEG_MNG_SOURCE}"
    "${KVAF_UMMU_MAPT_SOURCE}")
  if(NOT EXISTS "${KVAF_UMMU_SOURCE}")
    message(FATAL_ERROR
      "Missing source-matched UMMU input ${KVAF_UMMU_SOURCE}")
  endif()
endforeach()

function(kvaf_add_source_ummu_object TARGET_NAME)
  add_library(${TARGET_NAME} OBJECT
    source_matched_ummu.c
    source_matched_ummu_collections.c
  )
  set_target_properties(${TARGET_NAME} PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS YES
    C_VISIBILITY_PRESET hidden
    POSITION_INDEPENDENT_CODE YES
  )
  target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra)
  target_compile_definitions(${TARGET_NAME} PRIVATE
    KVAF_UMMU_FULL_UAPI_VERIFIED=1
    "KVAF_UMMU_COMMON_SOURCE=\"${KVAF_UMMU_COMMON_SOURCE}\""
    "KVAF_UMMU_LEVEL_BLOCK_SOURCE=\"${KVAF_UMMU_LEVEL_BLOCK_SOURCE}\""
    "KVAF_UMMU_SEG_MNG_SOURCE=\"${KVAF_UMMU_SEG_MNG_SOURCE}\""
    "KVAF_UMMU_MAPT_SOURCE=\"${KVAF_UMMU_MAPT_SOURCE}\""
  )
  target_include_directories(${TARGET_NAME} PRIVATE
    "${KVAF_UMMU_UAPI_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${KVAF_UMMU_SOURCE_DIR}"
    "${KVAF_UBENGINE_ROOT}/userspace/include/ummu"
  )
endfunction()

kvaf_add_source_ummu_object(kvaf_source_ummu_host)
kvaf_add_source_ummu_object(kvaf_source_ummu_device)

add_library(kvaf_udma_accessor_host OBJECT private_udma_accessor.c)
set_target_properties(kvaf_udma_accessor_host PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  C_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE YES
)
target_compile_options(kvaf_udma_accessor_host PRIVATE -Wall -Wextra)
target_compile_options(kvaf_udma_accessor_host PRIVATE
  -include
  "${CMAKE_CURRENT_SOURCE_DIR}/include/udma_public_abi_preinclude.h"
)
target_compile_definitions(kvaf_udma_accessor_host PRIVATE
  KVAF_UDMA_ACCESSOR_HOST ST64B
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
target_compile_options(kvaf_udma_accessor_device PRIVATE
  -include
  "${CMAKE_CURRENT_SOURCE_DIR}/include/udma_public_abi_preinclude.h"
)
target_compile_definitions(kvaf_udma_accessor_device PRIVATE
  KVAF_UDMA_ACCESSOR_DEVICE TGID_VA
)
target_include_directories(kvaf_udma_accessor_device PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  ${KVAF_UDMA_PRIVATE_INCLUDE_DIRS}
)

function(kvaf_add_urma_provider
    TARGET_NAME ACCESSOR_TARGET UDMA_TARGET UMMU_TARGET OUTPUT_BASENAME)
  add_library(${TARGET_NAME} SHARED
    $<TARGET_OBJECTS:kvaf_source_urma_core>
    $<TARGET_OBJECTS:kvaf_urma_provider_common>
    $<TARGET_OBJECTS:${ACCESSOR_TARGET}>
    $<TARGET_OBJECTS:${UDMA_TARGET}>
    $<TARGET_OBJECTS:${UMMU_TARGET}>
  )
  target_link_libraries(${TARGET_NAME} PRIVATE
    "${KVAF_URMA_COMMON_LIBRARY}"
    pthread
    m
    dl
    rt
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
  kvaf_source_udma_host
  kvaf_source_ummu_host
  kvaf_urma_provider_host)
kvaf_add_urma_provider(
  kvaf_urma_provider_device
  kvaf_udma_accessor_device
  kvaf_source_udma_device
  kvaf_source_ummu_device
  kvaf_urma_provider_device)

add_library(kvaf_public_abi_consumer_probe OBJECT
  public_abi_probe_consumer.c)
set_target_properties(kvaf_public_abi_consumer_probe PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  POSITION_INDEPENDENT_CODE YES
)
target_include_directories(kvaf_public_abi_consumer_probe PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${KVAF_UDMA_CORE_ROOT}/lib/urma/include"
  "${KVAF_UDMA_CORE_ROOT}/include"
  "${KVAF_UDMA_CORE_ROOT}/include/common"
)

add_library(kvaf_public_abi_udma_probe OBJECT public_abi_probe_udma.c)
set_target_properties(kvaf_public_abi_udma_probe PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  POSITION_INDEPENDENT_CODE YES
)
target_compile_options(kvaf_public_abi_udma_probe PRIVATE
  -include
  "${CMAKE_CURRENT_SOURCE_DIR}/include/udma_public_abi_preinclude.h"
)
target_include_directories(kvaf_public_abi_udma_probe PRIVATE
  ${KVAF_UDMA_PRIVATE_INCLUDE_DIRS}
)

add_executable(kvaf_urma_public_abi_probe
  public_abi_probe_main.c
  $<TARGET_OBJECTS:kvaf_public_abi_consumer_probe>
  $<TARGET_OBJECTS:kvaf_public_abi_udma_probe>
)
set_target_properties(kvaf_urma_public_abi_probe PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
)

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

add_executable(kvaf_urma_provider_aicpu_contract_probe
  provider_aicpu_contract_probe.c)
target_include_directories(kvaf_urma_provider_aicpu_contract_probe PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(kvaf_urma_provider_aicpu_contract_probe PRIVATE dl)
set_target_properties(kvaf_urma_provider_aicpu_contract_probe PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  SKIP_BUILD_RPATH YES
  BUILD_WITH_INSTALL_RPATH NO
  INSTALL_RPATH ""
)

add_executable(kvaf_ummu_uapi_probe ummu_uapi_probe.c)
target_include_directories(kvaf_ummu_uapi_probe PRIVATE
  "${KVAF_UMMU_UAPI_INCLUDE_DIR}"
  "${KVAF_UBENGINE_ROOT}/userspace/include/ummu"
)
set_target_properties(kvaf_ummu_uapi_probe PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  SKIP_BUILD_RPATH YES
  BUILD_WITH_INSTALL_RPATH NO
  INSTALL_RPATH ""
)
