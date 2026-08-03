cmake_minimum_required(VERSION 3.18)

project(kvaf_urma_perftest_provider LANGUAGES C)

set(KVAF_UMDK_ROOT "" CACHE PATH
  "Path to the paired openEuler UMDK source repository")
set(KVAF_LIBUMMU_ROOT "" CACHE PATH
  "Path to the libummu source repository pinned by UMDK")

if(NOT KVAF_UMDK_ROOT)
  message(FATAL_ERROR "KVAF_UMDK_ROOT is required")
endif()
if(NOT KVAF_LIBUMMU_ROOT)
  message(FATAL_ERROR "KVAF_LIBUMMU_ROOT is required")
endif()
get_filename_component(KVAF_UMDK_ROOT "${KVAF_UMDK_ROOT}" REALPATH)
get_filename_component(KVAF_LIBUMMU_ROOT "${KVAF_LIBUMMU_ROOT}" REALPATH)

set(KVAF_UMDK_URMA_ROOT "${KVAF_UMDK_ROOT}/src/urma")
set(KVAF_URMA_COMMON_SOURCE_DIR "${KVAF_UMDK_URMA_ROOT}/common")
set(KVAF_URMA_COMMON_INCLUDE_DIR
  "${KVAF_URMA_COMMON_SOURCE_DIR}/include")
set(KVAF_URMA_CORE_SOURCE_DIR
  "${KVAF_UMDK_URMA_ROOT}/lib/urma/core")
set(KVAF_URMA_PUBLIC_INCLUDE_DIR
  "${KVAF_URMA_CORE_SOURCE_DIR}/include")
set(KVAF_URMA_BOND_INCLUDE_DIR
  "${KVAF_UMDK_URMA_ROOT}/lib/urma/bond/include")
set(KVAF_PERFTEST_SOURCE_DIR
  "${KVAF_UMDK_URMA_ROOT}/tools/urma_perftest")
set(KVAF_UDMA_SOURCE_DIR "${KVAF_UMDK_URMA_ROOT}/hw/udma")
set(KVAF_UDMA_INCLUDE_DIR "${KVAF_UDMA_SOURCE_DIR}/include")
set(KVAF_UDMA_UAPI_INCLUDE_DIR
  "${KVAF_UDMA_SOURCE_DIR}/kernel_headers")
set(KVAF_UDMA_ABI_HEADER
  "${KVAF_UDMA_UAPI_INCLUDE_DIR}/udma_abi.h")
set(KVAF_UDMA_PRIVATE_HEADER
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_common.h")
set(KVAF_UDMA_JFR_SOURCE
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_jfr.c")

set(KVAF_UMMU_SOURCE_DIR "${KVAF_LIBUMMU_ROOT}")
set(KVAF_UMMU_API_INCLUDE_DIR "${KVAF_LIBUMMU_ROOT}/include")
set(KVAF_UMMU_UAPI_INCLUDE_DIR
  "${KVAF_LIBUMMU_ROOT}/kernel_headers")
set(KVAF_UMMU_UAPI_HEADER
  "${KVAF_UMMU_UAPI_INCLUDE_DIR}/ummu_core.h")

set(KVAF_REQUIRED_OFFICIAL_SOURCE_PATHS
  "${KVAF_UMDK_ROOT}/src/urma/bazel/urma_deps.bzl"
  "${KVAF_URMA_PUBLIC_INCLUDE_DIR}/urma_types.h"
  "${KVAF_URMA_PUBLIC_INCLUDE_DIR}/urma_provider.h"
  "${KVAF_URMA_COMMON_SOURCE_DIR}/ub_hmap.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_main.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_perf.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/ubagg_stats.c"
  "${KVAF_PERFTEST_SOURCE_DIR}/perftest_resources.c"
  "${KVAF_PERFTEST_SOURCE_DIR}/perftest_parameters.c"
  "${KVAF_PERFTEST_SOURCE_DIR}/perftest_mgmt_tcp.c"
  "${KVAF_PERFTEST_SOURCE_DIR}/perftest_mgmt_ub.c"
  "${KVAF_PERFTEST_SOURCE_DIR}/perftest_mgmt_ub.h"
  "${KVAF_UDMA_ABI_HEADER}"
  "${KVAF_UDMA_PRIVATE_HEADER}"
  "${KVAF_UDMA_JFR_SOURCE}"
  "${KVAF_UMMU_API_INCLUDE_DIR}/ummu_api.h"
  "${KVAF_UMMU_UAPI_HEADER}"
  "${KVAF_UMMU_SOURCE_DIR}/ummu_mapt.c"
)
foreach(KVAF_REQUIRED_PATH IN LISTS KVAF_REQUIRED_OFFICIAL_SOURCE_PATHS)
  if(NOT EXISTS "${KVAF_REQUIRED_PATH}")
    message(FATAL_ERROR
      "Incomplete paired UMDK/libummu source tree: ${KVAF_REQUIRED_PATH}")
  endif()
endforeach()

include(CheckCSourceCompiles)
set(KVAF_SAVED_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
set(KVAF_SAVED_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
set(CMAKE_REQUIRED_INCLUDES
  "${KVAF_UMMU_API_INCLUDE_DIR};${KVAF_UMMU_UAPI_INCLUDE_DIR}")
set(CMAKE_REQUIRED_FLAGS "-Werror")
unset(KVAF_UMMU_FULL_UAPI_COMPILES CACHE)
check_c_source_compiles([[
#include <stddef.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include "ummu_api.h"
int main(void) {
  struct ummu_tid_info info = {0};
  uint64_t caps = info.hw_cap;
  info.va = 0;
  info.size = 0;
  caps |= HW_CAP_EXPAN | HW_CAP_DOUBLE_PLBI | HW_CAP_KCMD_PLBI;
  caps |= HW_CAP_FREE_BIT | HW_CAP_PPLBI;
  return (int)(caps ^ (uint64_t)UMMU_IOCPLBI_VA ^
      (uint64_t)UMMU_IOCPLBI_ALL);
}
]] KVAF_UMMU_FULL_UAPI_COMPILES)
set(CMAKE_REQUIRED_INCLUDES "${KVAF_SAVED_REQUIRED_INCLUDES}")
set(CMAKE_REQUIRED_FLAGS "${KVAF_SAVED_REQUIRED_FLAGS}")
if(NOT KVAF_UMMU_FULL_UAPI_COMPILES)
  message(FATAL_ERROR
    "KVAF_LIBUMMU_ROOT does not provide the paired A5 UMMU ABI, including "
    "HW_CAP_FREE_BIT and HW_CAP_PPLBI. Probe log: "
    "${CMAKE_BINARY_DIR}/CMakeFiles/CMakeError.log")
endif()

file(STRINGS "${KVAF_UDMA_ABI_HEADER}" KVAF_UDMA_TID_SHIFT_LINE
  REGEX "^[ \t]*#define[ \t]+UDMA_TID_SHIFT[ \t]+8(U)?[ \t]*$")
file(STRINGS "${KVAF_UDMA_ABI_HEADER}" KVAF_UDMA_MAP_SHIFT_LINE
  REGEX "^[ \t]*#define[ \t]+MAP_INDEX_SHIFT[ \t]+4[ \t]*$")
file(STRINGS "${KVAF_UDMA_ABI_HEADER}" KVAF_UDMA_ACCESS_GUARD_LINE
  REGEX "^[ \t]*#define[ \t]+UDMA_SEGMENT_ACCESS_GUARD[ \t]+0xf[ \t]*$")
if(NOT KVAF_UDMA_TID_SHIFT_LINE OR
   NOT KVAF_UDMA_MAP_SHIFT_LINE OR
   NOT KVAF_UDMA_ACCESS_GUARD_LINE)
  message(FATAL_ERROR
    "The selected official UDMA source has an unexpected private/UAPI ABI")
endif()

foreach(KVAF_REQUIRED_PERFTEST_SOURCE
    perftest_log.c
    perftest_parameters.c
    perftest_resources.c
    perftest_mgmt_tcp.c
    perftest_mgmt_ub.c
    perftest_mgmt_ub.h)
  if(NOT EXISTS
      "${KVAF_PERFTEST_SOURCE_DIR}/${KVAF_REQUIRED_PERFTEST_SOURCE}")
    message(FATAL_ERROR
      "Missing official UMDK ${KVAF_REQUIRED_PERFTEST_SOURCE}")
  endif()
endforeach()

# Source-link the matching UMDK common library.  The provider must not resolve
# a system liburma_common.so built from an unrelated source snapshot.
set(KVAF_URMA_COMMON_SOURCES
  "${KVAF_URMA_COMMON_SOURCE_DIR}/ub_bitmap.c"
  "${KVAF_URMA_COMMON_SOURCE_DIR}/ub_dstring.c"
  "${KVAF_URMA_COMMON_SOURCE_DIR}/ub_get_clock.c"
  "${KVAF_URMA_COMMON_SOURCE_DIR}/ub_hash.c"
  "${KVAF_URMA_COMMON_SOURCE_DIR}/ub_hmap.c"
  "${KVAF_URMA_COMMON_SOURCE_DIR}/ub_util.c"
)

# Keep full warnings enabled for KVAF-owned code.  The selected, hash-pinned
# UMDK snapshot has a small set of known warnings in audited upstream files;
# attach the narrow suppressions to those source files only instead of
# weakening an entire KVAF target.
set_property(SOURCE
  "${KVAF_URMA_COMMON_SOURCE_DIR}/ub_dstring.c"
  APPEND PROPERTY COMPILE_OPTIONS
  "-Wno-sign-compare"
)
add_library(kvaf_source_urma_common OBJECT ${KVAF_URMA_COMMON_SOURCES})
set_target_properties(kvaf_source_urma_common PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  C_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE YES
)
target_compile_options(kvaf_source_urma_common PRIVATE -Wall -Wextra)
target_include_directories(kvaf_source_urma_common PRIVATE
  "${KVAF_URMA_COMMON_INCLUDE_DIR}")

# Keep the selected source list explicit.  In addition to the historical
# core, UMDK 26.06 adds urma_perf.c and ubagg_stats.c; omitting either would
# silently produce a partial liburma implementation.
set(KVAF_URMA_CORE_SOURCES
  "${KVAF_URMA_CORE_SOURCE_DIR}/ubagg_stats.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_cmd.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_cmd_tlv.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_cp_api.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_device.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_dp_api.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_format_convert.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_log.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_perf.c"
)
foreach(KVAF_URMA_CORE_SOURCE IN LISTS KVAF_URMA_CORE_SOURCES)
  if(NOT EXISTS "${KVAF_URMA_CORE_SOURCE}")
    message(FATAL_ERROR
      "Missing source-linked UMDK core input ${KVAF_URMA_CORE_SOURCE}")
  endif()
endforeach()

set_property(SOURCE
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_cmd.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_log.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_perf.c"
  APPEND PROPERTY COMPILE_OPTIONS
  "-Wno-sign-compare"
)
set_property(SOURCE
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_cp_api.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_device.c"
  "${KVAF_URMA_CORE_SOURCE_DIR}/urma_perf.c"
  APPEND PROPERTY COMPILE_OPTIONS
  "-Wno-unused-parameter"
)

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
  "${KVAF_URMA_PUBLIC_INCLUDE_DIR}"
  "${KVAF_URMA_CORE_SOURCE_DIR}"
  "${KVAF_URMA_COMMON_INCLUDE_DIR}"
  "${KVAF_URMA_BOND_INCLUDE_DIR}"
)

add_library(kvaf_urma_provider_common OBJECT
  urma_perftest_provider.c
  source_matched_perftest.c
  source_matched_communication.c
  "${KVAF_PERFTEST_SOURCE_DIR}/perftest_log.c"
  "${KVAF_PERFTEST_SOURCE_DIR}/perftest_parameters.c"
)
set_target_properties(kvaf_urma_provider_common PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  C_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE YES
)
target_compile_options(kvaf_urma_provider_common PRIVATE -Wall -Wextra)
target_include_directories(kvaf_urma_provider_common PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${KVAF_URMA_PUBLIC_INCLUDE_DIR}"
  "${KVAF_URMA_CORE_SOURCE_DIR}"
  "${KVAF_URMA_COMMON_INCLUDE_DIR}"
  "${KVAF_URMA_BOND_INCLUDE_DIR}"
  "${KVAF_PERFTEST_SOURCE_DIR}"
)
target_compile_definitions(kvaf_urma_provider_common PRIVATE
  "KVAF_PERFTEST_RESOURCES_SOURCE=\"${KVAF_PERFTEST_SOURCE_DIR}/perftest_resources.c\""
  "KVAF_PERFTEST_MGMT_TCP_SOURCE=\"${KVAF_PERFTEST_SOURCE_DIR}/perftest_mgmt_tcp.c\""
  "KVAF_PERFTEST_MGMT_UB_SOURCE=\"${KVAF_PERFTEST_SOURCE_DIR}/perftest_mgmt_ub.c\""
)

set(KVAF_UDMA_PRIVATE_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${KVAF_URMA_PUBLIC_INCLUDE_DIR}"
  "${KVAF_URMA_CORE_SOURCE_DIR}"
  "${KVAF_URMA_COMMON_INCLUDE_DIR}"
  "${KVAF_URMA_BOND_INCLUDE_DIR}"
  "${KVAF_UDMA_UAPI_INCLUDE_DIR}"
  "${KVAF_UDMA_INCLUDE_DIR}"
  "${KVAF_UDMA_SOURCE_DIR}"
  "${KVAF_UMMU_API_INCLUDE_DIR}"
  "${KVAF_UMMU_UAPI_INCLUDE_DIR}"
  "${KVAF_UMMU_SOURCE_DIR}"
)

set(KVAF_UDMA_COMMON_SOURCES
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_buf.c"
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_ctl.c"
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_ctrlq_tp.c"
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_db.c"
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_jfs.c"
  "${KVAF_UDMA_SOURCE_DIR}/udma_u_log.c"
)
foreach(KVAF_UDMA_SOURCE IN LISTS KVAF_UDMA_COMMON_SOURCES)
  if(NOT EXISTS "${KVAF_UDMA_SOURCE}")
    message(FATAL_ERROR "Missing official UDMA input ${KVAF_UDMA_SOURCE}")
  endif()
endforeach()

function(kvaf_add_source_udma_object TARGET_NAME ROLE_DEFINITION)
  add_library(${TARGET_NAME} OBJECT
    source_matched_udma_main.c
    source_matched_udma_ops.c
    source_matched_udma_segment.c
    source_matched_udma_tid.c
    source_matched_udma_jetty.c
    source_matched_udma_jfr.c
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
  target_compile_options(${TARGET_NAME} PRIVATE -Wall -Wextra)
  if(ROLE_DEFINITION)
    target_compile_definitions(${TARGET_NAME} PRIVATE ${ROLE_DEFINITION})
  endif()
  target_compile_definitions(${TARGET_NAME} PRIVATE
    "KVAF_UDMA_OPS_SOURCE=\"${KVAF_UDMA_SOURCE_DIR}/udma_u_ops.c\""
    "KVAF_UDMA_SEGMENT_SOURCE=\"${KVAF_UDMA_SOURCE_DIR}/udma_u_segment.c\""
    "KVAF_UDMA_TID_SOURCE=\"${KVAF_UDMA_SOURCE_DIR}/udma_u_tid.c\""
    "KVAF_UDMA_JETTY_SOURCE=\"${KVAF_UDMA_SOURCE_DIR}/udma_u_jetty.c\""
    "KVAF_UDMA_JFR_SOURCE=\"${KVAF_UDMA_JFR_SOURCE}\""
    "KVAF_UDMA_JFC_SOURCE=\"${KVAF_UDMA_SOURCE_DIR}/udma_u_jfc.c\""
  )
  target_include_directories(${TARGET_NAME} PRIVATE
    ${KVAF_UDMA_PRIVATE_INCLUDE_DIRS})
endfunction()

kvaf_add_source_udma_object(kvaf_source_udma_host "")
kvaf_add_source_udma_object(kvaf_source_udma_device "")

set(KVAF_UMMU_COMMON_SOURCE "${KVAF_UMMU_SOURCE_DIR}/ummu_common.c")
set(KVAF_UMMU_LEVEL_BLOCK_SOURCE
  "${KVAF_UMMU_SOURCE_DIR}/ummu_level_block.c")
set(KVAF_UMMU_SEG_MNG_SOURCE
  "${KVAF_UMMU_SOURCE_DIR}/ummu_seg_mng.c")
set(KVAF_UMMU_MAPT_SOURCE "${KVAF_UMMU_SOURCE_DIR}/ummu_mapt.c")
foreach(KVAF_UMMU_SOURCE
    "${KVAF_UMMU_COMMON_SOURCE}"
    "${KVAF_UMMU_LEVEL_BLOCK_SOURCE}"
    "${KVAF_UMMU_SEG_MNG_SOURCE}"
    "${KVAF_UMMU_MAPT_SOURCE}")
  if(NOT EXISTS "${KVAF_UMMU_SOURCE}")
    message(FATAL_ERROR
      "Missing paired libummu input ${KVAF_UMMU_SOURCE}")
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
    "${KVAF_UMMU_API_INCLUDE_DIR}"
    "${KVAF_UMMU_UAPI_INCLUDE_DIR}"
    "${KVAF_UMMU_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
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
target_compile_definitions(kvaf_udma_accessor_host PRIVATE
  KVAF_UDMA_ACCESSOR_HOST)
target_include_directories(kvaf_udma_accessor_host PRIVATE
  ${KVAF_UDMA_PRIVATE_INCLUDE_DIRS})

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
  KVAF_UDMA_ACCESSOR_DEVICE)
target_include_directories(kvaf_udma_accessor_device PRIVATE
  ${KVAF_UDMA_PRIVATE_INCLUDE_DIRS})

function(kvaf_add_urma_provider
    TARGET_NAME ACCESSOR_TARGET UDMA_TARGET UMMU_TARGET OUTPUT_BASENAME)
  add_library(${TARGET_NAME} SHARED
    $<TARGET_OBJECTS:kvaf_source_urma_common>
    $<TARGET_OBJECTS:kvaf_source_urma_core>
    $<TARGET_OBJECTS:kvaf_urma_provider_common>
    $<TARGET_OBJECTS:${ACCESSOR_TARGET}>
    $<TARGET_OBJECTS:${UDMA_TARGET}>
    $<TARGET_OBJECTS:${UMMU_TARGET}>
  )
  target_link_libraries(${TARGET_NAME} PRIVATE pthread m dl rt)
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
  "${KVAF_URMA_PUBLIC_INCLUDE_DIR}"
  "${KVAF_URMA_COMMON_INCLUDE_DIR}"
)

add_library(kvaf_public_abi_udma_probe OBJECT public_abi_probe_udma.c)
set_target_properties(kvaf_public_abi_udma_probe PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  POSITION_INDEPENDENT_CODE YES
)
target_include_directories(kvaf_public_abi_udma_probe PRIVATE
  ${KVAF_UDMA_PRIVATE_INCLUDE_DIRS})

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
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(kvaf_urma_provider_aicpu_contract_probe PRIVATE dl)
set_target_properties(kvaf_urma_provider_aicpu_contract_probe PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  SKIP_BUILD_RPATH YES
  BUILD_WITH_INSTALL_RPATH NO
  INSTALL_RPATH ""
)

add_executable(kvaf_ummu_uapi_probe official_ummu_uapi_probe.c)
target_include_directories(kvaf_ummu_uapi_probe PRIVATE
  "${KVAF_UMMU_API_INCLUDE_DIR}"
  "${KVAF_UMMU_UAPI_INCLUDE_DIR}"
)
set_target_properties(kvaf_ummu_uapi_probe PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  SKIP_BUILD_RPATH YES
  BUILD_WITH_INSTALL_RPATH NO
  INSTALL_RPATH ""
)

add_executable(kvaf_udma_uapi_probe udma_uapi_probe.c)
target_include_directories(kvaf_udma_uapi_probe PRIVATE
  "${KVAF_URMA_PUBLIC_INCLUDE_DIR}"
  "${KVAF_UDMA_UAPI_INCLUDE_DIR}")
set_target_properties(kvaf_udma_uapi_probe PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  SKIP_BUILD_RPATH YES
  BUILD_WITH_INSTALL_RPATH NO
  INSTALL_RPATH ""
)

add_executable(kvaf_udma_private_abi_probe udma_private_abi_probe.c)
target_include_directories(kvaf_udma_private_abi_probe PRIVATE
  ${KVAF_UDMA_PRIVATE_INCLUDE_DIRS})
set_target_properties(kvaf_udma_private_abi_probe PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
  SKIP_BUILD_RPATH YES
  BUILD_WITH_INSTALL_RPATH NO
  INSTALL_RPATH ""
)

message(STATUS "KVAF UMDK source root: ${KVAF_UMDK_ROOT}")
message(STATUS "KVAF paired libummu root: ${KVAF_LIBUMMU_ROOT}")
message(STATUS "KVAF liburma/common/UDMA/UMMU linkage: source-built internal")
