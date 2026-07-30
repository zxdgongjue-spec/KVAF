if(NOT DEFINED ASCEND_HOME_PATH)
  if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
  else()
    message(FATAL_ERROR
      "KVAF_BUILD_A5_AIV_URMA requires ASCEND_HOME_PATH; source the CANN "
      "set_env.sh before configuring")
  endif()
endif()

set(KVAF_ASC_DEVKIT_PATH "" CACHE PATH
  "Optional asc-devkit checkout that overrides CANN's adv_api/hcomm headers")
set(KVAF_A5_COMPILE_MODE "auto" CACHE STRING
  "Ascend compiler mode: auto, xasc, or xcce")
set_property(CACHE KVAF_A5_COMPILE_MODE PROPERTY STRINGS auto xasc xcce)

find_path(KVAF_ACL_INCLUDE_DIR acl/acl.h
  PATHS
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
  NO_DEFAULT_PATH
  REQUIRED
)
find_path(KVAF_HCOMM_RESOURCE_INCLUDE_DIR hcomm/hcomm_res.h
  PATHS
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/pkg_inc"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc"
  NO_DEFAULT_PATH
  REQUIRED
)
if(KVAF_ASC_DEVKIT_PATH)
  get_filename_component(KVAF_ASC_DEVKIT_PATH
    "${KVAF_ASC_DEVKIT_PATH}" ABSOLUTE)
  set(_KVAF_HCOMM_ADV_INCLUDE_DIR
    "${KVAF_ASC_DEVKIT_PATH}/include/adv_api")
  if(NOT EXISTS "${_KVAF_HCOMM_ADV_INCLUDE_DIR}/hcomm/hcomm.h")
    message(FATAL_ERROR
      "KVAF_ASC_DEVKIT_PATH does not contain include/adv_api/hcomm/hcomm.h: "
      "${KVAF_ASC_DEVKIT_PATH}")
  endif()
  # An explicitly supplied asc-devkit is authoritative. FORCE also replaces a
  # stale CMake cache entry left by a previous configure against CANN headers.
  set(KVAF_HCOMM_ADV_INCLUDE_DIR "${_KVAF_HCOMM_ADV_INCLUDE_DIR}"
    CACHE PATH "HCOMM AIV advanced API include directory" FORCE)
else()
  find_path(KVAF_HCOMM_ADV_INCLUDE_DIR hcomm/hcomm.h
    PATHS
      "${ASCEND_HOME_PATH}/include/adv_api"
      "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/adv_api"
    NO_DEFAULT_PATH
    REQUIRED
  )
endif()

set(_KVAF_HCOMM_HEADER
  "${KVAF_HCOMM_ADV_INCLUDE_DIR}/hcomm/hcomm.h")
set(_KVAF_HCOMM_COMMON_HEADER
  "${KVAF_HCOMM_ADV_INCLUDE_DIR}/hcomm/hcomm_common.h")
if(NOT EXISTS "${_KVAF_HCOMM_COMMON_HEADER}")
  message(FATAL_ERROR
    "Selected HCOMM AIV API is incomplete; missing "
    "${_KVAF_HCOMM_COMMON_HEADER}")
endif()
file(READ "${_KVAF_HCOMM_HEADER}" _KVAF_HCOMM_HEADER_CONTENT)
file(READ "${_KVAF_HCOMM_COMMON_HEADER}" _KVAF_HCOMM_COMMON_CONTENT)
foreach(_KVAF_REQUIRED_TOKEN IN ITEMS ReadNbi Drain)
  string(FIND "${_KVAF_HCOMM_HEADER_CONTENT}"
    "${_KVAF_REQUIRED_TOKEN}" _KVAF_REQUIRED_TOKEN_OFFSET)
  if(_KVAF_REQUIRED_TOKEN_OFFSET EQUAL -1)
    message(FATAL_ERROR
      "Selected HCOMM AIV API does not provide ${_KVAF_REQUIRED_TOKEN}: "
      "${_KVAF_HCOMM_HEADER}. KVAF requires the CANN 9.1.0 HCOMM "
      "Init/ReadNbi/Commit/Drain API; use the asc-devkit revision that "
      "matches the installed CANN package.")
  endif()
endforeach()
foreach(_KVAF_REQUIRED_TOKEN IN ITEMS
    ChannelHandle COMM_PROTOCOL_UBC_CTP)
  string(FIND "${_KVAF_HCOMM_COMMON_CONTENT}"
    "${_KVAF_REQUIRED_TOKEN}" _KVAF_REQUIRED_TOKEN_OFFSET)
  if(_KVAF_REQUIRED_TOKEN_OFFSET EQUAL -1)
    message(FATAL_ERROR
      "Selected HCOMM AIV API does not provide ${_KVAF_REQUIRED_TOKEN}: "
      "${_KVAF_HCOMM_COMMON_HEADER}. KVAF requires the CANN 9.1.0 HCOMM "
      "Init/ReadNbi/Commit/Drain API; use the asc-devkit revision that "
      "matches the installed CANN package.")
  endif()
endforeach()

message(STATUS "KVAF HCOMM AIV API: ${_KVAF_HCOMM_HEADER}")
message(STATUS
  "KVAF HCOMM host resources: "
  "${KVAF_HCOMM_RESOURCE_INCLUDE_DIR}/hcomm/hcomm_res.h")

find_library(KVAF_ASCENDCL_LIBRARY ascendcl
  PATHS
    "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
  NO_DEFAULT_PATH
  REQUIRED
)
find_library(KVAF_HCOMM_LIBRARY hcomm
  PATHS
    "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
  NO_DEFAULT_PATH
  REQUIRED
)

set(KVAF_A5_EFFECTIVE_COMPILE_MODE "${KVAF_A5_COMPILE_MODE}")
if(KVAF_A5_EFFECTIVE_COMPILE_MODE STREQUAL "auto")
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -v
    OUTPUT_VARIABLE KVAF_BISHENG_STDOUT
    ERROR_VARIABLE KVAF_BISHENG_STDERR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  set(KVAF_BISHENG_VERSION
    "${KVAF_BISHENG_STDOUT} ${KVAF_BISHENG_STDERR}")
  string(REGEX MATCH
    "([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])"
    KVAF_BISHENG_DATE "${KVAF_BISHENG_VERSION}")
  if(KVAF_BISHENG_DATE)
    string(REPLACE "-" "" KVAF_BISHENG_DATE_NUMBER "${KVAF_BISHENG_DATE}")
  else()
    set(KVAF_BISHENG_DATE_NUMBER 0)
  endif()
  if(KVAF_BISHENG_DATE_NUMBER GREATER_EQUAL 20260327)
    set(KVAF_A5_EFFECTIVE_COMPILE_MODE "xasc")
  else()
    set(KVAF_A5_EFFECTIVE_COMPILE_MODE "xcce")
  endif()
endif()

if(KVAF_A5_EFFECTIVE_COMPILE_MODE STREQUAL "xasc")
  set(KVAF_A5_COMPILE_OPTIONS
    -xasc
    --npu-arch=dav-3510
    --cce-auto-infer-kernel-type=false
  )
  set(KVAF_A5_ARCH_LINK_OPTIONS)
elseif(KVAF_A5_EFFECTIVE_COMPILE_MODE STREQUAL "xcce")
  set(KVAF_A5_COMPILE_OPTIONS
    -xcce
    --cce-aicore-arch=dav-c310
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
  )
  set(KVAF_A5_ARCH_LINK_OPTIONS --cce-aicore-arch=dav-c310)
else()
  message(FATAL_ERROR
    "KVAF_A5_COMPILE_MODE must be auto, xasc, or xcce")
endif()

message(STATUS
  "KVAF A5 AIV+SIMT+URMA compile mode: ${KVAF_A5_EFFECTIVE_COMPILE_MODE}")

add_executable(kvaf_a5_aiv_urma_h2d_smoke
  src/aiv_simt_urma_h2d_kernel.cpp
  src/hcomm_urma_resources.cpp
  smoke/aiv_simt_urma_h2d_smoke.cpp
)

target_include_directories(kvaf_a5_aiv_urma_h2d_smoke
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    # Keep the AIV API before CANN's general include directory. Some CANN
    # packages install another hcomm/hcomm.h there with an incompatible API.
    ${KVAF_HCOMM_ADV_INCLUDE_DIR}
    ${KVAF_ACL_INCLUDE_DIR}
    ${KVAF_HCOMM_RESOURCE_INCLUDE_DIR}
    "${ASCEND_HOME_PATH}/compiler/tikcpp"
    "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw"
    "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface"
    "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw/interface"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw/impl"
)

target_compile_options(kvaf_a5_aiv_urma_h2d_smoke
  PRIVATE
    ${KVAF_A5_COMPILE_OPTIONS}
    -Wno-ignored-attributes
)
target_link_options(kvaf_a5_aiv_urma_h2d_smoke
  PRIVATE
    --cce-fatobj-link
    ${KVAF_A5_ARCH_LINK_OPTIONS}
)
target_link_libraries(kvaf_a5_aiv_urma_h2d_smoke
  PRIVATE
    ${KVAF_ASCENDCL_LIBRARY}
    ${KVAF_HCOMM_LIBRARY}
    dl
    pthread
)
