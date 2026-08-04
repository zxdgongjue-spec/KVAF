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
  "Optional asc-devkit checkout that overrides the CANN device header tree")
set(KVAF_A5_COMPILE_MODE "auto" CACHE STRING
  "Ascend compiler mode: auto, xasc, or xcce")
set_property(CACHE KVAF_A5_COMPILE_MODE PROPERTY STRINGS auto xasc xcce)
option(KVAF_A5_BUILD_HCOMM_BACKEND
  "Build the legacy HCOMM resource-control smoke target" ON)
option(KVAF_A5_BUILD_DIRECT_URMA_BACKEND
  "Build the liburma/UDMA resource-control smoke target without HCOMM" OFF)

if(NOT KVAF_A5_BUILD_HCOMM_BACKEND AND
   NOT KVAF_A5_BUILD_DIRECT_URMA_BACKEND)
  message(FATAL_ERROR
    "Enable at least one A5 resource-control backend")
endif()

find_path(KVAF_ACL_INCLUDE_DIR acl/acl.h
  PATHS
    "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
  NO_DEFAULT_PATH
  REQUIRED
)
if(KVAF_A5_BUILD_DIRECT_URMA_BACKEND)
  find_path(KVAF_RUNTIME_INCLUDE_DIR runtime/rt_external_kernel.h
    PATHS
      "${ASCEND_HOME_PATH}/include"
      "${ASCEND_HOME_PATH}/pkg_inc"
      "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
      "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc"
    NO_DEFAULT_PATH
    REQUIRED
  )
endif()
if(KVAF_A5_BUILD_HCOMM_BACKEND)
  find_path(KVAF_HCOMM_RESOURCE_INCLUDE_DIR hcomm/hcomm_res.h
    PATHS
      "${ASCEND_HOME_PATH}/include"
      "${ASCEND_HOME_PATH}/pkg_inc"
      "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include"
      "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/pkg_inc"
    NO_DEFAULT_PATH
    REQUIRED
  )
endif()
set(KVAF_ASC_DEVKIT_INCLUDE_DIRS)
if(KVAF_ASC_DEVKIT_PATH)
  get_filename_component(KVAF_ASC_DEVKIT_PATH
    "${KVAF_ASC_DEVKIT_PATH}" ABSOLUTE)
  foreach(_KVAF_ASC_DEVKIT_REQUIRED_FILE IN ITEMS
      include/kernel_operator.h
      include/basic_api/kernel_basic_intf.h
      include/simt_api/asc_simt.h)
    if(NOT EXISTS
        "${KVAF_ASC_DEVKIT_PATH}/${_KVAF_ASC_DEVKIT_REQUIRED_FILE}")
      message(FATAL_ERROR
        "KVAF_ASC_DEVKIT_PATH is not a complete asc-devkit checkout; missing "
        "${KVAF_ASC_DEVKIT_PATH}/${_KVAF_ASC_DEVKIT_REQUIRED_FILE}")
    endif()
  endforeach()
  # Keep one complete AscendC tree together. The device data plane no longer
  # includes the version-sensitive hcomm/hcomm.h API. HCOMM remains a
  # resource-creation dependency only when the legacy backend is enabled.
  list(APPEND KVAF_ASC_DEVKIT_INCLUDE_DIRS
    "${KVAF_ASC_DEVKIT_PATH}/impl/adv_api"
    "${KVAF_ASC_DEVKIT_PATH}/impl/basic_api"
    "${KVAF_ASC_DEVKIT_PATH}/impl/basic_api/reg_compute"
    "${KVAF_ASC_DEVKIT_PATH}/impl/c_api"
    "${KVAF_ASC_DEVKIT_PATH}/impl/simt_api"
    "${KVAF_ASC_DEVKIT_PATH}/impl/utils"
    "${KVAF_ASC_DEVKIT_PATH}"
    "${KVAF_ASC_DEVKIT_PATH}/include"
    "${KVAF_ASC_DEVKIT_PATH}/include/adv_api"
    "${KVAF_ASC_DEVKIT_PATH}/include/adv_api/matmul"
    "${KVAF_ASC_DEVKIT_PATH}/include/aicpu_api"
    "${KVAF_ASC_DEVKIT_PATH}/include/basic_api"
    "${KVAF_ASC_DEVKIT_PATH}/include/basic_api/reg_compute"
    "${KVAF_ASC_DEVKIT_PATH}/include/c_api"
    "${KVAF_ASC_DEVKIT_PATH}/include/simt_api"
    "${KVAF_ASC_DEVKIT_PATH}/include/utils"
  )
else()
  foreach(_KVAF_CANN_REQUIRED_FILE IN ITEMS
      kernel_operator.h
      simt_api/asc_simt.h)
    if(NOT EXISTS "${KVAF_ACL_INCLUDE_DIR}/${_KVAF_CANN_REQUIRED_FILE}")
      message(FATAL_ERROR
        "Installed CANN device headers are incomplete; missing "
        "${KVAF_ACL_INCLUDE_DIR}/${_KVAF_CANN_REQUIRED_FILE}")
    endif()
  endforeach()
endif()

if(KVAF_ASC_DEVKIT_PATH)
  message(STATUS
    "KVAF AscendC device headers: ${KVAF_ASC_DEVKIT_PATH}")
else()
  message(STATUS
    "KVAF AscendC device headers: ${ASCEND_HOME_PATH}")
endif()
message(STATUS
  "KVAF AIV data plane: direct UBC/CTP SQ/CQ (no device HCOMM API)")
if(KVAF_A5_BUILD_HCOMM_BACKEND)
  message(STATUS
    "KVAF HCOMM host resources: "
    "${KVAF_HCOMM_RESOURCE_INCLUDE_DIR}/hcomm/hcomm_res.h")
endif()
if(KVAF_A5_BUILD_DIRECT_URMA_BACKEND)
  message(STATUS
    "KVAF direct URMA resources: source-linked liburma/UDMA/UMMU via "
    "Host+AICPU (no HCOMM or dynamic UDMA provider)")
endif()

find_library(KVAF_ASCENDCL_LIBRARY ascendcl
  PATHS
    "${ASCEND_HOME_PATH}/lib64"
    "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
  NO_DEFAULT_PATH
  REQUIRED
)
if(KVAF_A5_BUILD_HCOMM_BACKEND)
  find_library(KVAF_HCOMM_LIBRARY hcomm
    PATHS
      "${ASCEND_HOME_PATH}/lib64"
      "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
    NO_DEFAULT_PATH
    REQUIRED
  )
endif()

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

function(kvaf_configure_a5_target target)
  if(KVAF_ASC_DEVKIT_INCLUDE_DIRS)
    target_include_directories(${target}
      BEFORE PRIVATE
        ${KVAF_ASC_DEVKIT_INCLUDE_DIRS}
    )
  endif()

  target_include_directories(${target}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/provider/include
      ${CMAKE_CURRENT_SOURCE_DIR}/../include
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${KVAF_ACL_INCLUDE_DIR}
      "${ASCEND_HOME_PATH}/compiler/tikcpp"
      "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw"
      "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface"
      "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl"
      "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp"
      "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw"
      "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw/interface"
      "${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw/impl"
  )

  target_compile_options(${target}
    PRIVATE
      ${KVAF_A5_COMPILE_OPTIONS}
      -Wno-ignored-attributes
  )
  target_link_options(${target}
    PRIVATE
      --cce-fatobj-link
      ${KVAF_A5_ARCH_LINK_OPTIONS}
  )
  target_link_libraries(${target}
    PRIVATE
      ${KVAF_ASCENDCL_LIBRARY}
      dl
      pthread
  )
endfunction()

# Plain Host control binaries must not inherit AscendC device compilation or
# fat-object link options.  They launch already-packaged AICPU kernels through
# ACL/runtime only and contain no AIV/AIC translation unit.
function(kvaf_configure_a5_host_target target)
  target_include_directories(${target}
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/provider/include
      ${CMAKE_CURRENT_SOURCE_DIR}/../include
      ${CMAKE_CURRENT_SOURCE_DIR}/../../include
      ${KVAF_ACL_INCLUDE_DIR}
      ${KVAF_RUNTIME_INCLUDE_DIR}
  )
  target_compile_options(${target} PRIVATE -Wall -Wextra -Werror)
  target_link_libraries(${target}
    PRIVATE
      ${KVAF_ASCENDCL_LIBRARY}
      dl
      pthread
  )
endfunction()

if(KVAF_A5_BUILD_HCOMM_BACKEND)
  add_executable(kvaf_a5_aiv_urma_h2d_smoke
    src/aiv_simt_urma_h2d_kernel.cpp
    src/host_ddr_mapping.cpp
    src/hcomm_urma_resources.cpp
    smoke/aiv_simt_urma_h2d_smoke.cpp
  )
  kvaf_configure_a5_target(kvaf_a5_aiv_urma_h2d_smoke)
  target_include_directories(kvaf_a5_aiv_urma_h2d_smoke PRIVATE
    ${KVAF_HCOMM_RESOURCE_INCLUDE_DIR}
  )
  target_link_libraries(kvaf_a5_aiv_urma_h2d_smoke PRIVATE
    ${KVAF_HCOMM_LIBRARY}
  )

  # This deliberately contains no AIV kernel launch and creates no channel,
  # Jetty, SQ, CQ or WQE.  It is the first hardware-bearing stage for the
  # HCCP-backed resource lifecycle: Host DDR/HBM MR registration, descriptor
  # export where applicable, and strict resource-before-memory teardown.
  #
  # `hcomm_res.h` pulls in CANN's secure-C helpers.  On CANN 9.1 bisheng's
  # ordinary Host link path omits their runtime and the C++ standard runtime;
  # reuse the proven AscendC fat-object link path used by the HCOMM smoke
  # target.  This is a link/runtime choice only: this target has no device
  # kernel translation unit and never launches an AIV kernel.
  add_executable(kvaf_a5_hccp_resource_lifecycle_probe
    src/host_ddr_mapping.cpp
    src/hcomm_urma_resources.cpp
    smoke/hccp_resource_lifecycle_probe.cpp
  )
  kvaf_configure_a5_target(kvaf_a5_hccp_resource_lifecycle_probe)
  target_include_directories(kvaf_a5_hccp_resource_lifecycle_probe PRIVATE
    ${KVAF_HCOMM_RESOURCE_INCLUDE_DIR}
  )
  target_link_libraries(kvaf_a5_hccp_resource_lifecycle_probe PRIVATE
    ${KVAF_HCOMM_LIBRARY}
  )

  # The next HCCP stage is still strictly zero-WQE.  It creates the exact
  # asymmetric Host-CPU / Device-AIV channel that production will later hand
  # to AIV, then snapshots its queue metadata and tears it down.  It contains
  # no device kernel translation unit and never launches an AIV kernel.
  add_executable(kvaf_a5_hccp_aiv_channel_probe
    src/host_ddr_mapping.cpp
    src/hcomm_urma_resources.cpp
    smoke/hccp_aiv_channel_probe.cpp
  )
  kvaf_configure_a5_target(kvaf_a5_hccp_aiv_channel_probe)
  target_include_directories(kvaf_a5_hccp_aiv_channel_probe PRIVATE
    ${KVAF_HCOMM_RESOURCE_INCLUDE_DIR}
  )
  target_link_libraries(kvaf_a5_hccp_aiv_channel_probe PRIVATE
    ${KVAF_HCOMM_LIBRARY}
  )
endif()

if(KVAF_A5_BUILD_DIRECT_URMA_BACKEND)
  add_executable(kvaf_a5_aiv_direct_urma_h2d_smoke
    src/aiv_simt_urma_h2d_kernel.cpp
    src/host_ddr_mapping.cpp
    src/direct_urma_resources.cpp
    smoke/aiv_simt_direct_urma_h2d_smoke.cpp
  )
  kvaf_configure_a5_target(kvaf_a5_aiv_direct_urma_h2d_smoke)
  target_include_directories(kvaf_a5_aiv_direct_urma_h2d_smoke PRIVATE
    ${KVAF_RUNTIME_INCLUDE_DIR}
  )

  # This binary does not open an EID, URMA device, UMMU mapping, SQ or CQ.  It
  # proves only that one bounded AICPU KFC invocation and an AIV kernel on a
  # distinct stream can communicate through cache-maintained GM records.
  add_executable(kvaf_a5_aicpu_owner_probe
    src/aiv_simt_urma_h2d_kernel.cpp
    smoke/aicpu_owner_concurrency_probe.cpp
  )
  kvaf_configure_a5_target(kvaf_a5_aicpu_owner_probe)
  target_include_directories(kvaf_a5_aicpu_owner_probe PRIVATE
    ${KVAF_RUNTIME_INCLUDE_DIR}
  )

  # Staged AICPU Device-runtime compatibility probe.  Discover performs no
  # URMA init/resource calls, but its dlopen may run official constructors;
  # every mode is therefore owned by the guarded external supervisor.
  add_executable(kvaf_a5_aicpu_runtime_urma_probe
    smoke/aicpu_runtime_urma_probe.cpp
  )
  kvaf_configure_a5_host_target(kvaf_a5_aicpu_runtime_urma_probe)
endif()
