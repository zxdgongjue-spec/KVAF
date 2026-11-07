cmake_minimum_required(VERSION 3.18)

project(kvaf LANGUAGES CXX)

set(KVAF_TRANSPORT "urma" CACHE STRING "NPU data transport: urma or ubmem")
set_property(CACHE KVAF_TRANSPORT PROPERTY STRINGS urma ubmem)
if(KVAF_TRANSPORT STREQUAL "urma")
  set(KVAF_ENABLE_URMA 1)
  set(KVAF_ENABLE_UBMEM 0)
elseif(KVAF_TRANSPORT STREQUAL "ubmem")
  set(KVAF_ENABLE_URMA 0)
  set(KVAF_ENABLE_UBMEM 1)
else()
  message(FATAL_ERROR "KVAF_TRANSPORT must be urma or ubmem")
endif()

option(KVAF_BUILD_PYTHON "Build the optional Pybind Python registration module" OFF)
option(KVAF_ENABLE_AIV_BATCH_GET
       "Build the CANN-backed default BatchGet AccessOperator" OFF)
option(KVAF_ENABLE_AIV_BATCH_PUT
       "Build the CANN-backed Direct-HCCP BatchPut data plane"
       ${KVAF_ENABLE_AIV_BATCH_GET})
option(KVAF_ENABLE_AICPU_URMA "Build the optional AICPU UMDK GatherRead data plane" OFF)
option(KVAF_ENABLE_AICPU_CPU_PUSH "Build the Host direct-HCCP AICPU CPU push/scatter path" OFF)
option(KVAF_BUILD_REMOTE_CPU_SERVICE
       "Build the persistent CPU DDR service (URMA and UBmem)"
       OFF)
if(KVAF_ENABLE_AICPU_URMA)
  set(KVAF_ENABLE_NPU ON)
endif()
option(KVAF_ENABLE_NPU "Build CANN-backed NPU Runtime and Python APIs"
       ${KVAF_ENABLE_AIV_BATCH_GET})

if(KVAF_ENABLE_AIV_BATCH_GET AND NOT KVAF_ENABLE_NPU)
  message(FATAL_ERROR "KVAF_ENABLE_AIV_BATCH_GET requires KVAF_ENABLE_NPU")
endif()
if(KVAF_ENABLE_NPU AND KVAF_ENABLE_URMA AND NOT KVAF_ENABLE_AIV_BATCH_GET AND NOT KVAF_ENABLE_AICPU_URMA)
  message(FATAL_ERROR "The URMA NPU Runtime requires KVAF_ENABLE_AIV_BATCH_GET")
endif()
if(KVAF_ENABLE_UBMEM AND
   (KVAF_ENABLE_AIV_BATCH_GET OR KVAF_ENABLE_AIV_BATCH_PUT OR KVAF_BUILD_REMOTE_CPU_SERVICE))
  message(FATAL_ERROR "UBmem builds use KVAF_ENABLE_NPU without URMA AIV kernels or CPU service")
endif()

if(KVAF_ENABLE_AIV_BATCH_PUT AND NOT KVAF_ENABLE_AIV_BATCH_GET)
  message(FATAL_ERROR
    "KVAF_ENABLE_AIV_BATCH_PUT requires KVAF_ENABLE_AIV_BATCH_GET because "
    "DefaultAccessOperator is built by the CANN BatchGet integration")
endif()
if(KVAF_ENABLE_AICPU_CPU_PUSH AND NOT KVAF_ENABLE_AICPU_URMA)
  message(FATAL_ERROR "KVAF_ENABLE_AICPU_CPU_PUSH requires KVAF_ENABLE_AICPU_URMA")
endif()

# CANN/PyTorch NPU libraries use the legacy libstdc++ ABI. Keep this setting
# directory-wide so every KVAF target that exchanges STL-bearing public types
# (notably Status, requests and metadata vectors) has the same object layout.
# Applying it only to the binding and hardware operator lets Runtime construct
# ABI-1 objects that an ABI-0 callee later destroys, which corrupts the heap.
if(KVAF_ENABLE_NPU)
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
  message(STATUS "KVAF NPU C++ ABI: _GLIBCXX_USE_CXX11_ABI=0 (all targets)")
  if(NOT ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
      set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
    elseif(DEFINED ENV{ASCEND_HOME_PATH})
      set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
    else()
      set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
  endif()
  if(NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}/include/acl/acl.h")
    message(FATAL_ERROR "CANN ACL headers not found under ${ASCEND_CANN_PACKAGE_PATH}")
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)

add_subdirectory(common)
target_compile_definitions(kvaf_common INTERFACE
  KVAF_ENABLE_URMA=${KVAF_ENABLE_URMA}
  KVAF_ENABLE_UBMEM=${KVAF_ENABLE_UBMEM}
  KVAF_ENABLE_NPU=$<BOOL:${KVAF_ENABLE_NPU}>)
add_subdirectory(meta_manager)
add_subdirectory(access_operator)
add_subdirectory(runtime)

if(KVAF_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    RESULT_VARIABLE KVAF_PYBIND11_RESULT
    OUTPUT_VARIABLE KVAF_PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(NOT KVAF_PYBIND11_RESULT EQUAL 0)
    message(FATAL_ERROR "Unable to locate pybind11 from the selected Python interpreter")
  endif()

  list(APPEND CMAKE_PREFIX_PATH "${KVAF_PYBIND11_CMAKE_DIR}")
  find_package(pybind11 CONFIG REQUIRED)
  add_subdirectory(bindings)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
