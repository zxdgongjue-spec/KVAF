cmake_minimum_required(VERSION 3.18)

project(kvaf LANGUAGES CXX)

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
if(KVAF_ENABLE_AIV_BATCH_GET OR KVAF_ENABLE_AICPU_URMA)
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
  message(STATUS "KVAF AIV C++ ABI: _GLIBCXX_USE_CXX11_ABI=0 (all targets)")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)

add_subdirectory(common)
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
