cmake_minimum_required(VERSION 3.18)

project(kvaf LANGUAGES CXX)

option(KVAF_BUILD_PYTHON "Build the optional Pybind Python registration module" OFF)
option(KVAF_BUILD_A5_AIV_URMA
  "Build the Ascend 950 AIV+SIMT+URMA H2D backend and smoke executable" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

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
