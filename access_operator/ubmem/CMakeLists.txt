# UBmem is an A5 local Host-DDR data plane. It intentionally has no dependency
# on the Direct-HCCP, UMDK, liburma, or CPU-service targets.
set(KVAF_UBMEM_AICORE_ARCH "dav-c310-vec" CACHE STRING "UBmem A5 vector architecture")
if(NOT SOC_VERSION MATCHES "^Ascend950[A-Za-z0-9_]*$")
  message(FATAL_ERROR "UBmem requires explicit SOC_VERSION for the active Ascend950 device")
endif()
if(NOT KVAF_UBMEM_AICORE_ARCH STREQUAL "dav-c310-vec")
  message(FATAL_ERROR "UBmem currently supports only A5 dav-c310-vec")
endif()
set(_ubmem_ccec_dir "${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler/bin")
find_program(KVAF_UBMEM_CCEC NAMES ccec HINTS "${_ubmem_ccec_dir}" NO_DEFAULT_PATH)
find_program(KVAF_UBMEM_LLD NAMES ld.lld HINTS "${_ubmem_ccec_dir}" NO_DEFAULT_PATH)
find_path(KVAF_UBMEM_OPERATOR_INCLUDE NAMES kernel_operator.h HINTS
  "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw/interface"
  "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/tikcfw"
  "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw/interface"
  "${ASCEND_CANN_PACKAGE_PATH}/tikcpp/tikcfw"
  "${ASCEND_CANN_PACKAGE_PATH}/include/tikcpp/tikcfw/interface"
  "${ASCEND_CANN_PACKAGE_PATH}/include/tikcpp/tikcfw"
  "${ASCEND_CANN_PACKAGE_PATH}/include/ascendc"
  NO_DEFAULT_PATH)
if(NOT KVAF_UBMEM_CCEC OR NOT KVAF_UBMEM_LLD OR NOT KVAF_UBMEM_OPERATOR_INCLUDE)
  message(FATAL_ERROR "UBmem requires matching ccec, ld.lld, and kernel_operator.h under ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")
endif()
set(_ubmem_includes "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
                    "-I${KVAF_UBMEM_OPERATOR_INCLUDE}")
foreach(_root tools/tikcpp/tikcfw tikcpp/tikcfw include/tikcpp/tikcfw include/ascendc)
  foreach(_suffix "" /interface /impl /lib)
    list(APPEND _ubmem_includes "-I${ASCEND_CANN_PACKAGE_PATH}/${_root}${_suffix}")
  endforeach()
endforeach()
set(_ubmem_objects)
foreach(_operation get put)
  set(_source "${CMAKE_CURRENT_SOURCE_DIR}/kernel/batch_${_operation}.asc")
  set(_temporary "${CMAKE_CURRENT_BINARY_DIR}/batch_${_operation}.tmp.o")
  set(_object "${CMAKE_BINARY_DIR}/bin/kvaf_ubmem_batch_${_operation}.o")
  add_custom_command(OUTPUT "${_object}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_BINARY_DIR}/bin"
    COMMAND "${KVAF_UBMEM_CCEC}" -c -x cce -O2 "${_source}" -o "${_temporary}"
      "--cce-aicore-arch=${KVAF_UBMEM_AICORE_ARCH}"
      --cce-aicore-input-parameter-size=8192 --cce-aicore-only
      -mllvm -cce-aicore-function-stack-size=0x4000
      -mllvm -cce-aicore-stack-size=0x4000
      -mllvm -cce-aicore-record-overflow=false
      -mllvm -cce-aicore-addr-transform -mllvm -cce-aicore-jump-expand=true
      -std=c++17 -fstack-protector-all ${_ubmem_includes}
    COMMAND "${KVAF_UBMEM_LLD}" -m aicorelinux -Ttext=0 "${_temporary}" -static -o "${_object}"
    DEPENDS "${_source}" "${CMAKE_CURRENT_SOURCE_DIR}/kernel/copy_primitives.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/ubmem/transfer_abi.h"
    VERBATIM)
  list(APPEND _ubmem_objects "${_object}")
endforeach()
add_custom_target(kvaf_ubmem_kernels ALL DEPENDS ${_ubmem_objects})

add_library(kvaf_ubmem_access_operator host/access_operator.cc host/kernel_loader.cc)
target_include_directories(kvaf_ubmem_access_operator PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_libraries(kvaf_ubmem_access_operator PUBLIC kvaf_access_operator
  PRIVATE kvaf_meta_manager ascendcl)
target_link_directories(kvaf_ubmem_access_operator PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_compile_features(kvaf_ubmem_access_operator PUBLIC cxx_std_17)
add_dependencies(kvaf_ubmem_access_operator kvaf_ubmem_kernels)

if(BUILD_TESTING)
  add_executable(kvaf_ubmem_copy_model_test tests/copy_model_test.cc)
  target_include_directories(kvaf_ubmem_copy_model_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/model" "${CMAKE_CURRENT_SOURCE_DIR}/include")
  add_test(NAME kvaf_ubmem_copy_model_test COMMAND kvaf_ubmem_copy_model_test)
  add_executable(kvaf_ubmem_host_contract_test tests/host_contract_test.cc host/access_operator.cc)
  target_include_directories(kvaf_ubmem_host_contract_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/model" "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include" "${PROJECT_SOURCE_DIR}/common/include"
    "${PROJECT_SOURCE_DIR}/meta_manager/include")
  add_test(NAME kvaf_ubmem_host_contract_test COMMAND kvaf_ubmem_host_contract_test)
  # This fault test deliberately exercises the process-lifetime quarantine.
  set_tests_properties(kvaf_ubmem_host_contract_test PROPERTIES ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0")
endif()
