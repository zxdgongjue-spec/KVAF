find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_executable(kvaf_startup_selection_test startup_selection_test.cc)
target_include_directories(kvaf_startup_selection_test PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include ${PROJECT_SOURCE_DIR}/common/include)
add_test(NAME kvaf_startup_selection_test COMMAND kvaf_startup_selection_test)
add_test(NAME kvaf_aiv_test_cleanup_python
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_aiv_test_cleanup.py")

add_test(NAME kvaf_dsa_keys_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_dsa_keys.py")

add_executable(kvaf_batch_get_schedule_test batch_get_schedule_test.cc)
target_include_directories(kvaf_batch_get_schedule_test PRIVATE
  ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include)
add_test(NAME kvaf_batch_get_schedule_test COMMAND kvaf_batch_get_schedule_test)
add_executable(kvaf_batch_get_kernel_model_test batch_get_kernel_model_test.cc)
target_include_directories(kvaf_batch_get_kernel_model_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/aiv_model
  ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include)
add_test(NAME kvaf_batch_get_kernel_model_test COMMAND kvaf_batch_get_kernel_model_test)
add_test(NAME kvaf_batch_get_inflight_config_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_batch_get_inflight_config.py")

add_test(
  NAME kvaf_npu_eid_parser_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_npu_eids.py"
)

if(KVAF_BUILD_PYTHON)
  add_test(
    NAME kvaf_runtime_mock_python
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/test_runtime_mock.py"
  )
  set_tests_properties(kvaf_runtime_mock_python PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
  )

  if(KVAF_ENABLE_AIV_BATCH_GET)
    add_test(
      NAME kvaf_npu_batch_get_python
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/python/test_npu_batch_get.py"
    )
    set_tests_properties(kvaf_npu_batch_get_python PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
      TIMEOUT 240
    )

    if(KVAF_ENABLE_AIV_BATCH_PUT)
      add_test(
        NAME kvaf_npu_batch_put_python
        COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/python/test_npu_batch_put.py"
      )
      set_tests_properties(kvaf_npu_batch_put_python PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
        TIMEOUT 240
      )
    endif()
  endif()
else()
  message(STATUS
    "Skipping direct Runtime Python tests; configure with -DKVAF_BUILD_PYTHON=ON")
endif()

add_executable(kvaf_sequential_batch_get_kernel_model_test
  sequential_batch_get_kernel_model_test.cc
  ${PROJECT_SOURCE_DIR}/runtime/src/cpu_gather_executor.cc)
target_include_directories(kvaf_sequential_batch_get_kernel_model_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/aiv_model
  ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_SOURCE_DIR}/common/include)
target_link_libraries(kvaf_sequential_batch_get_kernel_model_test PRIVATE pthread)
add_test(NAME kvaf_sequential_batch_get_kernel_model_test COMMAND kvaf_sequential_batch_get_kernel_model_test)
add_test(NAME kvaf_aiv_sequential_hardware_python
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_aiv_sequential_batch_get.py")
set_tests_properties(kvaf_aiv_sequential_hardware_python PROPERTIES
  SKIP_RETURN_CODE 77 TIMEOUT 600 ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")

# Compile hardware Host sources even on development machines without CANN.
# Object-only: never link the declaration doubles into the production libraries.
if(NOT KVAF_ENABLE_AIV_BATCH_GET)
  add_test(NAME kvaf_aiv_kernel_build_python
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_aiv_kernel_build.py"
            "${CMAKE_COMMAND}")
  add_executable(kvaf_sequential_loader_test sequential_batch_get_loader_test.cc
    ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/host/execute_batch_get_loader.cpp
    ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/host/execute_sequential_batch_get_loader.cpp)
  target_include_directories(kvaf_sequential_loader_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fake_acl
    ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include
    ${PROJECT_SOURCE_DIR}/access_operator/include
    ${PROJECT_SOURCE_DIR}/common/include)
  foreach(scenario compat explicit custom relative missing)
    add_test(NAME kvaf_sequential_loader_${scenario} COMMAND kvaf_sequential_loader_test ${scenario})
  endforeach()
  add_executable(kvaf_sequential_host_test sequential_batch_get_host_test.cc
    ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/host/sequential_batch_get.cpp)
  target_include_directories(kvaf_sequential_host_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fake_acl
    ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include
    ${PROJECT_SOURCE_DIR}/access_operator/include
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_SOURCE_DIR}/common/include)
  add_test(NAME kvaf_sequential_host_test COMMAND kvaf_sequential_host_test)
  add_library(kvaf_sequential_host_compile_check OBJECT
    ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/host/sequential_batch_get.cpp
    ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/host/execute_sequential_batch_get_loader.cpp
    ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/host/batch_get_host.cpp
    ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_put/host/batch_put_host.cpp
    ${PROJECT_SOURCE_DIR}/access_operator/src/access_operator.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/direct_hccp/direct_hccp_remote_transport.cc
    ${PROJECT_SOURCE_DIR}/runtime/src/npu_runtime.cc)
  target_include_directories(kvaf_sequential_host_compile_check PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fake_acl
    ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include
    ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_put/include
    ${PROJECT_SOURCE_DIR}/access_operator/include
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_SOURCE_DIR}/meta_manager/include
    ${PROJECT_SOURCE_DIR}/common/include)
  target_compile_definitions(kvaf_sequential_host_compile_check PRIVATE KVAF_ENABLE_AIV_BATCH_PUT=1)
endif()
