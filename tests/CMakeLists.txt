find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(Threads REQUIRED)

add_executable(kvaf_initialize_gate_test initialize_gate_test.cc)
target_include_directories(kvaf_initialize_gate_test PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/src)
target_link_libraries(kvaf_initialize_gate_test PRIVATE kvaf_common Threads::Threads)
add_test(NAME kvaf_initialize_gate_test COMMAND kvaf_initialize_gate_test)
set_tests_properties(kvaf_initialize_gate_test PROPERTIES TIMEOUT 30)

add_executable(kvaf_append_only_meta_test append_only_meta_manager_test.cc)
target_link_libraries(kvaf_append_only_meta_test PRIVATE kvaf_meta_manager)
add_test(NAME kvaf_append_only_meta_test COMMAND kvaf_append_only_meta_test)
add_executable(kvaf_append_only_runtime_test append_only_runtime_test.cc)
target_link_libraries(kvaf_append_only_runtime_test PRIVATE kvaf_runtime)
add_test(NAME kvaf_append_only_runtime_test COMMAND kvaf_append_only_runtime_test)
add_test(NAME kvaf_batch_get_ranges_perf_host
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_batch_get_ranges_perf.py")
add_executable(kvaf_batch_get_ranges_runtime_test batch_get_ranges_runtime_test.cc)
target_link_libraries(kvaf_batch_get_ranges_runtime_test PRIVATE kvaf_runtime)
add_test(NAME kvaf_batch_get_ranges_runtime_test COMMAND kvaf_batch_get_ranges_runtime_test)
add_executable(kvaf_batch_get_ranges_operator_test batch_get_ranges_operator_test.cc
  ${PROJECT_SOURCE_DIR}/access_operator/src/access_operator.cc)
target_include_directories(kvaf_batch_get_ranges_operator_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/aiv_host_model
  ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include)
target_link_libraries(kvaf_batch_get_ranges_operator_test PRIVATE kvaf_meta_manager kvaf_access_operator)
add_test(NAME kvaf_batch_get_ranges_operator_test COMMAND kvaf_batch_get_ranges_operator_test)

add_executable(kvaf_npu_runtime_config_test npu_runtime_config_test.cc)
target_link_libraries(kvaf_npu_runtime_config_test PRIVATE kvaf_runtime)
add_test(NAME kvaf_npu_runtime_config_test COMMAND kvaf_npu_runtime_config_test)
add_test(NAME kvaf_build_script_python
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_build_script.py")

add_executable(kvaf_runtime_cleanup_test runtime_cleanup_test.cc)
target_link_libraries(kvaf_runtime_cleanup_test PRIVATE kvaf_runtime)
add_test(NAME kvaf_runtime_cleanup_test COMMAND kvaf_runtime_cleanup_test)

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
add_test(NAME kvaf_batch_get_unified_python
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_batch_get_unified.py")
add_test(NAME kvaf_batch_get_multicard_perf_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_batch_get_multicard_perf.py")

add_test(
  NAME kvaf_npu_eid_parser_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_npu_eids.py"
)

if(KVAF_BUILD_PYTHON)
  if(KVAF_ENABLE_AICPU_URMA)
    add_test(NAME kvaf_npu_aicpu_gather_ranges_python
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_npu_aicpu_gather_ranges.py")
    set_tests_properties(kvaf_npu_aicpu_gather_ranges_python PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
      SKIP_RETURN_CODE 77 TIMEOUT 300)
  endif()

  pybind11_add_module(_kvaf_constructor_test npu_constructor_binding.cc)
  target_include_directories(_kvaf_constructor_test PRIVATE ${PROJECT_SOURCE_DIR}/bindings/include)
  set_target_properties(_kvaf_constructor_test PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python")
  add_test(NAME kvaf_npu_constructor_python
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_npu_constructor.py")
  set_tests_properties(kvaf_npu_constructor_python PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python")

  add_test(
    NAME kvaf_runtime_mock_python
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/python/test_runtime_mock.py"
  )
  set_tests_properties(kvaf_runtime_mock_python PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
  )

  if(KVAF_ENABLE_ROCE_NPU)
    add_test(NAME kvaf_npu_roce_python
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_npu_batch_get.py"
              --transport roce)
    set_tests_properties(kvaf_npu_roce_python PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python" TIMEOUT 240)
  endif()

  if(KVAF_ENABLE_AIV_BATCH_GET)
    add_test(NAME kvaf_npu_append_only_python
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_npu_append_only.py")
    set_tests_properties(kvaf_npu_append_only_python PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
      SKIP_RETURN_CODE 77 TIMEOUT 300)
    add_test(NAME kvaf_npu_batch_get_ranges_python
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_npu_batch_get_ranges.py")
    set_tests_properties(kvaf_npu_batch_get_ranges_python PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
      SKIP_RETURN_CODE 77 TIMEOUT 300)
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
