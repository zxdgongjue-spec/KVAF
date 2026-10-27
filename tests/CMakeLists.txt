find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(Threads REQUIRED)

add_executable(kvaf_initialize_gate_test initialize_gate_test.cc)
target_include_directories(kvaf_initialize_gate_test PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/src)
target_link_libraries(kvaf_initialize_gate_test PRIVATE kvaf_common Threads::Threads)
add_test(NAME kvaf_initialize_gate_test COMMAND kvaf_initialize_gate_test)
set_tests_properties(kvaf_initialize_gate_test PROPERTIES TIMEOUT 30)

add_executable(kvaf_init_phase_timing_test init_phase_timing_test.cc)
target_include_directories(kvaf_init_phase_timing_test PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/src)
target_link_libraries(kvaf_init_phase_timing_test PRIVATE Threads::Threads)
add_test(NAME kvaf_init_phase_timing_test COMMAND kvaf_init_phase_timing_test)

add_executable(kvaf_cpu_memory_page_policy_test cpu_memory_page_policy_test.cc)
target_include_directories(kvaf_cpu_memory_page_policy_test PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/src)
target_link_libraries(kvaf_cpu_memory_page_policy_test PRIVATE kvaf_runtime)
add_test(NAME kvaf_cpu_memory_page_policy_test COMMAND kvaf_cpu_memory_page_policy_test)

add_executable(kvaf_runtime_init_profile_test runtime_init_profile_test.cc)
target_link_libraries(kvaf_runtime_init_profile_test PRIVATE kvaf_runtime)
add_test(NAME kvaf_runtime_init_profile_test COMMAND kvaf_runtime_init_profile_test)

add_test(NAME kvaf_ddr_init_diagnostics_python
  COMMAND "${Python3_EXECUTABLE}" -m unittest discover
          -s "${CMAKE_CURRENT_SOURCE_DIR}/diagnostics" -p "test_*ddr_init*.py")

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
