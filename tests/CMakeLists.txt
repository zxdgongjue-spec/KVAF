find_package(Python3 COMPONENTS Interpreter REQUIRED)

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
target_compile_definitions(kvaf_batch_get_kernel_model_test PRIVATE KVAF_BATCH_GET_LEAN_SCHEDULE=0)
add_test(NAME kvaf_batch_get_kernel_model_test COMMAND kvaf_batch_get_kernel_model_test)
add_executable(kvaf_batch_get_lean_kernel_model_test batch_get_kernel_model_test.cc)
target_include_directories(kvaf_batch_get_lean_kernel_model_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/aiv_model
  ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include)
target_compile_definitions(kvaf_batch_get_lean_kernel_model_test PRIVATE KVAF_BATCH_GET_LEAN_SCHEDULE=1)
add_test(NAME kvaf_batch_get_lean_kernel_model_test COMMAND kvaf_batch_get_lean_kernel_model_test)
foreach(tail_threshold IN ITEMS 32 128)
  set(tail_target kvaf_batch_get_tail_${tail_threshold}_kernel_model_test)
  add_executable(${tail_target} batch_get_kernel_model_test.cc)
  target_include_directories(${tail_target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/aiv_model
    ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include)
  target_compile_definitions(${tail_target} PRIVATE
    KVAF_BATCH_GET_LEAN_SCHEDULE=1 KVAF_BATCH_GET_MIN_TAIL_READS=${tail_threshold})
  add_test(NAME ${tail_target} COMMAND ${tail_target})
endforeach()
foreach(fast_case IN ITEMS trim_only group_only post_only untrimmed_group untrimmed_post untrimmed_both)
  set(fast_target kvaf_batch_get_${fast_case}_kernel_model_test)
  add_executable(${fast_target} batch_get_kernel_model_test.cc)
  target_include_directories(${fast_target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/aiv_model
    ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include)
  set(trim 1)
  set(group_fast 0)
  set(post_fast 0)
  if(fast_case MATCHES "^untrimmed")
    set(trim 0)
  endif()
  if(fast_case MATCHES "group|both")
    set(group_fast 1)
  endif()
  if(fast_case MATCHES "post|both")
    set(post_fast 1)
  endif()
  target_compile_definitions(${fast_target} PRIVATE KVAF_BATCH_GET_LEAN_SCHEDULE=${trim}
    KVAF_BATCH_GET_SINGLE_GROUP_FAST_PATH=${group_fast} KVAF_BATCH_GET_SINGLE_POST_FAST_PATH=${post_fast}
    KVAF_BATCH_GET_MIN_TAIL_READS=32)
  add_test(NAME ${fast_target} COMMAND ${fast_target})
endforeach()
add_test(NAME kvaf_batch_get_inflight_config_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_batch_get_inflight_config.py")
add_test(NAME kvaf_batch_get_csv_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/diagnostics/test_small_post_csv.py")

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
