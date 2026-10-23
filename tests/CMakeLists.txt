find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_executable(kvaf_cpu_push_get_protocol_test cpu_push_get_protocol_test.cc)
target_link_libraries(kvaf_cpu_push_get_protocol_test PRIVATE kvaf_common)
target_include_directories(kvaf_cpu_push_get_protocol_test PRIVATE
  ${PROJECT_SOURCE_DIR}/access_operator/demo/aicpu_batch_get/kernel)
add_test(NAME kvaf_cpu_push_get_protocol_test COMMAND kvaf_cpu_push_get_protocol_test)

# Mock the provider while exercising the production transport and kernel.
# No CANN installation, liburma or NPU is needed for these two tests.
find_path(KVAF_TEST_URMA_INCLUDE_DIR urma_api.h HINTS "${KVAF_URMA_ROOT}/include"
  "${PROJECT_SOURCE_DIR}/../third_party/umdk/include")
if(KVAF_TEST_URMA_INCLUDE_DIR)
  find_package(Threads REQUIRED)
  add_library(kvaf_cpu_push_get_mock STATIC cpu_push_get_mock.cc
    ../runtime/src/cpu_push_get_transport.cc ../runtime/src/cpu_push_get_worker.cc)
  target_link_libraries(kvaf_cpu_push_get_mock PUBLIC kvaf_common Threads::Threads)
  target_include_directories(kvaf_cpu_push_get_mock PUBLIC "${KVAF_TEST_URMA_INCLUDE_DIR}"
    "${PROJECT_SOURCE_DIR}/runtime/include" "${PROJECT_SOURCE_DIR}/runtime/src"
    "${PROJECT_SOURCE_DIR}/access_operator/include"
    "${PROJECT_SOURCE_DIR}/access_operator/demo"
    "${PROJECT_SOURCE_DIR}/access_operator/demo/aicpu_batch_get/kernel")
  add_executable(kvaf_cpu_push_get_transport_test cpu_push_get_transport_test.cc)
  target_link_libraries(kvaf_cpu_push_get_transport_test PRIVATE kvaf_cpu_push_get_mock)
  add_test(NAME kvaf_cpu_push_get_transport_test COMMAND kvaf_cpu_push_get_transport_test)
  add_executable(kvaf_cpu_push_get_pipeline_test cpu_push_get_pipeline_test.cc
    ../access_operator/demo/aicpu_batch_get/kernel/cpu_push_scatter.cc)
  target_compile_definitions(kvaf_cpu_push_get_pipeline_test PRIVATE KVAF_CPU_PUSH_HOST_TEST=1)
  target_link_libraries(kvaf_cpu_push_get_pipeline_test PRIVATE kvaf_cpu_push_get_mock)
  add_test(NAME kvaf_cpu_push_get_pipeline_test COMMAND kvaf_cpu_push_get_pipeline_test)
  set_tests_properties(kvaf_cpu_push_get_transport_test kvaf_cpu_push_get_pipeline_test PROPERTIES TIMEOUT 60)
endif()

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
