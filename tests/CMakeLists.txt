find_package(Python3 COMPONENTS Interpreter REQUIRED)

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
  endif()
else()
  message(STATUS
    "Skipping direct Runtime Python tests; configure with -DKVAF_BUILD_PYTHON=ON")
endif()

# Runtime tests are Python-only. Mock and NPU integration coverage call the
# pybind API directly; no subprocess driver is built or invoked.
