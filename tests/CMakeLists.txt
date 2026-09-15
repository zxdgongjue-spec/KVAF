find_package(Python3 COMPONENTS Interpreter REQUIRED)

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

# Host tests cover startup selection and the actual probe's request construction,
# data verification and lifetime ordering without linking CANN/UMDK.
add_executable(kvaf_startup_selection_test cpp/startup_selection_test.cc)
target_link_libraries(kvaf_startup_selection_test PRIVATE kvaf_runtime)
add_test(NAME kvaf_startup_selection COMMAND kvaf_startup_selection_test)

add_executable(kvaf_startup_bandwidth_probe_test cpp/startup_bandwidth_probe_test.cc)
target_include_directories(kvaf_startup_bandwidth_probe_test PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/stubs"
  "${CMAKE_CURRENT_SOURCE_DIR}/../runtime/src")
target_link_libraries(kvaf_startup_bandwidth_probe_test PRIVATE kvaf_runtime)
add_test(NAME kvaf_startup_bandwidth_probe COMMAND kvaf_startup_bandwidth_probe_test)
