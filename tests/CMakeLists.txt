find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_executable(kvaf_batch_get_compact_count_test batch_get_compact_count_test.cc)
target_include_directories(kvaf_batch_get_compact_count_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../access_operator/demo/aiv_batch_get/include)
target_link_libraries(kvaf_batch_get_compact_count_test PRIVATE kvaf_runtime)
target_compile_options(kvaf_batch_get_compact_count_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME kvaf_batch_get_compact_count COMMAND kvaf_batch_get_compact_count_test)
add_test(NAME kvaf_compact_count_capability
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_compact_count_capability.py")

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
      NAME kvaf_forward_resident_get_source_contract
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/python/test_forward_resident_get_source_contract.py"
    )

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

# Mock and NPU integration coverage call the pybind API directly. The compact
# count C++ test additionally checks opaque device forwarding without CANN.
