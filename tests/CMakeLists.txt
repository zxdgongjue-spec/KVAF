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

# Compile the real host operator against a private fake ACL/transport surface.
# This validates resource leases on developer machines without claiming any
# device transfer or kernel execution coverage.
add_executable(kvaf_fused_batch_get_context_test
  fused_batch_get_context_test.cc
  ../access_operator/src/access_operator.cc)
target_include_directories(kvaf_fused_batch_get_context_test PRIVATE
  fused_context_stubs
  ../access_operator/demo/aiv_batch_get/include)
target_link_libraries(kvaf_fused_batch_get_context_test PRIVATE
  kvaf_runtime kvaf_meta_manager kvaf_peer_link)
add_test(NAME kvaf_fused_batch_get_context_host
  COMMAND kvaf_fused_batch_get_context_test)

# Also compile the NpuRuntime forwarding API against the same declarations;
# this is a host syntax check and deliberately has no runtime/device backend.
add_library(kvaf_fused_npu_runtime_syntax OBJECT ../runtime/src/npu_runtime.cc)
target_include_directories(kvaf_fused_npu_runtime_syntax PRIVATE
  fused_context_stubs
  ../access_operator/demo/aiv_batch_get/include)
target_link_libraries(kvaf_fused_npu_runtime_syntax PRIVATE kvaf_runtime)

add_executable(kvaf_batch_get_device_control_test
  ../access_operator/demo/aiv_batch_get/kernel/tests/test_batch_get_device.cpp)
target_include_directories(kvaf_batch_get_device_control_test PRIVATE
  ../access_operator/demo/aiv_batch_get/kernel/tests
  ../access_operator/demo/aiv_batch_get/include)
add_test(NAME kvaf_batch_get_device_control_host
  COMMAND kvaf_batch_get_device_control_test)
