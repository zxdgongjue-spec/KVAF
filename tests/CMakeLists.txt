find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_executable(kvaf_runtime_mock_driver
  support/runtime_mock_driver.cc
)

target_link_libraries(kvaf_runtime_mock_driver
  PRIVATE
    kvaf_runtime
)

target_compile_features(kvaf_runtime_mock_driver PRIVATE cxx_std_17)

add_test(
  NAME kvaf_runtime_mock_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_runtime_mock.py"
          --driver "$<TARGET_FILE:kvaf_runtime_mock_driver>"
)

# This is a Python-owned NPU integration test for the merged AIV demo. It
# exits successfully with an explicit SKIP unless KVAF_RUN_NPU_TESTS=1 and
# the CANN-built binary/kernel paths are supplied by the target environment.
add_test(
  NAME kvaf_aiv_batch_get_npu_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_aiv_batch_get_npu.py"
)

if(KVAF_ENABLE_AIV_BATCH_GET)
  add_executable(kvaf_runtime_aiv_batch_get_runner
    support/runtime_aiv_batch_get_runner.cc
  )

  target_link_libraries(kvaf_runtime_aiv_batch_get_runner
    PRIVATE
      kvaf_runtime
      kvaf_aiv_batch_get_operator
      kvaf_execute_batch_get_host
      ascendcl
  )

  target_include_directories(kvaf_runtime_aiv_batch_get_runner
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../access_operator/demo/aiv_batch_get/include
      ${ASCEND_CANN_PACKAGE_PATH}/include
  )

  target_link_directories(kvaf_runtime_aiv_batch_get_runner
    PRIVATE
      ${ASCEND_CANN_PACKAGE_PATH}/lib64
  )

  target_compile_definitions(kvaf_runtime_aiv_batch_get_runner
    PRIVATE
      _GLIBCXX_USE_CXX11_ABI=0
  )

  target_compile_features(kvaf_runtime_aiv_batch_get_runner PRIVATE cxx_std_17)
  set_target_properties(kvaf_runtime_aiv_batch_get_runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
endif()
