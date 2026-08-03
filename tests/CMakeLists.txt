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
      NAME kvaf_aiv_batch_get_npu_python
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/python/test_aiv_batch_get_npu.py"
    )
    set_tests_properties(kvaf_aiv_batch_get_npu_python PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
      TIMEOUT 240
    )
  endif()
else()
  message(STATUS
    "Skipping direct Runtime Python tests; configure with -DKVAF_BUILD_PYTHON=ON")
endif()

# The old stdin/stdout test drivers are intentionally not part of the default
# build: tests call kvaf.Runtime directly. Keep them available only for
# targeted legacy diagnostics.
option(KVAF_BUILD_LEGACY_SUBPROCESS_TEST_DRIVERS
       "Build temporary legacy C++ Runtime test drivers" OFF)

if(KVAF_BUILD_LEGACY_SUBPROCESS_TEST_DRIVERS)
  add_executable(kvaf_runtime_mock_driver support/runtime_mock_driver.cc)
  target_link_libraries(kvaf_runtime_mock_driver PRIVATE kvaf_runtime)
  target_compile_features(kvaf_runtime_mock_driver PRIVATE cxx_std_17)

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
      PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )

    target_compile_definitions(kvaf_runtime_aiv_batch_get_runner
      PRIVATE _GLIBCXX_USE_CXX11_ABI=0
    )

    target_compile_features(kvaf_runtime_aiv_batch_get_runner PRIVATE cxx_std_17)
    set_target_properties(kvaf_runtime_aiv_batch_get_runner PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
  endif()
endif()
