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

# Runtime tests are Python-only. Mock and NPU integration coverage call the
# pybind API directly; no subprocess driver is built or invoked.

find_package(Threads REQUIRED)
add_executable(kvaf_cpu_gather_test cpu_gather_test.cc)
target_link_libraries(kvaf_cpu_gather_test PRIVATE kvaf_cpu_gather_plan)
add_test(NAME kvaf_cpu_gather COMMAND kvaf_cpu_gather_test)
if(KVAF_ENABLE_AICPU_URMA)
  add_executable(kvaf_aicpu_gather_read_client aicpu_gather_read_client.cc)
  target_include_directories(kvaf_aicpu_gather_read_client PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include" "${CMAKE_SOURCE_DIR}/access_operator/src")
  target_compile_definitions(kvaf_aicpu_gather_read_client PRIVATE KVAF_AICPU_URMA_CANN_ROOT="${ASCEND_CANN_PACKAGE_PATH}")
  target_link_libraries(kvaf_aicpu_gather_read_client PRIVATE kvaf_npu_runtime ascendcl)
  target_link_directories(kvaf_aicpu_gather_read_client PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
endif()

find_path(KVAF_DISCOVERY_TEST_URMA_INCLUDE_DIR urma_api.h HINTS "${KVAF_URMA_ROOT}/include" "${CMAKE_SOURCE_DIR}/../third_party/umdk/include")
if(KVAF_DISCOVERY_TEST_URMA_INCLUDE_DIR)
  add_library(kvaf_aicpu_umdk_mock SHARED aicpu_umdk_mock.cc)
  set_target_properties(kvaf_aicpu_umdk_mock PROPERTIES OUTPUT_NAME urma SUFFIX ".so.0" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/umdk-mock")
  target_include_directories(kvaf_aicpu_umdk_mock PRIVATE "${CMAKE_SOURCE_DIR}/access_operator/src" "${KVAF_DISCOVERY_TEST_URMA_INCLUDE_DIR}")
  add_executable(kvaf_aicpu_gather_integration_test aicpu_gather_integration_test.cc
    "${CMAKE_SOURCE_DIR}/access_operator/src/aicpu_urma/host.cc"
    "${CMAKE_SOURCE_DIR}/access_operator/src/aicpu_urma/umdk_session.cc"
    "${CMAKE_SOURCE_DIR}/access_operator/src/aicpu_urma/access_operator.cc"
    "${CMAKE_SOURCE_DIR}/access_operator/src/aicpu_urma/kernel.cc"
    "${CMAKE_SOURCE_DIR}/access_operator/src/aicpu_urma/umdk_transport.cc")
  target_compile_definitions(kvaf_aicpu_gather_integration_test PRIVATE KVAF_AICPU_URMA_HOST_TEST=1)
  target_include_directories(kvaf_aicpu_gather_integration_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/aicpu_umdk_host_stub"
    "${CMAKE_SOURCE_DIR}/access_operator/src" "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include"
    "${CMAKE_SOURCE_DIR}/runtime/include" "${KVAF_DISCOVERY_TEST_URMA_INCLUDE_DIR}")
  target_link_libraries(kvaf_aicpu_gather_integration_test PRIVATE kvaf_runtime kvaf_meta_manager kvaf_gather_host_plan Threads::Threads "${CMAKE_DL_LIBS}")
  add_dependencies(kvaf_aicpu_gather_integration_test kvaf_aicpu_umdk_mock)
  add_test(NAME kvaf_aicpu_gather_integration COMMAND kvaf_aicpu_gather_integration_test "$<TARGET_FILE:kvaf_aicpu_umdk_mock>")
  add_executable(kvaf_cpu_urma_channel_test cpu_urma_channel_test.cc)
  target_include_directories(kvaf_cpu_urma_channel_test PRIVATE "${CMAKE_SOURCE_DIR}/runtime/src" "${KVAF_DISCOVERY_TEST_URMA_INCLUDE_DIR}")
  add_test(NAME kvaf_cpu_urma_channel COMMAND kvaf_cpu_urma_channel_test)
endif()

add_executable(kvaf_peer_channel_tp_test peer_channel_tp_test.cc)
target_link_libraries(kvaf_peer_channel_tp_test PRIVATE kvaf_peer_link Threads::Threads)
add_test(NAME kvaf_peer_channel_tp COMMAND kvaf_peer_channel_tp_test)

add_test(NAME kvaf_aicpu_urma_install COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/aicpu_urma_install_test.sh")
