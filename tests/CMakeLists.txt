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

  if(KVAF_BUILD_REMOTE_CPU_SERVICE)
    add_test(
      NAME kvaf_cpu_runtime_npu_hbm_python
      COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/python/test_cpu_runtime_npu_hbm.py"
    )
    set_tests_properties(kvaf_cpu_runtime_npu_hbm_python PROPERTIES
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

if(KVAF_BUILD_HBM_RW_TEST)
  set(KVAF_URMA_ROOT "" CACHE PATH "UMDK source or installation root")
  find_path(KVAF_HBM_TEST_URMA_INCLUDE_DIR urma_api.h
    HINTS "${KVAF_URMA_ROOT}/include" "${CMAKE_SOURCE_DIR}/../third_party/umdk/include")
  find_library(KVAF_HBM_TEST_URMA_LIBRARY NAMES urma
    HINTS "${KVAF_URMA_ROOT}/lib64" "${KVAF_URMA_ROOT}/lib" /lib64 /usr/lib64)
  if(NOT KVAF_HBM_TEST_URMA_INCLUDE_DIR OR NOT KVAF_HBM_TEST_URMA_LIBRARY)
    message(FATAL_ERROR "KVAF_BUILD_HBM_RW_TEST requires urma_api.h and liburma")
  endif()

  add_executable(kvaf_npu_hbm_server npu_hbm_server.cc)
  target_include_directories(kvaf_npu_hbm_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/runtime/include
    ${CMAKE_SOURCE_DIR}/access_operator/include
    ${KVAF_HBM_TEST_URMA_INCLUDE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include)
  target_link_directories(kvaf_npu_hbm_server PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
  target_link_libraries(kvaf_npu_hbm_server PRIVATE
    kvaf_direct_hccp_session kvaf_peer_link ascendcl ${CMAKE_DL_LIBS} pthread)
  get_filename_component(KVAF_HBM_TEST_URMA_LIBRARY_DIR "${KVAF_HBM_TEST_URMA_LIBRARY}" DIRECTORY)

  set_property(TARGET kvaf_npu_hbm_server APPEND PROPERTY BUILD_RPATH
    "${KVAF_HBM_TEST_URMA_LIBRARY_DIR}" "${ASCEND_CANN_PACKAGE_PATH}/lib64")
endif()
