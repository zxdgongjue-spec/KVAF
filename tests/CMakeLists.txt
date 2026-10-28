find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_executable(kvaf_peer_eid_mapping_test peer_eid_mapping_test.cc)
target_link_libraries(kvaf_peer_eid_mapping_test PRIVATE kvaf_peer_link)
add_test(NAME kvaf_peer_eid_mapping_test COMMAND kvaf_peer_eid_mapping_test)
add_test(NAME kvaf_peer_eid_config_python
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_peer_eid_config.py")

add_executable(kvaf_startup_read_probe_test startup_read_probe_test.cc)
target_include_directories(kvaf_startup_read_probe_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/fake_acl
  ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include)
target_link_libraries(kvaf_startup_read_probe_test PRIVATE kvaf_runtime)
add_test(NAME kvaf_startup_read_probe_test COMMAND kvaf_startup_read_probe_test)

add_executable(kvaf_batch_get_peer_transport_test batch_get_peer_transport_test.cc)
target_include_directories(kvaf_batch_get_peer_transport_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/fake_acl
  ${PROJECT_SOURCE_DIR}/access_operator/include
  ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include
  ${PROJECT_SOURCE_DIR}/runtime/include)
target_link_libraries(kvaf_batch_get_peer_transport_test PRIVATE kvaf_common ${CMAKE_DL_LIBS} pthread)
add_test(NAME kvaf_batch_get_peer_transport_test COMMAND kvaf_batch_get_peer_transport_test)

if(KVAF_BUILD_REMOTE_CPU_SERVICE)
  add_executable(kvaf_cpu_region_endpoints_test cpu_region_endpoints_test.cc)
  target_include_directories(kvaf_cpu_region_endpoints_test PRIVATE "${KVAF_URMA_INCLUDE_DIR}")
  target_link_libraries(kvaf_cpu_region_endpoints_test PRIVATE kvaf_runtime kvaf_peer_link ${CMAKE_DL_LIBS})
  # The production service resolves expanded URMA APIs with dlsym.
  set_target_properties(kvaf_cpu_region_endpoints_test PROPERTIES ENABLE_EXPORTS ON)
  add_test(NAME kvaf_cpu_region_endpoints_test COMMAND kvaf_cpu_region_endpoints_test)
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
