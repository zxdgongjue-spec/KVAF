find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_executable(kvaf_npu_eid_discovery_test npu_eid_discovery_test.cc)
target_include_directories(kvaf_npu_eid_discovery_test PRIVATE ${PROJECT_SOURCE_DIR}/runtime/src)
target_link_libraries(kvaf_npu_eid_discovery_test PRIVATE kvaf_peer_link ${CMAKE_DL_LIBS})
add_test(NAME kvaf_npu_eid_discovery_test COMMAND kvaf_npu_eid_discovery_test)

add_executable(kvaf_offload_test offload_test.cc)
target_include_directories(kvaf_offload_test PRIVATE ${PROJECT_SOURCE_DIR}/runtime/src)
target_link_libraries(kvaf_offload_test PRIVATE kvaf_runtime kvaf_offload_control)
add_test(NAME kvaf_offload_test COMMAND kvaf_offload_test)
set_tests_properties(kvaf_offload_test PROPERTIES TIMEOUT 20)

add_executable(kvaf_offload_runtime_test offload_runtime_test.cc ../runtime/src/offload_runtime.cc)
target_include_directories(kvaf_offload_runtime_test PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/src ${CMAKE_CURRENT_SOURCE_DIR}/offload_fake_acl)
target_link_libraries(kvaf_offload_runtime_test PRIVATE kvaf_runtime kvaf_offload_control kvaf_peer_link)
add_test(NAME kvaf_offload_runtime_test COMMAND kvaf_offload_runtime_test)
set_tests_properties(kvaf_offload_runtime_test PROPERTIES TIMEOUT 30)

set(KVAF_TEST_URMA_INCLUDE_DIR "" CACHE PATH "Optional UMDK 24.12 headers for CPU Pull fake-provider test")
if(KVAF_TEST_URMA_INCLUDE_DIR)
  add_executable(kvaf_cpu_pull_urma_test cpu_pull_urma_test.cc ../runtime/src/cpu_pull_urma.cc)
  target_include_directories(kvaf_cpu_pull_urma_test PRIVATE
    ${PROJECT_SOURCE_DIR}/runtime/src ${PROJECT_SOURCE_DIR}/runtime/include ${KVAF_TEST_URMA_INCLUDE_DIR})
  target_link_libraries(kvaf_cpu_pull_urma_test PRIVATE kvaf_common ${CMAKE_DL_LIBS})
  set_target_properties(kvaf_cpu_pull_urma_test PROPERTIES ENABLE_EXPORTS ON)
  add_test(NAME kvaf_cpu_pull_urma_test COMMAND kvaf_cpu_pull_urma_test)
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

add_executable(kvaf_remote_offload_test remote_offload_test.cc
  ../runtime/src/npu_runtime.cc ../runtime/src/remote_offload.cc ../runtime/src/remote_pull_operator.cc)
target_include_directories(kvaf_remote_offload_test PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/src ${CMAKE_CURRENT_SOURCE_DIR}/offload_fake_acl
  ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include)
target_compile_definitions(kvaf_remote_offload_test PRIVATE KVAF_ENABLE_CPU_OFFLOAD=1)
target_link_libraries(kvaf_remote_offload_test PRIVATE kvaf_runtime kvaf_offload_control kvaf_peer_link ${CMAKE_DL_LIBS})
add_test(NAME kvaf_remote_offload_test COMMAND kvaf_remote_offload_test)
set_tests_properties(kvaf_remote_offload_test PROPERTIES TIMEOUT 30)
