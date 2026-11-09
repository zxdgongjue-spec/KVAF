find_package(Python3 COMPONENTS Interpreter REQUIRED)

find_package(Threads REQUIRED)
add_executable(kvaf_npu_runtime_config_test npu_runtime_config_test.cc)
target_link_libraries(kvaf_npu_runtime_config_test PRIVATE kvaf_runtime)
add_test(NAME kvaf_npu_runtime_config_test COMMAND kvaf_npu_runtime_config_test)
add_test(NAME kvaf_aicpu_configuration_python
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_aicpu_configuration.py")
add_test(NAME kvaf_build_script_python
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_build_script.py")
add_test(NAME kvaf_cache_execution_harness_python
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_cache_execution_harness.py")
set_tests_properties(kvaf_cache_execution_harness_python PROPERTIES TIMEOUT 30)
add_test(NAME kvaf_runtime_build_flags_python
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_runtime_build_flags.py")
set_tests_properties(kvaf_runtime_build_flags_python PROPERTIES
  ENVIRONMENT "KVAF_TEST_CMAKE_COMMAND=${CMAKE_COMMAND}" TIMEOUT 120)
add_executable(kvaf_aicpu_kernel_loader_test aicpu_kernel_loader_test.cc
  ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_urma/host/kernel_loader.cc)
target_include_directories(kvaf_aicpu_kernel_loader_test PRIVATE
  aicpu_loader_stub ${PROJECT_SOURCE_DIR}/access_operator/include)
target_link_libraries(kvaf_aicpu_kernel_loader_test PRIVATE kvaf_common Threads::Threads)
add_test(NAME kvaf_aicpu_kernel_loader_test COMMAND kvaf_aicpu_kernel_loader_test)

add_executable(kvaf_runtime_cleanup_test runtime_cleanup_test.cc)
target_link_libraries(kvaf_runtime_cleanup_test PRIVATE kvaf_runtime)
add_test(NAME kvaf_runtime_cleanup_test COMMAND kvaf_runtime_cleanup_test)

if(KVAF_BUILD_REMOTE_CPU_SERVICE AND NOT KVAF_CPU_ENABLE_URMA)
  add_executable(kvaf_cpu_backend_dispatch_test cpu_backend_dispatch_test.cc)
  target_link_libraries(kvaf_cpu_backend_dispatch_test PRIVATE kvaf_cpu_runtime)
  add_test(NAME kvaf_cpu_backend_dispatch_test COMMAND kvaf_cpu_backend_dispatch_test)
endif()

if(KVAF_BUILD_REMOTE_CPU_SERVICE)
  add_library(kvaf_ubmem_hal_stub SHARED ubmem_hal_stub.cc)
  target_include_directories(kvaf_ubmem_hal_stub PRIVATE ${PROJECT_SOURCE_DIR}/access_operator/include)
  target_link_libraries(kvaf_ubmem_hal_stub PRIVATE kvaf_common Threads::Threads)
  add_executable(kvaf_ubmem_hal_lifecycle_test ubmem_hal_lifecycle_test.cc)
  target_include_directories(kvaf_ubmem_hal_lifecycle_test PRIVATE ${PROJECT_SOURCE_DIR}/runtime/src)
  target_link_libraries(kvaf_ubmem_hal_lifecycle_test PRIVATE kvaf_cpu_runtime Threads::Threads ${CMAKE_DL_LIBS})
  # Do not link the stub: a link-time reference would hide premature dlclose.
  add_dependencies(kvaf_ubmem_hal_lifecycle_test kvaf_ubmem_hal_stub)
  foreach(scenario execution legacy concurrent)
    add_test(NAME kvaf_ubmem_hal_${scenario}_test COMMAND kvaf_ubmem_hal_lifecycle_test
      $<TARGET_FILE:kvaf_ubmem_hal_stub> ${scenario})
    set_tests_properties(kvaf_ubmem_hal_${scenario}_test PROPERTIES TIMEOUT 20)
  endforeach()
endif()

add_executable(kvaf_cpu_push_scatter_test cpu_push_scatter_test.cc
  ${PROJECT_SOURCE_DIR}/access_operator/demo/aicpu_batch_get/kernel/cpu_push_scatter.cc)
target_include_directories(kvaf_cpu_push_scatter_test PRIVATE
  ${PROJECT_SOURCE_DIR}/access_operator/demo ${PROJECT_SOURCE_DIR}/access_operator/include)
target_link_libraries(kvaf_cpu_push_scatter_test PRIVATE kvaf_common Threads::Threads)
add_test(NAME kvaf_cpu_push_scatter_test COMMAND kvaf_cpu_push_scatter_test)
add_test(NAME kvaf_cpu_push_input_bounds_test COMMAND kvaf_cpu_push_scatter_test --input-bounds)

add_executable(kvaf_cpu_push_shutdown_test cpu_push_shutdown_test.cc
  ${PROJECT_SOURCE_DIR}/access_operator/demo/aicpu_batch_get/host/cpu_push_host.cc)
target_include_directories(kvaf_cpu_push_shutdown_test PRIVATE cpu_push_host_stub
  ${PROJECT_SOURCE_DIR}/access_operator/include ${PROJECT_SOURCE_DIR}/access_operator/demo
  ${PROJECT_SOURCE_DIR}/access_operator/src ${PROJECT_SOURCE_DIR}/runtime/include)
target_link_libraries(kvaf_cpu_push_shutdown_test PRIVATE kvaf_common)
add_test(NAME kvaf_cpu_push_shutdown_test COMMAND kvaf_cpu_push_shutdown_test)

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
add_executable(kvaf_batch_put_kernel_model_test batch_put_kernel_model_test.cc)
target_include_directories(kvaf_batch_put_kernel_model_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/aiv_model
  ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_put/include)
add_test(NAME kvaf_batch_put_kernel_model_test COMMAND kvaf_batch_put_kernel_model_test)
set_tests_properties(kvaf_batch_put_kernel_model_test PROPERTIES TIMEOUT 10)
add_test(NAME kvaf_batch_get_inflight_config_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_batch_get_inflight_config.py")
add_test(NAME kvaf_batch_get_multicard_perf_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_batch_get_multicard_perf.py")

add_test(
  NAME kvaf_npu_eid_parser_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_npu_eids.py"
)

if(KVAF_BUILD_PYTHON)
  pybind11_add_module(_kvaf_constructor_test npu_constructor_binding.cc)
  target_include_directories(_kvaf_constructor_test PRIVATE ${PROJECT_SOURCE_DIR}/bindings/include)
  set_target_properties(_kvaf_constructor_test PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python")
  add_test(NAME kvaf_npu_constructor_python
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_npu_constructor.py")
  set_tests_properties(kvaf_npu_constructor_python PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python")

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

add_executable(kvaf_execution_dispatcher_test execution_dispatcher_test.cc)
target_link_libraries(kvaf_execution_dispatcher_test PRIVATE kvaf_runtime Threads::Threads)
add_test(NAME kvaf_execution_dispatcher_test COMMAND kvaf_execution_dispatcher_test)
set_tests_properties(kvaf_execution_dispatcher_test PROPERTIES TIMEOUT 20)
if(KVAF_BUILD_PYTHON)
  add_test(NAME kvaf_execution_config_python
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_execution_config.py")
  set_tests_properties(kvaf_execution_config_python PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
endif()

# Optional host-only validation against real/stub URMA headers; no liburma or NPU.
set(KVAF_TEST_URMA_INCLUDE_DIR "" CACHE PATH "URMA headers for host session registry tests")
if(KVAF_TEST_URMA_INCLUDE_DIR)
  add_executable(kvaf_aicpu_session_registry_test aicpu_session_registry_test.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_urma/kernel/session.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_urma/kernel/transfer.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_urma/kernel/control_kernel.cc)
  target_include_directories(kvaf_aicpu_session_registry_test PRIVATE
    ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_urma/kernel
    ${PROJECT_SOURCE_DIR}/access_operator/include ${KVAF_TEST_URMA_INCLUDE_DIR})
  target_link_libraries(kvaf_aicpu_session_registry_test PRIVATE kvaf_common Threads::Threads ${CMAKE_DL_LIBS})
  add_test(NAME kvaf_aicpu_session_registry_test COMMAND kvaf_aicpu_session_registry_test)
  set_tests_properties(kvaf_aicpu_session_registry_test PROPERTIES TIMEOUT 10)
endif()

add_executable(kvaf_cache_region_service_test cache_region_service_test.cc)
target_link_libraries(kvaf_cache_region_service_test PRIVATE kvaf_runtime Threads::Threads)
add_test(NAME kvaf_cache_region_service_test COMMAND kvaf_cache_region_service_test)
set_tests_properties(kvaf_cache_region_service_test PROPERTIES TIMEOUT 20)
add_executable(kvaf_execution_peer_recovery_test execution_peer_recovery_test.cc)
target_link_libraries(kvaf_execution_peer_recovery_test PRIVATE kvaf_runtime Threads::Threads)
add_test(NAME kvaf_execution_peer_recovery_test COMMAND kvaf_execution_peer_recovery_test)
set_tests_properties(kvaf_execution_peer_recovery_test PROPERTIES TIMEOUT 20)
set(KVAF_TEST_ACL_INCLUDE_DIR "" CACHE PATH "ACL headers for host mapping lease tests")
if(KVAF_TEST_ACL_INCLUDE_DIR)
  add_executable(kvaf_ubmem_mapping_cache_test ubmem_mapping_cache_test.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/ubmem/device_mapping.cc)
  target_include_directories(kvaf_ubmem_mapping_cache_test PRIVATE ${KVAF_TEST_ACL_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/access_operator/include)
  target_link_libraries(kvaf_ubmem_mapping_cache_test PRIVATE kvaf_ubmem_hal Threads::Threads)
  add_test(NAME kvaf_ubmem_mapping_cache_test COMMAND kvaf_ubmem_mapping_cache_test)
  set_tests_properties(kvaf_ubmem_mapping_cache_test PROPERTIES TIMEOUT 20)
endif()
