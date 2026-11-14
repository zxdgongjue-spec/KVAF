find_package(Python3 COMPONENTS Interpreter REQUIRED)

find_package(Threads REQUIRED)
add_executable(kvaf_direct_hccp_environment_test direct_hccp_environment_test.cc)
target_include_directories(kvaf_direct_hccp_environment_test PRIVATE aicpu_loader_stub
  ${PROJECT_SOURCE_DIR}/access_operator/include ${PROJECT_SOURCE_DIR}/runtime/include)
target_link_libraries(kvaf_direct_hccp_environment_test PRIVATE kvaf_common Threads::Threads)
add_test(NAME kvaf_direct_hccp_environment_test COMMAND kvaf_direct_hccp_environment_test)
set_tests_properties(kvaf_direct_hccp_environment_test PROPERTIES TIMEOUT 20)
add_executable(kvaf_execution_view_binding_test execution_view_binding_test.cc)
target_link_libraries(kvaf_execution_view_binding_test PRIVATE kvaf_access_operator Threads::Threads)
add_test(NAME kvaf_execution_view_binding_test COMMAND kvaf_execution_view_binding_test)
add_executable(kvaf_get_diagnostics_test get_diagnostics_test.cc)
target_link_libraries(kvaf_get_diagnostics_test PRIVATE kvaf_runtime Threads::Threads)
add_test(NAME kvaf_get_diagnostics_test COMMAND kvaf_get_diagnostics_test)
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
add_test(NAME kvaf_kernel_profile_sweep_python
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_kernel_profile_sweep.py")
set_tests_properties(kvaf_kernel_profile_sweep_python PROPERTIES TIMEOUT 30)
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
  ${PROJECT_SOURCE_DIR}/access_operator/algorithms/aicpu_batch_get/kernel/cpu_push_scatter.cc)
target_include_directories(kvaf_cpu_push_scatter_test PRIVATE
  ${PROJECT_SOURCE_DIR}/access_operator/algorithms ${PROJECT_SOURCE_DIR}/access_operator/include)
target_link_libraries(kvaf_cpu_push_scatter_test PRIVATE kvaf_common Threads::Threads)
add_test(NAME kvaf_cpu_push_scatter_test COMMAND kvaf_cpu_push_scatter_test)
add_test(NAME kvaf_cpu_push_request_shape_test COMMAND kvaf_cpu_push_scatter_test --request-shape)
set_tests_properties(kvaf_cpu_push_scatter_test kvaf_cpu_push_request_shape_test PROPERTIES TIMEOUT 180)

add_executable(kvaf_cpu_push_shutdown_test cpu_push_shutdown_test.cc
  ${PROJECT_SOURCE_DIR}/access_operator/algorithms/aicpu_batch_get/host/cpu_push_host.cc)
target_include_directories(kvaf_cpu_push_shutdown_test PRIVATE cpu_push_host_stub
  ${PROJECT_SOURCE_DIR}/access_operator/include ${PROJECT_SOURCE_DIR}/access_operator/algorithms
  ${PROJECT_SOURCE_DIR}/access_operator/src ${PROJECT_SOURCE_DIR}/runtime/include)
target_link_libraries(kvaf_cpu_push_shutdown_test PRIVATE kvaf_common)
add_test(NAME kvaf_cpu_push_shutdown_test COMMAND kvaf_cpu_push_shutdown_test)

add_test(NAME kvaf_dsa_keys_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_dsa_keys.py")

add_executable(kvaf_batch_get_schedule_test batch_get_schedule_test.cc)
target_include_directories(kvaf_batch_get_schedule_test PRIVATE
  ${PROJECT_SOURCE_DIR}/access_operator/algorithms/aiv_batch_get/include)
add_test(NAME kvaf_batch_get_schedule_test COMMAND kvaf_batch_get_schedule_test)
add_executable(kvaf_batch_get_kernel_model_test batch_get_kernel_model_test.cc)
target_include_directories(kvaf_batch_get_kernel_model_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/aiv_model
  ${PROJECT_SOURCE_DIR}/access_operator/algorithms/aiv_batch_get/include)
add_test(NAME kvaf_batch_get_kernel_model_test COMMAND kvaf_batch_get_kernel_model_test)
add_executable(kvaf_batch_put_kernel_model_test batch_put_kernel_model_test.cc)
target_include_directories(kvaf_batch_put_kernel_model_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/aiv_model
  ${PROJECT_SOURCE_DIR}/access_operator/algorithms/aiv_batch_put/include)
add_test(NAME kvaf_batch_put_kernel_model_test COMMAND kvaf_batch_put_kernel_model_test)
set_tests_properties(kvaf_batch_put_kernel_model_test PROPERTIES TIMEOUT 10)
add_test(NAME kvaf_batch_get_inflight_config_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_batch_get_inflight_config.py")
add_test(NAME kvaf_batch_get_unified_python
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_batch_get_unified.py")
add_test(NAME kvaf_batch_get_multicard_perf_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_batch_get_multicard_perf.py")
add_test(NAME kvaf_put_dispatch_trace_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_put_dispatch_trace.py")

add_test(NAME kvaf_perf_execution_python
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_perf_execution.py")

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
add_executable(kvaf_execution_resource_manager_test execution_resource_manager_test.cc)
target_link_libraries(kvaf_execution_resource_manager_test PRIVATE kvaf_runtime Threads::Threads)
add_test(NAME kvaf_execution_resource_manager_test COMMAND kvaf_execution_resource_manager_test)
set_tests_properties(kvaf_execution_resource_manager_test PROPERTIES TIMEOUT 20)
add_executable(kvaf_peer_execution_resources_test peer_execution_resources_test.cc)
target_include_directories(kvaf_peer_execution_resources_test PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/include)
target_link_libraries(kvaf_peer_execution_resources_test PRIVATE kvaf_access_operator)
add_test(NAME kvaf_peer_execution_resources_test COMMAND kvaf_peer_execution_resources_test)
set_tests_properties(kvaf_peer_execution_resources_test PROPERTIES TIMEOUT 20)
add_executable(kvaf_runtime_put_recovery_test runtime_put_recovery_test.cc)
target_link_libraries(kvaf_runtime_put_recovery_test PRIVATE kvaf_runtime Threads::Threads)
add_test(NAME kvaf_runtime_put_recovery_test COMMAND kvaf_runtime_put_recovery_test)
set_tests_properties(kvaf_runtime_put_recovery_test PROPERTIES TIMEOUT 20)
if(KVAF_BUILD_PYTHON)
  add_test(NAME kvaf_execution_config_python
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_execution_config.py")
  set_tests_properties(kvaf_execution_config_python PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
endif()

# Optional host-only validation against real/stub URMA headers; no liburma or NPU.
set(KVAF_TEST_URMA_INCLUDE_DIR "" CACHE PATH "URMA headers for host session registry tests")
if(KVAF_TEST_URMA_INCLUDE_DIR)
  add_executable(kvaf_aicpu_environment_test aicpu_environment_test.cc)
  target_include_directories(kvaf_aicpu_environment_test PRIVATE
    ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_urma/kernel
    ${PROJECT_SOURCE_DIR}/access_operator/include ${KVAF_TEST_URMA_INCLUDE_DIR})
  target_link_libraries(kvaf_aicpu_environment_test PRIVATE kvaf_common Threads::Threads ${CMAKE_DL_LIBS})
  add_test(NAME kvaf_aicpu_environment_test COMMAND kvaf_aicpu_environment_test)
  set_tests_properties(kvaf_aicpu_environment_test PROPERTIES TIMEOUT 20)
  add_executable(kvaf_aicpu_queue_profile_test aicpu_queue_profile_test.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_urma/kernel/session.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_urma/kernel/control_kernel.cc)
  target_include_directories(kvaf_aicpu_queue_profile_test PRIVATE
    ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_urma/kernel
    ${PROJECT_SOURCE_DIR}/access_operator/include ${KVAF_TEST_URMA_INCLUDE_DIR})
  target_link_libraries(kvaf_aicpu_queue_profile_test PRIVATE kvaf_common Threads::Threads ${CMAKE_DL_LIBS})
  add_test(NAME kvaf_aicpu_queue_profile_test COMMAND kvaf_aicpu_queue_profile_test)
  set_tests_properties(kvaf_aicpu_queue_profile_test PROPERTIES TIMEOUT 20)
  add_executable(kvaf_cpu_shared_registration_test cpu_shared_registration_test.cc)
  target_include_directories(kvaf_cpu_shared_registration_test PRIVATE
    ${PROJECT_SOURCE_DIR}/runtime/src ${PROJECT_SOURCE_DIR}/access_operator/include
    ${KVAF_TEST_URMA_INCLUDE_DIR})
  target_link_libraries(kvaf_cpu_shared_registration_test PRIVATE kvaf_common Threads::Threads)
  add_test(NAME kvaf_cpu_shared_registration_test COMMAND kvaf_cpu_shared_registration_test)
  set_tests_properties(kvaf_cpu_shared_registration_test PROPERTIES TIMEOUT 20)
  add_executable(kvaf_cpu_shared_memory_backend_test cpu_shared_memory_backend_test.cc
    ${PROJECT_SOURCE_DIR}/runtime/src/urma_cpu_backend.cc
    ${PROJECT_SOURCE_DIR}/runtime/src/cpu_urma_discovery.cc
    ${PROJECT_SOURCE_DIR}/runtime/src/cpu_push_get_transport.cc
    ${PROJECT_SOURCE_DIR}/runtime/src/cpu_push_get_worker.cc
    ${PROJECT_SOURCE_DIR}/runtime/src/cpu_gather_executor.cc)
  target_include_directories(kvaf_cpu_shared_memory_backend_test PRIVATE
    ${PROJECT_SOURCE_DIR}/runtime/src ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_SOURCE_DIR}/meta_manager/include ${PROJECT_SOURCE_DIR}/access_operator/include
    ${KVAF_TEST_URMA_INCLUDE_DIR})
  target_link_libraries(kvaf_cpu_shared_memory_backend_test PRIVATE
    kvaf_common Threads::Threads ${CMAKE_DL_LIBS})
  # The production backend resolves its expanded CTP symbols through dlsym.
  set_target_properties(kvaf_cpu_shared_memory_backend_test PROPERTIES ENABLE_EXPORTS ON)
  if(APPLE)
    target_compile_definitions(kvaf_cpu_shared_memory_backend_test PRIVATE MADV_HUGEPAGE=14)
  endif()
  add_test(NAME kvaf_cpu_shared_memory_backend_test COMMAND kvaf_cpu_shared_memory_backend_test)
  set_tests_properties(kvaf_cpu_shared_memory_backend_test PROPERTIES TIMEOUT 20)
  add_executable(kvaf_cpu_ddr_registration_test cpu_ddr_registration_test.cc)
  target_include_directories(kvaf_cpu_ddr_registration_test PRIVATE
    ${PROJECT_SOURCE_DIR}/runtime/src ${PROJECT_SOURCE_DIR}/access_operator/include
    ${KVAF_TEST_URMA_INCLUDE_DIR})
  target_link_libraries(kvaf_cpu_ddr_registration_test PRIVATE kvaf_common)
  add_test(NAME kvaf_cpu_ddr_registration_test COMMAND kvaf_cpu_ddr_registration_test)
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
  add_executable(kvaf_aicpu_gather_execution_test aicpu_gather_execution_test.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_urma/kernel/control_kernel.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_urma/kernel/transfer.cc
    ${PROJECT_SOURCE_DIR}/access_operator/algorithms/aicpu_batch_get/kernel/gather_read.cc)
  target_include_directories(kvaf_aicpu_gather_execution_test PRIVATE
    ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_urma/kernel
    ${PROJECT_SOURCE_DIR}/access_operator/algorithms/aicpu_batch_get/kernel
    ${PROJECT_SOURCE_DIR}/access_operator/algorithms ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_SOURCE_DIR}/access_operator/include ${KVAF_TEST_URMA_INCLUDE_DIR})
  target_link_libraries(kvaf_aicpu_gather_execution_test PRIVATE kvaf_common Threads::Threads ${CMAKE_DL_LIBS})
  add_test(NAME kvaf_aicpu_gather_execution_test COMMAND kvaf_aicpu_gather_execution_test)
  set_tests_properties(kvaf_aicpu_gather_execution_test PROPERTIES TIMEOUT 20)
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
  foreach(direction get put)
    add_executable(kvaf_aiv_${direction}_transport_close_test aiv_transport_close_test.cc)
    target_include_directories(kvaf_aiv_${direction}_transport_close_test PRIVATE
      ${KVAF_TEST_ACL_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/access_operator/include
      ${PROJECT_SOURCE_DIR}/access_operator/algorithms/aiv_batch_${direction}/include)
    target_link_libraries(kvaf_aiv_${direction}_transport_close_test PRIVATE kvaf_common Threads::Threads)
    add_test(NAME kvaf_aiv_${direction}_transport_close_test COMMAND kvaf_aiv_${direction}_transport_close_test)
    set_tests_properties(kvaf_aiv_${direction}_transport_close_test PROPERTIES TIMEOUT 20)
  endforeach()
  target_compile_definitions(kvaf_aiv_put_transport_close_test PRIVATE KVAF_TEST_CLOSE_PUT=1)
  add_executable(kvaf_aicpu_put_resolution_test aicpu_put_resolution_test.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_access_operator.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_execution_resources.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/aicpu_urma/host/session.cc
    ${PROJECT_SOURCE_DIR}/access_operator/algorithms/aicpu_batch_put/host/batch_put_host.cc)
  target_include_directories(kvaf_aicpu_put_resolution_test PRIVATE ${KVAF_TEST_ACL_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/access_operator/algorithms)
  add_test(NAME kvaf_aicpu_put_resolution_test COMMAND kvaf_aicpu_put_resolution_test)
  set_tests_properties(kvaf_aicpu_put_resolution_test PROPERTIES TIMEOUT 20)
  add_executable(kvaf_aicpu_gather_resources_test aicpu_gather_resources_test.cc
    ${PROJECT_SOURCE_DIR}/access_operator/algorithms/aicpu_batch_get/host/batch_get_host.cc)
  target_include_directories(kvaf_aicpu_gather_resources_test PRIVATE ${KVAF_TEST_ACL_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/access_operator/algorithms ${PROJECT_SOURCE_DIR}/access_operator/src)
  add_test(NAME kvaf_aicpu_gather_resources_test COMMAND kvaf_aicpu_gather_resources_test)
  set_tests_properties(kvaf_aicpu_gather_resources_test PROPERTIES TIMEOUT 20)
  add_executable(kvaf_ubmem_provider_resources_test ubmem_provider_resources_test.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/ubmem/access_operator.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/ubmem/execution_resources.cc)
  target_include_directories(kvaf_ubmem_provider_resources_test PRIVATE ${KVAF_TEST_ACL_INCLUDE_DIR})
  # Older host-test headers lack this production ACL launch attribute. Only the
  # fake launch entry is used here; never substitute fields in a hardware build.
  include(CheckCXXSourceCompiles)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_INCLUDES "${KVAF_TEST_ACL_INCLUDE_DIR}")
  check_cxx_source_compiles("#include <acl/acl.h>
    int main() { aclrtLaunchKernelAttr a{};
      a.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE; a.value.dynUBufSize = 0; }"
    KVAF_TEST_ACL_HAS_DYN_UBUF_SIZE)
  cmake_pop_check_state()
  if(NOT KVAF_TEST_ACL_HAS_DYN_UBUF_SIZE)
    target_compile_definitions(kvaf_ubmem_provider_resources_test PRIVATE
      ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE=ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE
      dynUBufSize=localMemorySize)
  endif()
  add_test(NAME kvaf_ubmem_provider_resources_test COMMAND kvaf_ubmem_provider_resources_test)
  set_tests_properties(kvaf_ubmem_provider_resources_test PROPERTIES TIMEOUT 20)
  add_executable(kvaf_aiv_provider_resources_test aiv_provider_resources_test.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/access_operator.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/aiv_execution_resources.cc)
  target_compile_definitions(kvaf_aiv_provider_resources_test PRIVATE KVAF_ENABLE_AIV_BATCH_PUT=1)
  target_include_directories(kvaf_aiv_provider_resources_test PRIVATE ${KVAF_TEST_ACL_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/access_operator/algorithms/aiv_batch_get/include
    ${PROJECT_SOURCE_DIR}/access_operator/algorithms/aiv_batch_put/include)
  foreach(provider_test kvaf_aicpu_put_resolution_test kvaf_aicpu_gather_resources_test kvaf_ubmem_provider_resources_test
                        kvaf_aiv_provider_resources_test)
    target_include_directories(${provider_test} PRIVATE
      ${PROJECT_SOURCE_DIR}/access_operator/include ${PROJECT_SOURCE_DIR}/meta_manager/include
      ${PROJECT_SOURCE_DIR}/runtime/include)
    target_link_libraries(${provider_test} PRIVATE kvaf_common Threads::Threads)
  endforeach()
  add_test(NAME kvaf_aiv_provider_resources_test COMMAND kvaf_aiv_provider_resources_test)
  set_tests_properties(kvaf_aiv_provider_resources_test PROPERTIES TIMEOUT 20)
  add_executable(kvaf_ubmem_mapping_cache_test ubmem_mapping_cache_test.cc
    ${PROJECT_SOURCE_DIR}/access_operator/src/ubmem/device_mapping.cc)
  target_include_directories(kvaf_ubmem_mapping_cache_test PRIVATE ${KVAF_TEST_ACL_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/access_operator/include)
  target_link_libraries(kvaf_ubmem_mapping_cache_test PRIVATE kvaf_ubmem_hal Threads::Threads)
  add_test(NAME kvaf_ubmem_mapping_cache_test COMMAND kvaf_ubmem_mapping_cache_test)
  set_tests_properties(kvaf_ubmem_mapping_cache_test PROPERTIES TIMEOUT 20)
endif()
