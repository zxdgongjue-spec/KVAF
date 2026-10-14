find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Build only the allocator and a private ACL stub. In particular, this target
# does not link Runtime, PeerLink, the CPU service or any device transport.
add_executable(kvaf_local_host_ddr_allocator_test
  local_host_ddr_allocator_test.cc
  ../runtime/src/local_host_ddr.cc)
target_include_directories(kvaf_local_host_ddr_allocator_test PRIVATE
  local_host_ddr_stubs
  ../runtime/include
  ../access_operator/include
  ../meta_manager/include)
target_link_libraries(kvaf_local_host_ddr_allocator_test PRIVATE kvaf_common)
add_test(NAME kvaf_local_host_ddr_allocator_host
  COMMAND kvaf_local_host_ddr_allocator_test)
add_test(NAME kvaf_local_host_ddr_allocator_thp
  COMMAND kvaf_local_host_ddr_allocator_test --thp)
set_tests_properties(kvaf_local_host_ddr_allocator_thp PROPERTIES
  SKIP_RETURN_CODE 77)

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
  ../access_operator/demo/aiv_batch_get/include
  ../access_operator/demo/aiv_batch_put/include)
target_link_libraries(kvaf_fused_batch_get_context_test PRIVATE
  kvaf_runtime kvaf_meta_manager kvaf_peer_link)
add_test(NAME kvaf_fused_batch_get_context_host
  COMMAND kvaf_fused_batch_get_context_test)

# Exercise the same real operator with both Get and Put fake transports. The
# small Put pool must not change leased Get geometry or DDR teardown ownership.
add_executable(kvaf_independent_batch_put_context_test
  fused_batch_get_context_test.cc
  ../access_operator/src/access_operator.cc)
target_include_directories(kvaf_independent_batch_put_context_test PRIVATE
  fused_context_stubs
  ../access_operator/demo/aiv_batch_get/include
  ../access_operator/demo/aiv_batch_put/include)
target_compile_definitions(kvaf_independent_batch_put_context_test PRIVATE
  KVAF_ENABLE_AIV_BATCH_PUT=1)
target_link_libraries(kvaf_independent_batch_put_context_test PRIVATE
  kvaf_runtime kvaf_meta_manager kvaf_peer_link)
add_test(NAME kvaf_independent_batch_put_context_host
  COMMAND kvaf_independent_batch_put_context_test)

# Also compile the NpuRuntime forwarding API against the same declarations;
# this is a host syntax check and deliberately has no runtime/device backend.
add_library(kvaf_fused_npu_runtime_syntax OBJECT ../runtime/src/npu_runtime.cc)
target_include_directories(kvaf_fused_npu_runtime_syntax PRIVATE
  fused_context_stubs
  ../access_operator/demo/aiv_batch_get/include
  ../access_operator/demo/aiv_batch_put/include)
target_link_libraries(kvaf_fused_npu_runtime_syntax PRIVATE kvaf_runtime)

add_executable(kvaf_batch_get_device_control_test
  ../access_operator/demo/aiv_batch_get/kernel/tests/test_batch_get_device.cpp)
target_include_directories(kvaf_batch_get_device_control_test PRIVATE
  ../access_operator/demo/aiv_batch_get/kernel/tests
  ../access_operator/demo/aiv_batch_get/include)
add_test(NAME kvaf_batch_get_device_control_host
  COMMAND kvaf_batch_get_device_control_test)

# Real local endpoint teardown with injected RA callbacks; no NPU is required.
add_executable(kvaf_local_host_transport_lifecycle_test
  local_host_transport_lifecycle_test.cc
  ../access_operator/src/direct_hccp/direct_hccp_local_host.cc)
target_include_directories(kvaf_local_host_transport_lifecycle_test PRIVATE
  ../access_operator/include)
add_test(NAME kvaf_local_host_transport_lifecycle
  COMMAND kvaf_local_host_transport_lifecycle_test)

# Runtime ownership and teardown with injected CPU-only dependencies.
add_executable(kvaf_runtime_core_test runtime_core_test.cc)
target_link_libraries(kvaf_runtime_core_test PRIVATE kvaf_runtime)
add_test(NAME kvaf_runtime_core_initialize_cleanup
  COMMAND kvaf_runtime_core_test)
