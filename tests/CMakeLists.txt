find_package(Python3 COMPONENTS Interpreter REQUIRED)

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

# CPU-only URMA lease ownership test; no NPU/transport correctness claim.
if(KVAF_ENABLE_URMA AND TARGET kvaf_runtime AND TARGET kvaf_peer_link)
  add_executable(kvaf_fused_batch_get_context_test
    fused_batch_get_context_test.cc ../access_operator/src/access_operator.cc)
  target_include_directories(kvaf_fused_batch_get_context_test PRIVATE
    fused_context_stubs ../access_operator/demo/aiv_batch_get/include)
  target_link_libraries(kvaf_fused_batch_get_context_test PRIVATE
    kvaf_runtime kvaf_meta_manager kvaf_peer_link)
  add_test(NAME kvaf_fused_batch_get_context_host
    COMMAND kvaf_fused_batch_get_context_test)
endif()

# Exercise host-DDR ownership and rollback with private ACL stubs, without NPU.
add_executable(kvaf_ubmem_host_arena_test
  ubmem_host_arena_test.cc ../runtime/src/ubmem/host_arena.cc
  ../runtime/src/ubmem/host_memory.cc)
target_include_directories(kvaf_ubmem_host_arena_test PRIVATE
  ubmem_host_arena_stubs ../runtime/include ../common/include
  ../access_operator/include ../meta_manager/include)
target_compile_features(kvaf_ubmem_host_arena_test PRIVATE cxx_std_17)
target_compile_definitions(kvaf_ubmem_host_arena_test PRIVATE KVAF_TEST_UBMEM_HOST_MEMORY=1)
add_test(NAME kvaf_ubmem_host_arena_host COMMAND kvaf_ubmem_host_arena_test)

# Exercise the real BindNuma implementation and its Linux syscall arguments.
# The arena lifecycle test replaces BindNuma itself and cannot catch ABI bugs.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(kvaf_ubmem_numa_syscall_test
    ubmem_numa_syscall_test.cc ../runtime/src/ubmem/host_memory.cc)
  target_include_directories(kvaf_ubmem_numa_syscall_test PRIVATE ../common/include)
  target_compile_features(kvaf_ubmem_numa_syscall_test PRIVATE cxx_std_17)
  target_link_options(kvaf_ubmem_numa_syscall_test PRIVATE "-Wl,--wrap=syscall")
  add_test(NAME kvaf_ubmem_numa_syscall_host COMMAND kvaf_ubmem_numa_syscall_test)
endif()

# Isolate the native UBmem lease lifecycle from both transport libraries and
# real ACL. Reuse the allocator driver model; no device correctness claim.
add_executable(kvaf_ubmem_fused_context_test
  ubmem_host_arena_test.cc
  ../runtime/src/ubmem/host_arena.cc ../runtime/src/ubmem/host_memory.cc
  ../runtime/src/npu_runtime.cc
  ../runtime/src/runtime.cc ../meta_manager/src/in_memory_meta_manager.cc)
target_include_directories(kvaf_ubmem_fused_context_test PRIVATE
  ubmem_host_arena_stubs ../runtime/include ../common/include
  ../access_operator/include ../access_operator/ubmem/include ../meta_manager/include)
target_compile_definitions(kvaf_ubmem_fused_context_test PRIVATE
  KVAF_ENABLE_URMA=0 KVAF_ENABLE_UBMEM=1 KVAF_TEST_UBMEM_FUSED_CONTEXT=1
  KVAF_TEST_UBMEM_HOST_MEMORY=1)
target_compile_features(kvaf_ubmem_fused_context_test PRIVATE cxx_std_17)
add_test(NAME kvaf_ubmem_fused_context_host COMMAND kvaf_ubmem_fused_context_test)

add_executable(kvaf_runtime_completion_test runtime_completion_test.cc)
target_link_libraries(kvaf_runtime_completion_test PRIVATE kvaf_runtime kvaf_meta_manager)
add_test(NAME kvaf_runtime_completion_host COMMAND kvaf_runtime_completion_test)

# Opt-in hardware smoke test; ordinary CTest runs remain host-only for UBmem.
option(KVAF_RUN_UBMEM_NPU_TEST "Register the UBmem NPU Put/Get smoke test" OFF)
if(KVAF_RUN_UBMEM_NPU_TEST)
  if(NOT (KVAF_ENABLE_NPU AND KVAF_ENABLE_UBMEM AND KVAF_BUILD_PYTHON))
    message(FATAL_ERROR "KVAF_RUN_UBMEM_NPU_TEST requires the UBmem NPU Python build")
  endif()
  add_test(NAME kvaf_ubmem_runtime_python
    COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_ubmem_runtime.py"
            --build-dir "${CMAKE_BINARY_DIR}")
  set_tests_properties(kvaf_ubmem_runtime_python PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
    TIMEOUT 120 LABELS "npu;ubmem")
endif()
