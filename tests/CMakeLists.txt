find_package(Python3 COMPONENTS Interpreter REQUIRED)

find_package(Threads REQUIRED)
add_executable(kvaf_npu_runtime_config_test npu_runtime_config_test.cc)
target_link_libraries(kvaf_npu_runtime_config_test PRIVATE kvaf_runtime)
add_test(NAME kvaf_npu_runtime_config_test COMMAND kvaf_npu_runtime_config_test)
add_test(NAME kvaf_aicpu_configuration_python
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_aicpu_configuration.py")
add_test(NAME kvaf_build_script_python
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_build_script.py")
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

add_executable(kvaf_cpu_push_scatter_test cpu_push_scatter_test.cc
  ${PROJECT_SOURCE_DIR}/access_operator/demo/aicpu_batch_get/kernel/cpu_push_scatter.cc)
target_include_directories(kvaf_cpu_push_scatter_test PRIVATE
  ${PROJECT_SOURCE_DIR}/access_operator/demo ${PROJECT_SOURCE_DIR}/access_operator/include)
target_link_libraries(kvaf_cpu_push_scatter_test PRIVATE kvaf_common Threads::Threads)
add_test(NAME kvaf_cpu_push_scatter_test COMMAND kvaf_cpu_push_scatter_test)

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

add_executable(kvaf_sequential_batch_get_kernel_model_test
  sequential_batch_get_kernel_model_test.cc
  ${PROJECT_SOURCE_DIR}/runtime/src/cpu_gather_executor.cc
  ${PROJECT_SOURCE_DIR}/runtime/src/cpu_push_gather_executor.cc)
target_include_directories(kvaf_sequential_batch_get_kernel_model_test PRIVATE
  aiv_model ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include
  ${PROJECT_SOURCE_DIR}/runtime/include ${PROJECT_SOURCE_DIR}/common/include)
target_compile_options(kvaf_sequential_batch_get_kernel_model_test PRIVATE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-UNDEBUG>)
target_link_libraries(kvaf_sequential_batch_get_kernel_model_test PRIVATE Threads::Threads)
add_test(NAME kvaf_sequential_batch_get_kernel_model_test COMMAND kvaf_sequential_batch_get_kernel_model_test)
foreach(mode scatter gather)
  add_test(NAME kvaf_aiv_${mode}_hardware_python
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_aiv_${mode}_batch_get.py")
  set_tests_properties(kvaf_aiv_${mode}_hardware_python PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 600 RESOURCE_LOCK kvaf_aiv_stage1_device
    ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python")
endforeach()
add_test(NAME kvaf_aiv_mode_isolation_python
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_aiv_mode_isolation.py")
if(NOT KVAF_ENABLE_AIV_BATCH_GET)
  add_test(NAME kvaf_aiv_kernel_build_python
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_aiv_kernel_build.py"
            "${CMAKE_COMMAND}")
  set(sequential_host_dir ${PROJECT_SOURCE_DIR}/access_operator/demo/aiv_batch_get)
  set(sequential_host_includes fake_acl ${sequential_host_dir}/include
    ${PROJECT_SOURCE_DIR}/access_operator/include ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_SOURCE_DIR}/common/include)
  add_executable(kvaf_sequential_loader_test sequential_batch_get_loader_test.cc
    ${sequential_host_dir}/host/execute_batch_get_loader.cpp
    ${sequential_host_dir}/host/execute_scatter_batch_get_loader.cpp
    ${sequential_host_dir}/host/execute_gather_batch_get_loader.cpp)
  target_include_directories(kvaf_sequential_loader_test PRIVATE ${sequential_host_includes})
  foreach(scenario compat legacy explicit custom relative missing wrong_symbol)
    add_test(NAME kvaf_sequential_loader_${scenario} COMMAND kvaf_sequential_loader_test ${scenario})
  endforeach()
  add_executable(kvaf_sequential_host_test sequential_batch_get_host_test.cc
    ${sequential_host_dir}/host/sequential_batch_get.cpp
    ${sequential_host_dir}/host/scatter_batch_get.cpp
    ${sequential_host_dir}/host/gather_batch_get.cpp)
  target_include_directories(kvaf_sequential_host_test PRIVATE ${sequential_host_includes})
  target_compile_options(kvaf_sequential_host_test PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-UNDEBUG>)
  add_test(NAME kvaf_sequential_host_test COMMAND kvaf_sequential_host_test)
endif()
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
