find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(Threads REQUIRED)

foreach(test npu_runtime_config runtime_cleanup)
  add_executable(kvaf_${test}_test ${test}_test.cc)
  target_link_libraries(kvaf_${test}_test PRIVATE kvaf_runtime)
  add_test(NAME kvaf_${test}_test COMMAND kvaf_${test}_test)
endforeach()

if(KVAF_BUILD_REMOTE_CPU_SERVICE AND NOT KVAF_CPU_ENABLE_URMA)
  add_executable(kvaf_cpu_backend_dispatch_test cpu_backend_dispatch_test.cc)
  target_link_libraries(kvaf_cpu_backend_dispatch_test PRIVATE kvaf_cpu_runtime)
  add_test(NAME kvaf_cpu_backend_dispatch_test COMMAND kvaf_cpu_backend_dispatch_test)
endif()

add_executable(kvaf_cpu_push_scatter_test cpu_push_scatter_test.cc
  ${PROJECT_SOURCE_DIR}/access_operator/algorithms/get/scatter/urma_aicpu/kernel/cpu_push_scatter.cc)
target_include_directories(kvaf_cpu_push_scatter_test PRIVATE
  ${PROJECT_SOURCE_DIR}/access_operator
  ${PROJECT_SOURCE_DIR}/access_operator/algorithms
  ${PROJECT_SOURCE_DIR}/access_operator/include)
target_link_libraries(kvaf_cpu_push_scatter_test PRIVATE kvaf_common Threads::Threads)
add_test(NAME kvaf_cpu_push_scatter_test COMMAND kvaf_cpu_push_scatter_test)

set(get_dir ${PROJECT_SOURCE_DIR}/access_operator/algorithms/get)
set(sequential_aiv_includes ${get_dir}/common/aiv/include ${get_dir}/common/aiv/kernel
  ${get_dir}/gather/urma_aiv/include ${get_dir}/scatter/urma_aiv/include)
add_executable(kvaf_sequential_batch_get_kernel_model_test
  sequential_batch_get_kernel_model_test.cc
  ${PROJECT_SOURCE_DIR}/runtime/cpu_service/gather/cpu_gather_executor.cc
  ${PROJECT_SOURCE_DIR}/runtime/cpu_service/push/cpu_push_gather_executor.cc)
target_include_directories(kvaf_sequential_batch_get_kernel_model_test PRIVATE
  aiv_model ${get_dir}/direct/urma_aiv/include ${sequential_aiv_includes}
  ${PROJECT_SOURCE_DIR}/runtime/cpu_service/push
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
  set(sequential_host_dir ${PROJECT_SOURCE_DIR}/access_operator/algorithms/get/direct/urma_aiv)
  set(sequential_host_includes fake_acl ${sequential_host_dir}/include ${sequential_aiv_includes}
    ${get_dir}/common/aiv/host ${PROJECT_SOURCE_DIR}/access_operator/providers/urma_aiv
    ${PROJECT_SOURCE_DIR}/access_operator/include ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_SOURCE_DIR}/common/include)
  add_executable(kvaf_sequential_loader_test sequential_batch_get_loader_test.cc
    ${sequential_host_dir}/host/execute_batch_get_loader.cpp
    ${get_dir}/scatter/urma_aiv/host/execute_scatter_batch_get_loader.cpp
    ${get_dir}/gather/urma_aiv/host/execute_gather_batch_get_loader.cpp)
  target_include_directories(kvaf_sequential_loader_test PRIVATE ${sequential_host_includes})
  foreach(scenario compat legacy explicit custom relative missing wrong_symbol)
    add_test(NAME kvaf_sequential_loader_${scenario} COMMAND kvaf_sequential_loader_test ${scenario})
  endforeach()
  add_executable(kvaf_sequential_host_test sequential_batch_get_host_test.cc
    ${PROJECT_SOURCE_DIR}/access_operator/providers/urma_aiv/sequential_batch_get.cc
    ${PROJECT_SOURCE_DIR}/access_operator/providers/urma_aiv/scatter_resources.cc
    ${PROJECT_SOURCE_DIR}/access_operator/providers/urma_aiv/gather_resources.cc
    ${get_dir}/scatter/urma_aiv/host/scatter_batch_get.cpp
    ${get_dir}/gather/urma_aiv/host/gather_batch_get.cpp)
  target_include_directories(kvaf_sequential_host_test PRIVATE ${sequential_host_includes})
  target_compile_options(kvaf_sequential_host_test PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-UNDEBUG>)
  add_test(NAME kvaf_sequential_host_test COMMAND kvaf_sequential_host_test)
endif()

if(KVAF_BUILD_PYTHON)
  pybind11_add_module(_kvaf_constructor_test npu_constructor_binding.cc)
  target_include_directories(_kvaf_constructor_test PRIVATE ${PROJECT_SOURCE_DIR}/bindings/include)
  set_target_properties(_kvaf_constructor_test PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/python")
  add_test(NAME kvaf_npu_constructor_python
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_npu_constructor.py")
  set_tests_properties(kvaf_npu_constructor_python PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python")
endif()
