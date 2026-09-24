find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_subdirectory(aicpu_batch_get)

add_executable(kvaf_aicpu_urma_wqe_test aicpu_urma_wqe_test.cc)
target_include_directories(kvaf_aicpu_urma_wqe_test PRIVATE "${CMAKE_SOURCE_DIR}/access_operator/src")
add_test(NAME kvaf_aicpu_urma_wqe COMMAND kvaf_aicpu_urma_wqe_test)

add_executable(kvaf_aicpu_urma_kernel_test aicpu_urma_kernel_test.cc
  "${CMAKE_SOURCE_DIR}/access_operator/src/aicpu_urma/kernel.cc")
target_compile_definitions(kvaf_aicpu_urma_kernel_test PRIVATE KVAF_AICPU_URMA_HOST_TEST=1)
target_include_directories(kvaf_aicpu_urma_kernel_test PRIVATE
  "${CMAKE_SOURCE_DIR}/access_operator/src"
  "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include"
  "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_put/include")
add_test(NAME kvaf_aicpu_urma_kernel COMMAND kvaf_aicpu_urma_kernel_test)

add_executable(kvaf_aicpu_urma_client_options_test aicpu_urma_client_options_test.cc)
add_test(NAME kvaf_aicpu_urma_client_options COMMAND kvaf_aicpu_urma_client_options_test)

add_test(NAME kvaf_aicpu_urma_install
  COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/aicpu_urma_install_test.sh")

add_executable(kvaf_aicpu_sgl_simulator aicpu_sgl_simulator.cc)
find_package(Threads REQUIRED)
find_package(OpenMP COMPONENTS CXX QUIET)
target_include_directories(kvaf_aicpu_sgl_simulator PRIVATE
  "${CMAKE_SOURCE_DIR}/access_operator/include"
  "${CMAKE_SOURCE_DIR}/runtime/include")
target_link_libraries(kvaf_aicpu_sgl_simulator PRIVATE kvaf_common Threads::Threads)
if(OpenMP_CXX_FOUND)
  target_link_libraries(kvaf_aicpu_sgl_simulator PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_features(kvaf_aicpu_sgl_simulator PRIVATE cxx_std_17)

add_test(NAME kvaf_aicpu_sgl_sim_single
  COMMAND kvaf_aicpu_sgl_simulator --rows 7 --row-bytes 128
          --sgl-size 1 --iterations 2 --wr-overhead-ns 100)
set_tests_properties(kvaf_aicpu_sgl_sim_single PROPERTIES
  PASS_REGULAR_EXPRESSION "AICPU_SGL_SIM_PASS.*wr_count_per_iteration=7")
add_test(NAME kvaf_aicpu_sgl_sim_partial
  COMMAND kvaf_aicpu_sgl_simulator --rows 7 --row-bytes 128
          --sgl-size 3 --iterations 2 --wr-overhead-ns 100)
set_tests_properties(kvaf_aicpu_sgl_sim_partial PROPERTIES
  PASS_REGULAR_EXPRESSION "AICPU_SEND_MODEL.*send_ns=671.68")
add_test(NAME kvaf_aicpu_sgl_sim_oversized
  COMMAND kvaf_aicpu_sgl_simulator --rows 7 --row-bytes 128
          --sgl-size 32 --iterations 1)
set_tests_properties(kvaf_aicpu_sgl_sim_oversized PROPERTIES
  PASS_REGULAR_EXPRESSION "AICPU_SGL_SIM_PASS.*wr_count_per_iteration=1")
add_test(NAME kvaf_aicpu_sgl_sim_invalid
  COMMAND kvaf_aicpu_sgl_simulator --sgl-size 0)
set_tests_properties(kvaf_aicpu_sgl_sim_invalid PROPERTIES WILL_FAIL TRUE)
add_test(NAME kvaf_aicpu_sgl_sim_parallel
  COMMAND kvaf_aicpu_sgl_simulator --rows 257 --row-bytes 4096
          --sgl-size 7 --iterations 3 --gather-threads 4 --gather-mode active)
set_tests_properties(kvaf_aicpu_sgl_sim_parallel PROPERTIES
  PASS_REGULAR_EXPRESSION "AICPU_SGL_SIM_PASS.*gather_threads_effective=4 gather_mode=active")
add_test(NAME kvaf_aicpu_sgl_sim_passive
  COMMAND kvaf_aicpu_sgl_simulator --rows 257 --row-bytes 4096
          --sgl-size 7 --iterations 3 --gather-threads 4 --gather-mode passive)
set_tests_properties(kvaf_aicpu_sgl_sim_passive PROPERTIES
  PASS_REGULAR_EXPRESSION "AICPU_SGL_SIM_PASS.*gather_threads_effective=4 gather_mode=passive")
if(OpenMP_CXX_FOUND)
  add_test(NAME kvaf_aicpu_sgl_sim_omp
    COMMAND kvaf_aicpu_sgl_simulator --rows 257 --row-bytes 4096
            --sgl-size 7 --iterations 3 --gather-threads 4 --gather-mode omp)
  set_tests_properties(kvaf_aicpu_sgl_sim_omp PROPERTIES
    PASS_REGULAR_EXPRESSION "AICPU_SGL_SIM_PASS.*gather_threads_effective=4 gather_mode=omp")
else()
  add_test(NAME kvaf_aicpu_sgl_sim_omp_unavailable
    COMMAND kvaf_aicpu_sgl_simulator --gather-mode omp)
  set_tests_properties(kvaf_aicpu_sgl_sim_omp_unavailable PROPERTIES
    WILL_FAIL TRUE)
endif()
add_test(NAME kvaf_aicpu_sgl_sim_small_fallback
  COMMAND kvaf_aicpu_sgl_simulator --rows 7 --row-bytes 128
          --sgl-size 3 --iterations 2 --gather-threads 8)
set_tests_properties(kvaf_aicpu_sgl_sim_small_fallback PROPERTIES
  PASS_REGULAR_EXPRESSION "AICPU_SGL_SIM_PASS.*gather_threads_effective=1")
add_test(NAME kvaf_aicpu_sgl_sim_invalid_threads
  COMMAND kvaf_aicpu_sgl_simulator --gather-threads 0)
set_tests_properties(kvaf_aicpu_sgl_sim_invalid_threads PROPERTIES WILL_FAIL TRUE)
add_test(NAME kvaf_aicpu_sgl_sim_invalid_mode
  COMMAND kvaf_aicpu_sgl_simulator --gather-mode unknown)
set_tests_properties(kvaf_aicpu_sgl_sim_invalid_mode PROPERTIES WILL_FAIL TRUE)
add_test(NAME kvaf_aicpu_sgl_sim_ns_log
  COMMAND kvaf_aicpu_sgl_simulator --rows 7 --row-bytes 128
          --sgl-size 3 --iterations 2 --gather-threads 1 --log-iterations 1)
set_tests_properties(kvaf_aicpu_sgl_sim_ns_log PROPERTIES
  PASS_REGULAR_EXPRESSION "ITERATION_NS index=2 gather_ns=[0-9]+ plan_ns=[0-9]+")
add_test(NAME kvaf_aicpu_sgl_sim_ns_summary
  COMMAND kvaf_aicpu_sgl_simulator --rows 7 --row-bytes 128
          --sgl-size 3 --iterations 2 --gather-threads 1)
set_tests_properties(kvaf_aicpu_sgl_sim_ns_summary PROPERTIES
  PASS_REGULAR_EXPRESSION "CPU_PREP_PER_ITERATION gather_avg_ns=[0-9]+\\.[0-9]+ gather_min_ns=[0-9]+ gather_max_ns=[0-9]+")
add_test(NAME kvaf_aicpu_sgl_sim_invalid_ns_log
  COMMAND kvaf_aicpu_sgl_simulator --log-iterations 2)
set_tests_properties(kvaf_aicpu_sgl_sim_invalid_ns_log PROPERTIES WILL_FAIL TRUE)

add_executable(kvaf_aiv_batch_roundtrip_options_test aiv_batch_roundtrip_options_test.cc)
add_test(NAME kvaf_aiv_batch_roundtrip_options COMMAND kvaf_aiv_batch_roundtrip_options_test)

if(KVAF_ENABLE_AIV_BATCH_PUT)
  add_executable(kvaf_aiv_batch_roundtrip aiv_batch_roundtrip.cc)
  target_compile_definitions(kvaf_aiv_batch_roundtrip PRIVATE
    "KVAF_AIV_NATIVE_CANN_ROOT=\"${ASCEND_CANN_PACKAGE_PATH}\""
    "KVAF_AIV_NATIVE_BUILD_DIR=\"${CMAKE_BINARY_DIR}\"")
  target_include_directories(kvaf_aiv_batch_roundtrip PRIVATE
    "${CMAKE_SOURCE_DIR}/runtime/src"
    "${ASCEND_CANN_PACKAGE_PATH}/include")
  target_link_directories(kvaf_aiv_batch_roundtrip PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
  target_link_libraries(kvaf_aiv_batch_roundtrip PRIVATE
    kvaf_peer_link kvaf_direct_hccp_session
    kvaf_execute_batch_put_host kvaf_execute_batch_get_host ascendcl ${CMAKE_DL_LIBS})
  target_compile_features(kvaf_aiv_batch_roundtrip PRIVATE cxx_std_17)
  set_property(TARGET kvaf_aiv_batch_roundtrip APPEND PROPERTY BUILD_RPATH
    "${ASCEND_CANN_PACKAGE_PATH}/lib64")
endif()

if(KVAF_ENABLE_AICPU_URMA)
  add_executable(kvaf_aicpu_urma_client aicpu_urma_client.cc)
  target_compile_definitions(kvaf_aicpu_urma_client PRIVATE
    "KVAF_AICPU_URMA_CANN_ROOT=\"${ASCEND_CANN_PACKAGE_PATH}\""
    "KVAF_AICPU_URMA_BUILD_DIR=\"${CMAKE_BINARY_DIR}\"")
  target_include_directories(kvaf_aicpu_urma_client PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
  target_link_directories(kvaf_aicpu_urma_client PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
  target_link_libraries(kvaf_aicpu_urma_client PRIVATE kvaf_npu_runtime ascendcl)
  target_compile_features(kvaf_aicpu_urma_client PRIVATE cxx_std_17)
  set_property(TARGET kvaf_aicpu_urma_client APPEND PROPERTY BUILD_RPATH
    "${ASCEND_CANN_PACKAGE_PATH}/lib64")
endif()

add_executable(kvaf_cpu_urma_link_selection_test cpu_urma_link_selection_test.cc)
target_include_directories(kvaf_cpu_urma_link_selection_test PRIVATE
  "${CMAKE_SOURCE_DIR}/access_operator/include")
target_link_libraries(kvaf_cpu_urma_link_selection_test PRIVATE kvaf_peer_link)
add_test(NAME kvaf_cpu_urma_link_selection COMMAND kvaf_cpu_urma_link_selection_test)

# This provider-double test needs only UMDK headers, not liburma or UB hardware.
find_path(KVAF_DISCOVERY_TEST_URMA_INCLUDE_DIR urma_api.h
  HINTS "${KVAF_URMA_INCLUDE_DIR}" "${KVAF_URMA_ROOT}/include"
        "${CMAKE_SOURCE_DIR}/../third_party/umdk/include")
if(KVAF_DISCOVERY_TEST_URMA_INCLUDE_DIR)
  add_executable(kvaf_cpu_urma_discovery_test cpu_urma_discovery_test.cc
    "${CMAKE_SOURCE_DIR}/access_operator/src/cpu_urma/cpu_urma_discovery.cc")
  target_include_directories(kvaf_cpu_urma_discovery_test PRIVATE
    "${CMAKE_SOURCE_DIR}/access_operator/include" "${KVAF_DISCOVERY_TEST_URMA_INCLUDE_DIR}")
  target_link_libraries(kvaf_cpu_urma_discovery_test PRIVATE kvaf_common)
  add_test(NAME kvaf_cpu_urma_discovery COMMAND kvaf_cpu_urma_discovery_test)
endif()

add_executable(kvaf_cpu_hbm_client_options_test cpu_hbm_client_options_test.cc)
target_include_directories(kvaf_cpu_hbm_client_options_test PRIVATE
  "${CMAKE_SOURCE_DIR}/access_operator/include")
target_link_libraries(kvaf_cpu_hbm_client_options_test PRIVATE kvaf_peer_link)
target_compile_features(kvaf_cpu_hbm_client_options_test PRIVATE cxx_std_17)
add_test(NAME kvaf_cpu_hbm_client_options COMMAND kvaf_cpu_hbm_client_options_test)

add_test(NAME kvaf_cpu_batch_put_metrics
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_cpu_batch_put_metrics.py")

add_test(NAME kvaf_npu_hbm_allocator
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_npu_hbm_allocator.py")

add_test(NAME kvaf_aiv_batch_get_bw
  COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/python/test_aiv_batch_get_bw.py")

add_executable(kvaf_cpu_urma_gather_builder_test cpu_urma_gather_builder_test.cc)
target_include_directories(kvaf_cpu_urma_gather_builder_test PRIVATE
  "${CMAKE_SOURCE_DIR}/access_operator/include")
target_link_libraries(kvaf_cpu_urma_gather_builder_test PRIVATE kvaf_peer_link)
target_compile_features(kvaf_cpu_urma_gather_builder_test PRIVATE cxx_std_17)
add_test(NAME kvaf_cpu_urma_gather_builder
  COMMAND kvaf_cpu_urma_gather_builder_test)

add_executable(kvaf_runtime_batch_put_prefix_test
  runtime_batch_put_prefix_test.cc)
target_link_libraries(kvaf_runtime_batch_put_prefix_test PRIVATE kvaf_runtime)
target_compile_features(kvaf_runtime_batch_put_prefix_test PRIVATE cxx_std_17)
add_test(NAME kvaf_runtime_batch_put_prefix
  COMMAND kvaf_runtime_batch_put_prefix_test)

add_executable(kvaf_npu_target_channel_test npu_target_channel_test.cc)
target_include_directories(kvaf_npu_target_channel_test PRIVATE
  "${CMAKE_SOURCE_DIR}/runtime/src"
  "${CMAKE_SOURCE_DIR}/access_operator/include"
  "${CMAKE_SOURCE_DIR}/common/include")
target_compile_features(kvaf_npu_target_channel_test PRIVATE cxx_std_17)
add_test(NAME kvaf_npu_target_channel COMMAND kvaf_npu_target_channel_test)

add_executable(kvaf_peer_hbm_eid_test peer_hbm_eid_test.cc)
target_include_directories(kvaf_peer_hbm_eid_test PRIVATE "${CMAKE_SOURCE_DIR}/runtime/src")
target_link_libraries(kvaf_peer_hbm_eid_test PRIVATE kvaf_peer_link)
add_test(NAME kvaf_peer_hbm_eid COMMAND kvaf_peer_hbm_eid_test)

add_executable(kvaf_cpu_urma_eid_perf_metrics_test cpu_urma_eid_perf_metrics_test.cc)
target_compile_features(kvaf_cpu_urma_eid_perf_metrics_test PRIVATE cxx_std_17)
add_test(NAME kvaf_cpu_urma_eid_perf_metrics COMMAND kvaf_cpu_urma_eid_perf_metrics_test)

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

# Python tests cover the public binding. The compact-prefix C++ test protects
# Runtime filtering before MetaManager and AccessOperator execution.

if(KVAF_BUILD_REMOTE_CPU_SERVICE)
  add_executable(kvaf_cpu_urma_cpu_bw cpu_urma_cpu_bw.cc)
  target_link_libraries(kvaf_cpu_urma_cpu_bw PRIVATE kvaf_cpu_urma_transport)
  target_compile_features(kvaf_cpu_urma_cpu_bw PRIVATE cxx_std_17)

  add_executable(kvaf_cpu_urma_eid_perf cpu_urma_eid_perf.cc)
  target_include_directories(kvaf_cpu_urma_eid_perf PRIVATE "${KVAF_URMA_INCLUDE_DIR}")
  target_link_libraries(kvaf_cpu_urma_eid_perf PRIVATE kvaf_cpu_urma_transport "${KVAF_URMA_LIBRARY}")
  target_compile_features(kvaf_cpu_urma_eid_perf PRIVATE cxx_std_17)
  # Standalone coverage uses the production transport owned by
  # access_operator; it does not carry a second URMA implementation.
  add_executable(kvaf_cpu_hbm_client cpu_hbm_client.cc)
  target_include_directories(kvaf_cpu_hbm_client PRIVATE
    "${KVAF_URMA_INCLUDE_DIR}")
  target_link_libraries(kvaf_cpu_hbm_client PRIVATE kvaf_cpu_urma_transport)
  target_compile_features(kvaf_cpu_hbm_client PRIVATE cxx_std_17)
  get_filename_component(KVAF_CPU_URMA_LIBRARY_DIR "${KVAF_URMA_LIBRARY}" DIRECTORY)
  set_property(TARGET kvaf_cpu_urma_cpu_bw APPEND PROPERTY BUILD_RPATH
    "${KVAF_CPU_URMA_LIBRARY_DIR}")
  set_property(TARGET kvaf_cpu_hbm_client APPEND PROPERTY BUILD_RPATH
    "${KVAF_CPU_URMA_LIBRARY_DIR}")
  set_property(TARGET kvaf_cpu_urma_eid_perf APPEND PROPERTY BUILD_RPATH
    "${KVAF_CPU_URMA_LIBRARY_DIR}")
endif()

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
