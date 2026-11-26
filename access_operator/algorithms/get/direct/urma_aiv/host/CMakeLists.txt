add_library(kvaf_execute_batch_get_host SHARED batch_get_host.cpp execute_batch_get_loader.cpp)
if(TARGET kvaf_access_operator)
  set(get_dir "${KVAF_ACCESS_OPERATOR_SOURCE_DIR}/algorithms/get")
  foreach(mode gather scatter)
    set(algorithm_dir "${get_dir}/${mode}/urma_aiv")
    target_sources(kvaf_execute_batch_get_host PRIVATE
      "${KVAF_ACCESS_OPERATOR_SOURCE_DIR}/providers/urma_aiv/${mode}_resources.cc"
      "${algorithm_dir}/host/${mode}_batch_get.cpp"
      "${algorithm_dir}/host/execute_${mode}_batch_get_loader.cpp")
    target_include_directories(kvaf_execute_batch_get_host PRIVATE "${algorithm_dir}/include")
  endforeach()
  target_include_directories(kvaf_execute_batch_get_host
    PUBLIC "${get_dir}/common/aiv/include"
    PRIVATE "${get_dir}/common/aiv/host" "${KVAF_ACCESS_OPERATOR_SOURCE_DIR}/providers/urma_aiv")
  target_link_libraries(kvaf_execute_batch_get_host PRIVATE kvaf_access_operator)
endif()
target_include_directories(kvaf_execute_batch_get_host
  PUBLIC "${KVAF_AIV_ALGORITHM_INCLUDE_DIR}"
  PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
target_compile_definitions(kvaf_execute_batch_get_host PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(kvaf_execute_batch_get_host PRIVATE -O2 -fno-common -fno-strict-aliasing)
target_link_directories(kvaf_execute_batch_get_host PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(kvaf_execute_batch_get_host PRIVATE
  kvaf_direct_hccp_session
  kvaf_peer_link
  -Wl,--no-as-needed c_sec unified_dlog ascendc_runtime ascendalog ascendcl runtime error_manager dl pthread)
if(TARGET kvaf_batch_get_kernels)
  add_dependencies(kvaf_execute_batch_get_host kvaf_batch_get_kernels)
endif()
add_custom_command(TARGET kvaf_execute_batch_get_host POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${KVAF_AIV_ALGORITHM_BIN_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:kvaf_execute_batch_get_host>"
          "${KVAF_AIV_ALGORITHM_BIN_DIR}/")
