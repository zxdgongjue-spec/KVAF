add_library(kvaf_execute_batch_get_host SHARED batch_get_host.cpp execute_batch_get_loader.cpp
  ../../../cpu_sgl_write/urma_aiv/host/loader.cpp
  ../../../cpu_sgl_push/host/loader.cpp)
target_include_directories(kvaf_execute_batch_get_host
  PUBLIC "${KVAF_AIV_ALGORITHM_INCLUDE_DIR}"
  PRIVATE "${KVAF_ACCESS_OPERATOR_SOURCE_DIR}/algorithms/get/cpu_sgl_write/urma_aiv/include"
          "${KVAF_ACCESS_OPERATOR_SOURCE_DIR}/../common/include"
  PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
target_compile_definitions(kvaf_execute_batch_get_host PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(kvaf_execute_batch_get_host PRIVATE -O2 -fno-common -fno-strict-aliasing)
target_link_directories(kvaf_execute_batch_get_host PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(kvaf_execute_batch_get_host PRIVATE
  kvaf_direct_hccp_session
  kvaf_peer_link
  -Wl,--no-as-needed c_sec unified_dlog ascendc_runtime ascendalog ascendcl runtime error_manager dl pthread)
if(TARGET kvaf_execute_batch_get_kernel)
  add_dependencies(kvaf_execute_batch_get_host kvaf_execute_batch_get_kernel)
endif()
add_custom_command(TARGET kvaf_execute_batch_get_host POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${KVAF_AIV_ALGORITHM_BIN_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:kvaf_execute_batch_get_host>"
          "${KVAF_AIV_ALGORITHM_BIN_DIR}/")
