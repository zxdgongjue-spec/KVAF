add_library(kvaf_execute_batch_get_host SHARED batch_get_host.cpp execute_batch_get_loader.cpp)
target_include_directories(kvaf_execute_batch_get_host
  PUBLIC "${KVAF_AIV_DEMO_INCLUDE_DIR}"
  PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
target_compile_definitions(kvaf_execute_batch_get_host PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(kvaf_execute_batch_get_host PRIVATE -O2 -fno-common -fno-strict-aliasing)
target_link_directories(kvaf_execute_batch_get_host PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(kvaf_execute_batch_get_host PRIVATE
  -Wl,--no-as-needed c_sec unified_dlog ascendc_runtime ascendalog ascendcl runtime error_manager dl pthread)
add_dependencies(kvaf_execute_batch_get_host kvaf_execute_batch_get_kernel)
add_custom_command(TARGET kvaf_execute_batch_get_host POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${KVAF_AIV_DEMO_BIN_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:kvaf_execute_batch_get_host>"
          "${KVAF_AIV_DEMO_BIN_DIR}/")
