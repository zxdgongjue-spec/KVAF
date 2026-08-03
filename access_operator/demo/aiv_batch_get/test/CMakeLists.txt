add_executable(kvaf_execute_batch_get_test execute_batch_get_main.cpp)
target_include_directories(kvaf_execute_batch_get_test PRIVATE
  "${KVAF_AIV_DEMO_INCLUDE_DIR}" "${ASCEND_CANN_PACKAGE_PATH}/include")
target_compile_definitions(kvaf_execute_batch_get_test PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(kvaf_execute_batch_get_test PRIVATE -O2 -Wall -Wextra -Werror)
target_link_directories(kvaf_execute_batch_get_test PRIVATE
  "${ASCEND_CANN_PACKAGE_PATH}/lib64" "${KVAF_AIV_DEMO_BIN_DIR}")
target_link_libraries(kvaf_execute_batch_get_test PRIVATE kvaf_execute_batch_get_host ascendcl pthread)
add_dependencies(kvaf_execute_batch_get_test kvaf_execute_batch_get_kernel)
set_target_properties(kvaf_execute_batch_get_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${KVAF_AIV_DEMO_BIN_DIR}")
