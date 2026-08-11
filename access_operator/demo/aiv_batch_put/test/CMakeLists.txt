add_executable(kvaf_execute_batch_put_test execute_batch_put_main.cpp)
target_include_directories(kvaf_execute_batch_put_test PRIVATE
  "${KVAF_AIV_DEMO_INCLUDE_DIR}" "${ASCEND_CANN_PACKAGE_PATH}/include")
target_compile_definitions(kvaf_execute_batch_put_test PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(kvaf_execute_batch_put_test PRIVATE -O2 -Wall -Wextra -Werror)
target_link_directories(kvaf_execute_batch_put_test PRIVATE
  "${ASCEND_CANN_PACKAGE_PATH}/lib64" "${KVAF_AIV_DEMO_BIN_DIR}")
target_link_libraries(kvaf_execute_batch_put_test PRIVATE
  kvaf_execute_batch_put_host ascendcl stdc++ pthread)
add_dependencies(kvaf_execute_batch_put_test kvaf_execute_batch_put_kernel)
set_target_properties(kvaf_execute_batch_put_test PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${KVAF_AIV_DEMO_BIN_DIR}")
