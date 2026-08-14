add_library(kvaf_direct_hccp_fake_ra SHARED fake_ra.cc)
set_target_properties(kvaf_direct_hccp_fake_ra PROPERTIES OUTPUT_NAME ra)
target_include_directories(kvaf_direct_hccp_fake_ra PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../aiv_batch_get/include")
target_compile_features(kvaf_direct_hccp_fake_ra PRIVATE cxx_std_17)

add_library(kvaf_direct_hccp_fake_runtime SHARED fake_runtime.cc)
set_target_properties(kvaf_direct_hccp_fake_runtime PROPERTIES OUTPUT_NAME ascendcl)
target_include_directories(kvaf_direct_hccp_fake_runtime PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../aiv_batch_get/include")
target_compile_features(kvaf_direct_hccp_fake_runtime PRIVATE cxx_std_17)

add_executable(kvaf_direct_hccp_session_test direct_hccp_session_test.cc)
add_dependencies(kvaf_direct_hccp_session_test
  kvaf_direct_hccp_fake_ra
  kvaf_direct_hccp_fake_runtime)
target_include_directories(kvaf_direct_hccp_session_test PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_libraries(kvaf_direct_hccp_session_test PRIVATE
  kvaf_direct_hccp_session
  dl)
target_compile_features(kvaf_direct_hccp_session_test PRIVATE cxx_std_17)

add_test(
  NAME kvaf_direct_hccp_session_reuses_existing_device_ra
  COMMAND kvaf_direct_hccp_session_test)
set_tests_properties(kvaf_direct_hccp_session_reuses_existing_device_ra PROPERTIES
  ENVIRONMENT "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:kvaf_direct_hccp_fake_ra>:$ENV{LD_LIBRARY_PATH}")
