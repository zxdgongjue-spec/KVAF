find_package(Threads REQUIRED)
add_library(kvaf_ubmem_hal STATIC hal.cc)
target_include_directories(kvaf_ubmem_hal PUBLIC ${PROJECT_SOURCE_DIR}/access_operator/include)
target_link_libraries(kvaf_ubmem_hal PUBLIC kvaf_common PRIVATE ${CMAKE_DL_LIBS})

add_library(kvaf_ubmem_cpu_runtime STATIC ${PROJECT_SOURCE_DIR}/runtime/src/ubmem_cpu_runtime.cc)
target_link_libraries(kvaf_ubmem_cpu_runtime PUBLIC kvaf_runtime kvaf_ubmem_hal kvaf_peer_link)

add_executable(kvaf_ubmem_cpu_service ${PROJECT_SOURCE_DIR}/runtime/src/ubmem_cpu_service_main.cc)
target_link_libraries(kvaf_ubmem_cpu_service PRIVATE kvaf_ubmem_cpu_runtime)
set_target_properties(kvaf_ubmem_cpu_service PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(KVAF_BUILD_UBMEM_NPU_TEST)
  include(${CMAKE_CURRENT_SOURCE_DIR}/npu.cmake)
endif()

if(BUILD_TESTING)
  add_library(kvaf_fake_hal SHARED ${PROJECT_SOURCE_DIR}/tests/ubmem/fake_hal.cc)
  target_link_libraries(kvaf_fake_hal PRIVATE kvaf_ubmem_hal)
  add_executable(kvaf_ubmem_control_test ${PROJECT_SOURCE_DIR}/tests/ubmem/control_test.cc
    ${PROJECT_SOURCE_DIR}/tests/ubmem/fake_acl.cc device_mapping.cc)
  target_include_directories(kvaf_ubmem_control_test PRIVATE ${PROJECT_SOURCE_DIR}/tests/ubmem/model)
  target_link_libraries(kvaf_ubmem_control_test PRIVATE kvaf_ubmem_cpu_runtime kvaf_fake_hal Threads::Threads ${CMAKE_DL_LIBS})
  add_test(NAME kvaf_ubmem_control_test COMMAND kvaf_ubmem_control_test $<TARGET_FILE:kvaf_fake_hal>)
  set_tests_properties(kvaf_ubmem_control_test PROPERTIES TIMEOUT 20)
  add_executable(kvaf_ubmem_kernel_test ${PROJECT_SOURCE_DIR}/tests/ubmem/kernel_test.cc)
  target_include_directories(kvaf_ubmem_kernel_test PRIVATE
    ${PROJECT_SOURCE_DIR}/tests/ubmem/model ${PROJECT_SOURCE_DIR}/access_operator/include)
  add_test(NAME kvaf_ubmem_kernel_test COMMAND kvaf_ubmem_kernel_test)
  add_test(NAME kvaf_ubmem_simt_kernel_test COMMAND kvaf_ubmem_kernel_test simt)
endif()
