add_library(kvaf_meta_manager
  src/in_memory_meta_manager.cc
)

target_include_directories(kvaf_meta_manager
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_compile_features(kvaf_meta_manager PUBLIC cxx_std_17)

if(BUILD_TESTING)
  add_executable(kvaf_meta_manager_test
    tests/in_memory_meta_manager_test.cc
  )

  add_executable(kvaf_meta_manager_perf
    tests/in_memory_meta_manager_perf.cc
  )

  target_link_libraries(kvaf_meta_manager_test
    PRIVATE
      kvaf_meta_manager
  )

  target_link_libraries(kvaf_meta_manager_perf
    PRIVATE
      kvaf_meta_manager
  )

  target_compile_features(kvaf_meta_manager_test PRIVATE cxx_std_17)
  target_compile_features(kvaf_meta_manager_perf PRIVATE cxx_std_17)

  add_test(NAME kvaf_meta_manager_test COMMAND kvaf_meta_manager_test)
endif()
