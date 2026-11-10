add_library(kvaf_meta_manager
  src/in_memory_meta_manager.cc
)

target_include_directories(kvaf_meta_manager
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kvaf_meta_manager PUBLIC kvaf_common)

target_compile_features(kvaf_meta_manager PUBLIC cxx_std_17)

# Python Runtime tests are the default test surface. Keep the original C++
# MetaManager checks available for focused legacy debugging only.
option(KVAF_BUILD_LEGACY_CPP_TESTS "Build legacy C++ MetaManager tests" OFF)

if(BUILD_TESTING AND KVAF_BUILD_LEGACY_CPP_TESTS)
  add_executable(kvaf_meta_manager_test
    tests/in_memory_meta_manager_test.cc
  )

  target_link_libraries(kvaf_meta_manager_test
    PRIVATE
      kvaf_meta_manager
  )

  target_compile_features(kvaf_meta_manager_test PRIVATE cxx_std_17)

  add_test(NAME kvaf_meta_manager_test COMMAND kvaf_meta_manager_test)
endif()
