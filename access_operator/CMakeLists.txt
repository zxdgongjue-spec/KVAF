add_library(kvaf_access_operator STATIC
  src/aiv_simt_urma_h2d.cpp
)

target_include_directories(kvaf_access_operator
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_compile_features(kvaf_access_operator PUBLIC cxx_std_17)

if(BUILD_TESTING)
  add_executable(kvaf_aiv_simt_urma_h2d_test
    tests/aiv_simt_urma_h2d_test.cpp
  )
  target_link_libraries(kvaf_aiv_simt_urma_h2d_test
    PRIVATE kvaf_access_operator
  )
  add_test(
    NAME kvaf_aiv_simt_urma_h2d_test
    COMMAND kvaf_aiv_simt_urma_h2d_test
  )
endif()

if(KVAF_BUILD_A5_AIV_URMA)
  add_subdirectory(a5)
endif()
