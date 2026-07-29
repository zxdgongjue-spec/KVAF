add_library(kvaf_access_operator INTERFACE)

target_include_directories(kvaf_access_operator
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_compile_features(kvaf_access_operator INTERFACE cxx_std_17)
