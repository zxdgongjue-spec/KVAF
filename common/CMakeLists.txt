add_library(kvaf_common INTERFACE)

target_include_directories(kvaf_common
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(kvaf_common INTERFACE cxx_std_17)
