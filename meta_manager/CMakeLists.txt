add_library(kvaf_meta_manager INTERFACE)

target_include_directories(kvaf_meta_manager
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_compile_features(kvaf_meta_manager INTERFACE cxx_std_17)
