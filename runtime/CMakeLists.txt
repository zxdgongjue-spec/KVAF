add_library(kvaf_runtime INTERFACE)

target_include_directories(kvaf_runtime
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(kvaf_runtime
  INTERFACE
    kvaf_meta_manager
    kvaf_access_operator
)

target_compile_features(kvaf_runtime INTERFACE cxx_std_17)
