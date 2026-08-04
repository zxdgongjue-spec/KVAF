add_library(kvaf_runtime
  src/runtime.cc
)

target_include_directories(kvaf_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(kvaf_runtime
  PUBLIC
    kvaf_meta_manager
    kvaf_access_operator
)

target_compile_features(kvaf_runtime PUBLIC cxx_std_17)
