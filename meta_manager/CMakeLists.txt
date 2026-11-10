add_library(kvaf_meta_manager
  src/in_memory_meta_manager.cc
)

target_include_directories(kvaf_meta_manager
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kvaf_meta_manager PUBLIC kvaf_common)

target_compile_features(kvaf_meta_manager PUBLIC cxx_std_17)
