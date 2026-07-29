pybind11_add_module(_kvaf python_module.cpp)

target_link_libraries(_kvaf PRIVATE kvaf_runtime)

target_include_directories(_kvaf
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../meta_manager/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../access_operator/include
)

set_target_properties(_kvaf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/kvaf"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/kvaf"
)

add_custom_command(TARGET _kvaf POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/kvaf
    ${CMAKE_BINARY_DIR}/python/kvaf
)
