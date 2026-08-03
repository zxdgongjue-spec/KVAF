pybind11_add_module(_kvaf python_module.cpp)

target_link_libraries(_kvaf PRIVATE kvaf_runtime)

target_include_directories(_kvaf
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../meta_manager/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../access_operator/include
)

if(KVAF_ENABLE_AIV_BATCH_GET)
  target_sources(_kvaf PRIVATE
    acl_python_utils.cc
    acl_graph_python.cc
    npu_python_runtime.cc
  )
  target_link_libraries(_kvaf
    PRIVATE
      kvaf_default_access_operator
      kvaf_execute_batch_get_host
      ascendcl
  )
  target_include_directories(_kvaf
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/../access_operator/demo/aiv_batch_get/include
      ${ASCEND_CANN_PACKAGE_PATH}/include
  )
  target_link_directories(_kvaf PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
  target_compile_definitions(_kvaf
    PRIVATE
      KVAF_ENABLE_AIV_BATCH_GET=1
      _GLIBCXX_USE_CXX11_ABI=0
  )
  # _kvaf directly links the AIV host shared object. Keep its build-tree
  # location in the extension RPATH so PYTHONPATH=build/python is sufficient
  # for the direct Python integration test.
  set_property(TARGET _kvaf APPEND PROPERTY BUILD_RPATH
    "$<TARGET_FILE_DIR:kvaf_execute_batch_get_host>"
  )
endif()

set_target_properties(_kvaf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/kvaf"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/kvaf"
)

add_custom_command(TARGET _kvaf POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/kvaf
    ${CMAKE_BINARY_DIR}/python/kvaf
  # Fail the build at the Python dynamic-loader boundary, rather than later
  # obscuring a missing shared library as a test-script import error.
  COMMAND ${CMAKE_COMMAND} -E env
    "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
    "${Python_EXECUTABLE}" -c
    "import kvaf"
)
