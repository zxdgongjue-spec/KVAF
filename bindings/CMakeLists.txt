pybind11_add_module(_kvaf src/python_module.cpp)

target_link_libraries(_kvaf PRIVATE kvaf_runtime)

target_include_directories(_kvaf
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../runtime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../meta_manager/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../access_operator/include
)

set(KVAF_PYTHON_BINDING_PREFLIGHT_ARGS)

if(KVAF_ENABLE_NPU)
  target_sources(_kvaf PRIVATE src/acl_python_utils.cc src/acl_graph_python.cc src/npu_python_runtime.cc)
  target_link_libraries(_kvaf PRIVATE kvaf_npu_runtime ascendcl)
  target_include_directories(_kvaf PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include")
  target_link_directories(_kvaf PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
  if(KVAF_ENABLE_AICPU_URMA)
    target_compile_definitions(_kvaf PRIVATE KVAF_ENABLE_AICPU_URMA=1)
  endif()
  if(KVAF_ENABLE_AIV_BATCH_GET)
    target_link_libraries(_kvaf PRIVATE kvaf_default_access_operator kvaf_execute_batch_get_host)
    target_compile_definitions(_kvaf PRIVATE KVAF_ENABLE_AIV_BATCH_GET=1)
    target_include_directories(_kvaf PRIVATE "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include")
    set_property(TARGET _kvaf APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:kvaf_execute_batch_get_host>")
    if(KVAF_ENABLE_AIV_BATCH_PUT)
      target_link_libraries(_kvaf PRIVATE kvaf_execute_batch_put_host)
      set_property(TARGET _kvaf APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:kvaf_execute_batch_put_host>")
    endif()
  endif()
  list(APPEND KVAF_PYTHON_BINDING_PREFLIGHT_ARGS --require-npu)
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
    "${Python_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/verify_python_binding.py"
    ${KVAF_PYTHON_BINDING_PREFLIGHT_ARGS}
)
