add_library(kvaf_access_operator INTERFACE)

target_include_directories(kvaf_access_operator
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_compile_features(kvaf_access_operator INTERFACE cxx_std_17)

option(KVAF_ENABLE_AIV_BATCH_GET
       "Build the CANN-backed AIV ExecuteBatchGet AccessOperator adapter" OFF)

if(KVAF_ENABLE_AIV_BATCH_GET)
  add_subdirectory(demo/aiv_batch_get)

  add_library(kvaf_aiv_batch_get_operator
    src/aiv_batch_get_access_operator.cc
  )

  target_include_directories(kvaf_aiv_batch_get_operator
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/demo/aiv_batch_get/include
      ${ASCEND_CANN_PACKAGE_PATH}/include
  )

  target_compile_definitions(kvaf_aiv_batch_get_operator
    PUBLIC
      _GLIBCXX_USE_CXX11_ABI=0
  )

  target_link_libraries(kvaf_aiv_batch_get_operator
    PUBLIC
      kvaf_access_operator
    PRIVATE
      kvaf_execute_batch_get_host
      ascendcl
  )

  target_link_directories(kvaf_aiv_batch_get_operator
    PRIVATE
      ${ASCEND_CANN_PACKAGE_PATH}/lib64
  )

  target_compile_features(kvaf_aiv_batch_get_operator PUBLIC cxx_std_17)
endif()
