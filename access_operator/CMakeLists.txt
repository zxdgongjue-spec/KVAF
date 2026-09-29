add_library(kvaf_access_operator INTERFACE)

target_include_directories(kvaf_access_operator
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kvaf_access_operator INTERFACE kvaf_common)

target_compile_features(kvaf_access_operator INTERFACE cxx_std_17)

if(KVAF_BUILD_REMOTE_CPU_SERVICE)
  add_subdirectory(src/cpu_urma)
endif()

if(KVAF_ENABLE_AIV_BATCH_GET OR KVAF_BUILD_HBM_RW_TEST OR KVAF_BUILD_AICPU_GET_TEST)
  add_subdirectory(src/direct_hccp)
endif()

if(KVAF_ENABLE_AIV_BATCH_GET)
  add_subdirectory(demo/aiv_batch_get)
  if(KVAF_ENABLE_AIV_BATCH_PUT)
    add_subdirectory(demo/aiv_batch_put)
  endif()
  if(KVAF_ENABLE_AICPU_URMA)
    add_subdirectory(src/aicpu_urma)
  endif()

  add_library(kvaf_default_access_operator
    src/access_operator.cc
  )

  target_include_directories(kvaf_default_access_operator
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/demo/aiv_batch_get/include
      ${ASCEND_CANN_PACKAGE_PATH}/include
  )

  if(KVAF_ENABLE_AIV_BATCH_PUT)
    target_include_directories(kvaf_default_access_operator
      PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/demo/aiv_batch_put/include)
  endif()

  if(KVAF_ENABLE_AICPU_URMA)
    target_link_libraries(kvaf_default_access_operator PRIVATE kvaf_aicpu_urma_host)
    target_compile_definitions(kvaf_default_access_operator PRIVATE KVAF_ENABLE_AICPU_URMA=1)
    target_include_directories(kvaf_default_access_operator PRIVATE
      src/aicpu_urma "${CMAKE_SOURCE_DIR}/runtime/include")
    target_include_directories(kvaf_execute_batch_put_host PRIVATE src/aicpu_urma)
    target_link_libraries(kvaf_execute_batch_put_host PRIVATE kvaf_aicpu_urma_host)
    target_compile_definitions(kvaf_execute_batch_put_host PRIVATE KVAF_ENABLE_AICPU_URMA=1)
  endif()

  target_compile_definitions(kvaf_default_access_operator
    PUBLIC
      _GLIBCXX_USE_CXX11_ABI=0
  )

  target_link_libraries(kvaf_default_access_operator
    PUBLIC
      kvaf_access_operator
    PRIVATE
      kvaf_meta_manager
      kvaf_execute_batch_get_host
      ascendcl
  )

  if(KVAF_ENABLE_AIV_BATCH_PUT)
    target_link_libraries(kvaf_default_access_operator
      PRIVATE kvaf_execute_batch_put_host)
    target_compile_definitions(kvaf_default_access_operator
      PRIVATE KVAF_ENABLE_AIV_BATCH_PUT=1)
  endif()

  target_link_directories(kvaf_default_access_operator
    PRIVATE
      ${ASCEND_CANN_PACKAGE_PATH}/lib64
  )

  target_compile_features(kvaf_default_access_operator PUBLIC cxx_std_17)
endif()
