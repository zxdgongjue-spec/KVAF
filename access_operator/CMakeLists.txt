add_library(kvaf_access_operator INTERFACE)

target_include_directories(kvaf_access_operator
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kvaf_access_operator INTERFACE kvaf_common)

target_compile_features(kvaf_access_operator INTERFACE cxx_std_17)

if((KVAF_ENABLE_AIV_BATCH_GET OR KVAF_ENABLE_AICPU_URMA) AND NOT ASCEND_CANN_PACKAGE_PATH)
  if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
  elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
  else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
  set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_CANN_PACKAGE_PATH}" CACHE PATH "CANN toolkit")
endif()
add_subdirectory(src/ubmem)

if(KVAF_ENABLE_AIV_BATCH_GET OR KVAF_ENABLE_AICPU_CPU_PUSH)
  add_subdirectory(src/direct_hccp)
endif()

if(KVAF_ENABLE_AIV_BATCH_GET)
  add_subdirectory(demo/aiv_batch_get)
  if(KVAF_ENABLE_AIV_BATCH_PUT)
    add_subdirectory(demo/aiv_batch_put)
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
      PRIVATE kvaf_execute_batch_put_host kvaf_direct_hccp_session)
    target_compile_definitions(kvaf_default_access_operator
      PRIVATE KVAF_ENABLE_AIV_BATCH_PUT=1)
  endif()

  target_link_directories(kvaf_default_access_operator
    PRIVATE
      ${ASCEND_CANN_PACKAGE_PATH}/lib64
  )

  target_compile_features(kvaf_default_access_operator PUBLIC cxx_std_17)
endif()

if(KVAF_ENABLE_AICPU_URMA)
  add_subdirectory(src/aicpu_urma)
  add_subdirectory(demo/aicpu_batch_get)
  add_subdirectory(demo/aicpu_batch_put)
  add_subdirectory(src/aicpu_urma/kernel)
  add_library(kvaf_aicpu_access_operator STATIC src/aicpu_access_operator.cc)
  if(KVAF_ENABLE_AICPU_CPU_PUSH)
    target_compile_definitions(kvaf_aicpu_access_operator PRIVATE KVAF_ENABLE_AICPU_CPU_PUSH=1)
  endif()
  target_link_libraries(kvaf_aicpu_access_operator PUBLIC kvaf_access_operator PRIVATE
    kvaf_aicpu_batch_get_host kvaf_aicpu_batch_put_host)
endif()
