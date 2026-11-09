add_library(kvaf_ubmem_hal STATIC hal.cc)
find_package(Threads REQUIRED)
target_include_directories(kvaf_ubmem_hal PUBLIC ${PROJECT_SOURCE_DIR}/access_operator/include)
target_link_libraries(kvaf_ubmem_hal PUBLIC kvaf_common PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

if(KVAF_ENABLE_AIV_BATCH_GET)
  include(${CMAKE_CURRENT_LIST_DIR}/npu.cmake)
endif()
