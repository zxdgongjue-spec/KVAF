add_library(kvaf_direct_hccp_session SHARED
  src/direct_hccp_session.cc
)

target_include_directories(kvaf_direct_hccp_session
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../aiv_batch_get/include"
)

target_compile_definitions(kvaf_direct_hccp_session PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(kvaf_direct_hccp_session PRIVATE -O2 -fno-common -fno-strict-aliasing)
target_link_libraries(kvaf_direct_hccp_session PRIVATE dl pthread)
target_compile_features(kvaf_direct_hccp_session PUBLIC cxx_std_17)
