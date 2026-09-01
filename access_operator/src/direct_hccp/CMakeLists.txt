add_library(kvaf_direct_hccp_session SHARED
  direct_hccp_session.cc
  direct_hccp_remote_transport.cc
)

target_include_directories(kvaf_direct_hccp_session
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/include"
    "${ASCEND_CANN_PACKAGE_PATH}/include"
)

target_compile_definitions(kvaf_direct_hccp_session PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(kvaf_direct_hccp_session PRIVATE -O2 -fno-common -fno-strict-aliasing)
target_link_directories(kvaf_direct_hccp_session PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(kvaf_direct_hccp_session PRIVATE kvaf_common ascendcl dl pthread)
target_compile_features(kvaf_direct_hccp_session PUBLIC cxx_std_17)
