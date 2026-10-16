add_executable(kvaf_aicpu_get_protocol_test protocol_test.cc)
target_include_directories(kvaf_aicpu_get_protocol_test PRIVATE
  "${CMAKE_SOURCE_DIR}/access_operator/include"
  "${CMAKE_SOURCE_DIR}/runtime/include"
  "${CMAKE_SOURCE_DIR}/common/include")
add_test(NAME kvaf_aicpu_get_protocol COMMAND kvaf_aicpu_get_protocol_test)
if(KVAF_BUILD_REMOTE_CPU_SERVICE)
  add_executable(kvaf_aicpu_get_cpu cpu_main.cc)
  target_link_libraries(kvaf_aicpu_get_cpu PRIVATE kvaf_cpu_urma_transport)
  # Preserve unoptimized URMA initialization ABI; optimize the caller only.
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(kvaf_aicpu_get_cpu PRIVATE -O2)
  endif()
endif()

if(KVAF_BUILD_AICPU_GET_TEST)
  if(NOT EXISTS "${ASCEND_CANN_PACKAGE_PATH}/include/acl/acl.h")
    message(FATAL_ERROR "AICPU test needs ASCEND_CANN_PACKAGE_PATH (recent CANN with binary-load APIs)")
  endif()
  find_path(KVAF_AICPU_TEST_URMA_INCLUDE_DIR urma_api.h
    HINTS "${KVAF_URMA_ROOT}/include" /usr/include /usr/local/include)
  if(NOT KVAF_AICPU_TEST_URMA_INCLUDE_DIR)
    message(FATAL_ERROR "AICPU test needs urma_api.h; set KVAF_URMA_ROOT")
  endif()
  add_executable(kvaf_aicpu_get_npu npu_main.cc)
  target_include_directories(kvaf_aicpu_get_npu PRIVATE
    "${CMAKE_SOURCE_DIR}/runtime/src"
    "${CMAKE_SOURCE_DIR}/access_operator/include"
    "${ASCEND_CANN_PACKAGE_PATH}/include"
    "${KVAF_AICPU_TEST_URMA_INCLUDE_DIR}")
  target_link_directories(kvaf_aicpu_get_npu PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
  target_link_libraries(kvaf_aicpu_get_npu PRIVATE
    kvaf_peer_link kvaf_direct_hccp_session ascendcl ${CMAKE_DL_LIBS} pthread)

  set(KVAF_AICPU_CXX "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++"
      CACHE FILEPATH "CANN device AICPU cross compiler")
  if(NOT EXISTS "${KVAF_AICPU_CXX}")
    message(FATAL_ERROR "Set KVAF_AICPU_CXX to the CANN AICPU cross compiler")
  endif()
  # Match hbm-sgl-probe: use the CANN device securec performance addon, not a
  # host securec library or an implementation silently replaced by std::memcpy.
  find_path(KVAF_AICPU_SECUREC_INCLUDE_DIR securec.h
    PATHS "${ASCEND_CANN_PACKAGE_PATH}/include"
          "${ASCEND_CANN_PACKAGE_PATH}/include/experiment"
          "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/include"
    NO_DEFAULT_PATH)
  find_file(KVAF_AICPU_SECUREC_LIBRARY NAMES libc_sec.so libsecurec.so
    PATHS "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/devlib"
          "${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/lib64"
          "${ASCEND_CANN_PACKAGE_PATH}/lib64/device"
          "${ASCEND_CANN_PACKAGE_PATH}/lib64"
    NO_DEFAULT_PATH)
  if(NOT KVAF_AICPU_SECUREC_INCLUDE_DIR OR NOT KVAF_AICPU_SECUREC_LIBRARY)
    message(FATAL_ERROR "AICPU securec_memcpy needs securec.h and the device AArch64 securec library. Set KVAF_AICPU_SECUREC_INCLUDE_DIR and KVAF_AICPU_SECUREC_LIBRARY.")
  endif()
  file(READ "${KVAF_AICPU_SECUREC_LIBRARY}" securec_elf_header OFFSET 0 LIMIT 20 HEX)
  if(NOT securec_elf_header MATCHES "^7f454c46" OR
     NOT securec_elf_header MATCHES "(b700|00b7)$")
    message(FATAL_ERROR "KVAF_AICPU_SECUREC_LIBRARY must be an AArch64 ELF library, not a host library")
  endif()
  get_filename_component(securec_library_name "${KVAF_AICPU_SECUREC_LIBRARY}" NAME)
  set(kernel_so "${CMAKE_CURRENT_BINARY_DIR}/aicpu_kernels_device/libkvaf_aicpu_test.so")
  set(kernel_tar "${CMAKE_CURRENT_BINARY_DIR}/kvaf-aicpu-test.tar.gz")
  configure_file(libkvaf_aicpu_test.json libkvaf_aicpu_test.json COPYONLY)
  add_custom_command(OUTPUT "${kernel_so}"
    BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/aicpu_kernels_device/${securec_library_name}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/aicpu_kernels_device"
    COMMAND "${KVAF_AICPU_CXX}" -std=c++17 -O3 -shared -fPIC
            -I "${CMAKE_CURRENT_SOURCE_DIR}" -I "${KVAF_AICPU_SECUREC_INCLUDE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cc" "${KVAF_AICPU_SECUREC_LIBRARY}"
            "-Wl,-rpath,$ORIGIN" -Wl,-z,defs -o "${kernel_so}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${KVAF_AICPU_SECUREC_LIBRARY}"
            "${CMAKE_CURRENT_BINARY_DIR}/aicpu_kernels_device/${securec_library_name}"
    DEPENDS kernel.cc pipeline.h scatter_copy.h output_verify.h abi.h
            "${KVAF_AICPU_SECUREC_LIBRARY}" "${KVAF_AICPU_SECUREC_INCLUDE_DIR}/securec.h" VERBATIM)

  add_custom_command(OUTPUT "${kernel_tar}"
    COMMAND ${CMAKE_COMMAND} -E tar czf "${kernel_tar}" aicpu_kernels_device
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${kernel_so}" VERBATIM)
  add_custom_target(kvaf_aicpu_get_kernel ALL DEPENDS "${kernel_tar}")
  add_dependencies(kvaf_aicpu_get_npu kvaf_aicpu_get_kernel)
endif()
