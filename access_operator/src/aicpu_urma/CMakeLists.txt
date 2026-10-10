if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
  if(DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
  elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
  else()
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
  endif()
endif()
set(KVAF_AICPU_CXX "${ASCEND_CANN_PACKAGE_PATH}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++"
    CACHE FILEPATH "AICPU aarch64 cross compiler")
if(NOT EXISTS "${KVAF_AICPU_CXX}")
  message(FATAL_ERROR "AICPU URMA requires the CANN aarch64 cross compiler: ${KVAF_AICPU_CXX}")
endif()

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
  message(FATAL_ERROR "AICPU memcpy_sp requires securec.h and the device AArch64 securec shared library. Set KVAF_AICPU_SECUREC_INCLUDE_DIR and KVAF_AICPU_SECUREC_LIBRARY if needed.")
endif()
file(READ "${KVAF_AICPU_SECUREC_LIBRARY}" securec_elf_header OFFSET 0 LIMIT 20 HEX)
if(NOT securec_elf_header MATCHES "^7f454c46" OR
   NOT securec_elf_header MATCHES "(b700|00b7)$")
  message(FATAL_ERROR "KVAF_AICPU_SECUREC_LIBRARY must be an AArch64 ELF library, not a host library: ${KVAF_AICPU_SECUREC_LIBRARY}")
endif()
get_filename_component(securec_library_name "${KVAF_AICPU_SECUREC_LIBRARY}" NAME)

add_library(kvaf_aicpu_urma_host STATIC host.cc)
target_include_directories(kvaf_aicpu_urma_host PUBLIC
  "${CMAKE_SOURCE_DIR}/access_operator/include"
  "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include"
  "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_put/include"
  "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_directories(kvaf_aicpu_urma_host PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_link_libraries(kvaf_aicpu_urma_host PRIVATE ascendcl)

set(kernel_dir "${CMAKE_CURRENT_BINARY_DIR}/aicpu_kernels_device")
set(kernel_so "${kernel_dir}/libkvaf_aicpu_urma.so")
set(kernel_tar "${CMAKE_CURRENT_BINARY_DIR}/kvaf-aicpu-urma.tar.gz")
configure_file(kernel.json kernel.json COPYONLY)
add_custom_command(OUTPUT "${kernel_so}"
  BYPRODUCTS "${kernel_dir}/${securec_library_name}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${kernel_dir}"
  COMMAND "${KVAF_AICPU_CXX}" -std=c++17 -O2 -shared -fPIC
          -I "${CMAKE_CURRENT_SOURCE_DIR}"
          -I "${CMAKE_SOURCE_DIR}/access_operator/include"
          -I "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include"
          -I "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_put/include"
          -I "${CMAKE_SOURCE_DIR}/runtime/include"
          -I "${KVAF_AICPU_SECUREC_INCLUDE_DIR}"
          "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cc"
          "${CMAKE_CURRENT_SOURCE_DIR}/sq_copy_benchmark.cc"
          "${KVAF_AICPU_SECUREC_LIBRARY}" "-Wl,-rpath,$ORIGIN" -Wl,-z,defs -o "${kernel_so}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${KVAF_AICPU_SECUREC_LIBRARY}" "${kernel_dir}/${securec_library_name}"
  DEPENDS kernel.cc wqe.h sq_copy.h gather_read.h sq_copy_benchmark.cc sq_copy_benchmark.h
          "${KVAF_AICPU_SECUREC_LIBRARY}"
          "${KVAF_AICPU_SECUREC_INCLUDE_DIR}/securec.h"
          "${CMAKE_SOURCE_DIR}/runtime/include/cpu_gather.h"
          "${CMAKE_SOURCE_DIR}/access_operator/include/gather_output.h"
          "${CMAKE_SOURCE_DIR}/access_operator/include/gather_read_storage.h"
          "${CMAKE_SOURCE_DIR}/access_operator/include/gather_read_profile.h" VERBATIM)
add_custom_command(OUTPUT "${kernel_tar}"
  COMMAND "${CMAKE_COMMAND}" -E tar czf "${kernel_tar}" aicpu_kernels_device
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "${kernel_so}" VERBATIM)
add_custom_target(kvaf_aicpu_urma_kernel DEPENDS "${kernel_tar}")
