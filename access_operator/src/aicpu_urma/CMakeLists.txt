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
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${kernel_dir}"
  COMMAND "${KVAF_AICPU_CXX}" -std=c++17 -O2 -shared -fPIC
          -I "${CMAKE_CURRENT_SOURCE_DIR}"
          -I "${CMAKE_SOURCE_DIR}/access_operator/include"
          -I "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_get/include"
          -I "${CMAKE_SOURCE_DIR}/access_operator/demo/aiv_batch_put/include"
          -I "${CMAKE_SOURCE_DIR}/runtime/include"
          "${CMAKE_CURRENT_SOURCE_DIR}/kernel.cc" -o "${kernel_so}"
  DEPENDS kernel.cc wqe.h gather_read.h
          "${CMAKE_SOURCE_DIR}/runtime/include/cpu_gather.h"
          "${CMAKE_SOURCE_DIR}/access_operator/include/gather_output.h"
          "${CMAKE_SOURCE_DIR}/access_operator/include/gather_read_profile.h" VERBATIM)
add_custom_command(OUTPUT "${kernel_tar}"
  COMMAND "${CMAKE_COMMAND}" -E tar czf "${kernel_tar}" aicpu_kernels_device
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "${kernel_so}" VERBATIM)
add_custom_target(kvaf_aicpu_urma_kernel DEPENDS "${kernel_tar}")
