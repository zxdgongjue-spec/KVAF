#pragma once
#include "get/common/aicpu/include/batch_get_abi.h"
namespace kvaf::cpu_push {
std::uint32_t ExecuteCpuPushScatter(KernelArgs*, BatchGetWorkerProfile*, BatchGetWorkerProfile**);
}  // namespace kvaf::cpu_push
