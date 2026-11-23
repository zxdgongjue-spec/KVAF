#pragma once
#include "get/common/aicpu/include/batch_get_abi.h"
namespace kvaf::cpu_push {
std::uint32_t ExecuteCpuPushScatter(KernelArgs*);
std::uint32_t ExecuteCpuSglWrite(CpuSglWriteArgs*);
}  // namespace kvaf::cpu_push
