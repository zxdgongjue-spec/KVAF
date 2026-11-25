#pragma once
#include "get/common/aicpu/include/batch_get_abi.h"
#include "cpu_sgl_push_abi.h"
namespace kvaf::cpu_push {
std::uint32_t ExecuteCpuPushScatter(KernelArgs*);
std::uint32_t ExecuteCpuSglWrite(CpuSglWriteArgs*);
std::uint32_t ExecuteCpuSglPush(CpuSglPushArgs*);
}  // namespace kvaf::cpu_push
