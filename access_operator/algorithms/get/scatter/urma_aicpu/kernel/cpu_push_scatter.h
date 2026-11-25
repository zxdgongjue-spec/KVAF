#pragma once
#include "get/common/aicpu/include/batch_get_abi.h"
namespace kvaf::cpu_push {
using MailboxTransfer = std::uint32_t (*)(const aicpu_urma::UmdkContext*, bool,
    const aicpu_urma::UmdkTransfer*, std::uint32_t, std::uint32_t);
std::uint32_t ExecuteCpuPushScatter(KernelArgs*, MailboxTransfer);
}  // namespace kvaf::cpu_push
