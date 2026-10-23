#pragma once
#include "aicpu_urma/control_abi.h"
namespace kvaf::aicpu_urma {
std::uint32_t UmdkTransferSync(const UmdkContext*, bool write, const UmdkTransfer*,
                               std::uint32_t count, std::uint32_t worker);
}  // namespace kvaf::aicpu_urma
