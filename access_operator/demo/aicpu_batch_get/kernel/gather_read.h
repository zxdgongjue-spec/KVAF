#pragma once
#include "aicpu_batch_get/include/batch_get_abi.h"
namespace kvaf::aicpu_urma {
std::uint32_t ExecuteGatherRead(UmdkBatchGetArgs*);
std::uint32_t ExecuteGatherScatter(UmdkBatchGetArgs*);
}  // namespace kvaf::aicpu_urma
