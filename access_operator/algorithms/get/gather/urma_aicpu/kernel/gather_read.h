#pragma once
#include "get/common/aicpu/include/batch_get_abi.h"
namespace kvaf::aicpu_urma {
std::uint32_t ExecuteGatherRead(UmdkBatchGetArgs*, BatchGetWorkerProfile*, BatchGetWorkerProfile**);
}  // namespace kvaf::aicpu_urma
