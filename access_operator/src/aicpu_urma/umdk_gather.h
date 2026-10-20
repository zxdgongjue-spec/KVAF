#pragma once
#include "batch_get_abi.h"
#include "gather_read.h"

namespace kvaf::aicpu_urma {
struct UmdkGatherMemory {
  std::uint64_t local_base, local_bytes, remote_base, remote_bytes;
};
struct UmdkTransfer {
  std::uint64_t remote, local, bytes;
};
struct UmdkBatchGetArgs {
  KvafExecuteBatchGetKernelArgs request;
  GatherReadArgs gather;
};
}  // namespace kvaf::aicpu_urma

extern "C" std::uint32_t KvafAicpuUmdkGatherMemory(const kvaf::aicpu_urma::GatherReadArgs*,
                                                   kvaf::aicpu_urma::UmdkGatherMemory*);
extern "C" std::uint32_t KvafAicpuUmdkGatherTransfer(const kvaf::aicpu_urma::GatherReadArgs*, bool,
                                                     const kvaf::aicpu_urma::UmdkTransfer*,
                                                     std::uint32_t, std::uint32_t);
extern "C" std::uint32_t KvafAicpuUmdkGatherScatter(const kvaf::aicpu_urma::GatherReadArgs*,
                                                    kvaf::aicpu_urma::GatherReadStatus*,
                                                    std::uint32_t, std::uint32_t, std::uint64_t,
                                                    std::uint32_t worker_index = 0);
