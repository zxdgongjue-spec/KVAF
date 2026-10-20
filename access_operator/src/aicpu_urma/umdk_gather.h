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
  std::uint64_t session, owner_pid;
};
static_assert(sizeof(UmdkTransfer) == 24U, "UMDK transfer ABI changed");
static_assert(sizeof(UmdkBatchGetArgs) == 128U, "UMDK BatchGet argument ABI changed");
}  // namespace kvaf::aicpu_urma

extern "C" std::uint32_t KvafAicpuUmdkGatherMemory(const kvaf::aicpu_urma::GatherReadArgs* args,
                                                   kvaf::aicpu_urma::UmdkGatherMemory* memory);

extern "C" std::uint32_t KvafAicpuUmdkGatherTransfer(
    const kvaf::aicpu_urma::GatherReadArgs* args, bool write,
    const kvaf::aicpu_urma::UmdkTransfer* transfers, std::uint32_t count, std::uint32_t worker,
    std::uint64_t* post_ns, std::uint64_t* cq_ns);

extern "C" std::uint32_t KvafAicpuUmdkGatherScatter(const kvaf::aicpu_urma::GatherReadArgs* args,
                                                    kvaf::aicpu_urma::GatherReadStatus* status,
                                                    std::uint32_t first, std::uint32_t last,
                                                    std::uint64_t remote_offset,
                                                    std::uint32_t worker_index = 0);
