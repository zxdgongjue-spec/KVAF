#pragma once

#include <memory>

#include "roce_options.h"
#include "runtime.h"

namespace kvaf {

// The allocator and operator share one TCP session and one RDMA connection.
std::shared_ptr<class RoceSession> CreateRoceSession(
    std::string address, RoceOptions options, std::uint32_t device_id);
std::unique_ptr<DdrMemoryAllocator> CreateRoceAllocator(std::shared_ptr<RoceSession> session);
std::unique_ptr<AccessOperator> CreateRoceOperator(std::shared_ptr<RoceSession> session);

}  // namespace kvaf
