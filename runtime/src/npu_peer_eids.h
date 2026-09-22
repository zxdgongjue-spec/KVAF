#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "status.h"

namespace kvaf {
// Both NPU initiator and HBM provider use the current Runtime discovery rules.
Status DiscoverNpuPeerEids(std::uint32_t device_id,
                           const std::vector<std::string>& configured,
                           std::vector<std::array<std::uint8_t, 16>>* eids);
}
