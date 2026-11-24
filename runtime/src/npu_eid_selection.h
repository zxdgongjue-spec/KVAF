#pragma once

#include <istream>
#include <optional>
#include "npu_runtime_config.h"

namespace kvaf::detail {
struct PairEids {
  std::optional<NpuEid> cpu;
  std::optional<NpuEid> npu;
};
PairEids ParsePairInfo(std::istream& input, std::uint32_t physical_device);
bool ResolvePhysicalDeviceId(std::uint32_t device, std::uint32_t* physical);
PairEids ReadDevicePair(std::uint32_t physical_device);
Status SelectNpuEids(std::uint32_t physical_device, const std::vector<std::string>& configured,
                     const PairEids& pair, std::vector<NpuEid>* selected);
std::string EidToHex(const NpuEid& eid);
}  // namespace kvaf::detail
