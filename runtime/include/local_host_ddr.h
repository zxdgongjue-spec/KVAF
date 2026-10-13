// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include "runtime.h"

namespace kvaf {
struct LocalHostDdrStats {
  std::uint64_t anon_huge_bytes{0};
  std::uint64_t smaps_covered_bytes{0};
};
std::unique_ptr<DdrMemoryAllocator> CreateLocalHostDdrAllocator(const std::string& mode);
LocalHostDdrStats LocalHostDdrPageStats(const DdrMemoryAllocator& allocator);
}  // namespace kvaf
