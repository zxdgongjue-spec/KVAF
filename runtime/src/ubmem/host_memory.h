#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>

#include "status.h"

namespace kvaf::ubmem_host {

constexpr std::size_t kThpAlignment = 2U * 1024U * 1024U;

// Allocation policy, not a guarantee about physical pages or NPU page tables.
bool ThpSupported();
Status AllocateThp(void** address, std::size_t bytes);
Status AdviseThp(void* address, std::size_t bytes);
Status BindNuma(void* address, std::size_t bytes, int node);
void Prefault(void* address, std::size_t bytes);
void Free(void* address);

struct ThpCoverage {
  bool available{false};
  bool exact{false};
  std::uint64_t measured_bytes{0};
  std::uint64_t anon_huge_bytes{0};
};

// Count only complete VMAs inside the arena. A partially overlapping VMA's
// aggregate AnonHugePages cannot be attributed to this allocation.
ThpCoverage ParseThpCoverage(std::istream& input, std::uintptr_t address,
                            std::size_t bytes);
ThpCoverage ReadThpCoverage(void* address, std::size_t bytes);

}  // namespace kvaf::ubmem_host
