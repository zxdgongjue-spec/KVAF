#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace kvaf::peer {

struct GatherSpan {
  std::uint64_t address;
  std::uint32_t length;
  std::uint32_t reserved;
};

static_assert(sizeof(GatherSpan) == 16U, "gather span ABI changed");

inline bool GatherRange(std::uint64_t base, std::uint64_t bytes,
                        std::uint64_t address, std::uint64_t length) {
  return length != 0U && address >= base && address - base <= bytes &&
         length <= bytes - (address - base);
}

inline std::uint64_t GatherControlOffset(std::uint64_t bytes) {
  return (bytes - 64U) & ~std::uint64_t{63U};
}

inline bool ValidateGather(std::uint64_t base, std::uint64_t bytes,
                           std::uint64_t staging, const std::vector<GatherSpan>& spans,
                           std::uint32_t* total_bytes) {
  if (total_bytes == nullptr || spans.empty() || bytes < 64U ||
      base > UINT64_MAX - bytes) return false;
  const auto data_bytes = GatherControlOffset(bytes);
  std::uint64_t total = 0U;
  for (const auto& span : spans) {
    if (span.reserved != 0U || !GatherRange(base, data_bytes, span.address, span.length) ||
        total > UINT32_MAX - span.length) return false;
    total += span.length;
  }
  if (!GatherRange(base, data_bytes, staging, total)) return false;
  for (const auto& span : spans) {
    if (span.address < staging + total && staging < span.address + span.length)
      return false;
  }
  *total_bytes = static_cast<std::uint32_t>(total);
  return true;
}

inline void GatherIntoStaging(void* memory, std::uint64_t base,
                              std::uint64_t staging, const std::vector<GatherSpan>& spans) {
  auto* destination = static_cast<std::uint8_t*>(memory) + (staging - base);
  for (const auto& span : spans) {
    std::memcpy(destination, static_cast<std::uint8_t*>(memory) + (span.address - base),
                span.length);
    destination += span.length;
  }
}

}  // namespace kvaf::peer
