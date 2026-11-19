#pragma once

#include <chrono>
#include <cstdint>

#include "requests.h"

namespace kvaf {
inline std::uint64_t GetHostTimingStart(const GetProviderTiming* timing) {
  if (!timing) return 0;
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline void GetHostTimingRecord(GetProviderTiming* timing,
                                std::uint64_t GetProviderTiming::*field,
                                std::uint64_t started) {
  if (timing) timing->*field += GetHostTimingStart(timing) - started;
}
}  // namespace kvaf
