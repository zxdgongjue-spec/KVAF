#pragma once

#include <cstddef>

#include "cpu_runtime.h"

namespace kvaf {
namespace detail {

enum class CpuMemoryAdvice { kNone, kHugePage, kNoHugePage };

struct CpuMemoryPagePolicy {
  std::size_t alignment;
  CpuMemoryAdvice advice;
  bool pre_touch;
};

// Both explicit huge-page policies touch the full allocation with the same
// alignment. Only the advice differs; origin retains its original behavior.
constexpr CpuMemoryPagePolicy MemoryPagePolicy(CpuMemoryPageMode mode) {
  switch (mode) {
    case CpuMemoryPageMode::kTransparentHugePage:
      return {2U * 1024U * 1024U, CpuMemoryAdvice::kHugePage, true};
    case CpuMemoryPageMode::kNoHugePage:
      return {2U * 1024U * 1024U, CpuMemoryAdvice::kNoHugePage, true};
    case CpuMemoryPageMode::kOrigin:
      return {4096U, CpuMemoryAdvice::kNone, false};
  }
  return {4096U, CpuMemoryAdvice::kNone, false};
}

}  // namespace detail
}  // namespace kvaf
