#pragma once
#include <cstddef>
#include "cpu_push_kernel_abi.h"

namespace kvaf::cpu_push {
// The shared prefix preserves Host completion/graph accounting. Each AIV core
// exclusively owns one epoch and builder line; no cross-core scalar atomics.
struct alignas(64) AivCoordination {
  Coordination shared;
  Word epochs[kMaxWorkers], builders[kMaxWorkers];
};
constexpr std::uint32_t kAivSglLocalBytes = 4096 * 8 + 256 * 4;
}  // namespace kvaf::cpu_push
