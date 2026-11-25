#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include "batch_get_profile_abi.h"

namespace kvaf {

// Caller-owned result, valid after a successful synchronous AICPU BatchGet.
struct BatchGetProfile {
  std::uint64_t prepare_ns{}, launch_ns{}, status_enqueue_ns{};
  std::uint64_t wait_ns{}, finish_ns{}, total_ns{};
  std::uint32_t worker_count{};
  std::array<BatchGetWorkerProfile, 8> workers{};
};

// Host-only trace for one Python BatchGet call. RuntimeCore borrows this only
// for the duration of BatchGet; execution resources never retain the pointer.
struct BatchGetHostTiming {
  std::uint64_t tensor_views_ns{}, request_validation_ns{}, stream_ns{}, refs_ns{};
  std::uint64_t binding_lock_ns{}, set_device_ns{}, core_call_ns{}, owner_update_ns{}, total_ns{};
  std::uint64_t core_lifecycle_ns{}, core_table_ns{}, core_direction_ns{}, core_previous_drain_ns{};
  std::uint64_t core_validate_ns{}, core_dispatch_ns{}, core_finish_ns{}, core_total_ns{};
};

inline std::uint64_t BatchGetProfileNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

// No clock reads when profiling is disabled. Nested durations are inclusive.
class BatchGetProfileScope {
 public:
  explicit BatchGetProfileScope(std::uint64_t* elapsed)
      : elapsed_(elapsed), start_(elapsed ? BatchGetProfileNowNs() : 0U) {}
  ~BatchGetProfileScope() {
    if (elapsed_) *elapsed_ += BatchGetProfileNowNs() - start_;
  }
 private:
  std::uint64_t* elapsed_;
  std::uint64_t start_;
};

}  // namespace kvaf
