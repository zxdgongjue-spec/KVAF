#pragma once

#include <chrono>
#include <cstdint>
#include <array>
#include <optional>
#include <vector>

#include "get_provider_timing.h"

namespace kvaf {

// Device timestamps share a clock across workers, not with the Host clock.
struct alignas(64) BatchGetWorkerProfile {
  std::uint64_t enter_ns{}, exit_ns{};
  std::uint64_t request_ns{}, input_cache_ns{}, descriptor_cache_ns{};
  std::uint64_t source_write_ns{}, notify_ns{}, read_ns{}, ack_ns{};
  std::uint64_t read_setup_ns{}, wr_build_ns{}, ready_wait_ns{}, post_ns{}, cq_wait_ns{};
  std::uint64_t bytes{}, spans{}, read_wqes{}, post_calls{}, ready_queries{};
  std::uint64_t scatter_ns{}, output_cache_ns{};
  std::uint64_t scatter_copies{}, first_post_wrs{};
  std::uint32_t first_chunk_bytes{}, tail_chunk_bytes{};
};

static_assert(sizeof(BatchGetWorkerProfile) == 192, "device worker profile stride changed");

struct GatherReadyObservation {
  std::uint32_t first_ready_bytes{}, ready_repeat_queries{};
};

struct GatherPullObservation {
  std::uint32_t ready_advances{}, planned_groups{};
};

// Caller-owned result, valid after a successful synchronous AICPU BatchGet.
struct BatchGetProfile {
  std::uint64_t prepare_ns{}, launch_ns{}, status_enqueue_ns{};
  std::uint64_t wait_ns{}, finish_ns{}, total_ns{};
  std::uint32_t worker_count{};
  std::vector<BatchGetWorkerProfile> workers;
  // Only Gather uses these diagnostics, with at most eight workers.
  std::array<std::optional<GatherReadyObservation>, 8> gather_ready_progress{};
  std::array<std::optional<GatherPullObservation>, 8> gather_pull_progress{};
};

// Host-only trace for one Python BatchGet call. RuntimeCore borrows this only
// for the duration of BatchGet; execution resources never retain the pointer.
struct BatchGetHostTiming {
  std::uint64_t tensor_views_ns{}, request_validation_ns{}, stream_ns{}, refs_ns{};
  std::uint64_t binding_lock_ns{}, set_device_ns{}, core_call_ns{}, owner_update_ns{}, total_ns{};
  std::uint64_t core_lifecycle_ns{}, core_table_ns{}, core_direction_ns{}, core_previous_drain_ns{};
  std::uint64_t core_validate_ns{}, core_dispatch_ns{}, core_finish_ns{}, core_total_ns{};
  std::uint64_t stream_sync_ns{};
  GetProviderTiming provider;
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
