#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#if defined(KVAF_AICPU_PROFILE_MONOTONIC_RAW)
#include <time.h>
#endif

namespace kvaf {

// Host-only durations; nested intervals may overlap device execution.
struct GetProviderTiming {
  std::uint64_t prepare_ns{}, submit_ns{}, completion_ns{}, cleanup_ns{}, lock_ns{};
  std::uint64_t aiv_capture_prepare_ns{}, aiv_capture_submit_ns{};
  std::uint64_t aiv_status_reset_ns{}, aiv_kernel_launch_ns{};
  std::uint64_t capture_check_ns{}, status_enqueue_ns{}, stream_sync_ns{}, status_check_ns{};
  std::uint64_t prepare_capture_ns{}, previous_completion_ns{}, input_build_ns{};
  std::uint64_t profile_buffers_ns{}, registration_ns{};
  std::uint64_t launch_begin_ns{}, launch_end_ns{}, sync_begin_ns{}, sync_end_ns{};
};

// Device timestamps share a clock across workers, not with the Host clock.
struct alignas(64) BatchGetWorkerProfile {
  std::uint64_t enter_ns{}, exit_ns{};
  std::uint64_t request_ns{}, input_cache_ns{}, descriptor_cache_ns{};
  std::uint64_t source_write_ns{}, notify_ns{}, read_ns{}, ack_ns{};
  std::uint64_t read_setup_ns{}, wr_build_ns{}, ready_wait_ns{}, post_ns{}, cq_wait_ns{};
  std::uint64_t bytes{}, spans{}, read_wqes{}, post_calls{}, ready_queries{};
  std::uint64_t scatter_ns{}, output_cache_ns{};
  std::uint64_t scatter_copies{};
  std::uint64_t entry_ns{}, return_ns{};
};

// Caller-owned result, valid after a successful synchronous AICPU BatchGet.
struct BatchGetProfile {
  std::uint64_t prepare_ns{}, launch_ns{}, status_enqueue_ns{};
  std::uint64_t wait_ns{}, finish_ns{}, total_ns{};
  std::uint32_t worker_count{};
  std::array<BatchGetWorkerProfile, 8> workers{};
};

// Host-only trace for one Python submit/drain pair. Lower layers borrow it
// only until each call returns; execution resources never retain the pointer.
struct BatchGetHostTiming {
  std::uint64_t tensor_views_ns{}, request_validation_ns{}, stream_ns{}, refs_ns{};
  std::uint64_t stream_select_ns{}, stream_handle_ns{};
  std::uint64_t binding_lock_ns{}, set_device_ns{}, core_call_ns{}, owner_update_ns{}, total_ns{};
  std::uint64_t gil_reacquire_ns{};
  std::uint64_t core_lifecycle_ns{}, core_table_ns{}, core_direction_ns{}, core_previous_drain_ns{};
  std::uint64_t core_validate_ns{}, core_dispatch_ns{}, core_finish_ns{}, core_total_ns{};
  std::uint64_t begin_ns{}, end_ns{}, drain_begin_ns{}, drain_end_ns{};
  std::uint64_t drain_binding_lock_ns{}, drain_set_device_ns{}, drain_core_call_ns{};
  std::uint64_t drain_owner_update_ns{}, drain_total_ns{}, drain_gil_reacquire_ns{};
  std::uint64_t core_drain_lock_ns{}, core_drain_operator_ns{}, core_drain_release_ns{}, core_drain_total_ns{};
  std::uint64_t dispatch_lock_ns{}, dispatch_resolve_ns{}, dispatch_output_check_ns{};
  std::uint64_t dispatch_previous_drain_ns{}, dispatch_context_ns{}, dispatch_describe_ns{};
  std::uint64_t dispatch_acquire_ns{}, dispatch_bind_ns{}, dispatch_warmup_ns{}, dispatch_provider_ns{};
  std::uint64_t dispatch_context_release_ns{}, dispatch_total_ns{};
  std::uint64_t dispatch_drain_lock_ns{}, dispatch_drain_provider_ns{};
  std::uint64_t dispatch_drain_release_ns{}, dispatch_drain_total_ns{};
  GetProviderTiming provider;
};

inline std::uint64_t BatchGetProfileNowNs() {
#if defined(KVAF_AICPU_PROFILE_MONOTONIC_RAW)
  // Match dev_start.log's clock_monotonic_raw anchor, independently of HWTS.
  timespec timestamp{};
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) return 0;
  return static_cast<std::uint64_t>(timestamp.tv_sec) * UINT64_C(1000000000) + timestamp.tv_nsec;
#else
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
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
