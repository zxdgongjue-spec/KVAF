#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace kvaf {

struct GatherReadWorkerTiming {
  std::uint64_t start_offset_ns{0U};
  std::uint64_t end_offset_ns{0U};
  std::uint64_t duration_ns{0U};
  std::uint64_t read_wqes{0U};
  std::uint64_t provider_post_calls{0U};
  std::uint64_t read_tasks{0U};
  std::uint64_t build_ns{0U};
  std::uint64_t post_ns{0U};
  std::uint64_t cq_ns{0U};
  std::uint64_t thread_cpu_ns{0U};
  std::uint64_t thread_cpu_available{0U};
  std::int32_t start_cpu{-1};
  std::int32_t end_cpu{-1};
  std::uint64_t cq_poll_calls{0U};
  std::uint64_t read_bytes{0U};
};

struct GatherReadTiming {
  std::uint64_t layout_compute_ns{
      0U};  // Dimensions only; GatherRead Host input validation is disabled.
  std::uint64_t configure_ns{0U};
  bool metadata_upload_async{false};
  std::uint64_t metadata_pack_ns{0U};
  std::uint64_t metadata_enqueue_ns{0U};
  std::uint64_t register_ns{0U};
  std::uint64_t launch_ns{0U};
  std::uint64_t sync_ns{0U};
  bool status_readback_async{false};
  std::uint64_t status_enqueue_ns{0U};
  std::uint64_t status_decode_ns{0U};
  std::uint64_t notify_ns{0U};
  std::uint64_t source_write_ns{0U};
  std::uint64_t source_write_post_ns{0U};
  std::uint64_t source_write_cq_ns{0U};
  std::uint64_t notify_post_ns{0U};
  std::uint64_t notify_cq_ns{0U};
  std::uint64_t ack_poll_ns{0U};
  std::uint64_t ack_started_ns{0U};
  std::uint64_t ack_post_ns{0U};
  std::uint64_t ack_cq_ns{0U};
  std::uint64_t ack_cache_ns{0U};
  std::uint64_t scatter_plan_ns{0U};
  std::uint64_t scatter_post_ns{0U};
  std::uint64_t scatter_cq_ns{0U};
  std::uint64_t scatter_total_ns{0U};
  std::uint64_t output_layout_ns{0U};
  std::uint64_t scatter_start_skew_ns{0U};
  std::uint64_t scatter_worker_ns{0U};
  std::uint64_t aicpu_cores{1U};
  bool profile_detail{false};
  std::array<GatherReadWorkerTiming, 8> workers{};
  std::uint64_t cpu_dispatch_ns{0U};
  std::uint64_t cpu_notify_spin_ns{0U};
  std::uint64_t cpu_source_cache_ns{0U};
  std::uint64_t cpu_copy_ns{0U};
  std::uint64_t cpu_staging_flush_ns{0U};
  std::uint64_t ack_polls{0U};
  std::uint64_t read_wqes{0U};
  std::uint64_t provider_post_calls{0U};
  std::uint64_t read_tasks{0U};
  std::uint32_t pipeline_chunk_bytes{0U};
  std::uint64_t ready_wait_ns{0U};
};

namespace peer {

struct GatherCpuProfile {
  std::uint64_t dispatch_ns;
  std::uint64_t notify_spin_ns;
  std::uint64_t source_cache_ns;
  std::uint64_t copy_ns;
  std::uint64_t staging_flush_ns;
};

struct GatherControl {
  std::uint64_t notify;
  std::uint64_t ack;
  std::uint64_t command;
  GatherCpuProfile profile;
};

static_assert(sizeof(GatherControl) == 64U, "Gather control ABI changed");
static_assert(offsetof(GatherControl, ack) == 8U, "Gather ack offset changed");
static_assert(offsetof(GatherControl, command) == 16U, "Gather command offset changed");
static_assert(offsetof(GatherControl, profile) == 24U, "Gather profile offset changed");

}  // namespace peer
}  // namespace kvaf
