#pragma once

#include <cstddef>
#include <cstdint>

namespace kvaf {

struct GatherReadTiming {
  std::uint64_t validate_ns{0U};
  std::uint64_t configure_ns{0U};
  std::uint64_t source_upload_ns{0U};
  std::uint64_t descriptor_upload_ns{0U};
  std::uint64_t register_ns{0U};
  std::uint64_t launch_ns{0U};
  std::uint64_t sync_ns{0U};
  std::uint64_t release_ns{0U};
  std::uint64_t status_copy_ns{0U};
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
  std::uint64_t scatter_cache_ns{0U};
  std::uint64_t scatter_plan_ns{0U};
  std::uint64_t scatter_post_ns{0U};
  std::uint64_t scatter_cq_ns{0U};
  std::uint64_t cpu_dispatch_ns{0U};
  std::uint64_t cpu_notify_spin_ns{0U};
  std::uint64_t cpu_source_cache_ns{0U};
  std::uint64_t cpu_copy_ns{0U};
  std::uint64_t cpu_staging_flush_ns{0U};
  std::uint64_t ack_polls{0U};
  std::uint64_t read_wqes{0U};
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
