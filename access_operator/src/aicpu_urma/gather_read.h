#pragma once

#include <cstdint>

#include "gather_output.h"
#include "gather_read_profile.h"

namespace kvaf::aicpu_urma {

constexpr std::uint64_t kGatherStatusMagic = 0x4b56414647415448ULL;
constexpr std::uint32_t kGatherScratchBytes = 64U;

struct GatherReadStatus {
  std::uint64_t magic;
  std::uint64_t result;
  std::uint64_t phase;
  std::uint64_t ack_seen;
  std::uint64_t sq_producer_index;
  std::uint64_t cq_consumer_index;
  std::uint64_t cqe_word;
  std::uint64_t sequence;
  std::uint64_t notify_ns;
  std::uint64_t source_write_ns;
  std::uint64_t source_write_post_ns;
  std::uint64_t source_write_cq_ns;
  std::uint64_t notify_post_ns;
  std::uint64_t notify_cq_ns;
  std::uint64_t ack_poll_ns;
  std::uint64_t ack_started_ns;
  std::uint64_t ack_post_ns;
  std::uint64_t ack_cq_ns;
  std::uint64_t ack_cache_ns;
  std::uint64_t scatter_cache_ns;
  std::uint64_t scatter_plan_ns;
  std::uint64_t scatter_post_ns;
  std::uint64_t scatter_cq_ns;
  std::uint64_t ack_polls;
  std::uint64_t read_wqes;
  kvaf::peer::GatherCpuProfile cpu;
};

struct GatherReadArgs {
  std::uint64_t channel_entity;
  std::uint64_t queue_state;
  std::uint64_t core_status;
  std::uint64_t output;
  std::uint64_t remote_staging;
  std::uint64_t remote_command;
  std::uint64_t remote_notify;
  std::uint64_t remote_ack;
  std::uint64_t output_spans;
  std::uint64_t source_descriptor;
  std::uint64_t remote_descriptor;
  std::uint64_t output_extent;
  std::uint64_t sequence;
  std::uint32_t bytes;
  std::uint32_t output_span_count;
  std::uint32_t sgl_size;
  std::uint32_t source_span_count;
};

static_assert(sizeof(GatherReadStatus) <= 384U, "GatherRead exceeds shared status buffer");
static_assert(sizeof(GatherReadArgs) == 120U, "GatherRead argument ABI changed");

}  // namespace kvaf::aicpu_urma
