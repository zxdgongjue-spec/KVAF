#pragma once

#include <cstdint>

#include "gather_output.h"
#include "gather_read_profile.h"
#include "gather_read_storage.h"

namespace kvaf::aicpu_urma {

constexpr std::uint64_t kGatherStatusMagic = 0x4b56414647415448ULL;
constexpr std::uint32_t kGatherScratchBytes = 64U;
constexpr std::uint32_t kGatherMaxSpans = 65536U;

struct GatherReadStatus {
  std::uint64_t magic;
  std::uint64_t result;
  std::uint64_t phase;
  std::uint64_t ack_seen;
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
  std::uint64_t scatter_plan_ns;
  std::uint64_t scatter_post_ns;
  std::uint64_t scatter_cq_ns;
  std::uint64_t ack_polls;
  std::uint64_t read_wqes;
  kvaf::peer::GatherCpuProfile cpu;
  std::uint64_t scatter_begin_ns;
  std::uint64_t scatter_end_ns;
  std::uint64_t output_layout_ns;
  std::uint64_t profile_detail;
  std::uint64_t provider_post_calls;
  std::uint64_t read_tasks;
  std::uint64_t ready_wait_ns;
  std::uint64_t thread_cpu_ns;
  std::uint64_t thread_cpu_available;
  std::int32_t start_cpu;
  std::int32_t end_cpu;
  std::uint64_t cq_poll_calls;
  std::uint64_t read_bytes;
};

struct alignas(64) GatherQueueState {
  std::uint64_t next_worker{0U};
  std::uint64_t staging_offset{0U};
  std::uint64_t pipeline_sequence{0U};
};

struct GatherReadArgs {
  std::uint64_t umdk_session{0U}, umdk_owner_pid{0U};
  std::uint64_t queue_state{0U}, core_status{0U};
  std::uint64_t output{0U}, remote_staging{0U};
  std::uint64_t remote_command{0U}, remote_notify{0U}, remote_ack{0U};
  std::uint64_t output_spans{0U}, source_descriptor{0U}, remote_descriptor{0U};
  std::uint64_t output_extent{0U}, sequence{0U};
  std::uint32_t bytes{0U}, output_span_count{0U}, sgl_size{6U}, source_span_count{0U};
  std::uint32_t worker_count{1U}, profile_detail{0U}, sq_batch_wqes{32U},
      pipeline_chunk_bytes{131072U};
};
static_assert(sizeof(GatherQueueState) == 64U);
static_assert(sizeof(GatherReadArgs) == 144U);
static_assert(sizeof(GatherReadStatus) <= 384U);
}  // namespace kvaf::aicpu_urma
