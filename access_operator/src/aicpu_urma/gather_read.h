#pragma once
#include <cstdint>

#include "gather_output.h"
#include "gather_workspace.h"

namespace kvaf::aicpu_urma {
constexpr std::uint64_t kGatherStatusMagic = 0x4b56414647415448ULL;
constexpr std::uint32_t kGatherMaxSpans = kGatherMaxRequests;

struct alignas(64) GatherReadStatus {
  std::uint64_t magic, result, phase, ack_seen, cqe_word, sequence;
};
struct alignas(64) GatherQueueState {
  std::uint64_t next_worker{0U}, staging_offset{0U}, pipeline_sequence{0U};
};
struct GatherReadArgs {
  std::uint64_t umdk_session{0U}, umdk_owner_pid{0U};
  std::uint64_t queue_state{0U}, core_status{0U};
  std::uint64_t output{0U}, output_extent{0U}, scratch{0U};
  std::uint64_t remote_staging{0U}, remote_descriptor{0U};
  std::uint64_t remote_command{0U}, remote_notify{0U}, remote_ack{0U};
  std::uint64_t output_spans{0U}, source_descriptor{0U}, sequence{0U};
  std::uint32_t bytes{0U}, output_span_count{0U}, source_span_count{0U};
  std::uint32_t sgl_size{6U}, worker_count{1U}, sq_batch_wqes{32U};
  std::uint32_t pipeline_chunk_bytes{kGatherChunkBytes};
};
// One stable registered HBM workspace, independent of caller output allocations.
constexpr std::uint64_t kGatherArgsOffset = 8U * 64U;
constexpr std::uint64_t kGatherDescriptorOffset = kGatherArgsOffset + 256U;
constexpr std::uint64_t kGatherTargetsOffset =
    kGatherDescriptorOffset + 64U + kGatherMaxSpans * 16ULL;
constexpr std::uint64_t kGatherWorkspaceBytes =
    kGatherTargetsOffset + kGatherMaxSpans * sizeof(GatherOutputSpan);
static_assert(sizeof(GatherReadArgs) <= 256U);
}  // namespace kvaf::aicpu_urma
