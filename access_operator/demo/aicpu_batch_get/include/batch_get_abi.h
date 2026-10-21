#pragma once
#include <cstdint>

#include "../../aiv_batch_get/include/batch_get_abi.h"
#include "gather_workspace.h"
#include "umdk/kernel_abi.h"
namespace kvaf {
constexpr std::uint32_t kGatherMaxSglSize = 6U;
struct GatherOutputSpan {
  std::uint64_t offset;
  std::uint32_t length;
  std::uint32_t reserved{0U};
};
static_assert(sizeof(GatherOutputSpan) == 16U);
}  // namespace kvaf

namespace kvaf::aicpu_urma {
constexpr std::uint64_t kGatherStatusMagic = 0x4b56414647415448ULL;
constexpr std::uint32_t kGatherMaxSpans = kGatherMaxRequests;

struct alignas(64) GatherReadStatus {
  std::uint64_t magic, result, phase, ack_seen, cqe_word, sequence;
};
struct alignas(64) GatherQueueState {
  std::uint64_t next_worker{0U};
};
struct GatherReadArgs {
  UmdkContext context;
  std::uint64_t queue_state{0U}, core_status{0U};
  std::uint64_t output{0U}, output_extent{0U}, scratch{0U};
  std::uint64_t remote_staging{0U}, remote_descriptor{0U};
  std::uint64_t remote_command{0U}, remote_notify{0U}, remote_ack{0U};
  std::uint64_t output_spans{0U}, source_descriptor{0U}, sequence{0U};
  std::uint64_t remote_data_bytes{0U};
  std::uint32_t bytes{0U}, output_span_count{0U}, source_span_count{0U};
  std::uint32_t sgl_size{6U}, worker_count{1U}, sq_batch_wqes{32U};
  std::uint32_t worker_index{0U};
  std::uint32_t pipeline_chunk_bytes{kGatherChunkBytes};
};
// One stable registered HBM workspace, independent of caller output allocations.
constexpr std::uint64_t kGatherDescriptorOffset = kGatherMaxWorkers * 64U;
inline constexpr std::uint64_t GatherTargetsOffset(std::uint32_t workers) {
  return kGatherDescriptorOffset + workers * GatherLaneDescriptorBytes(workers);
}
inline constexpr std::uint64_t GatherWorkspaceBytes(std::uint32_t workers) {
  return GatherTargetsOffset(workers) + workers * GatherLaneTargetsBytes(workers);
}
static_assert(sizeof(GatherReadArgs) <= 256U);
struct UmdkBatchGetArgs {
  KvafExecuteBatchGetKernelArgs request;
  GatherReadArgs gather;
};
}  // namespace kvaf::aicpu_urma
