#pragma once
#include <cstdint>

#include "gather_workspace.h"
#include "batch_get_profile.h"
#include "batch_get_ticket.h"
#include "aicpu_urma/control_abi.h"
#include "cpu_push_get_layout.h"
#include "cpu_push_kernel_abi.h"
namespace kvaf {
constexpr std::uint32_t kGatherMaxSglSize = 6U;
}  // namespace kvaf

namespace kvaf::aicpu_urma {
constexpr std::uint64_t kGatherStatusMagic = 0x4b56414647415448ULL;
constexpr std::uint32_t kGatherMaxSpans = kGatherMaxRequests;

enum GatherReadPhase : std::uint64_t {
  kWriteSourceList = 1,
  kNotifyCpu = 2,
  kWaitCpuAck = 3,
  kReadStaging = 4,
  kWaitReadyPrefix = 5,
};

enum GatherReadError : std::uint64_t {
  kInvalidGatherRequest = 2,
  kGatherTimeout = 9,
  kCpuGatherFailed = 10,
  kGatherTransportFailed = 16,
  kGatherSequenceExhausted = 17,
};

struct alignas(64) GatherReadStatus {
  std::uint64_t magic, result, phase, ack_seen, cqe_word, sequence;
};
struct alignas(64) GatherExecutionControl {
  std::uint64_t claimed{}, failed{}, reserved[6]{};
};
struct GatherReadArgs {
  UmdkContext context;
  std::uint64_t core_status{0U};
  std::uint64_t output{0U}, output_extent{0U}, scratch{0U};
  std::uint64_t remote_staging{0U}, remote_descriptor{0U};
  std::uint64_t remote_command{0U}, remote_notify{0U}, remote_ack{0U};
  // sequence is assigned in the worker's local copy, never by the Host.
  std::uint64_t source_descriptor{0U}, sequence{0U};
  std::uint32_t bytes{0U}, source_span_count{0U};
  std::uint32_t sgl_size{6U}, worker_count{1U}, sq_batch_wqes{32U};
  std::uint32_t worker_index{0U};
  std::uint32_t pipeline_chunk_bytes{kGatherChunkBytes};
  std::uint64_t profile{0U};
  std::uint64_t execution_control{0U};
};
// One stable registered HBM workspace, independent of caller output allocations.
constexpr std::uint64_t kGatherDescriptorOffset = kGatherMaxWorkers * 64U;
inline constexpr std::uint64_t GatherWorkspaceBytes(std::uint32_t workers) {
  return kGatherDescriptorOffset + workers * GatherLaneDescriptorBytes(workers);
}
static_assert(sizeof(GatherReadArgs) <= 256U);
using AicpuBatchGetInput = cpu_push::BatchGetInput;
struct UmdkBatchGetArgs {
  AicpuBatchGetInput request;
  GatherReadArgs gather;
};
}  // namespace kvaf::aicpu_urma
