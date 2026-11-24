#pragma once
#include <cstdint>

#include "gather_workspace.h"
#include "batch_get_profile.h"
#include "batch_get_ticket.h"
#include "aicpu_urma/control_abi.h"
#include "cpu_push_get_layout.h"
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
  std::uint64_t completed_requests, failed_request;
};
struct alignas(64) GatherExecutionControl {
  std::uint64_t claimed{}, failed{}, reserved[6]{};
  alignas(64) std::uint64_t window_sequence[kGatherMaxWorkers]{};
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
  std::uint32_t window_capacity{kDefaultGetWindow};
  std::uint64_t profile{0U};
  std::uint64_t execution_control{0U};
  std::uint32_t read_wave_wrs{kGatherDefaultReadWaveWrs};
};
// One stable registered HBM workspace, independent of caller output allocations.
constexpr std::uint64_t kGatherDescriptorOffset = kGatherMaxWorkers * 64U;
inline constexpr std::uint64_t GatherWorkspaceBytes(
    std::uint32_t workers, std::uint32_t capacity = kDefaultGetWindow) {
  return kGatherDescriptorOffset + workers * GatherLaneDescriptorBytes(workers, capacity);
}
static_assert(sizeof(GatherReadArgs) <= 256U);
static_assert(sizeof(GatherReadArgs) == 160U, "Gather V5 argument ABI changed");
struct AicpuBatchGetInput {
  // GatherRead with indices == 0 reads address_table[i] as a complete source
  // address; offsets is unused. Other callers use address_table[indices[i]] + offsets[i].
  // Caller guarantees valid indices, offsets, addresses and output rows, as in
  // the original kernels; device execution does not scan/validate each entry.
  std::uint64_t address_table{}, indices{}, offsets{}, miss_count{}, output_indices{};
  std::uint32_t table_capacity{}, request_capacity{}, token_bytes{};
  std::uint64_t valid_table{}, canonical_base{}, logical_bytes{}, block_bytes{};
};
struct UmdkBatchGetArgs {
  AicpuBatchGetInput request;
  GatherReadArgs gather;
};
}  // namespace kvaf::aicpu_urma

namespace kvaf::cpu_push {
struct KernelArgs {
  aicpu_urma::AicpuBatchGetInput request;
  std::uint64_t workspace{}, output{}, output_bytes{}, coordination{}, status{}, profile{};
  std::uint64_t sequence{};  // Assigned in each instance's local copy after claiming a ticket.
  std::uint32_t workers{}, timeout_ms{kTimeoutMs};
  std::uint32_t window_capacity{kDefaultGetWindow};
  std::uint64_t window_base{};
};
}  // namespace kvaf::cpu_push
