#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "gather_workspace.h"
#include "gather_pull_plan.h"
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
  std::uint64_t ready_profile_stamp;
  std::uint32_t ready_profile_first, ready_profile_second;
  std::uint64_t completed_requests, failed_request;
};
static_assert(sizeof(GatherReadStatus) == 128, "Gather worker status stride changed");
// Tag the paired profile timestamp. Previous status formats stored the untagged timestamp here.
constexpr std::uint64_t kGatherReadyProgressCookie = 0xa7b18d3252504731ULL;
inline std::optional<GatherReadyObservation> GatherReadyProgress(const GatherReadStatus& status,
                                                                 const BatchGetWorkerProfile& sample) {
  if (status.magic != kGatherStatusMagic || status.result || !sample.enter_ns || !sample.ready_queries ||
      status.ready_profile_stamp != (sample.enter_ns ^ kGatherReadyProgressCookie)) return std::nullopt;
  return GatherReadyObservation{status.ready_profile_first, status.ready_profile_second};
}
constexpr std::uint64_t kGatherPullProfileCookie = 0xb3a3d0ba50554c31ULL;
inline void StampGatherPullPlan(GatherReadStatus& status, const BatchGetWorkerProfile& sample,
                               std::uint32_t advances, std::uint32_t groups) {
  status.ready_profile_first = advances;
  status.ready_profile_second = groups;
  status.ready_profile_stamp = sample.enter_ns ^ kGatherPullProfileCookie;
}
inline std::optional<GatherPullObservation> GatherPullProgress(const GatherReadStatus& status,
                                                              const BatchGetWorkerProfile& sample) {
  if (status.magic != kGatherStatusMagic || status.result || !sample.enter_ns || !sample.ready_queries ||
      status.ready_profile_stamp != (sample.enter_ns ^ kGatherPullProfileCookie)) return std::nullopt;
  return GatherPullObservation{status.ready_profile_first, status.ready_profile_second};
}
struct alignas(64) GatherExecutionControl {
  std::uint64_t claimed{}, failed{};
  GatherChunkConfig chunks{};
  std::uint64_t reserved64{};
  std::uint32_t address_groups_per_post{kGatherDefaultAddressGroupsPerPost}, reserved32{};
  std::uint64_t reserved[3]{};
  alignas(64) std::uint64_t window_sequence[kGatherMaxWorkers]{};
};
static_assert(sizeof(GatherExecutionControl) == 128, "Gather execution control stride changed");
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
  // Zero selects the policy from each launch's total device miss_count.
  std::uint32_t read_wave_wrs{kGatherDefaultReadWaveWrs};
  std::uint32_t pull_groups{};
};
// One stable registered HBM workspace, independent of caller output allocations.
constexpr std::uint64_t kGatherDescriptorOffset = kGatherMaxWorkers * 64U;
inline constexpr std::uint64_t GatherWorkspaceBytes(
    std::uint32_t workers, std::uint32_t capacity = kDefaultGetWindow) {
  return kGatherDescriptorOffset + workers * GatherLaneDescriptorBytes(workers, capacity);
}
static_assert(sizeof(GatherReadArgs) == 160U, "Gather argument layout changed");
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
  aicpu_urma::UmdkContext mailbox_context{};
  std::uint64_t mailbox_source{}, mailbox_remote{};
};
}  // namespace kvaf::cpu_push
