#pragma once
#include <cstdint>

#include "gather_workspace.h"
#include "batch_get_profile.h"
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
};

struct alignas(64) GatherReadStatus {
  std::uint64_t magic, result, phase, ack_seen, cqe_word, sequence;
};
struct GatherReadArgs {
  UmdkContext context;
  std::uint64_t core_status{0U};
  std::uint64_t output{0U}, output_extent{0U}, scratch{0U};
  std::uint64_t remote_staging{0U}, remote_descriptor{0U};
  std::uint64_t remote_command{0U}, remote_notify{0U}, remote_ack{0U};
  std::uint64_t source_descriptor{0U}, sequence{0U};
  std::uint32_t bytes{0U}, source_span_count{0U};
  std::uint32_t sgl_size{6U}, worker_count{1U}, sq_batch_wqes{32U};
  std::uint32_t worker_index{0U};
  std::uint32_t pipeline_chunk_bytes{kGatherChunkBytes};
  std::uint64_t profile{0U};
};
// One stable registered HBM workspace, independent of caller output allocations.
constexpr std::uint64_t kGatherDescriptorOffset = kGatherMaxWorkers * 64U;
inline constexpr std::uint64_t GatherWorkspaceBytes(std::uint32_t workers) {
  return kGatherDescriptorOffset + workers * GatherLaneDescriptorBytes(workers);
}
static_assert(sizeof(GatherReadArgs) <= 256U);
struct AicpuBatchGetInput {
  // GatherRead with indices == 0 reads address_table[i] as a complete source
  // address; offsets is unused. Other callers use address_table[indices[i]] + offsets[i].
  std::uint64_t address_table{}, indices{}, offsets{}, miss_count{}, output_indices{};
  std::uint32_t table_capacity{}, request_capacity{}, token_bytes{};
  std::uint64_t valid_table{}, canonical_base{}, logical_bytes{}, block_bytes{};
};
inline bool ResolveBatchGetSource(const AicpuBatchGetInput& r, std::uint32_t row,
                                  std::uint64_t output_bytes, std::uint64_t* source) {
  if (!source || row >= r.request_capacity || !r.address_table || !r.output_indices ||
      !r.token_bytes)
    return false;
  const auto output = reinterpret_cast<const std::int32_t*>(r.output_indices)[row];
  if (output < 0 || output_bytes < r.token_bytes ||
      std::uint64_t(output) > (output_bytes - r.token_bytes) / r.token_bytes)
    return false;
  std::uint32_t index = row;
  std::uint64_t offset = 0;
  if (r.indices) {
    const auto selected = reinterpret_cast<const std::int32_t*>(r.indices)[row];
    if (selected < 0 || std::uint32_t(selected) >= r.table_capacity || !r.offsets) return false;
    index = selected;
    const auto displacement = reinterpret_cast<const std::int32_t*>(r.offsets)[row];
    if (displacement < 0) return false;
    offset = displacement;
    if (r.block_bytes && (r.block_bytes < r.token_bytes || offset > r.block_bytes - r.token_bytes))
      return false;
    if (r.valid_table && !reinterpret_cast<const std::uint8_t*>(r.valid_table)[index]) return false;
    if (r.canonical_base && !r.valid_table) return false;
  } else if (index >= r.table_capacity)
    return false;
  const auto base = reinterpret_cast<const std::uint64_t*>(r.address_table)[index];
  if (r.canonical_base && r.indices && r.block_bytes &&
      (base < r.canonical_base || base - r.canonical_base > r.logical_bytes ||
       r.block_bytes > r.logical_bytes - (base - r.canonical_base)))
    return false;
  if (offset > UINT64_MAX - base) return false;
  *source = base + offset;
  if (r.canonical_base &&
      (*source < r.canonical_base || *source - r.canonical_base > r.logical_bytes ||
       r.token_bytes > r.logical_bytes - (*source - r.canonical_base)))
    return false;
  return true;
}
struct UmdkBatchGetArgs {
  AicpuBatchGetInput request;
  GatherReadArgs gather;
};
}  // namespace kvaf::aicpu_urma

namespace kvaf::cpu_push {
struct KernelArgs {
  aicpu_urma::AicpuBatchGetInput request;
  std::uint64_t workspace{}, output{}, output_bytes{}, coordination{}, status{}, profile{};
  std::uint64_t sequence{};
  std::uint32_t workers{}, timeout_ms{kTimeoutMs};
};
}  // namespace kvaf::cpu_push
