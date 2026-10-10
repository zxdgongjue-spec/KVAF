#pragma once

#include <cstdint>
#include <cstring>

#include "gather_output.h"
#include "sq_copy.h"

namespace kvaf::aicpu_urma {

constexpr std::uint32_t kWqeBytes = 64;
constexpr std::uint32_t kReadOpcode = 6;
constexpr std::uint32_t kWriteOpcode = 3;
constexpr std::uint32_t kWriteWithNotifyOpcode = 5;
constexpr std::uint32_t kMaxReadBytes = 256U * 1024U * 1024U;
constexpr std::uint32_t kMaxScatterWqebbs = 3U;

struct Transfer {
  std::uint64_t remote_address;
  std::uint64_t local_address;
  std::uint32_t length;
  std::uint32_t local_token_id;
};

struct LocalSge {
  std::uint64_t address;
  std::uint32_t length;
  std::uint32_t token_id;
};

struct ScatterTransfer {
  std::uint64_t remote_address;
  std::uint32_t bytes;
  std::uint32_t sge_count;
  LocalSge sges[kGatherMaxSglSize];
};

constexpr std::uint32_t ScatterWqebbs(std::uint32_t sge_count) {
  return (48U + sge_count * 16U + kWqeBytes - 1U) / kWqeBytes;
}

constexpr std::uint32_t MaxScatterBatch(std::uint32_t sq_depth) {
  const auto safe_batch = sq_depth > 1U ? (sq_depth - 1U) / kMaxScatterWqebbs : 0U;
  return safe_batch < 64U ? safe_batch : 64U;
}

static_assert(ScatterWqebbs(kGatherMaxSglSize) == kMaxScatterWqebbs,
              "scatter WQE buffer is too small");

struct Endpoint {
  std::uint32_t tp_id;
  std::uint32_t remote_token_id;
  std::uint32_t remote_token_value;
  std::uint8_t remote_eid[16];
};

struct ScatterReadHeader {
  std::uint32_t words[10];
};

inline ScatterReadHeader PrepareScatterReadHeader(const Endpoint& endpoint) {
  ScatterReadHeader header{};
  header.words[1] = kReadOpcode << 8U;
  header.words[2] = endpoint.tp_id & 0x00ffffffU;
  header.words[3] = endpoint.remote_token_id & 0x000fffffU;
  std::memcpy(&header.words[4], endpoint.remote_eid, sizeof(endpoint.remote_eid));
  header.words[8] = endpoint.remote_token_value;
  return header;
}

constexpr std::uint32_t kScatterBatchWqes = 64U;
constexpr std::uint32_t kScatterWqeWords = kMaxScatterWqebbs * 16U;

struct ScatterReadTemplates {
  std::uint32_t words[kGatherMaxSglSize][kScatterWqeWords];
  std::uint32_t control[2];
};

struct ScatterReadBatch {
  ScatterReadTemplates templates;
  alignas(64) std::uint32_t words[kScatterBatchWqes * kScatterWqeWords];
  std::uint32_t sge_counts[kScatterBatchWqes];
  std::uint32_t initialized_wqes;
};

inline void PrepareScatterReadBatch(ScatterReadBatch& batch, const Endpoint& endpoint,
                                    std::uint32_t local_token_id) {
  batch.initialized_wqes = 0U;
  std::memset(&batch.templates, 0, sizeof(batch.templates));
  const auto header = PrepareScatterReadHeader(endpoint);
  for (std::uint32_t count = 1U; count <= kGatherMaxSglSize; ++count) {
    auto* words = batch.templates.words[count - 1U];
    std::memcpy(words, header.words, sizeof(header.words));
    words[2] |= count << 24U;
    for (std::uint32_t index = 0U; index < count; ++index)
      words[13U + index * 4U] = local_token_id;
  }
  const auto fixed = (1U << 18U) | (1U << 20U) | (1U << 28U) | (1U << 29U);
  batch.templates.control[0] = fixed | (1U << 16U);
  batch.templates.control[1] = fixed | (2U << 16U) | (1U << 19U) | (1U << 21U);
}

inline void PatchScatterReadWqe(ScatterReadBatch& batch, std::uint32_t batch_index,
                                std::uint32_t block_offset, const ScatterTransfer& transfer,
                                std::uint32_t owner, bool completion) {
  auto* words = batch.words + block_offset * 16U;
  if (batch_index >= batch.initialized_wqes ||
      batch.sge_counts[batch_index] != transfer.sge_count) {
    std::memcpy(words, batch.templates.words[transfer.sge_count - 1U],
                ScatterWqebbs(transfer.sge_count) * kWqeBytes);
    batch.sge_counts[batch_index] = transfer.sge_count;
    batch.initialized_wqes = batch_index + 1U;
  }
  words[0] = batch.templates.control[completion ? 1U : 0U] | ((owner & 1U) << 31U);
  words[10] = static_cast<std::uint32_t>(transfer.remote_address);
  words[11] = static_cast<std::uint32_t>(transfer.remote_address >> 32U);
  for (std::uint32_t index = 0U; index < transfer.sge_count; ++index) {
    const auto offset = 12U + index * 4U;
    words[offset] = transfer.sges[index].length;
    words[offset + 2U] = static_cast<std::uint32_t>(transfer.sges[index].address);
    words[offset + 3U] = static_cast<std::uint32_t>(transfer.sges[index].address >> 32U);
  }
}

inline bool CopySqWqebbs(void* destination, const void* source, std::uint32_t blocks,
                         GatherSqCopyMode mode) {
  return kvaf::ValidGatherSqCopyMode(mode) &&
      CopySqBytes(destination, source, static_cast<std::size_t>(blocks) * kWqeBytes);
}

inline std::uint32_t PlanContiguousRead(Transfer (&transfers)[16],
                                        std::uint64_t remote, std::uint64_t local,
                                        std::uint32_t bytes, std::uint32_t local_token_id) {
  std::uint32_t count = 0U;
  std::uint64_t offset = 0U;
  while (offset < bytes) {
    const auto remaining = static_cast<std::uint64_t>(bytes) - offset;
    const auto length = static_cast<std::uint32_t>(
        remaining < kMaxReadBytes ? remaining : kMaxReadBytes);
    transfers[count++] = {remote + offset, local + offset, length, local_token_id};
    offset += length;
  }
  return count;
}

inline void EncodeWqe(std::uint32_t (&words)[16], const Endpoint& endpoint,
                      const Transfer& transfer, std::uint32_t opcode,
                      std::uint32_t owner, bool completion) {
  std::memset(words, 0, sizeof(words));
  const std::uint32_t place_order = completion ? 2U : 1U;
  words[0] = (place_order << 16U) | (1U << 18U) |
             (static_cast<std::uint32_t>(completion) << 19U) | (1U << 20U) |
             (static_cast<std::uint32_t>(completion) << 21U) | (1U << 28U) |
             (1U << 29U) | ((owner & 1U) << 31U);
  words[1] = opcode << 8U;
  words[2] = (endpoint.tp_id & 0x00ffffffU) | 0x01000000U;
  words[3] = endpoint.remote_token_id & 0x000fffffU;
  std::memcpy(&words[4], endpoint.remote_eid, sizeof(endpoint.remote_eid));
  words[8] = endpoint.remote_token_value;
  words[10] = static_cast<std::uint32_t>(transfer.remote_address);
  words[11] = static_cast<std::uint32_t>(transfer.remote_address >> 32U);
  words[12] = transfer.length;
  words[13] = transfer.local_token_id;
  words[14] = static_cast<std::uint32_t>(transfer.local_address);
  words[15] = static_cast<std::uint32_t>(transfer.local_address >> 32U);
}

inline void EncodeScatterReadWqe(std::uint32_t* words,
                                 const ScatterReadHeader& header,
                                 const ScatterTransfer& transfer,
                                 std::uint32_t owner, bool completion) {
  std::memcpy(words, header.words, sizeof(header.words));
  const std::uint32_t place_order = completion ? 2U : 1U;
  words[0] = (place_order << 16U) | (1U << 18U) |
             (static_cast<std::uint32_t>(completion) << 19U) | (1U << 20U) |
             (static_cast<std::uint32_t>(completion) << 21U) | (1U << 28U) |
             (1U << 29U) | ((owner & 1U) << 31U);
  words[2] |= transfer.sge_count << 24U;
  words[10] = static_cast<std::uint32_t>(transfer.remote_address);
  words[11] = static_cast<std::uint32_t>(transfer.remote_address >> 32U);
  for (std::uint32_t index = 0U; index < transfer.sge_count; ++index) {
    const std::uint32_t base = 12U + index * 4U;
    words[base] = transfer.sges[index].length;
    words[base + 1U] = transfer.sges[index].token_id;
    words[base + 2U] = static_cast<std::uint32_t>(transfer.sges[index].address);
    words[base + 3U] = static_cast<std::uint32_t>(transfer.sges[index].address >> 32U);
  }
  const auto used_words = 12U + transfer.sge_count * 4U;
  const auto total_words = ScatterWqebbs(transfer.sge_count) * 16U;
  std::memset(words + used_words, 0, (total_words - used_words) * sizeof(*words));
}

inline void EncodeScatterReadWqe(std::uint32_t* words,
                                 const Endpoint& endpoint,
                                 const ScatterTransfer& transfer,
                                 std::uint32_t owner, bool completion) {
  EncodeScatterReadWqe(words, PrepareScatterReadHeader(endpoint), transfer, owner, completion);
}

inline void EncodeNotifyWqe(std::uint32_t (&words)[32], const Endpoint& endpoint,
                            std::uint64_t command_address, std::uint64_t notify_address,
                            std::uint64_t local_address, std::uint32_t local_token_id,
                            std::uint64_t sequence, std::uint32_t owner) {
  std::memset(words, 0, sizeof(words));
  words[0] = (2U << 16U) | (1U << 18U) | (1U << 19U) |
             (1U << 20U) | (1U << 21U) | (1U << 28U) |
             (1U << 29U) | ((owner & 1U) << 31U);
  words[1] = kWriteWithNotifyOpcode << 8U;
  words[2] = (endpoint.tp_id & 0x00ffffffU) | 0x01000000U;
  words[3] = endpoint.remote_token_id & 0x000fffffU;
  std::memcpy(&words[4], endpoint.remote_eid, sizeof(endpoint.remote_eid));
  words[8] = endpoint.remote_token_value;
  words[10] = static_cast<std::uint32_t>(command_address);
  words[11] = static_cast<std::uint32_t>(command_address >> 32U);
  words[12] = endpoint.remote_token_id & 0x000fffffU;
  words[13] = endpoint.remote_token_value;
  words[14] = static_cast<std::uint32_t>(notify_address);
  words[15] = static_cast<std::uint32_t>(notify_address >> 32U);
  words[16] = static_cast<std::uint32_t>(sequence);
  words[17] = static_cast<std::uint32_t>(sequence >> 32U);
  words[20] = 8U;
  words[21] = local_token_id;
  words[22] = static_cast<std::uint32_t>(local_address);
  words[23] = static_cast<std::uint32_t>(local_address >> 32U);
}

inline bool InRange(std::uint64_t base, std::uint64_t size,
                    std::uint64_t address, std::uint64_t length) {
  return length != 0U && address >= base && address - base <= size &&
         length <= size - (address - base);
}

}  // namespace kvaf::aicpu_urma
