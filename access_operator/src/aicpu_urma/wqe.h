#pragma once

#include <cstdint>
#include <cstring>

#include "gather_output.h"

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

static_assert(ScatterWqebbs(kGatherMaxSglSize) == kMaxScatterWqebbs,
              "scatter WQE buffer is too small");

struct Endpoint {
  std::uint32_t tp_id;
  std::uint32_t remote_token_id;
  std::uint32_t remote_token_value;
  std::uint8_t remote_eid[16];
};

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

inline void EncodeScatterReadWqe(std::uint32_t (&words)[kMaxScatterWqebbs * 16U],
                                 const Endpoint& endpoint,
                                 const ScatterTransfer& transfer,
                                 std::uint32_t owner, bool completion) {
  std::memset(words, 0, sizeof(words));
  const std::uint32_t place_order = completion ? 2U : 1U;
  words[0] = (place_order << 16U) | (1U << 18U) |
             (static_cast<std::uint32_t>(completion) << 19U) | (1U << 20U) |
             (static_cast<std::uint32_t>(completion) << 21U) | (1U << 28U) |
             (1U << 29U) | ((owner & 1U) << 31U);
  words[1] = kReadOpcode << 8U;
  words[2] = (endpoint.tp_id & 0x00ffffffU) | (transfer.sge_count << 24U);
  words[3] = endpoint.remote_token_id & 0x000fffffU;
  std::memcpy(&words[4], endpoint.remote_eid, sizeof(endpoint.remote_eid));
  words[8] = endpoint.remote_token_value;
  words[10] = static_cast<std::uint32_t>(transfer.remote_address);
  words[11] = static_cast<std::uint32_t>(transfer.remote_address >> 32U);
  for (std::uint32_t index = 0U; index < transfer.sge_count; ++index) {
    const std::uint32_t base = 12U + index * 4U;
    words[base] = transfer.sges[index].length;
    words[base + 1U] = transfer.sges[index].token_id;
    words[base + 2U] = static_cast<std::uint32_t>(transfer.sges[index].address);
    words[base + 3U] = static_cast<std::uint32_t>(transfer.sges[index].address >> 32U);
  }
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
