#pragma once

#include <cstdint>
#include <cstring>

namespace kvaf::aicpu_urma {

constexpr std::uint32_t kWqeBytes = 64;
constexpr std::uint32_t kReadOpcode = 6;
constexpr std::uint32_t kWriteOpcode = 3;

struct Transfer {
  std::uint64_t remote_address;
  std::uint64_t local_address;
  std::uint32_t length;
  std::uint32_t local_token_id;
};

struct Endpoint {
  std::uint32_t tp_id;
  std::uint32_t remote_token_id;
  std::uint32_t remote_token_value;
  std::uint8_t remote_eid[16];
};

inline void EncodeWqe(std::uint32_t (&words)[16], const Endpoint& endpoint,
                      const Transfer& transfer, std::uint32_t opcode,
                      std::uint32_t owner, bool completion) {
  std::memset(words, 0, sizeof(words));
  const std::uint32_t place_order = completion ? 2U : 1U;
  words[0] = (place_order << 16U) | (1U << 18U) | (1U << 20U) |
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

inline bool InRange(std::uint64_t base, std::uint64_t size,
                    std::uint64_t address, std::uint64_t length) {
  return length != 0U && address >= base && address - base <= size &&
         length <= size - (address - base);
}

}  // namespace kvaf::aicpu_urma
