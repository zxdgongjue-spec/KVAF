#pragma once

#include <cstdint>

#include "direct_hccp/direct_hccp_abi.h"

namespace kvaf::hbm_test {

constexpr std::uint32_t kMagic = 0x4b564842U;  // KVHB
constexpr std::uint16_t kVersion = 1U;
constexpr std::uint32_t kHdcType = 18U;

#pragma pack(push, 1)
struct Hello {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t reserved;
  std::uint64_t bytes;
  std::uint32_t device_id;
  std::uint32_t eid_index;
};

struct Export {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t reserved;
  std::uint64_t hbm_bytes;
  std::uint32_t token_value;
  std::uint32_t segment_key_bytes;
  std::uint32_t jetty_key_bytes;
  std::uint32_t reserved2;
  std::uint8_t peer_eid[direct_hccp::kEidBytes];
  std::uint8_t segment_key[direct_hccp::kMemKeyBytes];
  std::uint8_t jetty_key[direct_hccp::kQpKeyBytes];
};

struct ChannelRequest {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t reserved;
  std::uint64_t local_tp;
  std::uint32_t local_psn;
  std::uint32_t local_key_bytes;
  std::uint8_t local_eid[direct_hccp::kEidBytes];
  std::uint8_t local_key[direct_hccp::kQpKeyBytes];
};

struct ChannelResponse {
  std::int32_t status;
  std::uint32_t peer_psn;
  std::uint64_t peer_tp;
  std::uint32_t peer_key_bytes;
  std::uint8_t peer_key[direct_hccp::kQpKeyBytes];
};
#pragma pack(pop)

static_assert(sizeof(Export) == 240U, "HBM export ABI changed");

}  // namespace kvaf::hbm_test
