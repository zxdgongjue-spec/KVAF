#pragma once
// Only the public segment descriptor used by the fake RA provider. Hardware
// builds compile against real UMDK headers; this does not test provider ABI.
#include <cstdint>
struct urma_eid_t { std::uint8_t raw[16]; };
#pragma pack(push, 1)
struct urma_ubva_t { urma_eid_t eid; std::uint32_t uasid; std::uint64_t va; };
#pragma pack(pop)
struct urma_seg_t {
  urma_ubva_t ubva;
  std::uint64_t len;
  std::uint32_t attr;
  std::uint32_t token_id;
};
