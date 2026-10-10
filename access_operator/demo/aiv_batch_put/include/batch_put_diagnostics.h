#pragma once

#include <cstdint>
#include "batch_put_internal.h"

namespace kvaf::put_diag {
// ABI from HCCP inc/network/hccp_ctx.h. Optional symbol, never a required
// transport dependency. status is urma_cr_status, NOT the raw CQE status.
struct AuxInfoIn {
  uint32_t type;
  uint32_t status;
  uint8_t s_r;
  uint8_t union_padding[3];
  uint8_t reserved[7];
};
struct AuxInfoOut {
  uint32_t types[256];
  uint32_t values[256];
  uint32_t count;
};
static_assert(sizeof(AuxInfoIn) == 20);
static_assert(sizeof(AuxInfoOut) == 2052);
using GetAuxInfo = int (*)(void*, AuxInfoIn*, AuxInfoOut*);

inline bool InRange(uint64_t address, uint64_t length, uint64_t base, uint64_t size) {
  return address >= base && address - base <= size && length <= size - (address - base);
}

inline uint64_t RequestOffset(uint64_t lane_wqe_offset, uint32_t lane, uint32_t lanes) {
  // PostBatchPutRequests stripes 32-WQE posts across lanes.
  return ((lane_wqe_offset / 32U) * lanes + lane) * 32U + lane_wqe_offset % 32U;
}

struct WqeChecks {
  bool local_range;
  bool remote_range;
  bool local_token;
  bool remote_token;
  bool remote_eid;
  bool tpn;
};
inline WqeChecks CheckWqe(const uint32_t* w, const KvafBatchPutDirectUrmaEntity& e) {
  bool eid = true;
  for (uint32_t i = 0; i < 16; ++i)
    eid = eid && static_cast<uint8_t>(w[4 + i / 4] >> ((i % 4) * 8)) == e.remote_eid[i];
  return {InRange(uint64_t(w[14]) | (uint64_t(w[15]) << 32), w[12], e.local_hbm_address, e.local_hbm_size),
          InRange(uint64_t(w[10]) | (uint64_t(w[11]) << 32), w[12], e.remote_host_ddr_address, e.remote_host_ddr_size),
          w[13] == e.local_token_id,
          (w[3] & 0xfffffU) == e.remote_token_id && w[8] == e.remote_token_value,
          eid, (w[2] & 0xffffffU) == e.tp_id};
}
}  // namespace kvaf::put_diag
