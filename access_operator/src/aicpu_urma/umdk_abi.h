#pragma once

#include <urma_api.h>

#include <cstddef>
#include <cstdint>

namespace kvaf::aicpu_urma::umdk {

struct ImportFlag {
  std::uint64_t value, stag, dtag;
};
struct RemoteJettyEx {
  urma_jetty_id_t jetty_id;
  std::uint32_t trans_mode, policy, type, reserved_before_flag;
  ImportFlag flag;
  std::uint32_t tp_type, reserved_after_tp_type;
};
struct Sge {
  std::uint64_t addr;
  std::uint32_t len;
  urma_target_seg_t* tseg;
  void* user_tseg;
};
struct TpQuery {
  std::uint32_t flag, trans_mode;
  std::uint8_t local_eid[16], peer_eid[16];
};
struct TpInfo {
  std::uint64_t tp_handle;
};
struct ActiveTp {
  std::uint64_t tp_handle, peer_tp_handle, tag;
  std::uint32_t tx_psn, rx_psn;
  std::uint64_t reserved;
};
// Match the deployed HBM segment registration/import contract.
inline constexpr std::uint32_t MemRegFlags(bool non_pin) {
  return 1U | (0xEU << 5U) | (non_pin ? (1U << 11U) : 0U) | (1U << 13U);
}
inline constexpr std::uint32_t MemImportFlags() { return 0xEU << 1U; }
using GetTpList = int (*)(urma_context_t*, TpQuery*, std::uint32_t*, TpInfo*);
using ImportJettyEx = urma_target_jetty_t* (*)(urma_context_t*, RemoteJettyEx*, urma_token_t*,
                                               ActiveTp*);
static_assert(sizeof(Sge) == 32 && offsetof(Sge, tseg) == 16 && offsetof(Sge, user_tseg) == 24);
static_assert(sizeof(urma_jetty_id_t) + sizeof(urma_transport_mode_t) == 28);

}  // namespace kvaf::aicpu_urma::umdk
