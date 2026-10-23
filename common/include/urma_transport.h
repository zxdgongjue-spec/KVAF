#pragma once

#include <urma_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace kvaf::urma {

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
static_assert(sizeof(RemoteJettyEx) == 72 && offsetof(RemoteJettyEx, flag) == 40);
static_assert(sizeof(TpQuery) == 40 && sizeof(ActiveTp) == 40);

// Directly linked CPU and dynamically loaded AICPU use the same ABI/configuration.
struct NativeApi {
  decltype(&::urma_get_device_list) urma_get_device_list = &::urma_get_device_list;
  decltype(&::urma_free_device_list) urma_free_device_list = &::urma_free_device_list;
  decltype(&::urma_get_eid_list) urma_get_eid_list = &::urma_get_eid_list;
  decltype(&::urma_free_eid_list) urma_free_eid_list = &::urma_free_eid_list;
  decltype(&::urma_create_jfc) urma_create_jfc = &::urma_create_jfc;
  decltype(&::urma_create_jfr) urma_create_jfr = &::urma_create_jfr;
  decltype(&::urma_create_jetty) urma_create_jetty = &::urma_create_jetty;
};

template <typename Api, typename Visit>
bool EnumerateEndpoints(const Api& api, const char* requested_device, std::uint32_t eid_index,
                        Visit visit) {
  int count = 0;
  auto** devices = api.urma_get_device_list(&count);
  if (!devices) return false;
  bool done = false;
  for (int i = 0; i < count && !done; ++i) {
    auto* device = devices[i];
    if (!device || device->type != URMA_TRANSPORT_UB ||
        (*requested_device && std::strcmp(requested_device, device->name)))
      continue;
    std::uint32_t eid_count = 0;
    auto* eids = api.urma_get_eid_list(device, &eid_count);
    if (!eids) continue;
    for (std::uint32_t j = 0; j < eid_count && !done; ++j) {
      const auto& eid = eids[j];
      if ((eid_index != UINT32_MAX && eid.eid_index != eid_index) ||
          std::all_of(eid.eid.raw, eid.eid.raw + 16, [](auto b) { return b == 0; }))
        continue;
      done = visit(device, eid);
    }
    api.urma_free_eid_list(eids);
  }
  api.urma_free_device_list(devices);
  return true;
}

struct QueueOptions {
  std::uint32_t cq_depth, receive_depth, max_sge;
  bool lock_free;
};

template <typename Api>
const char* CreateQueues(const Api& api, urma_context_t* context, std::uint32_t token,
                         QueueOptions options, urma_jfc_t*& cq, urma_jfr_t*& rq,
                         urma_jetty_t*& jetty) {
  urma_jfc_cfg_t cq_cfg{};
  cq_cfg.depth = options.cq_depth;
  cq_cfg.flag.bs.lock_free = options.lock_free;
  cq = api.urma_create_jfc(context, &cq_cfg);
  if (!cq) return "urma_create_jfc";
  urma_jfr_cfg_t rq_cfg{};
  rq_cfg.depth = options.receive_depth;
  rq_cfg.trans_mode = URMA_TM_RM;
  rq_cfg.max_sge = 1;
  rq_cfg.jfc = cq;
  rq_cfg.token_value.token = token;
  rq_cfg.flag.bs.tag_matching = URMA_NO_TAG_MATCHING;
  rq = api.urma_create_jfr(context, &rq_cfg);
  if (!rq) return "urma_create_jfr";
  urma_jetty_cfg_t cfg{};
  cfg.flag.bs.share_jfr = URMA_SHARE_JFR;
  cfg.jfs_cfg.depth = 8192;
  cfg.jfs_cfg.trans_mode = URMA_TM_RM;
  cfg.jfs_cfg.flag.bs.lock_free = options.lock_free;
  cfg.jfs_cfg.max_sge = options.max_sge;
  if (options.lock_free) {
    cfg.jfs_cfg.max_rsge = 1;
    cfg.jfs_cfg.priority = URMA_MAX_PRIORITY;
    cfg.jfs_cfg.rnr_retry = URMA_TYPICAL_RNR_RETRY;
    cfg.jfs_cfg.err_timeout = URMA_TYPICAL_ERR_TIMEOUT;
  }
  cfg.jfs_cfg.jfc = cq;
  cfg.shared.jfr = rq;
  cfg.shared.jfc = cq;
  jetty = api.urma_create_jetty(context, &cfg);
  return jetty ? nullptr : "urma_create_jetty";
}

inline bool QueryTp(GetTpList query, urma_context_t* context, const std::uint8_t* peer_eid,
                    std::uint64_t* tp) {
  TpQuery cfg{};
  cfg.flag = 1;
  cfg.trans_mode = URMA_TM_RM;
  std::memcpy(cfg.local_eid, context->eid.raw, 16);
  std::memcpy(cfg.peer_eid, peer_eid, 16);
  TpInfo info{};
  std::uint32_t count = 1;
  if (query(context, &cfg, &count, &info) || !count || !info.tp_handle) return false;
  *tp = info.tp_handle;
  return true;
}

inline urma_target_jetty_t* ImportJetty(ImportJettyEx import, urma_context_t* context,
                                        const std::uint8_t* key, std::uint32_t token_value,
                                        std::uint64_t local_tp, std::uint64_t peer_tp,
                                        std::uint32_t tx_psn, std::uint32_t rx_psn) {
  RemoteJettyEx remote{};
  std::memcpy(&remote.jetty_id, key, sizeof(remote.jetty_id));
  std::memcpy(&remote.trans_mode, key + sizeof(remote.jetty_id), sizeof(remote.trans_mode));
  remote.policy = URMA_JETTY_GRP_POLICY_RR;
  remote.type = URMA_JETTY;
  urma_import_jetty_flag_t flag{};
  flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
  remote.flag.value = flag.value;
  remote.tp_type = 1;
  ActiveTp active{};
  active.tp_handle = local_tp;
  active.peer_tp_handle = peer_tp;
  active.tx_psn = tx_psn;
  active.rx_psn = rx_psn;
  urma_token_t token{};
  token.token = token_value;
  return import(context, &remote, &token, &active);
}

}  // namespace kvaf::urma
