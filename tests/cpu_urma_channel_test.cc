#include "cpu_urma_channel.h"

#include <cerrno>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
using kvaf::cpu_runtime_detail::UrmaChannel;
std::unordered_set<urma_jfc_t*> cqs;
std::unordered_set<urma_jfr_t*> jfrs;
std::unordered_set<urma_jetty_t*> jettys;
std::unordered_set<urma_target_jetty_t*> targets;
std::uint32_t next_id = 1U;
int fail_stage = 0;

void Check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
bool Fail(int stage) {
  if (fail_stage != stage) return false;
  errno = ENOSPC;
  return true;
}
void AttachPeer(UrmaChannel& channel) {
  channel.remote_jetty = new urma_target_jetty_t{};
  channel.remote_jetty->id = channel.local_jetty->jetty_id;
  targets.insert(channel.remote_jetty);
}
void CheckLive(std::size_t count) {
  Check(cqs.size() == count && jfrs.size() == count && jettys.size() == count &&
            targets.size() == count,
        "channel leaked or released another channel's resources");
}
}  // namespace

// Run the production channel owner against provider doubles, using UMDK's
// real declarations. Deletion checks enforce queue dependency ordering.
extern "C" {
urma_jfc_t* urma_create_jfc(urma_context_t* context, urma_jfc_cfg_t* cfg) {
  if (Fail(1)) return nullptr;
  Check(cfg->depth == 16384U, "CQ depth changed");
  auto* cq = new urma_jfc_t{};
  cq->urma_ctx = context;
  cq->jfc_cfg = *cfg;
  cq->jfc_id.id = next_id++;
  cqs.insert(cq);
  return cq;
}
urma_jfr_t* urma_create_jfr(urma_context_t* context, urma_jfr_cfg_t* cfg) {
  if (Fail(2)) return nullptr;
  Check(cqs.count(cfg->jfc) == 1U && cfg->depth == 8192U && cfg->trans_mode == URMA_TM_RM &&
            cfg->max_sge == 1U && cfg->flag.bs.tag_matching == URMA_NO_TAG_MATCHING,
        "JFR configuration or CQ ownership changed");
  auto* jfr = new urma_jfr_t{};
  jfr->urma_ctx = context;
  jfr->jfr_cfg = *cfg;
  jfr->jfr_id.id = next_id++;
  jfrs.insert(jfr);
  return jfr;
}
urma_jetty_t* urma_create_jetty(urma_context_t* context, urma_jetty_cfg_t* cfg) {
  if (Fail(3)) return nullptr;
  Check(jfrs.count(cfg->shared.jfr) == 1U && cqs.count(cfg->jfs_cfg.jfc) == 1U &&
            cfg->jfs_cfg.jfc == cfg->shared.jfc &&
            cfg->shared.jfr->jfr_cfg.jfc == cfg->shared.jfc && cfg->jfs_cfg.depth == 8192U &&
            cfg->jfs_cfg.trans_mode == URMA_TM_RM && cfg->jfs_cfg.max_sge == 1U &&
            cfg->flag.bs.share_jfr == URMA_SHARE_JFR,
        "Jetty is not bound to its own JFR and CQ");
  for (const auto* existing : jettys)
    Check(existing->jetty_cfg.jfs_cfg.jfc != cfg->jfs_cfg.jfc &&
              existing->jetty_cfg.shared.jfr != cfg->shared.jfr,
          "different channels share a CQ or JFR");
  auto* jetty = new urma_jetty_t{};
  jetty->urma_ctx = context;
  jetty->jetty_cfg = *cfg;
  jetty->jetty_id.id = next_id++;
  jettys.insert(jetty);
  return jetty;
}
urma_status_t urma_unimport_jetty(urma_target_jetty_t* target) {
  Check(targets.erase(target) == 1U, "peer Jetty released twice");
  delete target;
  return URMA_SUCCESS;
}
urma_status_t urma_delete_jetty(urma_jetty_t* jetty) {
  for (const auto* target : targets)
    Check(target->id.id != jetty->jetty_id.id, "local Jetty released before imported peer");
  Check(jettys.erase(jetty) == 1U, "Jetty released twice");
  delete jetty;
  return URMA_SUCCESS;
}
urma_status_t urma_delete_jfr(urma_jfr_t* jfr) {
  for (const auto* jetty : jettys)
    Check(jetty->jetty_cfg.shared.jfr != jfr, "JFR released before Jetty");
  Check(jfrs.erase(jfr) == 1U, "JFR released twice");
  delete jfr;
  return URMA_SUCCESS;
}
urma_status_t urma_delete_jfc(urma_jfc_t* cq) {
  for (const auto* jfr : jfrs) Check(jfr->jfr_cfg.jfc != cq, "CQ released before JFR");
  for (const auto* jetty : jettys)
    Check(jetty->jetty_cfg.jfs_cfg.jfc != cq, "CQ released before Jetty");
  Check(cqs.erase(cq) == 1U, "CQ released twice");
  delete cq;
  return URMA_SUCCESS;
}
}

int main() {
  try {
    urma_context_t context{};
    {
      std::vector<UrmaChannel> channels;
      std::array<std::uint8_t, 16> peer_eid{};
      peer_eid[15] = 7;
      // Four independent request chains need eight CPU-side channels.
      for (std::uint32_t index = 0U; index < 8U; ++index) {
        UrmaChannel channel;
        Check(channel.CreateLocal(&context, index + 100U), "channel creation failed");
        Check(channel.local_jfr->jfr_cfg.token_value.token == index + 100U,
              "channel token changed");
        const auto* shared =
            kvaf::cpu_runtime_detail::FindSharedChannelTp(channels, peer_eid.data(), 42);
        if (index == 0) {
          Check(shared == nullptr, "TP was reused before any channel connected");
          channel.tp = {true, peer_eid, 43, 42, 5, 1};
        } else {
          Check(shared != nullptr && shared->local == 43 && shared->local_psn == 5 &&
                    shared->peer_psn == 1,
                "CPU channel did not retain the shared TP/PSN pair");
          channel.tp = *shared;
        }
        AttachPeer(channel);
        channels.push_back(std::move(channel));
        Check(channel.local_jfc == nullptr && channel.local_jfr == nullptr &&
                  channel.local_jetty == nullptr && channel.remote_jetty == nullptr,
              "move did not transfer channel ownership");
        CheckLive(index + 1U);
      }
      auto other_eid = peer_eid;
      other_eid[15] = 8;
      Check(kvaf::cpu_runtime_detail::FindSharedChannelTp(channels, other_eid.data(), 42) ==
                    nullptr &&
                kvaf::cpu_runtime_detail::FindSharedChannelTp(channels, peer_eid.data(), 99) ==
                    nullptr,
            "TP reused for another EID or peer TP");
      for (const auto& channel : channels)
        Check(channel.tp.shared && channel.tp.local == 43 && channel.tp.peer == 42,
              "vector relocation lost TP ownership metadata");
      auto* survivor = channels[1].local_jfc;
      channels[0].Reset();
      channels[0].Reset();
      CheckLive(7U);
      Check(cqs.count(survivor) == 1U, "releasing one channel invalidated another CQ");

      for (fail_stage = 1; fail_stage <= 3; ++fail_stage) {
        UrmaChannel failed;
        Check(!failed.CreateLocal(&context, 123U) && errno == ENOSPC,
              "provider allocation failure was not propagated");
        Check(failed.local_jfc == nullptr && failed.local_jfr == nullptr &&
                  failed.local_jetty == nullptr,
              "partial allocation was not unwound");
        CheckLive(7U);
      }
      fail_stage = 0;
      {
        // TP lookup or peer import may fail after all local queues exist.
        UrmaChannel unconnected;
        Check(unconnected.CreateLocal(&context, 123U), "unconnected setup failed");
      }
      CheckLive(7U);

      Check(kvaf::cpu_runtime_detail::FindSharedChannelTp(channels, peer_eid.data(), 42) != nullptr,
            "releasing one channel invalidated the remaining shared TP binding");
      std::vector<UrmaChannel> another_region;
      Check(kvaf::cpu_runtime_detail::FindSharedChannelTp(another_region, peer_eid.data(), 42) ==
                nullptr,
            "TP binding crossed region lifetime");
      channels[1].tp.shared = false;
      channels[1].tp.peer = 99;
      Check(kvaf::cpu_runtime_detail::FindSharedChannelTp(channels, peer_eid.data(), 99) == nullptr,
            "legacy private-TP channel was used for shared TP lookup");
      channels[0] = std::move(channels[1]);
      CheckLive(7U);
      channels[0] = std::move(channels[2]);
      CheckLive(6U);
      while (!channels.empty()) channels.pop_back();
      CheckLive(0U);
    }
    CheckLive(0U);
    std::cout << "CPU URMA channel isolation and lifecycle: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
