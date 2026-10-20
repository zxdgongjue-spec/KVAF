#pragma once

#include <urma_api.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace kvaf::cpu_runtime_detail {

struct UrmaChannelTp {
  bool shared = false;
  std::array<std::uint8_t, 16> peer_eid{};
  std::uint64_t local = 0U, peer = 0U;
  std::uint32_t local_psn = 0U, peer_psn = 0U;

  bool Matches(const std::uint8_t* eid, std::uint64_t handle) const {
    return shared && peer == handle && std::equal(peer_eid.begin(), peer_eid.end(), eid);
  }
};

// A service channel owns its local queues and the imported peer Jetty.
// Moving a channel transfers ownership; teardown releases dependants first.
struct UrmaChannel {
  urma_jfc_t* local_jfc = nullptr;
  urma_jfr_t* local_jfr = nullptr;
  urma_jetty_t* local_jetty = nullptr;
  urma_target_jetty_t* remote_jetty = nullptr;
  UrmaChannelTp tp;

  UrmaChannel() = default;
  ~UrmaChannel() { Reset(); }
  UrmaChannel(const UrmaChannel&) = delete;
  UrmaChannel& operator=(const UrmaChannel&) = delete;
  UrmaChannel(UrmaChannel&& other) noexcept { Swap(other); }
  UrmaChannel& operator=(UrmaChannel&& other) noexcept {
    if (this != &other) {
      Reset();
      Swap(other);
    }
    return *this;
  }

  bool CreateLocal(urma_context_t* context, std::uint32_t token) {
    constexpr std::uint32_t jetty_depth = 8192U;
    constexpr std::uint32_t cq_depth = 16384U;
    urma_jfc_cfg_t cq_config{};
    cq_config.depth = cq_depth;
    local_jfc = urma_create_jfc(context, &cq_config);
    if (local_jfc == nullptr) return Fail("urma_create_jfc", cq_depth, token);

    urma_jfr_cfg_t jfr_config{};
    jfr_config.depth = jetty_depth;
    jfr_config.trans_mode = URMA_TM_RM;
    jfr_config.max_sge = 1U;
    jfr_config.jfc = local_jfc;
    jfr_config.token_value.token = token;
    jfr_config.flag.bs.tag_matching = URMA_NO_TAG_MATCHING;
    local_jfr = urma_create_jfr(context, &jfr_config);
    if (local_jfr == nullptr) return Fail("urma_create_jfr", jetty_depth, token);

    urma_jetty_cfg_t config{};
    config.flag.bs.share_jfr = URMA_SHARE_JFR;
    config.jfs_cfg.depth = jetty_depth;
    config.jfs_cfg.trans_mode = URMA_TM_RM;
    config.jfs_cfg.max_sge = 1U;
    config.jfs_cfg.jfc = local_jfc;
    config.shared.jfr = local_jfr;
    config.shared.jfc = local_jfc;
    local_jetty = urma_create_jetty(context, &config);
    if (local_jetty == nullptr) return Fail("urma_create_jetty", jetty_depth, token);
    return true;
  }

  void Reset() {
    if (remote_jetty != nullptr) {
      (void)urma_unimport_jetty(remote_jetty);
      remote_jetty = nullptr;
    }
    if (local_jetty != nullptr) {
      (void)urma_delete_jetty(local_jetty);
      local_jetty = nullptr;
    }
    if (local_jfr != nullptr) {
      (void)urma_delete_jfr(local_jfr);
      local_jfr = nullptr;
    }
    if (local_jfc != nullptr) {
      (void)urma_delete_jfc(local_jfc);
      local_jfc = nullptr;
    }
    tp = {};
  }

 private:
  void Swap(UrmaChannel& other) noexcept {
    std::swap(local_jfc, other.local_jfc);
    std::swap(local_jfr, other.local_jfr);
    std::swap(local_jetty, other.local_jetty);
    std::swap(remote_jetty, other.remote_jetty);
    std::swap(tp, other.tp);
  }

  bool Fail(const char* operation, std::uint32_t depth, std::uint32_t token) {
    const int error = errno;
    std::fprintf(stderr, "[KVAF_CPU] %s failed depth=%u token=0x%x errno=%d\n", operation, depth,
                 token, error);
    Reset();
    errno = error;
    return false;
  }
};

inline const UrmaChannelTp* FindSharedChannelTp(const std::vector<UrmaChannel>& channels,
                                                const std::uint8_t* peer_eid,
                                                std::uint64_t peer_tp) {
  for (const auto& channel : channels)
    if (channel.tp.Matches(peer_eid, peer_tp)) return &channel.tp;
  return nullptr;
}

}  // namespace kvaf::cpu_runtime_detail
