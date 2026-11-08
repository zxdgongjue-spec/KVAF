#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "urma_transport.h"

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
    const auto* error =
        urma::CreateQueues(urma::NativeApi{}, context, token, {16384U, 8192U, 1U, false}, local_jfc,
                           local_jfr, local_jetty);
    return error ? Fail(error, 8192U, token) : true;
  }

  void Reset() { (void)Close(); }

  bool Close() {
    if (remote_jetty != nullptr) {
      if (urma_unimport_jetty(remote_jetty) != URMA_SUCCESS) return false;
      remote_jetty = nullptr;
    }
    if (local_jetty != nullptr) {
      if (urma_delete_jetty(local_jetty) != URMA_SUCCESS) return false;
      local_jetty = nullptr;
    }
    if (local_jfr != nullptr) {
      if (urma_delete_jfr(local_jfr) != URMA_SUCCESS) return false;
      local_jfr = nullptr;
    }
    if (local_jfc != nullptr) {
      if (urma_delete_jfc(local_jfc) != URMA_SUCCESS) return false;
      local_jfc = nullptr;
    }
    tp = {};
    return true;
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
