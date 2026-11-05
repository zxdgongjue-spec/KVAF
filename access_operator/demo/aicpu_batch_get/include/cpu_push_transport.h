#pragma once

#include <array>
#include <memory>

#include "access_operator.h"

namespace kvaf::cpu_push {
struct Config {
  std::shared_ptr<peer::PeerLink> peer_link;
  std::string kernel_json;
  std::array<std::uint8_t, 16> local_eid{};
  std::uint32_t token_bytes{}, workers{1};
};
class CpuPushTransport {
 public:
  explicit CpuPushTransport(Config);
  ~CpuPushTransport();
  Status Initialize(const RemoteRegionDesc&);
  Status Execute(const BatchGetExecRequest&);
  Status Drain();
  Status Shutdown();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace kvaf::cpu_push
