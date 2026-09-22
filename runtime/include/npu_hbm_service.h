#pragma once

#include <memory>
#include <string>
#include <vector>
#include "runtime.h"

namespace kvaf {

struct NpuHbmServiceOptions {
  std::uint32_t device_id{0U};
  std::vector<std::string> npu_eids;
  bool huge_first{true};
  std::uint32_t max_sessions{64U};
  std::uint32_t max_channels_per_session{4096U};
};

// NPU-side provider. Each session owns its MR/token and target Jettys; NPU
// readers use Direct-HCCP AIV READ/WRITE without a CPU URMA data plane.
class NpuHbmService final : public Runtime {
 public:
  NpuHbmService();
  ~NpuHbmService() override;
  NpuHbmService(const NpuHbmService&) = delete;
  NpuHbmService& operator=(const NpuHbmService&) = delete;
  Status Initialize(const NpuHbmServiceOptions& options);
  std::size_t active_sessions() const;
  Status LinkTo(const LinkTarget&) override;
  Status AcceptLink(const LinkTarget&) override;
  Status ReleaseLink(LinkId id) override;
  Status ServeConnection(int fd) override;
  // Busy while a reader still owns a session. Readers must synchronize and
  // shut down before stopping this service.
  Status Shutdown() override;
  bool initialized() const override;
  RuntimeCapabilities capabilities() const override;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
