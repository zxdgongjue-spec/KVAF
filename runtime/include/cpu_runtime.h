#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "runtime.h"

namespace kvaf {

struct CpuRuntimeOptions {
  std::string ub_device;
  bool has_eid_index{false};
  std::uint32_t eid_index{0};
  // Request that UMDK pins the CPU DDR pages when registering each segment.
  // Kept opt-in because a KVAF service can register hundreds of GiB across
  // sessions, which consumes locked physical memory for the service lifetime.
  bool pin_ddr{false};
};

// CPU endpoint Runtime. It owns only CPU-side UMDK resources and sessions;
// RuntimeService owns the listening socket and calls ServeConnection.
class CpuRuntime final : public Runtime {
 public:
  CpuRuntime();
  ~CpuRuntime() override;

  CpuRuntime(const CpuRuntime&) = delete;
  CpuRuntime& operator=(const CpuRuntime&) = delete;

  Status Initialize(const CpuRuntimeOptions& options);

  Status LinkTo(const LinkTarget& target) override;
  Status AcceptLink(const LinkTarget& peer) override;
  Status ReleaseLink(LinkId link_id) override;
  Status ServeConnection(int connection_fd) override;
  Status Shutdown() override;
  bool initialized() const override;
  RuntimeCapabilities capabilities() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf
