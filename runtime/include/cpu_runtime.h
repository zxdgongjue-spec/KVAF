#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "runtime.h"

namespace kvaf {

enum class CpuMemoryPageMode {
  kDefault,
  kBasePage,
  kTransparentHugePage,
};

struct CpuRuntimeOptions {
  std::string ub_device;
  bool has_eid_index{false};
  std::uint32_t eid_index{0};
  CpuMemoryPageMode memory_page_mode{CpuMemoryPageMode::kTransparentHugePage};
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
