#pragma once

#include <cstdint>
#include <memory>

#include "runtime.h"

namespace kvaf {

struct NpuServiceOptions {
  std::uint32_t device_id{0U};
  // If false, the device is selected lazily from AllocateHbmRegion requests.
  bool has_device_id{false};
  // Index in the discovered NPU EID list. The HDC context index is fixed at
  // zero to match the existing AIV InitContext() implementation.
  std::uint32_t eid_index{0U};
  bool has_eid_index{false};
};

// NPU-side control service. It owns device HBM and exports provider-created
// segment/Jetty descriptors to a CPU Runtime client. The data plane remains
// initiated by the CPU peer; this service does not own MetaManager.
class NpuService final : public Runtime {
 public:
  NpuService();
  ~NpuService() override;

  NpuService(const NpuService&) = delete;
  NpuService& operator=(const NpuService&) = delete;

  Status Initialize(const NpuServiceOptions& options);

  Status LinkTo(const LinkTarget&) override;
  Status AcceptLink(const LinkTarget&) override;
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
