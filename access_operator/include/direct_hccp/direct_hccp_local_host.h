#pragma once

#include "direct_hccp/direct_hccp_remote_transport.h"

namespace kvaf::direct_hccp {

// In-process Host DDR endpoint. It owns RA resources, never the DDR allocation.
// A host/device Jetty pair belongs to exactly one AIV queue.
struct LocalHostWorker {
  void* cq{nullptr};
  CqInfo cq_info{};
  DeviceAivJetty jetty{};
  void* device_target{nullptr};
  QpImportInfoT device_import{};
};
struct LocalHostEndpoint {
  void* context{nullptr};
  void* token_handle{nullptr};
  DirectMemory memory{};
  std::array<std::uint8_t, kEidBytes> host_eid{};
  std::array<std::uint8_t, kEidBytes> device_eid{};
  std::array<LocalHostWorker, 96> workers{};
};

bool CreateLocalHostEndpoint(const DirectHccpApi& api, std::uint32_t physical_device_id,
                             void* ddr, std::uint64_t bytes, LocalHostEndpoint* endpoint);
bool CreateLocalHostChannel(const DirectHccpApi& api, std::uint32_t physical_device_id,
                            void* device_context, std::uint32_t device_token_value,
                            const QpCreateInfo& device_jetty, std::uint32_t queue_index,
                            LocalHostEndpoint* endpoint, PeerChannel* channel);
// Retire both directions before any dependent HBM/DDR registration is freed.
// On failure all remaining resources stay owned by the caller for quarantine.
bool RetireLocalAndDeviceQueues(const DirectHccpApi& api, void* device_context,
                                DeviceAivWorkerResources* workers, std::uint32_t count,
                                LocalHostEndpoint* endpoint);
bool ReleaseLocalHostEndpoint(const DirectHccpApi& api, LocalHostEndpoint* endpoint);

}  // namespace kvaf::direct_hccp
