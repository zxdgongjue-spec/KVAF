#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "access_operator.h"
#include "cpu_urma/cpu_urma_transport.h"

namespace kvaf {

// CPU data-plane AccessOperator. CPU BatchPut follows the AIV BatchGet data
// flow (CPU DRAM -> HBM), while CPU BatchGet follows the AIV BatchPut data
// flow (HBM -> CPU DRAM). Both paths implement the same Runtime-facing
// AccessOperator interface as the AIV backend.
class CpuUrmaAccessOperator final : public AccessOperator {
 public:
  explicit CpuUrmaAccessOperator(CpuUrmaTransportOptions options,
                                 std::shared_ptr<peer::PeerLink> link = nullptr);
  ~CpuUrmaAccessOperator() override;

  Status Initialize(const RemoteRegionDesc& remote_region) override;
  Status Shutdown() override;
  Status ValidateLink();
  Status ExecuteBatchGet(const BatchGetExecRequest& request) override;
  Status ExecuteBatchPut(const BatchPutExecRequest& request) override;

  CpuUrmaTransport::BatchTiming last_batch_timing() const;

 private:
  CpuUrmaTransportOptions options_;
  CpuUrmaTransport transport_;
  std::shared_ptr<peer::PeerLink> link_;
  RemoteRegionDesc region_{};
  mutable std::mutex batch_mutex_;
  std::vector<CpuUrmaTransport::HostSge> batch_buffers_;
  std::vector<std::uint64_t> batch_remote_offsets_;
  std::vector<std::uint8_t> batch_used_sources_;
  bool initialized_{false};
};

}  // namespace kvaf
