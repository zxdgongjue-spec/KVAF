#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "access_operator.h"
#include "cpu_urma_transport.h"

namespace kvaf {

// CPU data-plane AccessOperator. It reuses RuntimeCore's existing execution
// plans and translates each entry into one synchronous URMA operation.
class CpuUrmaAccessOperator final : public AccessOperator {
 public:
  explicit CpuUrmaAccessOperator(CpuUrmaTransportOptions options,
                                 std::shared_ptr<peer::PeerLink> link = nullptr);
  ~CpuUrmaAccessOperator() override;

  Status Initialize(const RemoteRegionDesc& remote_region) override;
  Status Shutdown() override;
  Status ExecuteBatchGet(const BatchGetExecRequest& request) override;
  Status ExecuteBatchPut(const BatchPutExecRequest& request) override;

 private:
  CpuUrmaTransportOptions options_;
  CpuUrmaTransport transport_;
  std::shared_ptr<peer::PeerLink> link_;
  RemoteRegionDesc region_{};
  std::mutex batch_mutex_;
  std::vector<CpuUrmaTransport::HostSge> batch_buffers_;
  std::vector<std::uint64_t> batch_remote_offsets_;
  bool initialized_{false};
};

}  // namespace kvaf
