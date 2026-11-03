#pragma once
#include "access_operator.h"
#include "peer_link.h"
#include "remote_offload.h"

namespace kvaf {
// A's dispatcher shares one Core/MetaManager/allocator. AIV and CPU Pull have
// separate connections but consume exactly the same RemoteRegionDesc.
class RemotePullOperator final : public AccessOperator {
 public:
  RemotePullOperator(AccessOperator* aiv, std::shared_ptr<peer::PeerLink> link);
  Status Initialize(const RemoteRegionDesc&) override;
  Status CanShutdown() const override;
  Status Shutdown() override;
  Status ExecuteBatchGet(const BatchGetExecRequest&) override;
  Status ExecuteBatchPut(const BatchPutExecRequest&) override;
  Status CompleteOffload(bool committed) override;
  const OffloadRegionTicket& ticket() const { return ticket_; }
  Status LastChecksum(std::uint64_t*) const;
  Status ValidateCpuSource(const CpuPullSourceDesc&) override;
  Status SubmitCpuPull(const CpuPullRequest&) override;
  Status QueryCpuPull(CpuPullState*) override;
  bool pending() const { return active_; }
 private:
  AccessOperator* aiv_{};
  std::shared_ptr<peer::PeerLink> link_;
  OffloadRegionTicket ticket_;
  remote_pull::RegionInfo region_{}; // Complete reader-local DDR description.
  CpuPullRequest current_;
  bool active_{};
  bool committed_{};
};
}
