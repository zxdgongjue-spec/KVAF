#pragma once
#include "access_operator.h"
#include "peer_link.h"
#include "remote_offload.h"
#include "remote_pull_service.h"

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
  Status SubmitCpuPullAndWait(const CpuPullRequest&, std::uint32_t, CpuPullState*) override;
  Status SubmitCpuPullAndWait(const CpuPullRequest&, std::uint32_t, CpuPullState*, CpuPullSubmitTiming*) override;
  Status QueryCpuPull(CpuPullState*) override;
  Status CancelCpuPull(const ReadyToken&) override;
  bool pending() const { return active_; }
 private:
  Status InspectTask(remote_pull::Command, CpuPullState*);
  Status BuildSpans(const CpuPullRequest&, std::vector<remote_pull::Span>*, bool prepared = false) const;
  Status SubmitPreparedCpuPullAndWait(const CpuPullRequest&, std::uint32_t, CpuPullState*, CpuPullSubmitTiming*) override;
  Status SubmitImpl(const CpuPullRequest&, remote_pull::Command, std::uint32_t, CpuPullState*, CpuPullSubmitTiming*, bool prepared);
  AccessOperator* aiv_{};
  std::shared_ptr<peer::PeerLink> link_;
  OffloadRegionTicket ticket_;
  remote_pull::RegionInfo region_{}; // Complete reader-local DDR description.
  CpuPullRequest current_;
  bool active_{};
  bool committed_{}, finished_{}, cancelling_{};
  std::uint64_t operation_{};
  mutable remote_pull::ControlChannel channel_;
};
}
