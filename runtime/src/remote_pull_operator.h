#pragma once
#include <optional>

#include "cpu_pull_executor.h"
#include "peer_link.h"
#include "remote_offload.h"
#include "remote_pull_service.h"

namespace kvaf {
// A's dispatcher shares one Core/MetaManager/allocator. AIV and CPU Pull have
// separate connections but consume exactly the same RemoteRegionDesc.
class RemotePullOperator final : public CpuPullExecutor {
 public:
  explicit RemotePullOperator(std::shared_ptr<peer::PeerLink> link, std::uint32_t candidate = 0);
  Status Initialize(const RemoteRegionDesc&);
  Status CanShutdown() const override;
  Status Shutdown() override;
  Status CompleteOffload(bool committed) override;
  const OffloadRegionTicket& ticket() const { return ticket_; }
  Status LastChecksum(std::uint64_t*) const override;
  Status ValidateCpuSource(const CpuPullSourceDesc&) override;
  Status SubmitCpuPull(const CpuPullRequest&) override;
  Status SubmitCpuPullAndWait(const CpuPullRequest&, std::uint32_t, CpuPullState*);
  Status SubmitCpuPullAndWait(const CpuPullRequest&, std::uint32_t, CpuPullState*, CpuPullSubmitTiming*);
  Status QueryCpuPull(CpuPullState*) override;
  Status CancelCpuPull(const ReadyToken&) override;
  bool pending() const override { return active_ || pending_cancel_.has_value(); }
  Status SubmitPreparedCpuPullAndWait(const CpuPullRequest&, std::uint32_t, CpuPullState*, CpuPullSubmitTiming*) override;
 private:
  Status InspectTask(remote_pull::Command, CpuPullState*);
  Status BuildSpans(const CpuPullRequest&, std::vector<remote_pull::Span>*, bool prepared = false) const;
  Status SubmitImpl(const CpuPullRequest&, remote_pull::Command, std::uint32_t, CpuPullState*, CpuPullSubmitTiming*, bool prepared);
  std::uint32_t candidate_{};
  std::shared_ptr<peer::PeerLink> link_;
  OffloadRegionTicket ticket_;
  remote_pull::RegionInfo region_{}; // Complete reader-local DDR description.
  CpuPullRequest current_;
  std::optional<CpuPullRequest> pending_cancel_; // Unsubmitted token awaiting a definitive Cancel reply.
  bool active_{};
  bool closed_{};
  bool committed_{}, finished_{}, cancelling_{};
  std::uint64_t operation_{};
  mutable remote_pull::ControlChannel channel_;
};
}
