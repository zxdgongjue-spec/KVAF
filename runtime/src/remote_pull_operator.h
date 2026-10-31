#pragma once
#include "access_operator.h"
#include "peer_link.h"
#include "remote_offload.h"
#include <unordered_map>

namespace kvaf {
// A's dispatcher shares one Core/MetaManager/allocator. AIV and CPU Pull have
// separate connections but consume exactly the same RemoteRegionDesc.
class RemotePullOperator final : public AccessOperator {
 public:
  RemotePullOperator(AccessOperator* aiv, std::shared_ptr<peer::PeerLink> link, std::uint32_t timeout_ms);
  Status Initialize(const RemoteRegionDesc&) override;
  Status CanShutdown() const override;
  Status Shutdown() override;
  Status ExecuteBatchGet(const BatchGetExecRequest&) override;
  Status ExecuteBatchPut(const BatchPutExecRequest&) override;
  Status ExecuteRemoteOffload(const RemoteOffloadExecRequest&) override;
  Status ResolveOffload() override;
  Status CompleteOffload(bool committed) override;
  // Called before PreparePut. Duplicate Ready messages return their original
  // outcome without reserving new slots or posting a second READ.
  Status Validate(const RemoteOffloadRequest&, bool* duplicate);
  const OffloadRegionTicket& ticket() const { return ticket_; }
  Status LastChecksum(std::uint64_t*) const;
  bool pending() const { return active_; }
 private:
  struct Record { RemoteOffloadRequest request; Status result; };
  AccessOperator* aiv_{};
  std::shared_ptr<peer::PeerLink> link_;
  OffloadRegionTicket ticket_;
  std::uint32_t timeout_ms_;
  RemoteOffloadRequest current_;
  bool active_{};
  std::unordered_map<std::uint64_t, Record> records_;
};
}
