#pragma once
#include "access_operator.h"
namespace kvaf {
struct UmdkAccessOperatorConfig {
  std::string kernel_json;
  std::string device;
  std::uint32_t eid_index{UINT32_MAX};
  std::uint32_t workers{1U};
  std::shared_ptr<PeerLink> peer_link;
};
// Opt-in AICPU data plane. Existing AIV operators and public BatchPut inputs
// are unchanged; setup WRITE, BatchPut/Get and GatherRead share this session.
class UmdkAccessOperator final : public AccessOperator {
 public:
  explicit UmdkAccessOperator(UmdkAccessOperatorConfig config);
  ~UmdkAccessOperator() override;
  Status Initialize(const RemoteRegionDesc&) override;
  Status Shutdown() override;
  Status ExecuteBatchPut(const BatchPutExecRequest&) override;
  Status ExecuteBatchGet(const BatchGetExecRequest&) override;
  Status ExecuteGatherRead(const CpuGatherReadRequest&) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace kvaf
