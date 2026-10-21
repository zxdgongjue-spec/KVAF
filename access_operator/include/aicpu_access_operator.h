#pragma once
#include "access_operator.h"
namespace kvaf {
struct UmdkAccessOperatorConfig {
  std::string kernel_json;
  std::string device;
  std::uint32_t eid_index{UINT32_MAX};
  std::uint32_t workers{1U};
  std::shared_ptr<PeerLink> peer_link;
  std::uint32_t token_bytes{0U};
};
// Runtime adapter for the AICPU demos. BatchGet, BatchPut and setup WRITE
// share one UMDK session.
class UmdkAccessOperator final : public AccessOperator {
 public:
  explicit UmdkAccessOperator(UmdkAccessOperatorConfig config);
  ~UmdkAccessOperator() override;
  Status Initialize(const RemoteRegionDesc&) override;
  Status Shutdown() override;
  Status ExecuteBatchPut(const BatchPutExecRequest&) override;
  Status ExecuteBatchGet(const BatchGetExecRequest&) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace kvaf
