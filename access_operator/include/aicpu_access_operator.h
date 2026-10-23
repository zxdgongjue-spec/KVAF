#pragma once
#include "access_operator.h"
#include <array>
namespace kvaf {
struct UmdkAccessOperatorConfig {
  std::string kernel_json;
  std::string device;
  std::uint32_t eid_index{UINT32_MAX};
  std::uint32_t workers{1U};
  std::shared_ptr<PeerLink> peer_link;
  std::uint32_t token_bytes{0U};
  std::string batch_get_mode{"gather_read"};
  std::array<std::uint8_t, 16> local_eid{};
};
// Runtime adapter for the AICPU demos. gather_read and BatchPut share UMDK;
// cpu_push_scatter owns an independent Host-side direct HCCP target.
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
