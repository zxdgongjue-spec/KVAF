#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "kvaf/requests.h"
#include "kvaf/status.h"
#include "kvaf/types.h"

namespace kvaf {

struct PutAccessDesc;

struct BatchPutExecRequest {
  const BatchPutRequest* request{nullptr};
  const PutAccessDesc* put_desc{nullptr};
};

// KV Access Operator 是 Runtime 的数据面依赖。Runtime 管理远端 DDR
// 生命周期和 Meta 可见性；Operator 只负责初始化链路并执行同步搬运。
class AccessOperator {
 public:
  virtual ~AccessOperator() = default;

  virtual Status Initialize(const RemoteRegionDesc& remote_region) = 0;
  virtual Status Shutdown() = 0;

  virtual Status ExecuteBatchGet(const BatchGetExecRequest& request) = 0;
  virtual Status ExecuteBatchPut(const BatchPutExecRequest& request) = 0;
};

// Configuration for the current NPU-backed default implementation. Transport
// 不绑定 ACL stream；每次 BatchGet 都通过 request.execution 接收调用方 stream。
struct AccessOperatorConfig {
  std::string kernel_object_path;
  std::uint32_t npu_core_count{0};
  // A captured aclrtLaunchKernelWithHostArgs task can outlive the C++ call
  // that submitted it. Enable this for caller-managed asynchronous execution
  // when the host argument record must remain valid until shutdown.
  bool retain_unsynchronized_launch_args{false};
};

// Default synchronous NPU data-plane implementation.
//
// Initialize creates one reusable transport for Runtime-owned Host DDR.
// ExecuteBatchGet obtains a LocalBuffer from the caller-owned output in the
// request, launches the current BatchGet demo and waits for completion.
// ExecuteBatchPut currently uses synchronous ACL device-to-host copies as a
// temporary NPU mock. It preserves the Runtime transaction contract but is
// not the eventual BatchPut data-plane implementation. This class is built
// only with KVAF_ENABLE_AIV_BATCH_GET=ON.
class DefaultAccessOperator final : public AccessOperator {
 public:
  explicit DefaultAccessOperator(AccessOperatorConfig config);
  ~DefaultAccessOperator() override;

  DefaultAccessOperator(const DefaultAccessOperator&) = delete;
  DefaultAccessOperator& operator=(const DefaultAccessOperator&) = delete;

  Status Initialize(const RemoteRegionDesc& remote_region) override;
  Status Shutdown() override;
  Status ExecuteBatchGet(const BatchGetExecRequest& request) override;
  Status ExecuteBatchPut(const BatchPutExecRequest& request) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf
