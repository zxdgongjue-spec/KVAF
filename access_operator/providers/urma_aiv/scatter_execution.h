#pragma once

#include <array>
#include <mutex>
#include <utility>
#include "direct_hccp/direct_hccp_session.h"
#include "direct_hccp/target_channel.h"
#include "scatter_batch_get.h"
#include "sequential_batch_get.h"

namespace kvaf {
namespace peer { struct PeerRegion; }

struct ScatterResources final : ExecutionResourceOwner {
  explicit ScatterResources(AccessOperatorConfig c) : config(std::move(c)) {}
  Status Initialize(const RemoteRegionDesc&) override;
  Status Close() override;
  Status SetProfiling(bool);
  ScatterLaunchView View();

  std::mutex mutex;
  ExecutionViewState views;
  ScatterInvocation fallback{this};
  ScatterInvocation* active{&fallback};
  ScatterInvocation* eager{&fallback};

 private:
  Status InitializePush(const peer::PeerRegion&, bool cpu_gather);
  uint64_t StatusBytes() const { return config.npu_core_count * sizeof(KvafSequentialStatus); }
  uint64_t ProfileBytes() const { return config.npu_core_count * sizeof(ScatterBatchGetCoreProfile); }

  AccessOperatorConfig config;
  RemoteRegionDesc region{};
  ScatterProgress progress;
  bool initialized{};
  KvafScatterBatchGetArgs args{};
  void* workspace{};
  void* status_device{};
  void* status_host{};
  direct_hccp::DirectHccpApi api{};
  direct_hccp::DirectHccpSession* session{};
  void* context{};
  void* token{};
  void* memory_handle{};
  void* mailbox_import{};
  uint64_t mailbox_context{};
  std::array<std::unique_ptr<direct_hccp::TargetChannel>, cpu_push::kMaxJettyCount> channels;
  uint32_t payload_jettys{cpu_push::kJettyCount}, payload_jettys_per_core{};
  bool push_open{};
};

class ScatterBatchGet final : public SequentialBatchGet {
 public:
  explicit ScatterBatchGet(AccessOperatorConfig config);
  ~ScatterBatchGet() override;
  std::unique_ptr<ExecutionResourceOwner> TakeResources() override;
  Status Initialize(const RemoteRegionDesc& region) override;
  Status ExecuteBatchGet(const BatchGetExecRequest& exec) override;
  Status DrainGet() override;
  Status Shutdown() override;
  Status SetBatchGetKernelProfiling(bool enabled) override;
  Status CopyScatterProfiles(std::vector<ScatterBatchGetCoreProfile>* profiles) const override;
  std::unique_ptr<ExecutionState> CreateExecutionState() override;
  Status BindExecutionState(ExecutionState*) override;
  Status PrepareGetGraph(const GetGraphOptions&) override;
  Status CheckBatchGetStatus(void*) override;
  Status FinishGetGraph(void*, bool discard) override;
  bool GetGraphNeedsRecovery() const override;
 private:
  std::unique_ptr<ScatterResources> standalone_;
  ScatterResources* resources_;
};
}  // namespace kvaf
