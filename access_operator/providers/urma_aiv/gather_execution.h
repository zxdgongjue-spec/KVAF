#pragma once

#include <mutex>
#include <utility>
#include "gather_batch_get.h"
#include "sequential_batch_get.h"

namespace kvaf {

struct GatherResources final : ExecutionResourceOwner {
  explicit GatherResources(AccessOperatorConfig c) : config(std::move(c)) {}
  Status Initialize(const RemoteRegionDesc&) override;
  Status Close() override;
  Status SetProfiling(bool);
  GatherLaunchView View();

  std::mutex mutex;
  ExecutionViewState views;
  GatherInvocation fallback{this};
  GatherInvocation* active{&fallback};
  GatherInvocation* eager{&fallback};

 private:
  Status InitializeGather();
  uint64_t StatusBytes() const { return config.npu_core_count * sizeof(KvafSequentialStatus); }
  uint64_t ProfileBytes() const { return config.npu_core_count * sizeof(KvafGatherBatchGetCoreProfile); }

  AccessOperatorConfig config;
  RemoteRegionDesc region{};
  GatherProgress progress;
  bool initialized{};
  KvafGatherBatchGetArgs args{};
  void* workspace{};
  void* status_device{};
  void* status_host{};
  void* queues{};
  void* profile_device{};
  void* profile_host{};
  bool profiling{}, check_async_status{}, gather_enabled{};
  GatherOutputRegistration outputs;
  KvafBatchGetLocalBuffer* workspace_buffer{};
};

class GatherBatchGet final : public SequentialBatchGet {
 public:
  explicit GatherBatchGet(AccessOperatorConfig config);
  ~GatherBatchGet() override;
  std::unique_ptr<ExecutionResourceOwner> TakeResources() override;
  Status Initialize(const RemoteRegionDesc& region) override;
  Status ExecuteBatchGet(const BatchGetExecRequest& exec) override;
  Status DrainGet() override;
  Status Shutdown() override;
  Status SetBatchGetKernelProfiling(bool enabled) override;
  Status CopyGatherBatchGetProfile(GatherBatchGetProfile* profile) const override;
  std::unique_ptr<ExecutionState> CreateExecutionState() override;
  Status BindExecutionState(ExecutionState*) override;
  Status PrepareGetGraph(const GetGraphOptions&) override;
  Status CheckBatchGetStatus(void*) override;
  Status FinishGetGraph(void*, bool discard) override;
  bool GetGraphNeedsRecovery() const override;
 private:
  std::unique_ptr<GatherResources> standalone_;
  GatherResources* resources_;
};
}  // namespace kvaf
