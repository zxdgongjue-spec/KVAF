#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime.h"
#include "npu_runtime_config.h"

namespace kvaf {

struct BatchGetKernelProfile;

struct NpuRuntimeOptions : ExecutionPolicy {
  // Providers are discovered from the build and prepared on first use.
  // Defaults select an algorithm; resource entries do not pre-enable providers.
  // Unspecified defaults prefer an available implementation for transport_mode.
  std::vector<std::pair<PutExecutionConfig, ExecutionResources>> put_resources;
  std::vector<std::pair<GetExecutionConfig, ExecutionResources>> get_resources;
  NpuRuntimeOptions(std::uint32_t token_bytes,
                    std::uint32_t tokens_per_block)
      : batch_get_token_bytes(token_bytes),
        batch_get_tokens_per_block(tokens_per_block) {}

  std::uint32_t batch_get_token_bytes;
  std::uint32_t batch_get_tokens_per_block;
  std::uint32_t device_id{0U};
  // Optional strict local-EID list. Empty uses the first automatically discovered
  // candidate. For AIV, multiple entries create simultaneous BatchGet
  // paths and stripe core/Jetty workers over them in list order. BatchPut
  // remains on the first EID. AICPU uses a single EID. AIV Get validates this
  // list against its own resource budget when that provider is first prepared.
  std::vector<std::string> npu_eids{};
  // Optional strict K-side UB device. No startup bandwidth tuning is run.
  std::string peer_ub_device{};
};

// NPU role Runtime. It owns the current MetaManager, NPU AccessOperator and
// DDR allocator, while Python remains responsible only for Tensor adaptation.
class NpuRuntime final : public Runtime {
 public:
  explicit NpuRuntime(NpuRuntimeOptions options);
  ~NpuRuntime() override;

  NpuRuntime(const NpuRuntime&) = delete;
  NpuRuntime& operator=(const NpuRuntime&) = delete;

  Status Initialize(std::uint64_t ddr_bytes, std::uint64_t slot_size);
  RuntimeCore& core();
  const RuntimeCore& core() const;

  std::uint32_t device_id() const;
  std::optional<GetExecutionConfig> default_get() const;
  const std::vector<std::string>& npu_eids() const;
  std::uint32_t batch_get_token_bytes() const;
  std::uint32_t batch_get_tokens_per_block() const;

  Status Warmup(PutExecutionConfig, const ExecutionWarmup& = {});
  Status Warmup(GetExecutionConfig, const ExecutionWarmup& = {});
  Status DrainGet(BatchGetHostTiming* = nullptr);
  Status Recover(PutExecutionConfig);
  Status Recover(GetExecutionConfig);

  Status SetBatchGetKernelProfiling(bool enabled, bool detailed = false,
                                    std::optional<GetExecutionConfig> config = std::nullopt);
  // Outside capture: synchronize the execution stream and report retained
  // AICPU protocol errors, including errors from earlier graph replays.
  Status CheckBatchGetStatus(void* stream,
                            std::optional<GetExecutionConfig> config = std::nullopt,
                            std::uint64_t graph_lease = 0);
  Status CopyBatchGetKernelProfiles(
      std::vector<BatchGetKernelProfile>* profiles,
      std::optional<GetExecutionConfig> config = std::nullopt,
      std::uint64_t graph_lease = 0) const;

  Status LinkTo(const LinkTarget& target) override;
  Status AcceptLink(const LinkTarget& peer) override;
  Status ReleaseLink(LinkId link_id) override;
  Status ServeConnection(int connection_fd) override;
  Status Shutdown() override;
  bool initialized() const override;
  RuntimeCapabilities capabilities() const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf
