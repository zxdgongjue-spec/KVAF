#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "npu_service.h"
#include "runtime.h"
#include "cpu_gather.h"

namespace kvaf {

struct CpuGatherReadRequest {
  std::vector<peer::GatherSpan> spans;
  std::uint64_t staging_address{0U};
  MutableBufferView output;
  void* stream{nullptr};
};

struct NpuRuntimeOptions {
  NpuRuntimeOptions(std::string kernel_object_path,
                    std::uint32_t token_bytes,
                    std::uint32_t tokens_per_block)
      : batch_get_kernel_object_path(std::move(kernel_object_path)),
        batch_get_token_bytes(token_bytes),
        batch_get_tokens_per_block(tokens_per_block) {}

  std::string batch_get_kernel_object_path;
  std::uint32_t batch_get_token_bytes;
  std::uint32_t batch_get_tokens_per_block;
  std::string batch_put_kernel_object_path;
  std::string aicpu_urma_kernel_json_path;
  std::uint32_t device_id{0U};
  std::uint32_t npu_core_count{1U};
  std::uint32_t jettys_per_core{2U};
};

// NPU role Runtime. It owns the current MetaManager, NPU AccessOperator and
// DDR allocator, while Python remains responsible only for Tensor adaptation.
class NpuRuntime final : public Runtime {
 public:
  explicit NpuRuntime(NpuRuntimeOptions options);
  // HBM provider mode. The service owns HBM and accepts CPU Runtime links;
  // the AIV kernel configuration is not required in this mode.
  explicit NpuRuntime(NpuServiceOptions options);
  ~NpuRuntime() override;

  NpuRuntime(const NpuRuntime&) = delete;
  NpuRuntime& operator=(const NpuRuntime&) = delete;

  Status Initialize(std::uint64_t ddr_bytes, std::uint64_t slot_size);
  Status InitializeHbmService();
  Status GatherRead(const CpuGatherReadRequest& request);
  RuntimeCore& core();
  const RuntimeCore& core() const;

  std::uint32_t device_id() const;
  std::uint32_t npu_core_count() const;
  std::uint32_t jettys_per_core() const;
  std::uint32_t batch_get_token_bytes() const;
  std::uint32_t batch_get_tokens_per_block() const;

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
