#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "npu_service.h"
#include "runtime.h"
#include "cpu_gather.h"
#include "gather_output.h"
#include "gather_read_profile.h"

namespace kvaf {

struct CpuGatherReadRequest {
  std::vector<peer::GatherSpan> spans;
  std::uint64_t staging_address{0U};
  std::uint64_t source_descriptor_address{0U};
  MutableBufferView output;
  std::vector<GatherOutputSpan> output_spans;
  std::uint32_t sgl_size{kGatherMaxSglSize};
  void* stream{nullptr};
  GatherReadTiming* timing{nullptr};
  bool profile_detail{false};
  std::uint32_t sq_batch_wqes{8U};
  std::uint32_t read_task_spans{0U};
  GatherSqCopyMode sq_copy_mode{GatherSqCopyMode::kMemcpy};
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
  // Remote memory owner; device_id above always selects the local NPU.
  DeviceKind peer_device{DeviceKind::kCpu};
  std::uint32_t peer_device_id{0U};
};

// NPU role Runtime. It owns the current MetaManager, NPU AccessOperator and
// remote memory allocator, while Python remains responsible only for Tensor adaptation.
class NpuRuntime final : public Runtime {
 public:
  explicit NpuRuntime(NpuRuntimeOptions options);
  // HBM provider mode. The service owns HBM and accepts CPU or NPU Runtime links;
  // the AIV kernel configuration is not required in this mode.
  explicit NpuRuntime(NpuServiceOptions options);
  ~NpuRuntime() override;

  NpuRuntime(const NpuRuntime&) = delete;
  NpuRuntime& operator=(const NpuRuntime&) = delete;

  // Allocates CPU DRAM or NPU HBM according to options.peer_device.
  Status Initialize(std::uint64_t remote_bytes, std::uint64_t slot_size);
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
