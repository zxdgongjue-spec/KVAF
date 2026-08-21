#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "runtime.h"

namespace kvaf {

struct NpuRuntimeOptions {
  std::string batch_get_kernel_object_path;
  std::string batch_put_kernel_object_path;
  std::uint32_t device_id{0U};
  std::uint32_t npu_core_count{1U};
  std::uint32_t jettys_per_core{2U};
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
  std::uint32_t npu_core_count() const;
  std::uint32_t jettys_per_core() const;

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
