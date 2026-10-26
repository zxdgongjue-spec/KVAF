#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime.h"
#include "peer_link.h"

namespace kvaf {

struct BatchGetKernelProfile;

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
  std::uint32_t device_id{0U};
  std::uint32_t npu_core_count{2U};
  std::uint32_t jettys_per_core{2U};
  // Total submitted, unconfirmed READs per BatchGet launch; 0 disables control.
  std::uint32_t batch_get_max_inflight_total{2048U};
  // Optional strict local-EID list. Empty preserves automatic discovery and
  // selects one working EID. Multiple entries create simultaneous BatchGet
  // paths and stripe core/Jetty workers over them in list order. BatchPut
  // remains on the first EID.
  std::vector<std::string> npu_eids{};
  // Opt-in startup BatchGet tuning over adaptive_peer_ub_devices. A K-side
  // explicit device uses the native first-working path without tuning.
  bool adaptive_ub_device{false};
  // BatchGet request count used by adaptive tuning. Performance scripts can
  // override it independently from the measured shape.
  std::uint32_t adaptive_ub_miss_count{2048U};
  // Optional N-side peer-device allowlist for adaptive tuning. It is ignored
  // when the K-side service applies an explicit --ub-device filter.
  std::vector<std::string> adaptive_peer_ub_devices{};
  // Optional strict K-side UB device selected by an earlier Runtime process.
  // Its EIDs remain connectivity fallbacks, but no bandwidth tuning is run.
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
  // Attach without reallocating memory or writing startup probe data. The
  // caller must coordinate slot ownership and publication across runtimes.
  Status InitializeShared(const peer::SharedRegionHandle& handle);
  Status GetPeerRegion(peer::PeerRegion* region) const;
  RuntimeCore& core();
  const RuntimeCore& core() const;

  std::uint32_t device_id() const;
  std::uint32_t npu_core_count() const;
  std::uint32_t jettys_per_core() const;
  const std::vector<std::string>& npu_eids() const;
  std::uint32_t batch_get_token_bytes() const;
  std::uint32_t batch_get_tokens_per_block() const;

  Status SetBatchGetKernelProfiling(bool enabled);
  Status CopyBatchGetKernelProfiles(
      std::vector<BatchGetKernelProfile>* profiles) const;

  Status LinkTo(const LinkTarget& target) override;
  Status AcceptLink(const LinkTarget& peer) override;
  Status ReleaseLink(LinkId link_id) override;
  Status ServeConnection(int connection_fd) override;
  Status Shutdown() override;
  bool initialized() const override;
  RuntimeCapabilities capabilities() const override;

 private:
  Status InitializeImpl(std::uint64_t ddr_bytes, std::uint64_t slot_size,
                        const peer::SharedRegionHandle* shared);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf
