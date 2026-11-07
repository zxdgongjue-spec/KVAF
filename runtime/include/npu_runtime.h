#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "access_operator.h"
#include "runtime.h"
#include "ubmem/fused_context.h"
#include "ubmem/host_arena.h"
#include "npu_runtime_config.h"

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
  Transport transport{Transport::kUrma};
  std::uint32_t device_id{0U};
  std::uint32_t npu_core_count{2U};
  std::uint32_t jettys_per_core{2U};
  // Empty selects AIV. gather_read / cpu_push_scatter select AICPU Get and Put.
  // The AICPU kernel package is resolved from the CANN installation internally.
  std::string aicpu_batch_get_mode{};
  // Total submitted, unconfirmed READs per BatchGet launch; 0 disables control.
  std::uint32_t batch_get_max_inflight_total{2048U};
  // Optional strict local-EID list. Empty preserves automatic discovery and
  // selects one working EID. For AIV, multiple entries create simultaneous BatchGet
  // paths and stripe core/Jetty workers over them in list order. BatchPut
  // remains on the first EID. Both AICPU modes try explicit EIDs one at a time
  // as connectivity fallbacks; Get and Put share the selected EID.
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
  // Caller already initialized device RA (for example through HCCL). Borrow
  // it without calling RaInit again; the caller retains its lifetime.
  bool reuse_device_ra{false};
  UbmemPageMode ubmem_page_mode{UbmemPageMode::kThp};
  int ubmem_numa_node{-1};  // THP arena only; -1 inherits the caller policy.
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
  Transport transport() const;
  bool uses_aicpu() const;
  std::uint32_t npu_core_count() const;
  std::uint32_t jettys_per_core() const;
  const std::vector<std::string>& npu_eids() const;
  std::uint32_t batch_get_token_bytes() const;
  std::uint32_t batch_get_tokens_per_block() const;

  Status SetBatchGetKernelProfiling(bool enabled, bool detailed = false);
  Status CopyBatchGetKernelProfiles(
      std::vector<BatchGetKernelProfile>* profiles) const;

  // Acquire once before fused launches. KVAF registers this caller-owned HBM
  // output and exclusively leases its BatchGet queues; ordinary BatchGet and
  // another acquisition are rejected until release. All uses of the returned
  // context must run sequentially on one caller stream.
  Status AcquireFusedBatchGetContext(void* output, std::uint64_t output_bytes,
                                    FusedBatchGetContext* context);
  Status RegisterFusedBatchGetOutput(
      const FusedBatchGetContext& owner, void* output,
      std::uint64_t output_bytes, FusedBatchGetContext* view);
  Status ReleaseFusedBatchGetContext(const FusedBatchGetContext& context,
                                    void* stream);

  // UBmem leases the local arena and caller output storage, independently of
  // the URMA queue ABI. Launches use the acquire stream; release additionally
  // drains the device to cover graph replay streams. No replay after release.
  Status AcquireUbmemFusedContext(void* output, std::uint64_t output_bytes,
                                  void* stream, UbmemFusedContext* context);
  Status RegisterUbmemFusedOutput(const UbmemFusedContext& owner, void* output,
                                  std::uint64_t output_bytes,
                                  UbmemFusedContext* view);
  Status RegisterUbmemFusedAccessTable(const UbmemFusedContext& owner,
                                      const AccessTableHandle& table);
  Status ReleaseUbmemFusedContext(const UbmemFusedContext& owner, void* stream);

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
