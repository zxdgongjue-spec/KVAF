#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime.h"
#include "remote_offload.h"
#include "hbm_read.h"

namespace kvaf {

struct BatchGetKernelProfile;

struct NpuRuntimeOptions {
  NpuRuntimeOptions(std::string kernel_object_path = "",
                    std::uint32_t token_bytes = 0,
                    std::uint32_t tokens_per_block = 0)
      : batch_get_kernel_object_path(std::move(kernel_object_path)),
        batch_get_token_bytes(token_bytes),
        batch_get_tokens_per_block(tokens_per_block) {}

  std::string batch_get_kernel_object_path;
  std::uint32_t batch_get_token_bytes;
  std::uint32_t batch_get_tokens_per_block;
  std::string batch_put_kernel_object_path;
  std::string batch_get_raw_addr_kernel_object_path;
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
  bool enable_cpu_offload{false};
  std::uint32_t cpu_candidate{UINT32_MAX}; // Pin the shared pool, or discover.
  std::uint32_t offload_timeout_ms{30000};
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
  Status GetOffloadRegion(OffloadRegionTicket*) const;
  // Local-source convenience path uses this same Core and shared DDR pool.
  // C++ callers retain input storage until ReleaseOffloadSource or Shutdown
  // succeeds, including after a completed batch (the registration is cached).
  Status Offload(const BatchPutRequest&, OffloadExecutionOptions = {});
  Status ReleaseOffloadSource(); // Releases the cached local registration after resolution.
  Status OffloadFrom(const std::vector<KeyId>& keys, const HbmReadyBatch&);
  Status ResolveOffload();
  bool pending_offload() const;
  // Diagnostic RPC: hashes current DDR after commit, outside Offload's data path.
  // Call before freeing/overwriting destination slots, the source's next Ready,
  // or source detach. Synchronous like Offload; not a historical checksum cache.
  Status LastOffloadChecksum(std::uint64_t*) const;

  Status AttachCpuSource(const CpuPullSourceDesc&, CpuPullSourceHandle*);
  Status PreparePut(const std::vector<KeyId>&, const std::vector<std::uint64_t>&, PutHandle*);
  Status SubmitCpuPull(const CpuPullSourceHandle&, const ReadyToken&, const PutHandle&,
                       Uint64ArrayView src, Uint64ArrayView dst, Uint64ArrayView lengths, CpuPullTask*);
  Status QueryCpuPull(const CpuPullTask&, CpuPullState*);
  Status WaitCpuPull(const CpuPullTask&, std::uint32_t timeout_ms, CpuPullState*);
  Status CommitPut(const PutHandle&);
  Status AbortPut(const PutHandle&);

  // Remote producer HBM: direct READ ranges, no AccessTable or DDR metadata.
  Status AttachHbm(const std::string& address, const HbmReadTicket&, LinkId*,
                   const std::string& local_eid = "");
  Status BatchGetRawAddr(LinkId, const HbmReadBatch&, MutableBufferView output,
                         const std::vector<std::uint64_t>& output_offsets, void* stream);
  Status BatchGetRawAddr(LinkId, const ReadyToken&, NpuInt64ArrayView src, NpuInt64ArrayView dst,
                         NpuInt64ArrayView lengths, MutableBufferView output, void* stream);
  Status ResolveRawAddr(LinkId); // Never resubmits DMA; retries control/cleanup only.
  Status DetachHbm(LinkId);
  bool pending_raw_addr() const;

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
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kvaf
