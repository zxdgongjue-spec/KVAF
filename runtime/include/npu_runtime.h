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
  // Synchronous local-HBM -> shared-DDR convenience API. Success includes
  // metadata commit. Failed unresolved DMA retains input/slots and blocks reuse;
  // Shutdown queries the original task and retries cleanup without resubmitting.
  // Input registration is cached until a different source or Shutdown; native
  // callers must keep the input allocation alive for that registration lifetime.
  Status BatchCpuPull(const BatchPutRequest&, BatchPutExecutionOptions = {});
  bool pending_offload() const;
  // Diagnostic RPC: hashes current DDR after commit, outside CPU Pull's data path.
  // Call before freeing/overwriting destination slots, the source's next Ready,
  // or source detach. Synchronous; not a historical checksum cache.
  Status LastOffloadChecksum(std::uint64_t*) const;

  Status AttachCpuSource(const CpuPullSourceDesc&);
  Status PreparePut(const std::vector<KeyId>&, const std::vector<std::uint64_t>&, PutHandle*);
  // Resolves the attached source from ReadyToken. dst_addrs are absolute DDR addresses.
  Status SubmitCpuPull(const ReadyToken&, const PutHandle&,
                       Uint64ArrayView src_offsets, Uint64ArrayView dst_addrs,
                       Uint64ArrayView lengths);
  Status QueryCpuPull(const PutHandle&, CpuPullState*);
  Status WaitCpuPull(const PutHandle&, std::uint32_t timeout_ms, CpuPullState*);
  Status CommitPut(const PutHandle&);
  Status AbortPut(const PutHandle&);

  // Save the remote source connection, selected by ReadyToken on each READ.
  // No AccessTable or DDR metadata is used for this path.
  Status AttachHbm(const std::string& address, const HbmReadTicket&,
                   const std::string& local_eid = "");
  // As with BatchGet, output registrations are cached until detach/shutdown.
  // Keep output storage alive and at a stable address until then. Callers guarantee
  // valid, disjoint ranges and lengths in 1..UINT32_MAX within device READ limits.
  // Each request produces one WQE; validate_ranges checks lengths and bounds.
  Status BatchGetRawAddr(const ReadyToken&, NpuInt64ArrayView src, NpuInt64ArrayView dst,
                         NpuInt64ArrayView lengths, MutableBufferView output, void* stream,
                         HbmReadProfile* profile = nullptr, bool validate_ranges = false);
  Status ResolveRawAddr(const ReadyToken&); // Retries this batch's control/cleanup, never DMA.
  // Also cleans up a failed Attach using the original export ticket.
  Status DetachHbm(const HbmReadTicket&);
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
