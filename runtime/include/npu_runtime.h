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

// Optional Host wall-clock timings for one synchronous BatchCpuPull call.
// Successful calls only; stages are inside total_us, never CPU-service clocks.
struct CpuPullProfile {
  double validation_us{}, prepare_put_us{}, dst_offsets_us{}, submit_cpu_pull_us{};
  double wait_cpu_pull_us{}, commit_put_us{}, total_us{};
};

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

// NPU role Runtime. HBM readers are independent of the optional DDR module
// (MetaManager, AccessOperator and allocator). Python only adapts Tensors.
class NpuRuntime final : public Runtime {
 public:
  explicit NpuRuntime(NpuRuntimeOptions options);
  ~NpuRuntime() override;

  NpuRuntime(const NpuRuntime&) = delete;
  NpuRuntime& operator=(const NpuRuntime&) = delete;

  // Base device initialization; no CPU service or DDR allocation is required.
  Status Initialize();
  // Compatibility entrypoint: base initialization followed by InitializeDdr.
  Status Initialize(std::uint64_t ddr_bytes, std::uint64_t slot_size);
  Status InitializeDdr(std::uint64_t ddr_bytes, std::uint64_t slot_size);
  // Keep HBM reader connections and the configured CPU target alive.
  Status ShutdownDdr();
  bool ddr_initialized() const;
  RuntimeCore& core();
  const RuntimeCore& core() const;
  Status GetOffloadRegion(OffloadRegionTicket*) const;
  // One source span per key, from the CPU source MR selected by ReadyToken. Prepare,
  // Submit, Wait and Commit are internal; success includes metadata publication.
  // The producer owns registration, synchronization, Ready and ReleaseBatch on
  // both local and remote NPUs. UINT32_MAX uses options.offload_timeout_ms.
  Status BatchCpuPull(const std::vector<KeyId>& keys, const ReadyToken&,
                      Uint64ArrayView src_offsets, const std::vector<std::uint64_t>& lengths,
                      std::uint32_t timeout_ms = UINT32_MAX, CpuPullProfile* profile = nullptr);
  // Finish the internally retained batch, never replay READ. Explicit PutHandle
  // transactions still use Query/Wait/Commit/Abort with their original handle.
  Status ResolveCpuPull(std::uint32_t timeout_ms = UINT32_MAX);
  bool pending_offload() const;
  // Diagnostic RPC: hashes current DDR after commit, outside CPU Pull's data path.
  // Call before freeing/overwriting destination slots, the source's next Ready,
  // or source detach. Synchronous; not a historical checksum cache.
  Status LastOffloadChecksum(std::uint64_t*) const;

  Status PreparePut(const std::vector<KeyId>&, const std::vector<std::uint64_t>&, PutHandle*);
  // Uses the source identity carried by ReadyToken. dst_offsets are byte offsets
  // from this Runtime's DDR base, within the given Put's reserved ranges.
  Status SubmitCpuPull(const ReadyToken&, const PutHandle&,
                       Uint64ArrayView src_offsets, Uint64ArrayView dst_offsets,
                       Uint64ArrayView lengths);
  Status QueryCpuPull(const PutHandle&, CpuPullState*);
  Status WaitCpuPull(const PutHandle&, std::uint32_t timeout_ms, CpuPullState*);
  Status CommitPut(const PutHandle&);
  Status AbortPut(const PutHandle&);

  // One receiver listener automatically attaches incoming producer sources.
  // Saved connections are selected by ReadyToken; no AccessTable/DDR is used.
  Status ListenHbm(const std::string& listen_address, const std::string& advertise_address,
                   const std::string& local_eid, HbmReceiverTicket*);
  // Fence this published batch against late READs and confirm cancellation to P.
  Status CancelRawAddr(const ReadyToken&);
  // As with BatchGet, output registrations are cached until detach/shutdown.
  // Keep output storage alive and at a stable address until then. Callers guarantee
  // valid, disjoint ranges and lengths in 1..UINT32_MAX within device READ limits.
  // Each request produces one WQE; validate_ranges checks lengths and bounds.
  Status BatchGetRawAddr(const ReadyToken&, NpuInt64ArrayView src, NpuInt64ArrayView dst,
                         NpuInt64ArrayView lengths, MutableBufferView output, void* stream,
                         HbmReadProfile* profile = nullptr, bool validate_ranges = false);
  Status ResolveRawAddr(const ReadyToken&); // Retries this batch's control/cleanup, never DMA.
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
