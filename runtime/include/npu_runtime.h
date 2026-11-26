#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "runtime.h"
#include "npu_runtime_config.h"
#include "scatter_batch_get_profile.h"
#include "remote_offload.h"
#include "hbm_read.h"

namespace kvaf {

struct BatchGetKernelProfile;
struct GatherBatchGetProfile;

struct CpuPullOptions {
  std::uint32_t timeout_ms{30000};
};
struct NpuPullOptions {
  std::string kernel_path;
  std::uint32_t cores{2}, jettys{2}, max_inflight_total{0};
  // Per-source cache capacity for distinct output address/size registrations.
  std::uint32_t max_output_bindings{64};
  bool diagnostics{false};
};
struct NpuRuntimeOptions : ExecutionPolicy {
  // Providers are discovered from the build and prepared on first use.
  // Defaults select an algorithm; resource entries do not pre-enable providers.
  // Unspecified defaults prefer an available implementation for transport_mode.
  std::vector<std::pair<PutExecutionConfig, ExecutionResources>> put_resources;
  std::vector<std::pair<GetExecutionConfig, ExecutionResources>> get_resources;
  NpuRuntimeOptions() = default;
  NpuRuntimeOptions(std::uint32_t token_bytes,
                    std::uint32_t tokens_per_block)
      : batch_get_token_bytes(token_bytes),
        batch_get_tokens_per_block(tokens_per_block) {}

  std::uint32_t batch_get_token_bytes{0};
  std::uint32_t batch_get_tokens_per_block{0};
  std::uint32_t device_id{0U};
  // Optional strict local-EID list. Empty uses the first automatically discovered
  // candidate. For AIV, multiple entries create simultaneous BatchGet
  // paths and stripe core/Jetty workers over them in list order. BatchPut
  // remains on the first EID. AICPU uses a single EID. AIV Get validates this
  // list against its own resource budget when that provider is first prepared.
  std::vector<std::string> npu_eids{};
  // A device name pins the peer. An ordered CSV selects remote AIV/Gather Get FEs.
  // Empty selects automatically; KVAF_DEFAULT_PEER_FE_COUNT controls remote Get.
  std::string peer_ub_device{};
  std::uint32_t aiv_gather_sgl_size{6U};
  bool adaptive_ub_device{false};
  std::uint32_t adaptive_ub_miss_count{2048U};
  std::vector<std::string> adaptive_peer_ub_devices;
  CpuPullOptions cpu_pull_options;
  NpuPullOptions npu_pull_options;
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
  // Effective Get budget after DDR peer discovery, including caller overrides.
  ExecutionResources GetResources(GetExecutionConfig config) const;
  RuntimeCore& core();
  const RuntimeCore& core() const;
  // A local CPU route uses the source card's pair_info. Omission on the first
  // call selects the runtime's device. Later calls reuse the same DDR binding.
  Status GetOffloadRegion(OffloadRegionTicket*,
                          std::optional<std::uint32_t> source_physical_device_id = std::nullopt);
  // One source span per key, from the CPU source MR selected by ReadyToken. Prepare,
  // Submit-and-wait uses one persistent control RPC; success includes metadata publication.
  // The producer owns registration, synchronization, Ready and ReleaseBatch on
  // both local and remote NPUs. UINT32_MAX uses options.cpu_pull_options.timeout_ms.
  Status BatchCpuPull(const std::vector<KeyId>& keys, const ReadyToken&,
                      Uint64ArrayView src_offsets, const std::vector<std::uint64_t>& lengths,
                      std::uint32_t timeout_ms = UINT32_MAX, CpuPullProfile* profile = nullptr);
  // Finish the internally retained batch, never replay READ. Explicit PutHandle
  // transactions still use Query/Wait/Commit/Abort with their original handle.
  Status ResolveCpuPull(std::uint32_t timeout_ms = UINT32_MAX);
  // Fence an unused batch or abort its original Put. Retain both ends and retry on InDoubt.
  Status CancelCpuPull(const ReadyToken&);
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
  // Fence future Host submissions; queued work/graphs still need caller completion.
  Status CancelRawAddr(const ReadyToken&);
  // As with BatchGet, output registrations are cached until detach/shutdown.
  // Keep output storage alive and at a stable address until then. Callers guarantee
  // valid, disjoint ranges and lengths in 1..UINT32_MAX within device READ limits.
  // Enqueue one WQE per request on the caller's stream, like ordinary BatchGet.
  // Default success means submitted, not completed. Warm up registrations before
  // graph capture; keep metadata/output/source and communication resources alive
  // until all work finishes and captured graphs are destroyed. The caller owns
  // synchronization and producer reuse. Serialize work on each source connection,
  // including across streams. Detach/shutdown does not synchronize or destroy
  // application graphs. Optional synchronize waits on this stream.
  // validate_ranges checks lengths/bounds on device; no Host status readback.
  Status BatchGetRawAddr(const ReadyToken&, NpuInt64ArrayView src, NpuInt64ArrayView dst,
                         NpuInt64ArrayView lengths, MutableBufferView output, void* stream,
                         HbmReadProfile* profile = nullptr, bool validate_ranges = false,
                         bool synchronize = false);
  // Reports quarantined submission/wait failures; never replays READ or notifies.
  Status ResolveRawAddr(const ReadyToken&);
  // Failure quarantine only, not a completion query for normally queued work.
  bool pending_raw_addr() const;

  std::uint32_t device_id() const;
  std::optional<GetExecutionConfig> default_get() const;
  const std::vector<std::string>& npu_eids() const;
  std::uint32_t batch_get_token_bytes() const;
  std::uint32_t batch_get_tokens_per_block() const;

  Status Warmup(PutExecutionConfig, const ExecutionWarmup& = {});
  Status Warmup(GetExecutionConfig, const ExecutionWarmup& = {});
  Status DrainGet();
  Status Recover(PutExecutionConfig);
  Status Recover(GetExecutionConfig);

  Status SetScatterProfiling(bool enabled);
  Status CopyScatterProfiles(std::vector<ScatterBatchGetCoreProfile>* profiles) const;
  Status SetGatherBatchGetProfiling(bool enabled);
  Status CopyGatherBatchGetProfile(GatherBatchGetProfile* profile) const;
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
