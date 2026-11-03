#include "npu_runtime.h"
#include "npu_eid_discovery.h"
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
#include "hbm_read_client.h"
#endif
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
#include "remote_pull_operator.h"
#endif

#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "batch_get_abi.h"
#include "access_operator.h"
#include "peer_link.h"
#include "meta_manager.h"
#include "startup_bandwidth_probe.h"
#include "startup_selection.h"

namespace kvaf {
namespace {

Status Internal(std::string message) { return Status::Internal(std::move(message)); }

#if defined(KVAF_ENABLE_CPU_OFFLOAD)
class ScopedCpuPullTimer {
 public:
  explicit ScopedCpuPullTimer(double* elapsed) : elapsed_(elapsed) {
    if (elapsed_) start_ = std::chrono::steady_clock::now();
  }
  ~ScopedCpuPullTimer() {
    if (elapsed_) *elapsed_ += std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - start_).count();
  }
 private:
  double* elapsed_;
  std::chrono::steady_clock::time_point start_{};
};
#endif

class PeerDdrAllocator final : public DdrMemoryAllocator {
 public:
  explicit PeerDdrAllocator(std::shared_ptr<peer::PeerLink> link)
      : link_(std::move(link)) {}

  void SetCandidate(std::uint32_t candidate_id) { candidate_id_ = candidate_id; }
  bool has_allocation() const { return allocated_.base != 0U; }
  Status ReleaseAllocation() {
    if (!has_allocation()) return Status::OK();
    auto status = link_->ReleaseRegion(allocated_.base);
    if (status.ok()) allocated_ = {};
    return status;
  }

  Status Allocate(const RuntimeInitRequest& request, RemoteRegionDesc* region) override {
    if (link_ == nullptr || region == nullptr || request.ddr_device != DeviceKind::kCpu ||
        request.ddr_bytes == 0U || request.slot_size == 0U ||
        request.ddr_bytes % request.slot_size != 0U) {
      return Status::InvalidArgument("invalid peer DDR allocation request");
    }
    if (allocated_.base != 0U) return Status::InvalidState("peer DDR is already allocated");
    Status status = link_->AllocateRegion(request.ddr_bytes, request.slot_size, candidate_id_, &allocated_);
    if (!status.ok()) return status;
    *region = RemoteRegionDesc{allocated_.base, allocated_.length, request.slot_size, DeviceKind::kCpu};
    return Status::OK();
  }

  Status Release(const RemoteRegionDesc& region) override {
    if (allocated_.base == 0U || allocated_.base != region.base) {
      return Status::NotFound("peer DDR allocation was not found");
    }
    return ReleaseAllocation();
  }

 private:
  std::shared_ptr<peer::PeerLink> link_;
  std::uint32_t candidate_id_{0U};
  peer::PeerRegion allocated_{};
};

using npu_eid::NpuLocalEidCandidate;
using npu_eid::DiscoverNpuLocalEidCandidates;
using npu_eid::ParseConfiguredNpuLocalEids;

// Keep this aligned with the BatchPut AIV kernel. A real 128-KiB KV dump
// spreads roughly one 32-WQE post across every AIV core. The startup probe
// deliberately uses the same fan-out, but only 64 bytes per WQE: it validates
// every queue/CQ lane without retaining an 80+ MiB source buffer for process
// lifetime. The extra one-row post reaches each configured additional jetty.
constexpr std::uint64_t kBatchPutProbeWqesPerPost = 32U;
constexpr std::uint64_t kBatchPutProbeBytesPerRow = 64U;

std::uint64_t BatchPutStartupProbeRows(std::uint64_t ddr_bytes,
                                       std::uint64_t slot_size,
                                       std::uint32_t npu_core_count,
                                       std::uint32_t jettys_per_core) {
  if (ddr_bytes == 0U || slot_size == 0U || npu_core_count == 0U ||
      jettys_per_core == 0U) {
    return 0U;
  }
  const std::uint64_t rows_per_core =
      kBatchPutProbeWqesPerPost + static_cast<std::uint64_t>(jettys_per_core - 1U);
  const std::uint64_t desired_rows =
      static_cast<std::uint64_t>(npu_core_count) * rows_per_core;
  return std::min(ddr_bytes / slot_size, desired_rows);
}

}  // namespace

struct NpuRuntime::Impl {
  explicit Impl(NpuRuntimeOptions init_options) : options(std::move(init_options)) {
    config.batch_get_kernel_object_path = options.batch_get_kernel_object_path;
    config.batch_put_kernel_object_path = options.batch_put_kernel_object_path;
    config.npu_core_count = options.npu_core_count;
    config.jettys_per_core = options.jettys_per_core;
    config.batch_get_max_inflight_total = options.batch_get_max_inflight_total;
    // BatchPut is intentionally unchanged and keeps its existing two-Jetty
    // maximum when BatchGet selects three to sixty-four lanes.
    config.batch_put_jettys_per_core =
        std::min(options.jettys_per_core, std::uint32_t{2U});
  }

  void CreateCore() {
    meta_manager = CreateInMemoryMetaManager();
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
    if (!config.batch_get_kernel_object_path.empty()) access_operator = std::make_unique<DefaultAccessOperator>(config);
#endif
    AccessOperator* selected = access_operator.get();
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
    if (options.enable_cpu_offload) {
      remote_operator = std::make_unique<RemotePullOperator>(selected, peer_link);
      selected = remote_operator.get();
    }
#endif
    if (!selected) throw std::logic_error("no enabled NPU data plane");
    runtime_core = std::make_unique<RuntimeCore>(*meta_manager, *selected, *allocator);
  }

  void ResetDdrCore() {
    runtime_core.reset();
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
    remote_operator.reset();
#endif
    access_operator.reset();
    meta_manager.reset();
  }

  void ReleaseProbe() {
    bandwidth_probe.reset();
    if (probe_buffer != nullptr) {
      (void)aclrtFree(probe_buffer);
      probe_buffer = nullptr;
    }
  }

#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  std::unique_ptr<HbmReader> hbm_reader;
#endif
  NpuRuntimeOptions options;
  AccessOperatorConfig config;
  bool base_initialized{false};
  bool closing{false};
  std::unique_ptr<MetaManager> meta_manager;
  std::unique_ptr<DdrMemoryAllocator> allocator;
  PeerDdrAllocator* peer_allocator{nullptr};
  std::shared_ptr<peer::PeerLink> peer_link;
  std::unique_ptr<AccessOperator> access_operator;
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  std::unique_ptr<RemotePullOperator> remote_operator;
  PutHandle batch_pull_put;
  bool batch_pull_abort{};
  Status batch_pull_failure;

  // The original Put owns the request and the irreversible metadata decision.
  // Recovery and shutdown never resubmit READ or release the producer's source.
  Status FinishBatchCpuPull(std::uint32_t timeout_ms, CpuPullProfile* profile = nullptr) {
    if (!batch_pull_put.id()) return Status::InvalidState("no pending batch_cpu_pull to resolve");
    const auto pending = [](const Status& status) {
      return Status::InDoubt("CPU Pull unresolved; retain producer data and call resolve_cpu_pull: " + status.message());
    };
    try {
      if (!batch_pull_abort) {
        CpuPullState state = CpuPullState::InDoubt;
        Status status;
        {
          ScopedCpuPullTimer timer(profile ? &profile->wait_cpu_pull_us : nullptr);
          status = runtime_core->WaitCpuPull(batch_pull_put, timeout_ms, &state);
        }
        if (!status.ok()) {
          if (state != CpuPullState::Failed) return pending(status);
          batch_pull_abort = true;
          batch_pull_failure = status;
        }
      }
      Status status;
      {
        ScopedCpuPullTimer timer(profile ? &profile->commit_put_us : nullptr);
        status = batch_pull_abort ? runtime_core->AbortPut(batch_pull_put)
                                  : runtime_core->CommitPut(batch_pull_put);
      }
      if (!status.ok()) return pending(status);
      const auto result = batch_pull_abort ? batch_pull_failure : Status::OK();
      batch_pull_put = {};
      batch_pull_abort = false;
      batch_pull_failure = Status::OK();
      return result;
    } catch (const std::exception& e) {
      return pending(Status::Internal(e.what()));
    }
  }
#endif
  std::unique_ptr<RuntimeCore> runtime_core;
  void* probe_buffer{nullptr};
  std::unique_ptr<detail::StartupBandwidthProbe> bandwidth_probe;
};

NpuRuntime::NpuRuntime(NpuRuntimeOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}

Status NpuRuntime::Initialize() {
  if (impl_->closing) return Status::InvalidState("Runtime shutdown is incomplete; retry shutdown");
  if (initialized()) return Status::OK();
  const bool get = !impl_->config.batch_get_kernel_object_path.empty();
  const bool put = !impl_->config.batch_put_kernel_object_path.empty();
  const bool raw = !impl_->options.batch_get_raw_addr_kernel_object_path.empty();
#if !defined(KVAF_ENABLE_CPU_OFFLOAD)
  if (impl_->options.enable_cpu_offload) return Status::NotImplemented("CPU Offload was not built");
#endif
#if !defined(KVAF_ENABLE_AIV_BATCH_GET)
  if (get || raw) return Status::NotImplemented("AIV support was not built");
#endif
#if !defined(KVAF_ENABLE_AIV_BATCH_PUT)
  if (put) return Status::NotImplemented("AIV BatchPut support was not built");
#endif
  if (impl_->options.enable_cpu_offload && !impl_->options.offload_timeout_ms)
    return Status::InvalidArgument("offload timeout must be positive");
  if ((!get && !raw && !impl_->options.enable_cpu_offload) || (put && !get))
    return Status::InvalidArgument("configure raw READ, DDR AIV kernels or CPU Pull");
  if ((get || raw) && (impl_->options.npu_core_count == 0U ||
      impl_->options.npu_core_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES ||
      impl_->options.jettys_per_core == 0U ||
      impl_->options.jettys_per_core > KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE ||
      (impl_->options.batch_get_max_inflight_total != 0U &&
       impl_->options.batch_get_max_inflight_total < 2U * impl_->options.npu_core_count)))
    return Status::InvalidArgument("invalid NPU core/Jetty/inflight configuration");
  if (raw && !get && std::uint64_t{impl_->options.npu_core_count} * impl_->options.jettys_per_core > 4U)
    return Status::InvalidArgument("raw READ supports 1..4 total Jettys");
  if (get && (impl_->options.batch_get_token_bytes == 0U ||
      impl_->options.batch_get_tokens_per_block == 0U ||
      (impl_->options.adaptive_ub_device &&
       !impl_->options.peer_ub_device.empty()) ||
      (impl_->options.adaptive_ub_device &&
       (impl_->options.adaptive_ub_miss_count == 0U ||
        impl_->options.adaptive_ub_miss_count >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))))) {
    return Status::InvalidArgument("invalid DDR BatchGet configuration");
  }
  if (aclrtSetDevice(impl_->options.device_id) != ACL_SUCCESS) {
    return Internal("aclrtSetDevice failed for NpuRuntime");
  }
  impl_->base_initialized = true;
  return Status::OK();
}

NpuRuntime::~NpuRuntime() {
  if (impl_ != nullptr) {
    if (!Shutdown().ok()) { (void)impl_.release(); return; }
    impl_->runtime_core.reset();
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
    impl_->remote_operator.reset();
#endif
    impl_->access_operator.reset();
    impl_->ReleaseProbe();
  }
}

Status NpuRuntime::LinkTo(const LinkTarget& target) {
  if (target.address.empty()) return Status::InvalidArgument("NpuRuntime LinkTo requires peer host:port");
  if (impl_->closing) return Status::InvalidState("Runtime shutdown is incomplete; retry shutdown");
  if (ddr_initialized()) return Status::InvalidState("NpuRuntime LinkTo must run before initialize_ddr");
  if (impl_->peer_link != nullptr) return Status::InvalidState("NpuRuntime already has a PeerLink");
  if (impl_->config.batch_get_kernel_object_path.empty() && !impl_->options.enable_cpu_offload)
    return Status::InvalidArgument("configure DDR AIV kernels or CPU Pull before link_to");
  if (impl_->config.batch_put_kernel_object_path.empty() && !impl_->options.enable_cpu_offload) {
    return Status::InvalidArgument("PeerLink requires the AIV BatchPut kernel for RMA validation");
  }
  impl_->peer_link = std::make_shared<peer::PeerLink>(target.address);
  impl_->config.peer_link = impl_->peer_link;
  auto allocator = std::make_unique<PeerDdrAllocator>(impl_->peer_link);
  impl_->peer_allocator = allocator.get();
  impl_->allocator = std::move(allocator);
  return Status::OK();
}

Status NpuRuntime::Initialize(std::uint64_t ddr_bytes, std::uint64_t slot_size) {
  const auto status = Initialize();
  return status.ok() ? InitializeDdr(ddr_bytes, slot_size) : status;
}

Status NpuRuntime::InitializeDdr(std::uint64_t ddr_bytes, std::uint64_t slot_size) {
  const RuntimeInitRequest request{ddr_bytes, slot_size, DeviceKind::kCpu};
  if (!initialized()) return Status::NotInitialized("call initialize before initialize_ddr");
  if (impl_->closing) return Status::InvalidState("Runtime shutdown is incomplete; retry shutdown");
  if (ddr_initialized()) return Status::InvalidState("DDR has already been initialized");
  if (impl_->peer_link == nullptr || impl_->peer_allocator == nullptr) {
    return Status::InvalidState("NpuRuntime requires link_to before initialize_ddr");
  }
  if (ddr_bytes == 0U || slot_size == 0U || ddr_bytes % slot_size != 0U) {
    return Status::InvalidArgument("DDR size must be a nonzero multiple of slot_size");
  }

  if (impl_->peer_allocator->has_allocation())
    return Status::InDoubt("previous DDR cleanup is unresolved; retry shutdown_ddr before initialize_ddr");
  if (aclrtSetDevice(device_id()) != ACL_SUCCESS) return Internal("select NPU for DDR initialization");
  std::uint32_t candidate_count = 0U;
  std::uint32_t device_count = 0U;
  bool device_filter_applied = false;
  Status last = impl_->peer_link->GetCandidateCount(
      &candidate_count, &device_count, &device_filter_applied);
  if (!last.ok()) return last;
  if (candidate_count == 0U) {
    return Status::NotFound("peer Runtime returned no UMDK/EID candidates");
  }
  if (impl_->options.cpu_candidate != UINT32_MAX && impl_->options.cpu_candidate >= candidate_count)
    return Status::InvalidArgument("CPU candidate is outside the advertised list");
  if (impl_->config.batch_get_kernel_object_path.empty()) {
    // Card A can coordinate a card B source without loading any AIV binary.
    for (std::uint32_t candidate = 0; candidate < candidate_count; ++candidate) {
      if (impl_->options.cpu_candidate != UINT32_MAX && candidate != impl_->options.cpu_candidate) continue;
      impl_->peer_allocator->SetCandidate(candidate);
      impl_->runtime_core.reset();
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
      impl_->remote_operator.reset();
#endif
      impl_->access_operator.reset(); impl_->CreateCore();
      last = impl_->runtime_core->Initialize(request);
      if (last.ok() || last.code() == StatusCode::kInDoubt) return last;
      if (impl_->peer_allocator->has_allocation()) return Status::InDoubt("shared DDR allocation cleanup unresolved");
    }
    return last;
  }
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  std::vector<NpuLocalEidCandidate> npu_candidates;
  std::string discovery_error;
  const bool explicit_eids = !impl_->options.npu_eids.empty();
  if (explicit_eids) {
    if (!ParseConfiguredNpuLocalEids(impl_->options.device_id,
                                     impl_->options.npu_eids,
                                     &npu_candidates, &discovery_error)) {
      return Status::InvalidArgument(discovery_error);
    }
  } else if (!DiscoverNpuLocalEidCandidates(
                 impl_->options.device_id, &npu_candidates,
                 &discovery_error)) {
    return Status::NotFound(discovery_error);
  }
  const std::size_t worker_count =
      static_cast<std::size_t>(impl_->options.npu_core_count) *
      impl_->options.jettys_per_core;
  if (explicit_eids && npu_candidates.size() > worker_count) {
    return Status::InvalidArgument(
        "npu_eids count exceeds npu_core_count * jettys_per_core workers");
  }
  const bool fixed_peer_device = !impl_->options.peer_ub_device.empty();
  const bool has_n_side_allowlist =
      !impl_->options.adaptive_peer_ub_devices.empty();
  const bool tune_candidates = impl_->options.adaptive_ub_device &&
                               !device_filter_applied && has_n_side_allowlist &&
                               impl_->options.cpu_candidate == UINT32_MAX;
  if (impl_->options.adaptive_ub_device && !tune_candidates) {
    std::fprintf(stderr,
                 "[KVAF_AIV] UB_DEVICE_ADAPTIVE_SKIPPED device_id=%u "
                 "reason=%s\n",
                 impl_->options.device_id,
                 device_filter_applied ? "k_side_device_filter"
                                       : "no_n_side_candidates");
  }
  const detail::StartupProbeConfig probe_config{
      impl_->options.batch_get_token_bytes,
      impl_->options.batch_get_tokens_per_block,
      impl_->options.adaptive_ub_miss_count};
  detail::StartupProbeLayout probe_layout{};
  if (tune_candidates) {
    last = detail::BuildStartupProbeLayout(probe_config, &probe_layout);
    if (!last.ok()) return last;
  }
  // An explicit N-side list remains one strict multi-EID group. Automatically
  // discovered EIDs are connectivity fallbacks.
  const std::size_t npu_group_count = explicit_eids ? 1U : npu_candidates.size();
  std::vector<std::string> peer_candidate_device_names;
  std::vector<bool> peer_candidate_allowed;
  if (fixed_peer_device) {
    peer_candidate_allowed.assign(candidate_count, false);
    std::uint32_t fixed_candidate_count = 0U;
    for (std::uint32_t peer_candidate = 0U; peer_candidate < candidate_count;
         ++peer_candidate) {
      peer::PeerCandidateInfo info{};
      last = impl_->peer_link->GetCandidateInfo(peer_candidate, &info);
      if (!last.ok()) return last;
      if (info.candidate_id != peer_candidate || info.device_id >= device_count) {
        return Status::Internal("peer Runtime returned invalid candidate metadata");
      }
      if (info.device_name == impl_->options.peer_ub_device) {
        peer_candidate_allowed[peer_candidate] = true;
        ++fixed_candidate_count;
      }
    }
    if (fixed_candidate_count == 0U) {
      return Status::NotFound("peer Runtime did not advertise fixed UB device " +
                              impl_->options.peer_ub_device);
    }
  }
  auto initialize_candidate = [&](const detail::StartupCandidate& candidate,
                                  bool measure, double* cycles) -> Status {
    const std::uint32_t peer_candidate = candidate.peer;
    const RuntimeInitRequest candidate_request = measure
        ? RuntimeInitRequest{probe_layout.ddr_bytes,
                             probe_layout.block_bytes,
                             DeviceKind::kCpu}
        : request;
    const auto ddr_bytes = candidate_request.ddr_bytes;
    const auto slot_size = candidate_request.slot_size;
    impl_->peer_allocator->SetCandidate(peer_candidate);
    Status last;
    for (const std::size_t npu_group : std::array<std::size_t, 1U>{candidate.local_group}) {
      const std::size_t group_begin = explicit_eids ? 0U : npu_group;
      const std::size_t group_size = explicit_eids ? npu_candidates.size() : 1U;
      impl_->runtime_core.reset();
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
      impl_->remote_operator.reset();
#endif
      impl_->access_operator.reset();
      impl_->ReleaseProbe();
      impl_->config.peer_local_eids.clear();
      impl_->config.peer_local_eids.reserve(group_size);
      for (std::size_t offset = 0U; offset < group_size; ++offset) {
        const NpuLocalEidCandidate& candidate = npu_candidates[group_begin + offset];
        impl_->config.peer_local_eids.push_back(candidate.eid);
      }
      // BatchPut intentionally remains on the established single-EID path.
      // Multi-EID striping applies only to BatchGet and uses the vector above.
      impl_->config.peer_local_eid = impl_->config.peer_local_eids.front();
      impl_->CreateCore();
      last = impl_->runtime_core->Initialize(candidate_request);
      if (last.ok()) {
        const std::uint64_t probe_bytes =
            std::min<std::uint64_t>(slot_size, kBatchPutProbeBytesPerRow);
        const std::uint64_t probe_rows = BatchPutStartupProbeRows(
            ddr_bytes, slot_size, impl_->options.npu_core_count,
            impl_->config.batch_put_jettys_per_core);
        const std::uint64_t probe_total_bytes = probe_rows * probe_bytes;
        aclrtStream stream = nullptr;
        if (probe_bytes == 0U || probe_rows == 0U ||
            aclrtMalloc(&impl_->probe_buffer, probe_total_bytes,
                        ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
            impl_->probe_buffer == nullptr ||
            aclrtMemset(impl_->probe_buffer, probe_total_bytes, 0xA5U,
                        probe_total_bytes) != ACL_SUCCESS ||
            aclrtCreateStream(&stream) != ACL_SUCCESS || stream == nullptr) {
          last = Internal("failed to prepare peer candidate RMA probe");
        } else {
          std::vector<KeyId> probe_keys;
          std::vector<std::uint64_t> probe_offsets;
          std::vector<std::uint64_t> probe_lengths;
          probe_keys.reserve(static_cast<std::size_t>(probe_rows));
          probe_offsets.reserve(static_cast<std::size_t>(probe_rows));
          probe_lengths.reserve(static_cast<std::size_t>(probe_rows));
          for (std::uint64_t row = 0U; row < probe_rows; ++row) {
            // MetaManager maps each bounded DSA key directly to its slot.
            probe_keys.push_back(row);
            probe_offsets.push_back(row * probe_bytes);
            probe_lengths.push_back(probe_bytes);
          }
          last = impl_->runtime_core->BatchPut(
              BatchPutRequest{probe_keys,
                              ConstBufferView{impl_->probe_buffer,
                                              static_cast<std::size_t>(probe_total_bytes),
                                              DeviceKind::kNpu},
                              probe_offsets, probe_lengths},
              BatchPutExecutionOptions{stream});
          if (last.ok()) {
            last = impl_->runtime_core->BatchFree(BatchFreeRequest{probe_keys});
          }
        }
        if (stream != nullptr && aclrtDestroyStream(stream) != ACL_SUCCESS && last.ok()) {
          last = Internal("failed to destroy peer candidate probe stream");
        }
      }
      if (last.ok() && measure) {
        impl_->bandwidth_probe =
            std::make_unique<detail::StartupBandwidthProbe>(probe_config);
        last = impl_->bandwidth_probe->Run(*impl_->runtime_core, *static_cast<DefaultAccessOperator*>(impl_->access_operator.get()), cycles);
        if (last.ok()) {
          std::fprintf(stderr,
                       "[KVAF_AIV] UB_DEVICE_PROBE_RESULT device_id=%u "
                       "ub_device=%s miss_count=%u cycles=%.0f\n",
                       impl_->options.device_id,
                       peer_candidate_device_names[peer_candidate].c_str(),
                       probe_config.miss_count, *cycles);
          return last;
        }
      }
      if (last.ok()) {
        if (tune_candidates) {
          std::fprintf(stderr,
                       "[KVAF_AIV] UB_DEVICE_SELECTED device_id=%u "
                       "ub_device=%s cycles=%.0f\n",
                       impl_->options.device_id,
                       peer_candidate_device_names[peer_candidate].c_str(),
                       *cycles);
        } else if (fixed_peer_device) {
          std::fprintf(stderr,
                       "[KVAF_AIV] UB_DEVICE_REUSED device_id=%u ub_device=%s\n",
                       impl_->options.device_id,
                       impl_->options.peer_ub_device.c_str());
        }
        return Status::OK();
      }
      return last;
    }
    return last;
  };
  auto cleanup = [&]() -> Status {
    if (impl_->runtime_core->initialized()) {
      const Status status = impl_->runtime_core->Shutdown();
      if (!status.ok()) return status;
    }
    if (impl_->peer_allocator->has_allocation()) {
      return Status::Internal("peer candidate left DDR allocated");
    }
    // Registered source/output buffers outlive the transports even on errors.
    impl_->runtime_core.reset();
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
    impl_->remote_operator.reset();
#endif
    impl_->access_operator.reset();
    impl_->ReleaseProbe();
    return Status::OK();
  };
  if (!tune_candidates) {
    last = Status::NotFound("no CPU candidate matches the selected UB device");
    for (std::uint32_t peer_candidate = 0U; peer_candidate < candidate_count;
         ++peer_candidate) {
      if (impl_->options.cpu_candidate != UINT32_MAX && peer_candidate != impl_->options.cpu_candidate) continue;
      if (fixed_peer_device && !peer_candidate_allowed[peer_candidate]) continue;
      for (std::size_t npu_group = 0U; npu_group < npu_group_count; ++npu_group) {
        last = initialize_candidate(
            detail::StartupCandidate{peer_candidate, npu_group}, false, nullptr);
        if (last.ok() || last.code() == StatusCode::kInDoubt) return last;
        const Status released = cleanup();
        if (!released.ok()) return released;
      }
    }
    if (!last.ok() && fixed_peer_device) {
      return Status::Internal("failed to initialize fixed peer UB device " +
                              impl_->options.peer_ub_device + ": " +
                              last.message());
    }
    return last;
  }

  peer_candidate_device_names.resize(candidate_count);
  std::vector<std::vector<detail::StartupCandidate>> all_device_candidates(
      device_count);
  std::vector<std::string> all_device_names(device_count);
  for (std::uint32_t peer_candidate = 0U; peer_candidate < candidate_count;
       ++peer_candidate) {
    peer::PeerCandidateInfo info{};
    last = impl_->peer_link->GetCandidateInfo(peer_candidate, &info);
    if (!last.ok()) return last;
    if (info.candidate_id != peer_candidate || info.device_id >= device_count) {
      return Status::Internal("peer Runtime returned invalid candidate metadata");
    }
    peer_candidate_device_names[peer_candidate] = info.device_name;
    std::string& known_name = all_device_names[info.device_id];
    if (known_name.empty()) {
      known_name = info.device_name;
    } else if (known_name != info.device_name) {
      return Status::Internal("peer Runtime returned inconsistent candidate device grouping");
    }
    auto& group = all_device_candidates[info.device_id];
    for (std::size_t npu_group = 0U; npu_group < npu_group_count; ++npu_group) {
      group.push_back(detail::StartupCandidate{peer_candidate, npu_group});
    }
  }
  for (std::size_t device = 0U; device < all_device_candidates.size(); ++device) {
    if (all_device_candidates[device].empty() ||
        all_device_names[device].empty()) {
      return Status::Internal("peer Runtime returned incomplete candidate device grouping");
    }
  }
  std::vector<std::size_t> selected_device_indices;
  for (std::size_t device = 0U; device < all_device_names.size(); ++device) {
    if (std::find(impl_->options.adaptive_peer_ub_devices.begin(),
                  impl_->options.adaptive_peer_ub_devices.end(),
                  all_device_names[device]) !=
        impl_->options.adaptive_peer_ub_devices.end()) {
      selected_device_indices.push_back(device);
    }
  }
  if (selected_device_indices.empty()) {
    return Status::NotFound(
        "none of the script-configured adaptive UB devices were advertised by the peer");
  }
  std::vector<std::vector<detail::StartupCandidate>> device_candidates;
  std::vector<std::string> device_names;
  device_candidates.reserve(selected_device_indices.size());
  device_names.reserve(selected_device_indices.size());
  for (const std::size_t device : selected_device_indices) {
    device_candidates.push_back(std::move(all_device_candidates[device]));
    device_names.push_back(std::move(all_device_names[device]));
  }
  for (std::size_t device = 0U; device < device_candidates.size(); ++device) {
    std::fprintf(stderr,
                 "[KVAF_AIV] UB_DEVICE_PROBE_PLAN device_id=%u ub_device=%s "
                 "pairs=%zu token_bytes=%u tokens_per_block=%u miss_count=%u "
                 "aiv_cores=%u jettys_per_core=%u warmup=%u samples=%u\n",
                 impl_->options.device_id, device_names[device].c_str(),
                 device_candidates[device].size(), probe_config.token_bytes,
                 probe_config.tokens_per_block, probe_config.miss_count,
                 impl_->options.npu_core_count, impl_->options.jettys_per_core,
                 detail::kStartupWarmup, detail::kStartupSamples);
  }
  return detail::SelectStartupDeviceCandidate(
      device_candidates, initialize_candidate, cleanup);
#else
  return Status::NotImplemented("AIV support was not built");
#endif
}

RuntimeCore& NpuRuntime::core() {
  if (!ddr_initialized()) {
    throw std::logic_error("DDR is not initialized; call initialize_ddr before DDR operations");
  }
  return *impl_->runtime_core;
}
const RuntimeCore& NpuRuntime::core() const {
  if (!ddr_initialized()) {
    throw std::logic_error("DDR is not initialized; call initialize_ddr before DDR operations");
  }
  return *impl_->runtime_core;
}
std::uint32_t NpuRuntime::device_id() const { return impl_->options.device_id; }
std::uint32_t NpuRuntime::npu_core_count() const { return impl_->options.npu_core_count; }
std::uint32_t NpuRuntime::jettys_per_core() const { return impl_->options.jettys_per_core; }
const std::vector<std::string>& NpuRuntime::npu_eids() const {
  return impl_->options.npu_eids;
}
std::uint32_t NpuRuntime::batch_get_token_bytes() const {
  return impl_->options.batch_get_token_bytes;
}
std::uint32_t NpuRuntime::batch_get_tokens_per_block() const {
  return impl_->options.batch_get_tokens_per_block;
}

Status NpuRuntime::SetBatchGetKernelProfiling(bool enabled) {
  if (!ddr_initialized() || impl_->access_operator == nullptr) {
    return Status::NotInitialized(
        "initialize_ddr before configuring BatchGet profiling");
  }
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  return static_cast<DefaultAccessOperator*>(impl_->access_operator.get())->SetBatchGetKernelProfiling(enabled);
#else
  return Status::NotImplemented("AIV support was not built");
#endif
}

Status NpuRuntime::CopyBatchGetKernelProfiles(
    std::vector<BatchGetKernelProfile>* profiles) const {
  if (!ddr_initialized() || impl_->access_operator == nullptr) {
    return Status::NotInitialized(
        "initialize_ddr before reading BatchGet profiling");
  }
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  return static_cast<DefaultAccessOperator*>(impl_->access_operator.get())->CopyBatchGetKernelProfiles(profiles);
#else
  return Status::NotImplemented("AIV support was not built");
#endif
}

Status NpuRuntime::GetOffloadRegion(OffloadRegionTicket* ticket) const {
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  if (!ticket || !ddr_initialized() || !impl_->remote_operator) return Status::InvalidState("CPU Offload is not initialized");
  *ticket = impl_->remote_operator->ticket(); return Status::OK();
#else
  return Status::NotImplemented("CPU Offload was not built");
#endif
}
Status NpuRuntime::BatchCpuPull(const std::vector<KeyId>& keys, const ReadyToken& ready,
                                Uint64ArrayView src_offsets, const std::vector<std::uint64_t>& lengths,
                                std::uint32_t timeout_ms, CpuPullProfile* profile) {
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  if (profile) *profile = {};
  ScopedCpuPullTimer total(profile ? &profile->total_us : nullptr);
  {
    ScopedCpuPullTimer timer(profile ? &profile->validation_us : nullptr);
    if (!ddr_initialized() || !impl_->remote_operator) return Status::NotInitialized("initialize_ddr before batch_cpu_pull");
    if (impl_->closing) return Status::InvalidState("Runtime shutdown is incomplete; retry shutdown");
    if (pending_offload()) return Status::Busy("finish the existing Put or resolve_cpu_pull before starting a new batch");
    if (keys.empty() || keys.size() > remote_pull::kMaxSpans || !src_offsets.data ||
        keys.size() != src_offsets.size || keys.size() != lengths.size() ||
        reinterpret_cast<std::uintptr_t>(src_offsets.data) % alignof(std::uint64_t))
      return Status::InvalidArgument("batch_cpu_pull requires nonempty, equally sized keys, src_offsets and lengths");
    const auto* source = std::get_if<CpuPullSourceDesc>(&ready.source);
    if (!source || !ready.sequence || !source->bytes_for(ready.mr_id))
      return Status::InvalidArgument("batch_cpu_pull requires a CPU ReadyToken with a valid MR");
    auto status = impl_->remote_operator->ValidateCpuSource(*source);
    if (!status.ok()) return status;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      if (!lengths[i] || !offload::Contains(source->bytes_for(ready.mr_id), src_offsets.data[i], lengths[i]))
        return Status::InvalidArgument("batch_cpu_pull source range is outside the selected MR");
    }
  }
  Status status;
  {
    ScopedCpuPullTimer timer(profile ? &profile->prepare_put_us : nullptr);
    status = impl_->runtime_core->PreparePut(keys, lengths, &impl_->batch_pull_put);
  }
  if (!status.ok()) return status;
  impl_->batch_pull_abort = false;
  impl_->batch_pull_failure = Status::OK();
  try {
    std::vector<std::uint64_t> destinations;
    {
      ScopedCpuPullTimer timer(profile ? &profile->dst_offsets_us : nullptr);
      destinations.reserve(impl_->batch_pull_put.entries().size());
      const auto ddr_base = impl_->runtime_core->remote_region().base;
      for (const auto& entry : impl_->batch_pull_put.entries())
        destinations.push_back(entry.address - ddr_base);
    }
    {
      ScopedCpuPullTimer timer(profile ? &profile->submit_cpu_pull_us : nullptr);
      status = impl_->runtime_core->SubmitCpuPull(ready, impl_->batch_pull_put, src_offsets, destinations, lengths);
    }
  } catch (const std::exception& e) {
    status = impl_->batch_pull_put.submitted() ? Status::InDoubt(e.what()) : Status::Internal(e.what());
  }
  if (!status.ok()) {
    if (status.code() == StatusCode::kInDoubt)
      return Status::InDoubt("CPU Pull submission unresolved; retain producer data and call resolve_cpu_pull: " + status.message());
    impl_->batch_pull_abort = true;
    impl_->batch_pull_failure = status;
  }
  return impl_->FinishBatchCpuPull(timeout_ms == UINT32_MAX ? impl_->options.offload_timeout_ms : timeout_ms, profile);
#else
  return Status::NotImplemented("CPU Offload was not built");
#endif
}
Status NpuRuntime::ResolveCpuPull(std::uint32_t timeout_ms) {
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  if (!ddr_initialized()) return Status::NotInitialized("initialize_ddr before resolve_cpu_pull");
  return impl_->FinishBatchCpuPull(timeout_ms == UINT32_MAX ? impl_->options.offload_timeout_ms : timeout_ms);
#else
  return Status::NotImplemented("CPU Offload was not built");
#endif
}
bool NpuRuntime::pending_offload() const {
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  if (impl_->batch_pull_put.id()) return true;
#endif
  return impl_->runtime_core && impl_->runtime_core->pending_offload();
}
Status NpuRuntime::LastOffloadChecksum(std::uint64_t* checksum) const {
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  return impl_->remote_operator ? impl_->remote_operator->LastChecksum(checksum) : Status::NotInitialized("CPU Offload is not initialized");
#else
  return Status::NotImplemented("CPU Offload was not built");
#endif
}

Status NpuRuntime::ListenHbm(const std::string& address, const std::string& advertised,
                             const std::string& local_eid, HbmReceiverTicket* receiver) {
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  if (!initialized()) return Status::NotInitialized("initialize NpuRuntime before listen_hbm");
  if (impl_->closing) return Status::InvalidState("Runtime shutdown is incomplete; retry shutdown");
  if (!impl_->hbm_reader) {
    auto kernel = impl_->options.batch_get_raw_addr_kernel_object_path;
    if (kernel.empty()) {
      const auto& get = impl_->options.batch_get_kernel_object_path;
      if (get.empty()) return Status::InvalidArgument("configure batch_get_raw_addr_kernel_object_path for NPU Pull");
      const auto slash = get.find_last_of('/');
      kernel = (slash == std::string::npos ? std::string{} : get.substr(0, slash + 1)) +
          "kvaf_execute_batch_get_raw_addr_kernels.o";
    }
    impl_->hbm_reader = std::make_unique<HbmReader>(device_id(), npu_core_count(), jettys_per_core(),
        impl_->options.batch_get_max_inflight_total, kernel);
  }
  return impl_->hbm_reader->Listen(address, advertised, local_eid, receiver);
#else
  return Status::NotImplemented("raw-address READ requires AIV BatchGet build");
#endif
}
Status NpuRuntime::PreparePut(const std::vector<KeyId>& keys,const std::vector<std::uint64_t>& lengths,PutHandle* put) {
  if (!ddr_initialized()) return Status::NotInitialized("initialize_ddr before PreparePut");
  if (impl_->closing) return Status::InvalidState("Runtime shutdown is incomplete; retry shutdown");
  if (pending_offload()) return Status::Busy("finish the existing Put or resolve_cpu_pull before starting a new batch");
  return core().PreparePut(keys,lengths,put);
}
Status NpuRuntime::SubmitCpuPull(const ReadyToken& ready,const PutHandle& put,
                                Uint64ArrayView src_offsets,Uint64ArrayView dst_offsets,Uint64ArrayView lengths) {
  if (!ddr_initialized()) return Status::NotInitialized("initialize_ddr before CPU Pull");
  if (impl_->closing) return Status::InvalidState("Runtime shutdown is incomplete; retry shutdown");
  return core().SubmitCpuPull(ready,put,src_offsets,dst_offsets,lengths);
}
Status NpuRuntime::QueryCpuPull(const PutHandle& put,CpuPullState* state) {
  if (!ddr_initialized()) return Status::NotInitialized("DDR is not initialized");
  return core().QueryCpuPull(put,state);
}
Status NpuRuntime::WaitCpuPull(const PutHandle& put,std::uint32_t timeout,CpuPullState* state) {
  if (!ddr_initialized()) return Status::NotInitialized("DDR is not initialized");
  return core().WaitCpuPull(put,timeout,state);
}
Status NpuRuntime::CommitPut(const PutHandle& put) {
  if (!ddr_initialized()) return Status::NotInitialized("DDR is not initialized");
  return core().CommitPut(put);
}
Status NpuRuntime::AbortPut(const PutHandle& put) {
  if (!ddr_initialized()) return Status::NotInitialized("DDR is not initialized");
  return core().AbortPut(put);
}
Status NpuRuntime::BatchGetRawAddr(const ReadyToken& ready,NpuInt64ArrayView src,NpuInt64ArrayView dst,
                                  NpuInt64ArrayView lengths,MutableBufferView output,void* stream,HbmReadProfile* profile,
                                  bool validate_ranges) {
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  if (!initialized() || !impl_->hbm_reader) return Status::NotInitialized("attach source HBM before raw READ");
  if (impl_->closing) return Status::InvalidState("Runtime shutdown is incomplete; retry shutdown");
  return impl_->hbm_reader->Read(ready,src,dst,lengths,output,stream,profile,validate_ranges);
#else
  return Status::NotImplemented("raw-address READ requires AIV BatchGet build");
#endif
}
Status NpuRuntime::ResolveRawAddr(const ReadyToken& ready) {
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  if (impl_->hbm_reader) return impl_->hbm_reader->Resolve(ready);
#endif
  return Status::NotInitialized("no HBM reader");
}
Status NpuRuntime::CancelRawAddr(const ReadyToken& ready) {
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  if(!impl_->hbm_reader)return Status::NotInitialized("NPU Pull is not initialized");
  return impl_->hbm_reader->Cancel(ready);
#else
  return Status::NotImplemented("raw-address READ requires AIV BatchGet build");
#endif
}
bool NpuRuntime::pending_raw_addr() const {
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  return impl_->hbm_reader && impl_->hbm_reader->pending();
#else
  return false;
#endif
}

Status NpuRuntime::AcceptLink(const LinkTarget&) {
  return Status::NotImplemented("NpuRuntime does not accept links in the CPU--NPU deployment");
}

Status NpuRuntime::ReleaseLink(LinkId link_id) {
  if (link_id != 1U || impl_->peer_link == nullptr) return Status::NotFound("PeerLink was not found");
  const Status shutdown = ShutdownDdr();
  if (!shutdown.ok()) return shutdown;
  impl_->peer_link.reset();
  impl_->config.peer_link.reset();
  impl_->config.peer_local_eid = {};
  impl_->config.peer_local_eids.clear();
  impl_->peer_allocator = nullptr;
  impl_->allocator.reset();
  impl_->runtime_core.reset();
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  impl_->remote_operator.reset();
#endif
  impl_->access_operator.reset();
  return Status::OK();
}

Status NpuRuntime::ServeConnection(int) {
  return Status::NotImplemented("NpuRuntime has no incoming control service in this deployment");
}

Status NpuRuntime::Shutdown() {
  if (impl_ == nullptr) return Status::OK();
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  if (impl_->hbm_reader) {
    // HBM cleanup may partially close connections. Block new HBM work until
    // the whole shutdown finishes. A DDR-only CanShutdown rejection retains
    // the existing behavior: callers may finish/release the blocking batch.
    impl_->closing = true;
    auto status = impl_->hbm_reader->Shutdown();
    if (!status.ok()) return status;
    impl_->hbm_reader.reset();
  }
#endif
  auto status = ShutdownDdr();
  if (!status.ok()) return status;
  impl_->base_initialized = false;
  impl_->closing = false;
  return Status::OK();
}

Status NpuRuntime::ShutdownDdr() {
  if (impl_ == nullptr) return Status::OK();
  // Cleanup may be called while the application is working on another card.
  // Select the owning device for teardown, then restore the caller's device.
  struct RestoreDevice {
    int32_t previous{-1};
    ~RestoreDevice() { if (previous >= 0) (void)aclrtSetDevice(previous); }
  } restore;
  if (impl_->runtime_core || impl_->probe_buffer) {
    (void)aclrtGetDevice(&restore.previous);
    if (aclrtSetDevice(device_id()) != ACL_SUCCESS) return Internal("select NPU for DDR cleanup");
  }
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  if (impl_->batch_pull_put.id()) {
    // Do one query, without blocking shutdown on an active DMA. Once resolved,
    // source detach remains the producer's responsibility, even on the same NPU.
    const auto status = impl_->FinishBatchCpuPull(0);
    if (impl_->batch_pull_put.id()) return status;
  }
#endif
  Status status;
  if (impl_->runtime_core && impl_->runtime_core->initialized()) {
    status = impl_->runtime_core->Shutdown();
  } else if (impl_->peer_allocator && impl_->peer_allocator->has_allocation()) {
    // Core initialization may have failed and its DDR-release reply may have
    // been lost. Retire any partial operator before retrying that allocation.
    AccessOperator* selected = impl_->access_operator.get();
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
    if (impl_->remote_operator) selected = impl_->remote_operator.get();
#endif
    if (selected) status = selected->Shutdown();
    if (status.ok()) status = impl_->peer_allocator->ReleaseAllocation();
  }
  if (status.ok()) {
    impl_->ResetDdrCore();
    impl_->ReleaseProbe();
  }
  return status;
}

bool NpuRuntime::initialized() const {
  return impl_ != nullptr && impl_->base_initialized;
}

bool NpuRuntime::ddr_initialized() const {
  return initialized() && impl_->runtime_core != nullptr && impl_->runtime_core->initialized();
}

RuntimeCapabilities NpuRuntime::capabilities() const {
  return RuntimeCapabilities{!impl_->config.batch_get_kernel_object_path.empty(), !impl_->config.batch_put_kernel_object_path.empty(), false, true, impl_->options.enable_cpu_offload};
}

}  // namespace kvaf
