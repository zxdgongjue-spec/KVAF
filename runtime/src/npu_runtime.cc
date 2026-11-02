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
  std::unique_ptr<MetaManager> meta_manager;
  std::unique_ptr<DdrMemoryAllocator> allocator;
  PeerDdrAllocator* peer_allocator{nullptr};
  std::shared_ptr<peer::PeerLink> peer_link;
  std::unique_ptr<AccessOperator> access_operator;
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  std::unique_ptr<RemotePullOperator> remote_operator;
  std::unique_ptr<CpuPullHbmSource> local_source;
  ConstBufferView local_input{};
  PutHandle local_put;
  ReadyToken local_ready;
  bool local_commit{}, local_retire{};

  // No new submission is performed here. Shutdown can finish an acknowledged
  // decision or abort a failed call only after the original DMA is terminal.
  Status FinishLocal() {
    if (!local_put.id()) return Status::OK();
    auto status = local_commit ? runtime_core->CommitPut(local_put) : runtime_core->AbortPut(local_put);
    if (!status.ok()) return Status::InDoubt("local CPU Pull is unresolved; retain source and DDR: " + status.message());
    if (local_source && local_source->frozen()) {
      status = local_commit ? local_source->ReleaseBatch(local_ready) : local_source->CancelBatch(local_ready);
      if (!status.ok()) return Status::InDoubt("local CPU Pull source acknowledgement pending: " + status.message());
    }
    if (local_source && (local_retire || !local_source->registered())) {
      status = local_source->Shutdown();
      if (!status.ok()) return Status::InDoubt("local CPU Pull source cleanup pending: " + status.message());
      local_source.reset(); local_input = {};
    }
    local_put = {}; local_ready = {}; local_commit = local_retire = false;
    return Status::OK();
  }
#endif
  std::unique_ptr<RuntimeCore> runtime_core;
  void* probe_buffer{nullptr};
  std::unique_ptr<detail::StartupBandwidthProbe> bandwidth_probe;
};

NpuRuntime::NpuRuntime(NpuRuntimeOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {
#if !defined(KVAF_ENABLE_CPU_OFFLOAD)
  if (impl_->options.enable_cpu_offload) throw std::invalid_argument("CPU Offload was not built");
#endif
#if !defined(KVAF_ENABLE_AIV_BATCH_GET)
  if (!impl_->config.batch_get_kernel_object_path.empty()) throw std::invalid_argument("AIV support was not built");
#endif
  if (!impl_->options.offload_timeout_ms) throw std::invalid_argument("offload timeout must be positive");
  if (impl_->config.batch_get_kernel_object_path.empty()) {
    if (!impl_->options.enable_cpu_offload || !impl_->config.batch_put_kernel_object_path.empty())
      throw std::invalid_argument("configure AIV kernels or enable CPU Offload");
  } else if (impl_->options.npu_core_count == 0U ||
      impl_->options.npu_core_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES ||
      impl_->options.jettys_per_core == 0U ||
      impl_->options.jettys_per_core > KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE ||
      (impl_->options.batch_get_max_inflight_total != 0U &&
       impl_->options.batch_get_max_inflight_total < 2U * impl_->options.npu_core_count) ||
      impl_->options.batch_get_token_bytes == 0U ||
      impl_->options.batch_get_tokens_per_block == 0U ||
      (impl_->options.adaptive_ub_device &&
       !impl_->options.peer_ub_device.empty()) ||
      (impl_->options.adaptive_ub_device &&
       (impl_->options.adaptive_ub_miss_count == 0U ||
        impl_->options.adaptive_ub_miss_count >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())))) {
    throw std::invalid_argument("invalid NPU Runtime configuration");
  }
  if (aclrtSetDevice(impl_->options.device_id) != ACL_SUCCESS) {
    throw std::runtime_error("aclrtSetDevice failed for NpuRuntime");
  }
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
  if (initialized()) return Status::InvalidState("NpuRuntime LinkTo must run before initialize");
  if (impl_->peer_link != nullptr) return Status::InvalidState("NpuRuntime already has a PeerLink");
  if (impl_->config.batch_put_kernel_object_path.empty() && !impl_->options.enable_cpu_offload) {
    return Status::InvalidArgument("PeerLink requires the AIV BatchPut kernel for RMA validation");
  }
  impl_->peer_link = std::make_shared<peer::PeerLink>(target.address);
  impl_->config.peer_link = impl_->peer_link;
  auto allocator = std::make_unique<PeerDdrAllocator>(impl_->peer_link);
  impl_->peer_allocator = allocator.get();
  impl_->allocator = std::move(allocator);
  impl_->runtime_core.reset();
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  impl_->remote_operator.reset();
#endif
  impl_->access_operator.reset();
  impl_->CreateCore();
  return Status::OK();
}

Status NpuRuntime::Initialize(std::uint64_t ddr_bytes, std::uint64_t slot_size) {
  const RuntimeInitRequest request{ddr_bytes, slot_size, DeviceKind::kCpu};
  if (initialized()) return Status::InvalidState("NpuRuntime has already been initialized");
  if (impl_->peer_link == nullptr || impl_->peer_allocator == nullptr) {
    return Status::InvalidState("NpuRuntime requires LinkTo before Initialize");
  }
  if (ddr_bytes == 0U || slot_size == 0U || ddr_bytes % slot_size != 0U) {
    return Status::InvalidArgument("DDR size must be a nonzero multiple of slot_size");
  }

  if (impl_->peer_allocator->has_allocation())
    return Status::InDoubt("previous DDR cleanup is unresolved; retry Shutdown before Initialize");
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
  if (impl_->runtime_core == nullptr) {
    throw std::logic_error("NpuRuntime requires LinkTo before core access");
  }
  return *impl_->runtime_core;
}
const RuntimeCore& NpuRuntime::core() const {
  if (impl_->runtime_core == nullptr) {
    throw std::logic_error("NpuRuntime requires LinkTo before core access");
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
  if (impl_->access_operator == nullptr) {
    return Status::NotInitialized(
        "NpuRuntime requires LinkTo before configuring BatchGet profiling");
  }
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  return static_cast<DefaultAccessOperator*>(impl_->access_operator.get())->SetBatchGetKernelProfiling(enabled);
#else
  return Status::NotImplemented("AIV support was not built");
#endif
}

Status NpuRuntime::CopyBatchGetKernelProfiles(
    std::vector<BatchGetKernelProfile>* profiles) const {
  if (impl_->access_operator == nullptr) {
    return Status::NotInitialized(
        "NpuRuntime requires LinkTo before reading BatchGet profiling");
  }
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  return static_cast<DefaultAccessOperator*>(impl_->access_operator.get())->CopyBatchGetKernelProfiles(profiles);
#else
  return Status::NotImplemented("AIV support was not built");
#endif
}

Status NpuRuntime::GetOffloadRegion(OffloadRegionTicket* ticket) const {
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  if (!ticket || !initialized() || !impl_->remote_operator) return Status::InvalidState("CPU Offload is not initialized");
  *ticket = impl_->remote_operator->ticket(); return Status::OK();
#else
  return Status::NotImplemented("CPU Offload was not built");
#endif
}
Status NpuRuntime::BatchCpuPull(const BatchPutRequest& request, BatchPutExecutionOptions execution) {
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  if (!initialized() || !impl_->remote_operator) return Status::InvalidState("CPU Pull is not initialized");
  if (pending_offload()) return Status::Busy("finish the existing transaction or shutdown the unresolved BatchCpuPull");
  int32_t current = -1;
  if (aclrtGetDevice(&current) != ACL_SUCCESS || current != static_cast<int32_t>(device_id()))
    return Status::InvalidState("select the Runtime NPU before BatchCpuPull");
  if (!request.input.valid() || request.input.device != DeviceKind::kNpu || !execution.stream ||
      request.keys.empty() || request.keys.size() > remote_pull::kMaxSpans ||
      request.keys.size() != request.input_offsets.size() || request.keys.size() != request.lengths.size())
    return Status::InvalidArgument("BatchCpuPull requires local NPU input, a stream and equally sized metadata");
  for (std::size_t i = 0; i < request.keys.size(); ++i)
    if (!offload::Contains(request.input.bytes, request.input_offsets[i], request.lengths[i]))
      return Status::InvalidArgument("BatchCpuPull source range is outside its HBM Tensor");
  auto status = impl_->runtime_core->PreparePut(request.keys, request.lengths, &impl_->local_put);
  if (!status.ok()) return status;
  impl_->local_commit = false; impl_->local_ready = {};
  auto execute = [&]() -> Status {
    if (impl_->local_source && (request.input.data != impl_->local_input.data || request.input.bytes != impl_->local_input.bytes)) {
      impl_->local_retire = true;
      auto result = impl_->local_source->Shutdown(); if (!result.ok()) return result;
      impl_->local_source.reset(); impl_->local_input = {};
    }
    if (!impl_->local_source) {
      impl_->local_retire = true;
      impl_->local_source = std::make_unique<CpuPullHbmSource>(device_id(),
          impl_->options.npu_eids.empty() ? "" : impl_->options.npu_eids.front());
      impl_->local_input = request.input;
      auto result = impl_->local_source->Register(impl_->remote_operator->ticket(), request.input, execution.stream);
      if (!result.ok()) return result;
    }
    impl_->local_retire = false;
    // Like BatchPut, preceding production on the supplied stream must finish
    // before transport reads the source. Identity-only Ready itself does not wait.
    if (aclrtSynchronizeStream(static_cast<aclrtStream>(execution.stream)) != ACL_SUCCESS)
      return Status::Internal("BatchCpuPull producer stream synchronization failed");
    auto result = impl_->local_source->Ready(&impl_->local_ready); if (!result.ok()) return result;
    CpuPullSourceDesc desc;
    result = impl_->local_source->GetDescriptor(&desc); if (!result.ok()) return result;
    result = impl_->runtime_core->AttachCpuSource(desc); if (!result.ok()) return result;
    std::vector<std::uint64_t> destinations;
    for (const auto& entry : impl_->local_put.entries())
      destinations.push_back(entry.address);
    CpuPullTask task;
    result = impl_->runtime_core->SubmitCpuPull(impl_->local_ready, impl_->local_put,
        request.input_offsets, destinations, request.lengths, &task);
    if (!result.ok()) return result;
    CpuPullState state;
    result = impl_->runtime_core->WaitCpuPull(task, impl_->options.offload_timeout_ms, &state);
    if (!result.ok()) return result;
    impl_->local_commit = true; // Preserve this decision across a lost Commit ACK.
    return Status::OK();
  };
  try { status = execute(); }
  catch (const std::exception& e) { status = Status::InDoubt(std::string("BatchCpuPull interrupted: ") + e.what()); }
  const auto finished = impl_->FinishLocal();
  // Failed calls never create another READ. If cleanup cannot prove completion,
  // the retained transaction blocks new work until Shutdown can safely retire it.
  return finished.ok() ? status : finished;
#else
  return Status::NotImplemented("CPU Pull was not built");
#endif
}
bool NpuRuntime::pending_offload() const {
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  if (impl_->local_put.id()) return true;
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

Status NpuRuntime::AttachHbm(const std::string& address, const HbmReadTicket& ticket,
                             const std::string& local_eid) {
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  if (!initialized()) return Status::NotInitialized("initialize NpuRuntime before attach_hbm");
  if (!impl_->hbm_reader) {
    auto kernel = impl_->options.batch_get_raw_addr_kernel_object_path;
    if (kernel.empty()) {
      const auto& get = impl_->options.batch_get_kernel_object_path;
      const auto slash = get.find_last_of('/');
      kernel = (slash == std::string::npos ? std::string{} : get.substr(0, slash + 1)) +
          "kvaf_execute_batch_get_raw_addr_kernels.o";
    }
    impl_->hbm_reader = std::make_unique<HbmReader>(device_id(), npu_core_count(), jettys_per_core(),
        impl_->options.batch_get_max_inflight_total, kernel);
  }
  return impl_->hbm_reader->Attach(address, ticket, local_eid);
#else
  return Status::NotImplemented("raw-address READ requires AIV BatchGet build");
#endif
}
Status NpuRuntime::AttachCpuSource(const CpuPullSourceDesc& source) {
  if (!initialized()) return Status::NotInitialized("initialize Runtime before attaching a CPU source");
  return core().AttachCpuSource(source);
}
Status NpuRuntime::PreparePut(const std::vector<KeyId>& keys,const std::vector<std::uint64_t>& lengths,PutHandle* put) {
  if (!initialized()) return Status::NotInitialized("initialize Runtime before PreparePut");
  if (pending_offload()) return Status::Busy("finish the existing transaction or shutdown the unresolved BatchCpuPull");
  return core().PreparePut(keys,lengths,put);
}
Status NpuRuntime::SubmitCpuPull(const ReadyToken& ready,const PutHandle& put,
                                Uint64ArrayView src_offsets,Uint64ArrayView dst_addrs,Uint64ArrayView lengths,CpuPullTask* task) {
  if (!initialized()) return Status::NotInitialized("initialize Runtime before CPU Pull");
  return core().SubmitCpuPull(ready,put,src_offsets,dst_addrs,lengths,task);
}
Status NpuRuntime::QueryCpuPull(const CpuPullTask& task,CpuPullState* state) {
  if (!initialized()) return Status::NotInitialized("Runtime is not initialized");
  return core().QueryCpuPull(task,state);
}
Status NpuRuntime::WaitCpuPull(const CpuPullTask& task,std::uint32_t timeout,CpuPullState* state) {
  if (!initialized()) return Status::NotInitialized("Runtime is not initialized");
  return core().WaitCpuPull(task,timeout,state);
}
Status NpuRuntime::CommitPut(const PutHandle& put) {
  if (!initialized()) return Status::NotInitialized("Runtime is not initialized");
  return core().CommitPut(put);
}
Status NpuRuntime::AbortPut(const PutHandle& put) {
  if (!initialized()) return Status::NotInitialized("Runtime is not initialized");
  return core().AbortPut(put);
}
Status NpuRuntime::BatchGetRawAddr(const ReadyToken& ready,NpuInt64ArrayView src,NpuInt64ArrayView dst,
                                  NpuInt64ArrayView lengths,MutableBufferView output,void* stream,HbmReadProfile* profile,
                                  bool validate_ranges) {
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  if (!initialized() || !impl_->hbm_reader) return Status::NotInitialized("attach source HBM before raw READ");
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
Status NpuRuntime::DetachHbm(const HbmReadTicket& ticket) {
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  if (impl_->hbm_reader) return impl_->hbm_reader->Detach(ticket);
#endif
  return Status::NotInitialized("no HBM reader");
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
  const Status shutdown = Shutdown();
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
  if (impl_->hbm_reader) { auto status = impl_->hbm_reader->Shutdown(); if (!status.ok()) return status; }
#endif
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  auto source_status = impl_->FinishLocal(); if (!source_status.ok()) return source_status;
  if (impl_->local_source) {
    source_status = impl_->local_source->Shutdown(); if (!source_status.ok()) return source_status;
    impl_->local_source.reset(); impl_->local_input = {};
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
  if (status.ok()) impl_->ReleaseProbe();
  return status;
}

bool NpuRuntime::initialized() const {
  return impl_ != nullptr && impl_->runtime_core != nullptr && impl_->runtime_core->initialized();
}

RuntimeCapabilities NpuRuntime::capabilities() const {
  return RuntimeCapabilities{!impl_->config.batch_get_kernel_object_path.empty(), !impl_->config.batch_put_kernel_object_path.empty(), false, true, impl_->options.enable_cpu_offload};
}

}  // namespace kvaf
