#include "npu_runtime.h"
#include "npu_eid_selection.h"
#include "execution_dispatcher.h"
#include "session_access_operator.h"
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
#include "remote_pull_operator.h"
#endif
#if defined(KVAF_ENABLE_NPU_PULL)
#include "hbm_read_client.h"
#include "npu_kernel_paths.h"
#endif

#include <acl/acl.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "access_operator.h"
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
#include "ubmem/access_operator.h"
#include "sequential_batch_get.h"
#include "startup_bandwidth_probe.h"
#include "ubmem/device_mapping.h"
#endif
#include "peer_link.h"
#include "meta_manager.h"
#include "gather_workspace.h"
#if defined(KVAF_ENABLE_AICPU_URMA)
#include "aicpu_access_operator.h"
#endif

namespace kvaf {
namespace {

class PeerDdrAllocator final : public DdrMemoryAllocator {
 public:
  explicit PeerDdrAllocator(std::shared_ptr<peer::PeerLink> link, TransportMode mode)
      : link_(std::move(link)), mode_(mode) {}

  Status Allocate(const RuntimeInitRequest& request, RemoteRegionDesc* region) override {
    if (link_ == nullptr || region == nullptr || request.ddr_device != DeviceKind::kCpu ||
        request.ddr_bytes == 0U || request.slot_size == 0U ||
        request.ddr_bytes % request.slot_size != 0U) {
      return Status::InvalidArgument("invalid peer DDR allocation request");
    }
    if (allocated_.base != 0U) return Status::InvalidState("peer DDR is already allocated");
    const Status status = link_->AllocateCacheRegion(
        request.ddr_bytes, request.slot_size, 0, mode_, &allocated_);
    if (!status.ok()) return status;
    *region = RemoteRegionDesc{allocated_.base, request.ddr_bytes, request.slot_size, DeviceKind::kCpu};
    return Status::OK();
  }

  Status Release(const RemoteRegionDesc& region) override {
    if (allocated_.base == 0U || allocated_.base != region.base) {
      return Status::NotFound("peer DDR allocation was not found");
    }
    const Status status = link_->ReleaseRegion(allocated_.base);
    if (status.ok()) allocated_ = {};
    return status;
  }

 private:
  std::shared_ptr<peer::PeerLink> link_;
  TransportMode mode_{TransportMode::Urma};
  peer::PeerRegion allocated_{};
};

// Register only compiled implementations. Registration stores factories and does
// not load kernels, open communication sessions, or reserve algorithm workspace.
std::vector<PutExecutionConfig> BuiltPutConfigs(TransportMode mode) {
  std::vector<PutExecutionConfig> result;
  auto add = [&](PutExecutionConfig c) {
    if (ValidateExecution(mode, c).ok()) result.push_back(c);
  };
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
  add({Transport::Urma, Executor::Aiv});
#endif
#if defined(KVAF_ENABLE_AICPU_URMA)
  add({Transport::Urma, Executor::Aicpu});
#endif
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  add({Transport::Ubmem, Executor::Aiv});
#endif
  return result;
}
std::vector<GetExecutionConfig> BuiltGetConfigs(TransportMode mode) {
  std::vector<GetExecutionConfig> result;
  auto add = [&](GetExecutionConfig c) {
    if (ValidateExecution(mode, c).ok()) result.push_back(c);
  };
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  add({Transport::Urma, Executor::Aiv, GetAlgorithm::Direct});
  add({Transport::Urma, Executor::Aiv, GetAlgorithm::Gather});
  add({Transport::Urma, Executor::Aiv, GetAlgorithm::Scatter});
#endif
#if defined(KVAF_ENABLE_AICPU_URMA)
  add({Transport::Urma, Executor::Aicpu, GetAlgorithm::Gather});
#if defined(KVAF_ENABLE_AICPU_CPU_PUSH)
  add({Transport::Urma, Executor::Aicpu, GetAlgorithm::Scatter});
#endif
#endif
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  add({Transport::Ubmem, Executor::Aiv, GetAlgorithm::Direct});
#endif
  return result;
}

}  // namespace

struct NpuRuntime::Impl {
  explicit Impl(NpuRuntimeOptions init_options) : options(std::move(init_options)) {
    // Honor explicit choices, including providers not compiled into this build.
    // An unavailable provider fails on first use, not at construction.
    const auto puts = BuiltPutConfigs(options.transport_mode);
    const auto gets = BuiltGetConfigs(options.transport_mode);
    if (!options.default_put && !puts.empty()) options.default_put = puts.front();
    if (!options.default_get && !gets.empty()) options.default_get = gets.front();
    auto normalized = NormalizeExecutionPolicy(&options);
    if (!normalized.ok()) throw std::invalid_argument(normalized.message());
    auto validate_resources = [&](const auto& entries) {
      for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto valid = ValidateExecution(options.transport_mode, entries[i].first);
        if (!valid.ok()) throw std::invalid_argument(valid.message());
        for (std::size_t j = 0; j < i; ++j)
          if (entries[i].first == entries[j].first)
            throw std::invalid_argument("duplicate execution resource budget");
      }
    };
    validate_resources(options.put_resources);
    validate_resources(options.get_resources);
  }

  void CreateCore() {
    meta_manager = CreateInMemoryMetaManager();
    auto dispatch = std::make_unique<ExecutionDispatcher>(options);
    for (auto c : BuiltPutConfigs(options.transport_mode)) {
      auto status = dispatch->RegisterPut(c, [this, c]() {
        return MakeProvider(c.transport, c.executor, GetAlgorithm::Direct, false);
      });
      if (!status.ok()) throw std::invalid_argument(status.message());
    }
    for (auto c : BuiltGetConfigs(options.transport_mode)) {
      auto status = dispatch->RegisterGet(c, [this, c]() {
        return MakeProvider(c.transport, c.executor, c.algorithm, true);
      });
      if (!status.ok()) throw std::invalid_argument(status.message());
    }
    dispatcher = std::move(dispatch);
    runtime_core = std::make_unique<RuntimeCore>(*meta_manager, *dispatcher, *allocator);
  }

  ExecutionResources Resources(Transport transport, Executor executor, GetAlgorithm algorithm,
                               bool get) const {
    const ExecutionResources* selected = nullptr;
    if (get) {
      for (const auto& entry : options.get_resources)
        if (entry.first == GetExecutionConfig{transport, executor, algorithm})
          selected = &entry.second;
    } else {
      for (const auto& entry : options.put_resources)
        if (entry.first == PutExecutionConfig{transport, executor}) selected = &entry.second;
    }
    auto budget = detail::ResolveExecutionResources(transport, executor, get, peer_candidates.local, selected);
    if (get && transport == Transport::Urma && executor == Executor::Aiv && algorithm != GetAlgorithm::Direct) {
      if (!selected || selected->kernel_path.empty()) {
        auto path = budget.kernel_path;
        const auto separator = path.find_last_of("/\\");
        const auto stem = algorithm == GetAlgorithm::Scatter ? "kvaf_scatter_batch_get_kernels.o"
                                                            : "kvaf_gather_batch_get_kernels.o";
        budget.kernel_path = (separator == std::string::npos ? "" : path.substr(0, separator + 1)) + stem;
      }
      if (!selected || !selected->max_inflight_total) budget.max_inflight_total = 0;
    }
    if (get && executor == Executor::Aicpu && algorithm == GetAlgorithm::Gather && !(selected && selected->jettys))
      budget.jettys = 1;
    return budget;
  }
  void ValidateProviderResources(Transport t, Executor e, GetAlgorithm algorithm, bool get,
                                 const ExecutionResources& budget) const {
    const ExecutionResources* explicit_resources = nullptr;
    if (get) {
      for (const auto& entry : options.get_resources)
        if (entry.first == GetExecutionConfig{t, e, algorithm}) explicit_resources = &entry.second;
    } else {
      for (const auto& entry : options.put_resources)
        if (entry.first == PutExecutionConfig{t, e}) explicit_resources = &entry.second;
    }
    if (explicit_resources) {
      const auto& r = *explicit_resources;
      if (r.jettys && (t != Transport::Urma || (e != Executor::Aiv && !(get && algorithm == GetAlgorithm::Gather))))
        throw std::invalid_argument("Jetty budget requires URMA AIV or AICPU Gather");
      if (r.max_inflight_total && (!get || e != Executor::Aiv || t != Transport::Urma))
        throw std::invalid_argument("in-flight limit is configurable only for URMA AIV Get");
      if (r.cpu_threads && (!get || algorithm != GetAlgorithm::Gather || e != Executor::Aicpu))
        throw std::invalid_argument("CPU thread budget requires AICPU Gather");
      if (!r.kernel_path.empty() && e == Executor::Aicpu)
        throw std::invalid_argument("AICPU kernel package is resolved through the CANN installation");
      if (r.gather_ddr_bytes && (!get || t != Transport::Urma || e != Executor::Aicpu ||
                                  algorithm != GetAlgorithm::Gather))
        throw std::invalid_argument("gather_ddr_bytes requires AICPU Gather Get");
      if (r.scatter_hbm_bytes && (!get || t != Transport::Urma || e != Executor::Aicpu ||
                                   algorithm != GetAlgorithm::Scatter))
        throw std::invalid_argument("scatter_hbm_bytes requires AICPU Scatter Get");
      if (r.gather_read_wave_wrs && (!get || t != Transport::Urma || e != Executor::Aicpu ||
                                      algorithm != GetAlgorithm::Gather))
        throw std::invalid_argument("gather_read_wave_wrs requires AICPU Gather Get");
      if (r.gather_address_groups_per_post && (!get || t != Transport::Urma || e != Executor::Aicpu ||
                                               algorithm != GetAlgorithm::Gather))
        throw std::invalid_argument("gather_address_groups_per_post requires AICPU Gather Get");
    }
    if (budget.gather_read_wave_wrs && !ValidGatherReadWaveWrs(*budget.gather_read_wave_wrs))
      throw std::invalid_argument("gather_read_wave_wrs must be in [1, 4096]");
    if (budget.gather_address_groups_per_post && !*budget.gather_address_groups_per_post)
      throw std::invalid_argument("gather_address_groups_per_post must be a positive integer");
    const bool aicpu_scatter = get && IsAicpuScatterGet({t, e, algorithm});
    if (!budget.cores || (!aicpu_scatter && budget.cores > (e == Executor::Aicpu ? 8U : 48U)) ||
        !budget.request_capacity ||
        (budget.request_capacity > 65536U &&
         !(get && IsWindowedGet({t, e, algorithm}))) || budget.cpu_threads > 64U ||
        (t == Transport::Urma && e == Executor::Aiv &&
         (!budget.jettys || budget.jettys > (get ? 64U : 2U) ||
          (get && *budget.max_inflight_total && *budget.max_inflight_total < 2U * budget.cores))))
      throw std::invalid_argument("invalid execution resource budget");
    if (get && t == Transport::Urma && e == Executor::Aiv && algorithm != GetAlgorithm::Direct &&
        (budget.cores > 8U || options.npu_eids.size() > 1 ||
         GatherStagingBytes(options.batch_get_token_bytes) > UINT32_MAX ||
         (algorithm == GetAlgorithm::Gather && (!options.aiv_gather_sgl_size || options.aiv_gather_sgl_size > 6))))
      throw std::invalid_argument("invalid AIV Gather/Scatter resource budget");
    if (e != Executor::Aicpu) {
      if (budget.kernel_path.empty())
        throw std::invalid_argument("selected AIV provider requires its kernel path");
      if (!std::ifstream(budget.kernel_path, std::ios::binary).good())
        throw std::invalid_argument("selected provider kernel is not readable: " + budget.kernel_path);
    }
  }
  void EnsureAicpuPackage() {
    // Put and Get can first use AICPU concurrently. Publish one complete package
    // description for this Runtime/device; the loader owns device-safe loading.
    std::lock_guard<std::mutex> lock(aicpu_package_mutex);
    if (aicpu_package_ready) return;
    detail::AicpuPackageSearch search;
    if (const char* path = std::getenv("ASCEND_HOME_PATH")) search.ascend_home_path = path;
    if (const char* path = std::getenv("ASCEND_CANN_PACKAGE_PATH")) search.ascend_cann_package_path = path;
    detail::AicpuKernelPackage selected;
    const auto status = detail::ResolveAicpuKernelPackage(search, &selected);
    if (!status.ok()) throw std::invalid_argument("AICPU package: " + status.message());
    kernel_package = std::move(selected);
    aicpu_package_ready = true;
    std::fprintf(stderr, "[KVAF] AICPU_PACKAGE root=%s json=%s\n",
        kernel_package.cann_root.c_str(), kernel_package.json_path.c_str());
  }
  std::unique_ptr<AccessOperator> MakeProvider(Transport transport, Executor executor,
                                               GetAlgorithm algorithm, bool get) {
    execution::Request q;
    q.transport = transport;
    q.executor = executor;
    q.algorithm = algorithm;
    q.put = !get;
    const auto budget = Resources(transport, executor, algorithm, get);
    if (get && (!options.batch_get_token_bytes || !options.batch_get_tokens_per_block))
      throw std::invalid_argument("ordinary cache Get requires a token layout");
    ValidateProviderResources(transport, executor, algorithm, get, budget);
    std::vector<peer::PeerCandidateInfo> candidates;
    std::vector<std::uint32_t> gather_candidates;
    if (transport == Transport::Urma) {
      EnsureCacheEids();
      const bool gather = get && executor == Executor::Aicpu && algorithm == GetAlgorithm::Gather;
      const bool direct = get && executor == Executor::Aiv && algorithm == GetAlgorithm::Direct;
      if (gather || direct) {
        if (gather && (!budget.jettys || budget.jettys > 8))
          throw std::invalid_argument("Gather supports 1..8 data Jettys per worker");
        const auto workers = std::uint64_t{budget.cores} * (direct ? budget.jettys : 1U);
        std::vector<peer::PeerCandidateInfo> active;
        const auto status = cpu_fes.ForWorkers(workers, &active);
        if (!status.ok()) throw std::invalid_argument(status.message());
        std::fprintf(stderr, "[KVAF] GET_FE_SELECTION algorithm=%s requested=%u candidates=%zu active=%zu workers=%llu\n",
                     gather ? "gather" : "direct", cpu_fes.requested_count, cpu_fes.read_fes.size(), active.size(),
                     static_cast<unsigned long long>(workers));
        if (gather) {
          // All Gather lanes share one execution and one staging allocation.
          // Additional FEs register that same workspace, rather than opening more gather pools.
          candidates.push_back(active.front());
          for (const auto& candidate : active) gather_candidates.push_back(candidate.candidate_id);
        } else {
          candidates = std::move(active);
        }
      } else {
        candidates.push_back(cpu_fes.primary);
      }
    }
    q.workers = budget.cores;
    q.jettys = budget.jettys;
    q.request_capacity = budget.request_capacity;
    q.token_bytes = options.batch_get_token_bytes;
    q.cpu_threads = budget.cpu_threads;
    q.auxiliary_bytes = 0;
    const auto eids = ProviderEids(transport, executor, get, budget);
    if (!eids.empty()) q.local_eid = eids.front();
    return std::make_unique<SessionAccessOperator>(
        peer_link, q, std::move(candidates),
        [this, transport, executor, algorithm, get, gather_candidates](const PeerExecutionResources::Links& links) {
          return MakeLocalProvider(transport, executor, algorithm, get, links, gather_candidates);
        },
        budget.request_capacity);
  }
  void EnsureCacheEids() {
    std::lock_guard<std::mutex> lock(cache_eid_mutex);
    if (cache_eids_ready) return;
    peer::PeerCandidateTable table;
    table.local = peer_candidates.local;
    auto status = peer_link->GetCacheCandidateCount(&table);
    if (!status.ok()) throw std::invalid_argument(status.message());
    std::uint32_t physical_device;
    if (!detail::ResolvePhysicalDeviceId(options.device_id, &physical_device))
      throw std::invalid_argument("cannot resolve physical NPU from device " + std::to_string(options.device_id));
    const auto& cpu_device = adaptive_cpu_device.empty() ? options.peer_ub_device : adaptive_cpu_device;
    const bool automatic_cpu = !table.device_fixed && cpu_device.empty();
    detail::PairEids pair;
    if (table.local && (options.npu_eids.empty() || automatic_cpu))
      pair = detail::ReadDevicePair(physical_device);
    status = detail::SelectNpuEids(physical_device, options.npu_eids, pair, &local_eids);
    if (!status.ok()) throw std::invalid_argument(status.message());
    status = peer_link->SelectCpuFes(table, cpu_device, pair.cpu,
                                     local_eids.size() > 1, &cpu_fes);
    if (!status.ok()) throw std::invalid_argument(status.message());
    const auto& cpu = cpu_fes.primary;
    const char* reason = "first_candidate";
    if (!automatic_cpu) reason = "fixed_device";
    else if (table.local) reason = pair.cpu && *pair.cpu == cpu.eid ? "pair_match" : "pair_fallback";
    std::fprintf(stderr, "[KVAF] PEER_LINK_EID_SELECTED scene=%s physical_device=%u candidate=%u "
                 "ub_device=%s eid_index=%u cpu_eid=%s npu_eid=%s cpu_selection=%s\n",
                 table.local ? "local" : "remote", physical_device, cpu.candidate_id,
                 cpu.device_name.c_str(), cpu.eid_index, detail::EidToHex(cpu.eid).c_str(),
                 detail::EidToHex(local_eids.front()).c_str(), reason);
    cache_eids_ready = true;
  }
  std::vector<detail::NpuEid> ProviderEids(Transport transport, Executor executor, bool get,
                                         const ExecutionResources& budget) {
    std::vector<detail::NpuEid> selected;
    const auto status = detail::SelectProviderEids(transport, executor, get,
        !options.npu_eids.empty(), budget, local_eids, &selected);
    if (!status.ok()) throw std::invalid_argument(status.message());
    return selected;
  }
  std::unique_ptr<AccessOperator> MakeLocalProvider(Transport transport, Executor executor,
                                                    GetAlgorithm algorithm, bool get,
                                                    const PeerExecutionResources::Links& links,
                                                    const std::vector<std::uint32_t>& gather_candidates) {
    const auto& link = links.front();
    const auto budget = Resources(transport, executor, algorithm, get);
    const auto eids = ProviderEids(transport, executor, get, budget);
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
    if (transport == Transport::Ubmem) {
      return std::make_unique<ubmem::AccessOperator>(
          ubmem::OperatorOptions{get ? budget.kernel_path : "", get ? "" : budget.kernel_path,
                                 options.device_id, budget.cores, link, mappings});
    }
    if (executor == Executor::Aiv) {
      AccessOperatorConfig c;
      c.peer_local = peer_candidates.local;
      c.peer_link = link;
      if (links.size() > 1) c.peer_get_links = links;
      c.peer_local_eids = eids;
      c.peer_local_eid = eids.front();
      c.npu_core_count = budget.cores;
      c.jettys_per_core = budget.jettys;
      c.batch_put_jettys_per_core = std::min(budget.jettys, 2U);
      c.batch_get_max_inflight_total = *budget.max_inflight_total;
      if (get)
        c.batch_get_kernel_object_path = budget.kernel_path;
      else
        c.batch_put_kernel_object_path = budget.kernel_path;
      if (get && algorithm != GetAlgorithm::Direct) {
        c.aiv_batch_get_mode = algorithm == GetAlgorithm::Scatter ? "cpu_push_scatter" : "gather_read";
        c.aiv_gather_sgl_size = options.aiv_gather_sgl_size;
        c.batch_get_token_bytes = options.batch_get_token_bytes;
        return CreateSequentialBatchGet(std::move(c));
      }
      c.enable_get = get;
      if (get) c.batch_put_kernel_object_path.clear();
      return std::make_unique<DefaultAccessOperator>(std::move(c));
    }
#endif
#if defined(KVAF_ENABLE_AICPU_URMA)
    EnsureAicpuPackage();
    return std::make_unique<UmdkAccessOperator>(UmdkAccessOperatorConfig{
        kernel_package.json_path, budget.cores, link, options.batch_get_token_bytes,
        algorithm == GetAlgorithm::Scatter ? "cpu_push_scatter" : "gather_read",
        eids.front(), get, budget.jettys, gather_candidates,
        {},
        budget.gather_ddr_bytes, budget.scatter_hbm_bytes,
        budget.gather_read_wave_wrs.value_or(kGatherDefaultReadWaveWrs),
        budget.gather_address_groups_per_post});
#else
    throw std::invalid_argument("requested execution provider was not built");
#endif
  }

  std::atomic<bool> base_initialized{false};
  std::atomic<bool> closing{false};
  std::atomic<bool> ddr_closing{false};
  mutable std::mutex base_mutex;
#if defined(KVAF_ENABLE_NPU_PULL)
  std::shared_ptr<HbmReader> hbm_reader;
  std::shared_ptr<HbmReader> Reader() const {
    std::lock_guard<std::mutex> lock(base_mutex);
    return hbm_reader;
  }
#endif
  NpuRuntimeOptions options;
  detail::AicpuKernelPackage kernel_package;
  std::mutex aicpu_package_mutex;
  bool aicpu_package_ready{false};
  peer::CpuFeSelection cpu_fes;
  peer::PeerCandidateTable peer_candidates;
  std::vector<detail::NpuEid> local_eids;
  std::mutex cache_eid_mutex;
  bool cache_eids_ready{false};
  std::string adaptive_cpu_device;
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  std::unique_ptr<detail::StartupBandwidthProbe> bandwidth_probe;
#endif
  std::unique_ptr<MetaManager> meta_manager;
  std::unique_ptr<DdrMemoryAllocator> allocator;
  std::shared_ptr<peer::PeerLink> peer_link;
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  std::shared_ptr<ubmem::DeviceMappingCache> mappings =
      std::make_shared<ubmem::DeviceMappingCache>();
#endif
  std::unique_ptr<ExecutionDispatcher> dispatcher;
  std::unique_ptr<RuntimeCore> runtime_core;
  std::mutex cpu_pull_prepare_mutex;
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  std::unique_ptr<RemotePullOperator> cpu_pull;
  std::optional<std::uint32_t> pull_source_physical;
#endif
};

NpuRuntime::NpuRuntime(NpuRuntimeOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {
  const auto& config = impl_->options;
  if (config.adaptive_ub_device) {
    const GetExecutionConfig direct{Transport::Urma, Executor::Aiv, GetAlgorithm::Direct};
    if (config.transport_mode != TransportMode::Urma || !(config.default_get == direct) ||
        !config.peer_ub_device.empty() || !config.adaptive_ub_miss_count ||
        config.adaptive_ub_miss_count > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("adaptive CPU selection requires URMA AIV Direct, a valid miss count and no peer pin");
    }
  }
  if ((config.batch_get_token_bytes == 0) != (config.batch_get_tokens_per_block == 0))
    throw std::invalid_argument("invalid NPU Runtime token layout");
  if (aclrtSetDevice(config.device_id) != ACL_SUCCESS)
    throw std::runtime_error("aclrtSetDevice failed for NpuRuntime");
  const auto mode = config.transport_mode == TransportMode::Ubmem ? "ubmem" :
      config.transport_mode == TransportMode::UrmaAndUbmem ? "urma+ubmem" : "urma";
  std::fprintf(stderr, "[KVAF] NPU_RUNTIME transport_mode=%s device=%u\n", mode, config.device_id);
}

NpuRuntime::~NpuRuntime() {
  if (impl_ != nullptr) {
    const auto status = Shutdown();
    if (!status.ok()) { (void)impl_.release(); return; }
    impl_->runtime_core.reset();
    impl_->dispatcher.reset();
  }
}

Status NpuRuntime::LinkTo(const LinkTarget& target) {
  if (target.address.empty()) return Status::InvalidArgument("NpuRuntime LinkTo requires peer host:port");
  if (ddr_initialized()) return Status::InvalidState("NpuRuntime LinkTo must run before initialize_ddr");
  if (impl_->peer_link != nullptr) return Status::InvalidState("NpuRuntime already has a PeerLink");
  impl_->peer_link = std::make_shared<peer::PeerLink>(target.address);
  impl_->allocator = std::make_unique<PeerDdrAllocator>(impl_->peer_link, impl_->options.transport_mode);
  impl_->runtime_core.reset();
  impl_->dispatcher.reset();
  impl_->CreateCore();
  return Status::OK();
}

Status NpuRuntime::Initialize() {
  std::lock_guard<std::mutex> lock(impl_->base_mutex);
  if (impl_->closing) return Status::InvalidState("retry incomplete Runtime shutdown");
  if (impl_->base_initialized) return Status::OK();
  if (aclrtSetDevice(impl_->options.device_id) != ACL_SUCCESS)
    return Status::Internal("aclrtSetDevice failed");
  impl_->base_initialized = true;
  return Status::OK();
}
Status NpuRuntime::Initialize(std::uint64_t bytes, std::uint64_t slot) {
  auto status = Initialize();
  return status.ok() ? InitializeDdr(bytes, slot) : status;
}
Status NpuRuntime::InitializeDdr(std::uint64_t ddr_bytes, std::uint64_t slot_size) {
  if (!initialized()) return Status::NotInitialized("initialize base device before DDR");
  const RuntimeInitRequest request{ddr_bytes, slot_size, DeviceKind::kCpu};
  if (ddr_initialized()) return Status::InvalidState("DDR has already been initialized");
  if (impl_->peer_link == nullptr || impl_->allocator == nullptr) {
    return Status::InvalidState("NpuRuntime requires LinkTo before Initialize");
  }
  if (ddr_bytes == 0U || slot_size == 0U || ddr_bytes % slot_size != 0U) {
    return Status::InvalidArgument("DDR size must be a nonzero multiple of slot_size");
  }

  if (impl_->runtime_core != nullptr && impl_->runtime_core->needs_cleanup()) {
    const Status status = ShutdownDdr();
    if (!status.ok()) return status;
  }
  // Cache allocation does not initialize a transport or prepare an algorithm.
  // Keep the entire explicit EID list until the selected provider applies its
  // own routing and worker budget; the default Get must not choose that policy.
  impl_->local_eids.clear();
  impl_->cpu_fes = {};
  impl_->cache_eids_ready = false;
  if (impl_->options.transport_mode != TransportMode::Ubmem) {
    auto status = impl_->peer_link->DiscoverCandidates(&impl_->peer_candidates);
    if (!status.ok()) return status;
  }
  const auto initialize = [&](const RuntimeInitRequest& cache) {
    impl_->runtime_core.reset();
    impl_->dispatcher.reset();
    impl_->cache_eids_ready = false;
    impl_->CreateCore();
    return impl_->runtime_core->Initialize(cache);
  };
  impl_->adaptive_cpu_device.clear();
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  if (impl_->options.adaptive_ub_device && !impl_->options.adaptive_peer_ub_devices.empty()) {
    peer::PeerCandidateTable table;
    auto status = impl_->peer_link->GetCacheCandidateCount(&table);
    if (!status.ok()) return status;
    if (!table.device_fixed) {
      const detail::StartupProbeConfig probe{
          impl_->options.batch_get_token_bytes, impl_->options.batch_get_tokens_per_block,
          impl_->options.adaptive_ub_miss_count};
      detail::StartupProbeLayout layout{};
      status = detail::BuildStartupProbeLayout(probe, &layout);
      if (!status.ok()) return status;
      const auto& devices = impl_->options.adaptive_peer_ub_devices;
      std::vector<std::vector<detail::StartupCandidate>> candidates;
      candidates.reserve(devices.size());
      for (std::uint32_t index = 0; index < devices.size(); ++index) {
        const auto& device = devices[index];
        if (device.empty() || device.find(',') != std::string::npos)
          return Status::InvalidArgument("adaptive CPU candidate must name one UB device");
        candidates.push_back({{index, 0U}});
      }
      const RuntimeInitRequest scratch{layout.ddr_bytes, layout.block_bytes, DeviceKind::kCpu};
      status = detail::SelectStartupDeviceCandidate(candidates,
          [&](const detail::StartupCandidate& candidate, bool measure, double* cycles) {
            const auto& device = devices[candidate.peer];
            impl_->adaptive_cpu_device = device;
            auto trial = initialize(measure ? scratch : request);
            if (!trial.ok()) return trial;
            if (measure) {
              impl_->bandwidth_probe = std::make_unique<detail::StartupBandwidthProbe>(probe);
              trial = impl_->bandwidth_probe->Run(*this, cycles);
              if (!trial.ok()) return trial;
              std::fprintf(stderr, "[KVAF_AIV] UB_DEVICE_PROBE_RESULT device_id=%u ub_device=%s "
                                   "miss_count=%u cycles=%.0f\n",
                           impl_->options.device_id, device.c_str(), probe.miss_count, *cycles);
            } else {
              std::fprintf(stderr, "[KVAF_AIV] UB_DEVICE_SELECTED device_id=%u ub_device=%s cycles=%.0f\n",
                           impl_->options.device_id, device.c_str(), *cycles);
            }
            return trial;
          }, [&] { return ShutdownDdr(); });
      if (!status.ok()) impl_->adaptive_cpu_device.clear();
      return status;
    }
  }
#endif
  return initialize(request);
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
ExecutionResources NpuRuntime::GetResources(GetExecutionConfig config) const {
  if (!ddr_initialized()) throw std::logic_error("Get resources require initialized DDR");
  return impl_->Resources(config.transport, config.executor, config.algorithm, true);
}
const std::vector<std::string>& NpuRuntime::npu_eids() const {
  return impl_->options.npu_eids;
}
std::uint32_t NpuRuntime::batch_get_token_bytes() const {
  return impl_->options.batch_get_token_bytes;
}
std::uint32_t NpuRuntime::batch_get_tokens_per_block() const {
  return impl_->options.batch_get_tokens_per_block;
}

Status NpuRuntime::DrainGet() {
  if (!ddr_initialized()) return Status::NotInitialized("initialize DDR before draining Get");
  return core().DrainGet();
}
Status NpuRuntime::Recover(PutExecutionConfig c) {
  if (aclrtSetDevice(impl_->options.device_id) != ACL_SUCCESS)
    return Status::Internal("aclrtSetDevice failed");
  if (!ddr_initialized()) return Status::NotInitialized("initialize DDR before recovery");
  return core().RecoverPut(c);
}
Status NpuRuntime::Recover(GetExecutionConfig c) {
  if (aclrtSetDevice(impl_->options.device_id) != ACL_SUCCESS)
    return Status::Internal("aclrtSetDevice failed");
  if (!ddr_initialized()) return Status::NotInitialized("initialize DDR before recovery");
  return core().RecoverGet(c);
}
Status NpuRuntime::Warmup(PutExecutionConfig c, const ExecutionWarmup& request) {
  if (!ddr_initialized()) return Status::NotInitialized("initialize DDR before warmup");
  return impl_->dispatcher->Warmup(c, request);
}
Status NpuRuntime::Warmup(GetExecutionConfig c, const ExecutionWarmup& request) {
  if (!ddr_initialized()) return Status::NotInitialized("initialize DDR before warmup");
  return impl_->dispatcher->Warmup(c, request);
}

Status NpuRuntime::SetBatchGetKernelProfiling(
    bool enabled, bool detailed, std::optional<GetExecutionConfig> config) {
  if (!ddr_initialized() || !impl_->dispatcher)
    return Status::NotInitialized("initialize Runtime before configuring Get profiling");
  return impl_->dispatcher->ConfigureGetProfiling(config, enabled, detailed);
}

Status NpuRuntime::CheckBatchGetStatus(void* stream, std::optional<GetExecutionConfig> config,
                                      std::uint64_t graph_lease) {
  if (!ddr_initialized() || !impl_->dispatcher)
    return Status::NotInitialized("NpuRuntime is not initialized");
  return impl_->dispatcher->CheckGetStatus(config, stream, graph_lease);
}

Status NpuRuntime::CopyBatchGetKernelProfiles(
    std::vector<BatchGetKernelProfile>* profiles, std::optional<GetExecutionConfig> config,
    std::uint64_t graph_lease) const {
  if (!ddr_initialized() || !impl_->dispatcher)
    return Status::NotInitialized("initialize Runtime before reading Get profiling");
  return impl_->dispatcher->CopyGetProfiles(config, profiles, graph_lease);
}

Status NpuRuntime::GetOffloadRegion(OffloadRegionTicket* out,
                                    std::optional<std::uint32_t> source_physical_device_id) {
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  if (!out) return Status::InvalidArgument("missing CPU Pull region ticket");
  std::lock_guard<std::mutex> lock(impl_->cpu_pull_prepare_mutex);
  if (!ddr_initialized()) return Status::NotInitialized("initialize DDR before CPU Pull");
  if (impl_->closing || impl_->ddr_closing) return Status::InvalidState("Runtime or DDR is closing");
  if (impl_->options.transport_mode == TransportMode::Ubmem)
    return Status::InvalidState("CPU Pull requires DDR URMA capability");

  // A CSV configures cache Direct Get only. Pull always selects a single endpoint.
  const auto& configured = impl_->options.peer_ub_device;
  const std::string device = configured.find(',') == std::string::npos ? configured : "";
  const auto& table = impl_->peer_candidates;
  const bool match_source = table.local && !table.device_fixed && device.empty();
  auto& pull = impl_->cpu_pull;
  if (!pull || (match_source && source_physical_device_id &&
                impl_->pull_source_physical != source_physical_device_id)) {
    detail::PairEids pair;
    if (match_source) {
      if (!source_physical_device_id) {
        std::uint32_t physical{};
        if (!detail::ResolvePhysicalDeviceId(impl_->options.device_id, &physical))
          return Status::Internal("cannot resolve runtime NPU physical device for CPU Pull");
        source_physical_device_id = physical;
      }
      pair = detail::ReadDevicePair(*source_physical_device_id);
    }
    peer::CpuFeSelection selection;
    auto status = impl_->peer_link->SelectCpuFes(table, device, pair.cpu, /*single_cpu_fe=*/true, &selection);
    if (!status.ok()) return status;
    const auto& cpu = selection.primary;
    if (pull && (pull->candidate().candidate_id != cpu.candidate_id || pull->candidate().eid != cpu.eid))
      return Status::InvalidState("source requires a different CPU EID for an already bound CPU Pull region");
    if (!pull) {
      const char* reason = !device.empty() || table.device_fixed ? "fixed_device" :
          !match_source ? "first_candidate" : pair.cpu == cpu.eid ? "pair_match" : "pair_fallback";
      std::fprintf(stderr, "[KVAF] CPU_PULL_EID_SELECTED scene=%s candidate=%u ub_device=%s eid_index=%u "
                   "cpu_eid=%s cpu_selection=%s\n", table.local ? "local" : "remote", cpu.candidate_id,
                   cpu.device_name.c_str(), cpu.eid_index, detail::EidToHex(cpu.eid).c_str(), reason);
      pull = std::make_unique<RemotePullOperator>(impl_->peer_link, cpu);
      impl_->pull_source_physical = source_physical_device_id;
    }
  }
  if (!pull->ticket().region_id) {
    auto status = core().SetCpuPullExecutor(pull.get());
    if (!status.ok()) return status;
    status = pull->Initialize(core().remote_region());
    if (!status.ok()) return status;
  }
  *out = pull->ticket();
  return Status::OK();
#else
  return Status::NotImplemented("CPU Pull was not built");
#endif
}
Status NpuRuntime::BatchCpuPull(const std::vector<KeyId>& keys,const ReadyToken& ready,
    Uint64ArrayView src,const std::vector<std::uint64_t>& lengths,std::uint32_t timeout,CpuPullProfile* profile) {
  if(!ddr_initialized())return Status::NotInitialized("initialize DDR before CPU Pull");
  if(impl_->closing)return Status::InvalidState("retry Runtime shutdown");
  if(timeout==UINT32_MAX)timeout=impl_->options.cpu_pull_options.timeout_ms;
  return core().BatchCpuPull(keys,ready,src,lengths,timeout,profile);
}
Status NpuRuntime::ResolveCpuPull(std::uint32_t timeout) {
  if(!ddr_initialized())return Status::NotInitialized("DDR is not initialized");
  if(timeout==UINT32_MAX)timeout=impl_->options.cpu_pull_options.timeout_ms;
  return core().ResolveCpuPull(timeout);
}
Status NpuRuntime::CancelCpuPull(const ReadyToken& ready) {
  if(!ddr_initialized())return Status::NotInitialized("DDR is not initialized");
  return core().CancelCpuPull(ready);
}
bool NpuRuntime::pending_offload() const { return ddr_initialized() && core().pending_offload(); }
Status NpuRuntime::LastOffloadChecksum(std::uint64_t* out) const {
  if(!ddr_initialized())return Status::NotInitialized("DDR is not initialized");
  return core().LastOffloadChecksum(out);
}
Status NpuRuntime::PreparePut(const std::vector<KeyId>& keys,const std::vector<std::uint64_t>& lengths,PutHandle* out) {
  if(!ddr_initialized())return Status::NotInitialized("DDR is not initialized");
  if(impl_->closing)return Status::InvalidState("retry Runtime shutdown");
  return core().PreparePut(keys,lengths,out);
}
Status NpuRuntime::SubmitCpuPull(const ReadyToken& ready,const PutHandle& put,Uint64ArrayView src,
    Uint64ArrayView dst,Uint64ArrayView lengths) {
  if(!ddr_initialized())return Status::NotInitialized("DDR is not initialized");
  if(impl_->closing)return Status::InvalidState("retry Runtime shutdown");
  return core().SubmitCpuPull(ready,put,src,dst,lengths);
}
Status NpuRuntime::QueryCpuPull(const PutHandle& put,CpuPullState* state) {
  if(!ddr_initialized())return Status::NotInitialized("DDR is not initialized");
  return core().QueryCpuPull(put,state);
}
Status NpuRuntime::WaitCpuPull(const PutHandle& put,std::uint32_t timeout,CpuPullState* state) {
  if(!ddr_initialized())return Status::NotInitialized("DDR is not initialized");
  return core().WaitCpuPull(put,timeout,state);
}
Status NpuRuntime::CommitPut(const PutHandle& put) {
  if(!ddr_initialized())return Status::NotInitialized("DDR is not initialized");
  return core().CommitPut(put);
}
Status NpuRuntime::AbortPut(const PutHandle& put) {
  if(!ddr_initialized())return Status::NotInitialized("DDR is not initialized");
  return core().AbortPut(put);
}

Status NpuRuntime::ListenHbm(const std::string& address,const std::string& advertised,
    const std::string& local_eid,HbmReceiverTicket* receiver) {
#if defined(KVAF_ENABLE_NPU_PULL)
  std::lock_guard<std::mutex> lock(impl_->base_mutex);
  if(!initialized())return Status::NotInitialized("initialize base device before listen_hbm");
  if(impl_->closing)return Status::InvalidState("retry Runtime shutdown");
  if(!impl_->hbm_reader) {
    const auto& o=impl_->options.npu_pull_options;
    auto kernel=o.kernel_path;
    if(kernel.empty())kernel=detail::kDefaultRawReadKernelPath;
    if(kernel.empty())return Status::InvalidArgument("configure npu_pull_options.kernel_path");
    impl_->hbm_reader=std::make_shared<HbmReader>(device_id(),o.cores,o.jettys,
        o.max_inflight_total,kernel,o.max_output_bindings,o.diagnostics);
  }
  return impl_->hbm_reader->Listen(address,advertised,local_eid,receiver);
#else
  return Status::NotImplemented("NPU Pull was not built");
#endif
}
Status NpuRuntime::BatchGetRawAddr(const ReadyToken& ready,NpuInt64ArrayView src,NpuInt64ArrayView dst,
    NpuInt64ArrayView lengths,MutableBufferView output,void* stream,HbmReadProfile* profile,
    bool validate_ranges,bool synchronize) {
#if defined(KVAF_ENABLE_NPU_PULL)
  auto reader=impl_->Reader();
  if(!initialized() || !reader)return Status::NotInitialized("listen and register source before raw READ");
  if(impl_->closing)return Status::InvalidState("retry Runtime shutdown");
  return reader->Read(ready,src,dst,lengths,output,stream,profile,validate_ranges,
                      synchronize);
#else
  return Status::NotImplemented("NPU Pull was not built");
#endif
}
Status NpuRuntime::ResolveRawAddr(const ReadyToken& ready) {
#if defined(KVAF_ENABLE_NPU_PULL)
  auto reader=impl_->Reader();
  if(reader)return reader->Resolve(ready);
#endif
  return Status::NotInitialized("NPU Pull is not initialized");
}
Status NpuRuntime::CancelRawAddr(const ReadyToken& ready) {
#if defined(KVAF_ENABLE_NPU_PULL)
  auto reader=impl_->Reader();
  if(reader)return reader->Cancel(ready);
#endif
  return Status::NotInitialized("NPU Pull is not initialized");
}
bool NpuRuntime::pending_raw_addr() const {
#if defined(KVAF_ENABLE_NPU_PULL)
  auto reader=impl_->Reader();
  return reader && reader->pending();
#else
  return false;
#endif
}

Status NpuRuntime::SetScatterProfiling(bool enabled) {
  if (!ddr_initialized()) return Status::NotInitialized("initialize DDR before Scatter profiling");
  return impl_->dispatcher->WithSequentialGet(GetAlgorithm::Scatter, true,
      [enabled](AccessOperator& op) { return op.SetScatterProfiling(enabled); });
}

Status NpuRuntime::CopyScatterProfiles(std::vector<ScatterBatchGetCoreProfile>* out) const {
  if (!ddr_initialized()) return Status::NotInitialized("initialize DDR before Scatter profiling");
  return impl_->dispatcher->WithSequentialGet(GetAlgorithm::Scatter, false,
      [out](AccessOperator& op) { return op.CopyScatterProfiles(out); });
}

Status NpuRuntime::SetGatherBatchGetProfiling(bool enabled) {
  if (!ddr_initialized()) return Status::NotInitialized("initialize DDR before Gather profiling");
  return impl_->dispatcher->WithSequentialGet(GetAlgorithm::Gather, true,
      [enabled](AccessOperator& op) { return op.SetGatherBatchGetProfiling(enabled); });
}

Status NpuRuntime::CopyGatherBatchGetProfile(GatherBatchGetProfile* out) const {
  if (!ddr_initialized()) return Status::NotInitialized("initialize DDR before Gather profiling");
  return impl_->dispatcher->WithSequentialGet(GetAlgorithm::Gather, false,
      [out](AccessOperator& op) { return op.CopyGatherBatchGetProfile(out); });
}

Status NpuRuntime::AcceptLink(const LinkTarget&) {
  return Status::NotImplemented("NpuRuntime does not accept links in the CPU--NPU deployment");
}

Status NpuRuntime::ReleaseLink(LinkId link_id) {
  if (link_id != 1U || impl_->peer_link == nullptr) return Status::NotFound("PeerLink was not found");
  const Status shutdown = ShutdownDdr();
  if (!shutdown.ok()) return shutdown;
  impl_->runtime_core.reset();
  impl_->dispatcher.reset();
  impl_->allocator.reset();
  impl_->peer_link.reset();
  impl_->local_eids.clear();
  return Status::OK();
}

Status NpuRuntime::ServeConnection(int) {
  return Status::NotImplemented("NpuRuntime has no incoming control service in this deployment");
}

Status NpuRuntime::Shutdown() {
  if (impl_ == nullptr) return Status::OK();
  impl_->closing = true;
  if (impl_->base_initialized && aclrtSetDevice(impl_->options.device_id)!=ACL_SUCCESS)
    return Status::Internal("select Runtime device for shutdown");
#if defined(KVAF_ENABLE_NPU_PULL)
  if (auto reader = impl_->Reader()) {
    auto s = reader->Shutdown();
    if (!s.ok()) return s;
    std::lock_guard<std::mutex> lock(impl_->base_mutex);
    impl_->hbm_reader.reset();
  }
#endif
  auto s = ShutdownDdr();
  if (!s.ok()) return s;
  impl_->base_initialized = false;
  impl_->closing = false;
  return Status::OK();
}
Status NpuRuntime::ShutdownDdr() {
  if (impl_ == nullptr) return Status::OK();
  std::lock_guard<std::mutex> prepare_lock(impl_->cpu_pull_prepare_mutex);
  impl_->ddr_closing=true;
  if (impl_->runtime_core && impl_->runtime_core->needs_cleanup() &&
      aclrtSetDevice(impl_->options.device_id)!=ACL_SUCCESS)
    return Status::Internal("select Runtime device for DDR shutdown");
  Status status = impl_->runtime_core != nullptr && impl_->runtime_core->needs_cleanup()
                      ? impl_->runtime_core->Shutdown()
                      : Status::OK();
  if (status.ok() && impl_->peer_link)
    status = impl_->peer_link->CloseExecution();
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  if (status.ok()) impl_->cpu_pull.reset();
#endif
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  if (status.ok()) impl_->bandwidth_probe.reset();
#endif
  if (status.ok()) impl_->ddr_closing=false;
  return status;
}

bool NpuRuntime::initialized() const { return impl_ && impl_->base_initialized; }
bool NpuRuntime::ddr_initialized() const {
  return impl_ != nullptr && impl_->runtime_core != nullptr && impl_->runtime_core->initialized();
}

RuntimeCapabilities NpuRuntime::capabilities() const {
  RuntimeCapabilities result{!BuiltGetConfigs(impl_->options.transport_mode).empty(),
      !BuiltPutConfigs(impl_->options.transport_mode).empty(), false, true};
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  result.cpu_offload=impl_->options.transport_mode!=TransportMode::Ubmem;
#endif
#if defined(KVAF_ENABLE_NPU_PULL)
  result.npu_pull=true;
#endif
  return result;
}

std::optional<GetExecutionConfig> NpuRuntime::default_get() const {
  return impl_->options.default_get;
}

}  // namespace kvaf
