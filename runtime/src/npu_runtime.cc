#include "npu_runtime.h"
#include "npu_eid_selection.h"
#include "execution_dispatcher.h"
#include "session_access_operator.h"

#include <acl/acl.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
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

using AclrtGetPhyDevIdByLogicDevId = int (*)(int32_t, int32_t*);

bool ResolvePhysicalDeviceId(std::uint32_t logical_device_id, std::uint32_t* physical_device_id) {
  if (physical_device_id == nullptr ||
      logical_device_id > static_cast<std::uint32_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }
  auto get_physical = reinterpret_cast<AclrtGetPhyDevIdByLogicDevId>(
      dlsym(RTLD_DEFAULT, "aclrtGetPhyDevIdByLogicDevId"));
  if (get_physical == nullptr) {
    void* ascendcl = dlopen("libascendcl.so", RTLD_NOW | RTLD_NOLOAD);
    if (ascendcl == nullptr) ascendcl = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
    if (ascendcl != nullptr) {
      get_physical = reinterpret_cast<AclrtGetPhyDevIdByLogicDevId>(
          dlsym(ascendcl, "aclrtGetPhyDevIdByLogicDevId"));
    }
  }
  int32_t physical = -1;
  if (get_physical == nullptr ||
      get_physical(static_cast<int32_t>(logical_device_id), &physical) != 0 || physical < 0) {
    return false;
  }
  *physical_device_id = static_cast<std::uint32_t>(physical);
  return true;
}

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
    return detail::ResolveExecutionResources(transport, executor, get, selected);
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
      if (r.jettys && (e != Executor::Aiv || t != Transport::Urma))
        throw std::invalid_argument("Jetty budget is configurable only for URMA AIV");
      if (r.max_inflight_total && (!get || e != Executor::Aiv || t != Transport::Urma))
        throw std::invalid_argument("in-flight limit is configurable only for URMA AIV Get");
      if (r.cpu_threads && (!get || algorithm != GetAlgorithm::Gather || e != Executor::Aicpu))
        throw std::invalid_argument("CPU thread budget requires AICPU Gather");
      if (!r.kernel_path.empty() && e == Executor::Aicpu)
        throw std::invalid_argument("AICPU kernel package is resolved through the CANN installation");
    }
    if (!budget.cores || budget.cores > (e == Executor::Aicpu ? 8U : 48U) ||
        !budget.request_capacity || budget.request_capacity > 65536U || budget.cpu_threads > 64U ||
        (t == Transport::Urma && e == Executor::Aiv &&
         (!budget.jettys || budget.jettys > (get ? 64U : 2U) ||
          (get && *budget.max_inflight_total && *budget.max_inflight_total < 2U * budget.cores))))
      throw std::invalid_argument("invalid execution resource budget");
    if (e == Executor::Aicpu) {
      if (get && algorithm == GetAlgorithm::Gather &&
          GatherStagingBytes(options.batch_get_token_bytes) > UINT32_MAX)
        throw std::invalid_argument("AICPU Gather payload must fit uint32");
    } else {
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
    ValidateProviderResources(transport, executor, algorithm, get, budget);
    q.candidate = selected_candidate;
    q.workers = budget.cores;
    q.jettys = budget.jettys;
    q.request_capacity = budget.request_capacity;
    q.token_bytes = options.batch_get_token_bytes;
    q.cpu_threads = budget.cpu_threads;
    q.auxiliary_bytes = 0;
    const auto eids = ProviderEids(transport, executor, get, budget);
    if (!eids.empty()) q.local_eid = eids.front();
    return std::make_unique<SessionAccessOperator>(
        peer_link, q,
        [this, transport, executor, algorithm, get](std::shared_ptr<peer::PeerLink> link) {
          return MakeLocalProvider(transport, executor, algorithm, get, std::move(link));
        },
        budget.request_capacity);
  }
  std::vector<detail::NpuEid> ProviderEids(Transport transport, Executor executor, bool get,
                                         const ExecutionResources& budget) const {
    std::vector<detail::NpuEid> selected;
    const auto status = detail::SelectProviderEids(transport, executor, get,
        !options.npu_eids.empty(), budget, local_eids, &selected);
    if (!status.ok()) throw std::invalid_argument(status.message());
    return selected;
  }
  std::unique_ptr<AccessOperator> MakeLocalProvider(Transport transport, Executor executor,
                                                    GetAlgorithm algorithm, bool get,
                                                    std::shared_ptr<peer::PeerLink> link) {
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
      c.peer_link = link;
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
        eids.front(), get});
#else
    throw std::invalid_argument("requested execution provider was not built");
#endif
  }

  NpuRuntimeOptions options;
  detail::AicpuKernelPackage kernel_package;
  std::mutex aicpu_package_mutex;
  bool aicpu_package_ready{false};
  std::uint32_t selected_candidate = 0;
  std::vector<detail::NpuEid> local_eids;
  std::unique_ptr<MetaManager> meta_manager;
  std::unique_ptr<DdrMemoryAllocator> allocator;
  std::shared_ptr<peer::PeerLink> peer_link;
#if defined(KVAF_ENABLE_AIV_BATCH_GET)
  std::shared_ptr<ubmem::DeviceMappingCache> mappings =
      std::make_shared<ubmem::DeviceMappingCache>();
#endif
  std::unique_ptr<ExecutionDispatcher> dispatcher;
  std::unique_ptr<RuntimeCore> runtime_core;
};

NpuRuntime::NpuRuntime(NpuRuntimeOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {
  const auto& config = impl_->options;
  if (config.batch_get_token_bytes == 0 || config.batch_get_tokens_per_block == 0)
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
  if (initialized()) return Status::InvalidState("NpuRuntime LinkTo must run before initialize");
  if (impl_->peer_link != nullptr) return Status::InvalidState("NpuRuntime already has a PeerLink");
  impl_->peer_link = std::make_shared<peer::PeerLink>(target.address);
  impl_->allocator = std::make_unique<PeerDdrAllocator>(impl_->peer_link, impl_->options.transport_mode);
  impl_->runtime_core.reset();
  impl_->dispatcher.reset();
  impl_->CreateCore();
  return Status::OK();
}

Status NpuRuntime::Initialize(std::uint64_t ddr_bytes, std::uint64_t slot_size) {
  const RuntimeInitRequest request{ddr_bytes, slot_size, DeviceKind::kCpu};
  if (initialized()) return Status::InvalidState("NpuRuntime has already been initialized");
  if (impl_->peer_link == nullptr || impl_->allocator == nullptr) {
    return Status::InvalidState("NpuRuntime requires LinkTo before Initialize");
  }
  if (ddr_bytes == 0U || slot_size == 0U || ddr_bytes % slot_size != 0U) {
    return Status::InvalidArgument("DDR size must be a nonzero multiple of slot_size");
  }

  if (impl_->runtime_core != nullptr && impl_->runtime_core->needs_cleanup()) {
    const Status status = Shutdown();
    if (!status.ok()) return status;
  }
  // Cache allocation does not initialize a transport or prepare an algorithm.
  // Keep the entire explicit EID list until the selected provider applies its
  // own routing and worker budget; the default Get must not choose that policy.
  impl_->local_eids.clear();
  impl_->selected_candidate = 0;
  if (impl_->options.transport_mode != TransportMode::Ubmem) {
    peer::PeerCandidateTable table;
    auto status = impl_->peer_link->DiscoverCandidates(&table);
    if (!status.ok()) return status;
    const auto& options = impl_->options;
    std::uint32_t physical_device;
    if (!ResolvePhysicalDeviceId(options.device_id, &physical_device))
      return Status::InvalidArgument("cannot resolve physical NPU from device " + std::to_string(options.device_id));
    const bool automatic_cpu = !table.device_fixed && options.peer_ub_device.empty();
    detail::PairEids pair;
    if (table.local && (options.npu_eids.empty() || automatic_cpu))
      pair = detail::ReadDevicePair(physical_device);
    status = detail::SelectNpuEids(physical_device, options.npu_eids, pair, &impl_->local_eids);
    if (!status.ok()) return status;
    peer::PeerCandidateInfo cpu;
    status = impl_->peer_link->SelectCpuCandidate(table, options.peer_ub_device, pair.cpu, &cpu);
    if (!status.ok()) return status;
    impl_->selected_candidate = cpu.candidate_id;
    const char* reason = "first_candidate";
    if (!automatic_cpu) reason = "fixed_device";
    else if (table.local) reason = pair.cpu && *pair.cpu == cpu.eid ? "pair_match" : "pair_fallback";
    std::fprintf(stderr, "[KVAF] PEER_LINK_EID_SELECTED scene=%s physical_device=%u candidate=%u "
                 "ub_device=%s eid_index=%u cpu_eid=%s npu_eid=%s cpu_selection=%s\n",
                 table.local ? "local" : "remote", physical_device, cpu.candidate_id,
                 cpu.device_name.c_str(), cpu.eid_index, detail::EidToHex(cpu.eid).c_str(),
                 detail::EidToHex(impl_->local_eids.front()).c_str(), reason);
  }
  impl_->runtime_core.reset();
  impl_->dispatcher.reset();
  impl_->CreateCore();
  return impl_->runtime_core->Initialize(request);
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
  if (!initialized()) return Status::NotInitialized("initialize Runtime before draining Get");
  return core().DrainGet();
}
Status NpuRuntime::Recover(PutExecutionConfig c) {
  if (aclrtSetDevice(impl_->options.device_id) != ACL_SUCCESS)
    return Status::Internal("aclrtSetDevice failed");
  if (!initialized()) return Status::NotInitialized("initialize Runtime before recovery");
  return core().RecoverPut(c);
}
Status NpuRuntime::Recover(GetExecutionConfig c) {
  if (aclrtSetDevice(impl_->options.device_id) != ACL_SUCCESS)
    return Status::Internal("aclrtSetDevice failed");
  if (!initialized()) return Status::NotInitialized("initialize Runtime before recovery");
  return core().RecoverGet(c);
}
Status NpuRuntime::Warmup(PutExecutionConfig c, const ExecutionWarmup& request) {
  if (!initialized()) return Status::NotInitialized("initialize DDR before warmup");
  return impl_->dispatcher->Warmup(c, request);
}
Status NpuRuntime::Warmup(GetExecutionConfig c, const ExecutionWarmup& request) {
  if (!initialized()) return Status::NotInitialized("initialize DDR before warmup");
  return impl_->dispatcher->Warmup(c, request);
}

Status NpuRuntime::SetBatchGetKernelProfiling(
    bool enabled, bool detailed, std::optional<GetExecutionConfig> config) {
  if (!initialized() || !impl_->dispatcher)
    return Status::NotInitialized("initialize Runtime before configuring Get profiling");
  return impl_->dispatcher->ConfigureGetProfiling(config, enabled, detailed);
}

Status NpuRuntime::CheckBatchGetStatus(void* stream, std::optional<GetExecutionConfig> config,
                                      std::uint64_t graph_lease) {
  if (!initialized() || !impl_->dispatcher)
    return Status::NotInitialized("NpuRuntime is not initialized");
  return impl_->dispatcher->CheckGetStatus(config, stream, graph_lease);
}

Status NpuRuntime::CopyBatchGetKernelProfiles(
    std::vector<BatchGetKernelProfile>* profiles, std::optional<GetExecutionConfig> config,
    std::uint64_t graph_lease) const {
  if (!initialized() || !impl_->dispatcher)
    return Status::NotInitialized("initialize Runtime before reading Get profiling");
  return impl_->dispatcher->CopyGetProfiles(config, profiles, graph_lease);
}

Status NpuRuntime::AcceptLink(const LinkTarget&) {
  return Status::NotImplemented("NpuRuntime does not accept links in the CPU--NPU deployment");
}

Status NpuRuntime::ReleaseLink(LinkId link_id) {
  if (link_id != 1U || impl_->peer_link == nullptr) return Status::NotFound("PeerLink was not found");
  const Status shutdown = Shutdown();
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
  Status status = impl_->runtime_core != nullptr && impl_->runtime_core->needs_cleanup()
                      ? impl_->runtime_core->Shutdown()
                      : Status::OK();
  if (status.ok() && impl_->peer_link)
    status = impl_->peer_link->CloseExecution();
  return status;
}

bool NpuRuntime::initialized() const {
  return impl_ != nullptr && impl_->runtime_core != nullptr && impl_->runtime_core->initialized();
}

RuntimeCapabilities NpuRuntime::capabilities() const {
  return RuntimeCapabilities{!BuiltGetConfigs(impl_->options.transport_mode).empty(),
      !BuiltPutConfigs(impl_->options.transport_mode).empty(), false, true};
}

std::optional<GetExecutionConfig> NpuRuntime::default_get() const {
  return impl_->options.default_get;
}

}  // namespace kvaf
