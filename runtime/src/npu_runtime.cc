#include "npu_runtime.h"

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
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "batch_get_abi.h"
#include "access_operator.h"
#include "peer_link.h"
#include "meta_manager.h"
#include "npu_eid_discovery.h"

namespace kvaf {
namespace {

using npu_eid_detail::ReadHcclClosEids;
using npu_eid_detail::ReadPairInfoDeviceEid;

Status Internal(std::string message) { return Status::Internal(std::move(message)); }

class PeerDdrAllocator final : public DdrMemoryAllocator {
 public:
  explicit PeerDdrAllocator(std::shared_ptr<peer::PeerLink> link)
      : link_(std::move(link)) {}

  void SetCandidate(std::uint32_t candidate_id) { candidate_id_ = candidate_id; }
  bool has_allocation() const { return allocated_.base != 0U; }

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
    const Status status = link_->ReleaseRegion(allocated_.base);
    if (status.ok()) allocated_ = {};
    return status;
  }

 private:
  std::shared_ptr<peer::PeerLink> link_;
  std::uint32_t candidate_id_{0U};
  peer::PeerRegion allocated_{};
};

using AclrtGetPhyDevIdByLogicDevId = int (*)(int32_t, int32_t*);

struct NpuLocalEidCandidate {
  std::array<std::uint8_t, peer::kEidBytes> eid{};
  std::string source;
};

std::string EidToHex(const std::array<std::uint8_t, peer::kEidBytes>& eid) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string text;
  text.reserve(eid.size() * 2U);
  for (std::uint8_t byte : eid) {
    text.push_back(kHex[byte >> 4U]);
    text.push_back(kHex[byte & 0x0fU]);
  }
  return text;
}

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

void AddNpuLocalEidCandidate(
    std::uint32_t physical_device_id,
    const std::array<std::uint8_t, peer::kEidBytes>& eid,
    std::string source, std::vector<NpuLocalEidCandidate>* candidates) {
  if (std::find_if(candidates->begin(), candidates->end(),
                   [&eid](const NpuLocalEidCandidate& candidate) {
                     return candidate.eid == eid;
                   }) != candidates->end()) {
    return;
  }
  candidates->push_back(NpuLocalEidCandidate{eid, std::move(source)});
  std::fprintf(stderr,
               "[KVAF_AIV] PEER_LINK_NPU_EID_DISCOVERED physical_device=%u source=%s eid=%s\n",
               physical_device_id, candidates->back().source.c_str(),
               EidToHex(candidates->back().eid).c_str());
}

bool DiscoverNpuLocalEidCandidates(std::uint32_t logical_device_id,
                                   std::vector<NpuLocalEidCandidate>* candidates,
                                   std::string* error) {
  if (candidates == nullptr || error == nullptr) return false;
  candidates->clear();
  std::uint32_t physical_device_id = 0U;
  const char* configured = std::getenv("KVAF_HCCL_ROOTINFO_PATH");
  const std::string rootinfo = configured != nullptr && configured[0] != '\0'
                                   ? configured
                                   : "/etc/hccl_rootinfo.json";
  std::string clos_error;
  std::vector<std::array<std::uint8_t, peer::kEidBytes>> clos_eids;
  if (!ResolvePhysicalDeviceId(logical_device_id, &physical_device_id)) {
    *error = "cannot resolve physical device from logical device " + std::to_string(logical_device_id);
    return false;
  } else {
    (void)ReadHcclClosEids(rootinfo, physical_device_id, &clos_eids, &clos_error);
  }
  for (std::size_t index = 0; index < clos_eids.size(); ++index) {
    AddNpuLocalEidCandidate(physical_device_id, clos_eids[index],
                            "hccl_clos_pg_" + std::to_string(index), candidates);
  }
  if (clos_eids.empty()) {
    std::fprintf(stderr, "[KVAF_AIV] PEER_LINK_NPU_EID_DISCOVERY_UNAVAILABLE path=%s reason=%s\n",
                 rootinfo.c_str(), clos_error.c_str());
  }
  std::array<std::uint8_t, peer::kEidBytes> pair_info_eid{};
  std::string pair_info_error;
  if (ReadPairInfoDeviceEid(physical_device_id, &pair_info_eid, &pair_info_error)) {
    AddNpuLocalEidCandidate(physical_device_id, pair_info_eid,
                            "asdrv_pair_info_device_eid", candidates);
  } else {
    std::fprintf(stderr, "[KVAF_AIV] PEER_LINK_NPU_EID_FALLBACK_UNAVAILABLE reason=%s\n",
                 pair_info_error.c_str());
  }
  if (!candidates->empty()) return true;
  *error = "no usable NPU EID candidate; CLOS=" + clos_error + "; pair_info=" + pair_info_error;
  return false;
}

}  // namespace

struct NpuRuntime::Impl {
  explicit Impl(NpuRuntimeOptions init_options) : options(std::move(init_options)) {
    config.batch_get_kernel_object_path = options.batch_get_kernel_object_path;
    config.batch_put_kernel_object_path = options.batch_put_kernel_object_path;
    config.aicpu_urma_kernel_json_path = options.aicpu_urma_kernel_json_path;
    config.npu_core_count = options.npu_core_count;
    config.jettys_per_core = options.jettys_per_core;
  }

#if defined(KVAF_BUILD_NPU_SERVICE)
  explicit Impl(NpuServiceOptions init_options)
      : options("", 0U, 0U), hbm_options(std::move(init_options)), hbm_mode(true) {}
#endif

  void CreateCore() {
    meta_manager = CreateInMemoryMetaManager();
    access_operator = std::make_unique<DefaultAccessOperator>(config);
    runtime_core = std::make_unique<RuntimeCore>(*meta_manager, *access_operator, *allocator);
  }

  void ReleaseProbe() {
    if (probe_buffer != nullptr) {
      (void)aclrtFree(probe_buffer);
      probe_buffer = nullptr;
    }
  }

  NpuRuntimeOptions options;
  AccessOperatorConfig config;
  std::unique_ptr<MetaManager> meta_manager;
  std::unique_ptr<DdrMemoryAllocator> allocator;
  PeerDdrAllocator* peer_allocator{nullptr};
  std::shared_ptr<peer::PeerLink> peer_link;
  std::unique_ptr<DefaultAccessOperator> access_operator;
  std::unique_ptr<RuntimeCore> runtime_core;
#if defined(KVAF_BUILD_NPU_SERVICE)
  std::unique_ptr<NpuService> hbm_service;
  NpuServiceOptions hbm_options{};
#endif
  bool hbm_mode{false};
  void* probe_buffer{nullptr};
};

NpuRuntime::NpuRuntime(NpuRuntimeOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {
  if ((impl_->config.batch_get_kernel_object_path.empty() &&
       impl_->config.aicpu_urma_kernel_json_path.empty()) || impl_->options.npu_core_count == 0U ||
      impl_->options.npu_core_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES ||
      impl_->options.jettys_per_core == 0U ||
      impl_->options.jettys_per_core > KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE ||
      impl_->options.batch_get_token_bytes == 0U ||
      impl_->options.batch_get_tokens_per_block == 0U) {
    throw std::invalid_argument("invalid NPU Runtime configuration");
  }
  if (aclrtSetDevice(impl_->options.device_id) != ACL_SUCCESS) {
    throw std::runtime_error("aclrtSetDevice failed for NpuRuntime");
  }
}

NpuRuntime::NpuRuntime(NpuServiceOptions options)
#if defined(KVAF_BUILD_NPU_SERVICE)
    : impl_(std::make_unique<Impl>(std::move(options))) {
  impl_->hbm_service = std::make_unique<NpuService>();
#else
    : impl_(std::make_unique<Impl>(NpuRuntimeOptions("", 0U, 0U))) {
  (void)options;
  throw std::runtime_error("NpuRuntime HBM provider was not built");
#endif
}

NpuRuntime::~NpuRuntime() {
  if (impl_ != nullptr) {
    (void)Shutdown();
    impl_->runtime_core.reset();
    impl_->access_operator.reset();
    impl_->ReleaseProbe();
  }
}

Status NpuRuntime::LinkTo(const LinkTarget& target) {
  if (impl_->hbm_mode) return Status::InvalidState("HBM provider NpuRuntime cannot LinkTo");
  if (target.address.empty()) return Status::InvalidArgument("NpuRuntime LinkTo requires peer host:port");
  if (initialized()) return Status::InvalidState("NpuRuntime LinkTo must run before initialize");
  if (impl_->peer_link != nullptr) return Status::InvalidState("NpuRuntime already has a PeerLink");
  if (impl_->config.batch_put_kernel_object_path.empty() &&
      impl_->config.aicpu_urma_kernel_json_path.empty()) {
    return Status::InvalidArgument("PeerLink requires the AIV BatchPut kernel for RMA validation");
  }
  impl_->peer_link = std::make_shared<peer::PeerLink>(target.address);
  impl_->config.peer_link = impl_->peer_link;
  auto allocator = std::make_unique<PeerDdrAllocator>(impl_->peer_link);
  impl_->peer_allocator = allocator.get();
  impl_->allocator = std::move(allocator);
  impl_->runtime_core.reset();
  impl_->access_operator.reset();
  impl_->CreateCore();
  return Status::OK();
}

Status NpuRuntime::Initialize(std::uint64_t ddr_bytes, std::uint64_t slot_size) {
  if (impl_->hbm_mode) return Status::InvalidState("HBM provider NpuRuntime uses InitializeHbmService");
  const RuntimeInitRequest request{ddr_bytes, slot_size, DeviceKind::kCpu};
  if (initialized()) return Status::InvalidState("NpuRuntime has already been initialized");
  if (impl_->peer_link == nullptr || impl_->peer_allocator == nullptr ||
      impl_->runtime_core == nullptr) {
    return Status::InvalidState("NpuRuntime requires LinkTo before Initialize");
  }
  if (!impl_->config.aicpu_urma_kernel_json_path.empty()) {
    const std::string& path = impl_->config.aicpu_urma_kernel_json_path;
    struct stat kernel_file {};
    if (stat(path.c_str(), &kernel_file) != 0 || !S_ISREG(kernel_file.st_mode) ||
        access(path.c_str(), R_OK) != 0) {
      return Status::InvalidArgument("AICPU URMA kernel JSON is missing or unreadable: " + path);
    }
  }

  std::uint32_t candidate_count = 0U;
  Status last = impl_->peer_link->GetCandidateCount(&candidate_count);
  if (!last.ok()) return last;
  if (candidate_count == 0U) {
    return Status::NotFound("peer Runtime returned no UMDK/EID candidates");
  }
  std::vector<NpuLocalEidCandidate> npu_candidates;
  std::string discovery_error;
  if (!DiscoverNpuLocalEidCandidates(impl_->options.device_id, &npu_candidates,
                                     &discovery_error)) {
    return Status::NotFound(discovery_error);
  }
  const char* probe_prefix = impl_->config.aicpu_urma_kernel_json_path.empty()
                                 ? "[KVAF_AIV]" : "[KVAF]";
  for (std::uint32_t peer_candidate = 0U; peer_candidate < candidate_count; ++peer_candidate) {
    impl_->peer_allocator->SetCandidate(peer_candidate);
    for (std::size_t npu_index = 0; npu_index < npu_candidates.size(); ++npu_index) {
      const NpuLocalEidCandidate& candidate = npu_candidates[npu_index];
      impl_->runtime_core.reset();
      impl_->access_operator.reset();
      impl_->ReleaseProbe();
      impl_->config.peer_local_eid = candidate.eid;
      impl_->CreateCore();
      std::fprintf(stderr,
                   "%s PEER_LINK_EID_PAIR_ATTEMPT peer_candidate=%u/%u npu_candidate=%zu/%zu "
                   "npu_source=%s npu_eid=%s\n",
                   probe_prefix, peer_candidate, candidate_count, npu_index, npu_candidates.size(), candidate.source.c_str(),
                   EidToHex(candidate.eid).c_str());
      last = impl_->runtime_core->Initialize(request);
      if (last.ok()) {
        const std::uint64_t probe_bytes = std::min<std::uint64_t>(slot_size, 64U);
        aclrtStream stream = nullptr;
        std::uint8_t pattern = 0xA5U;
        if (probe_bytes == 0U ||
            aclrtMalloc(&impl_->probe_buffer, probe_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
            impl_->probe_buffer == nullptr ||
            aclrtMemcpy(impl_->probe_buffer, probe_bytes, &pattern, sizeof(pattern),
                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
            aclrtCreateStream(&stream) != ACL_SUCCESS || stream == nullptr) {
          last = Internal("failed to prepare peer candidate RMA probe");
        } else {
          std::fprintf(stderr,
                       "%s PEER_LINK_EID_PAIR_RMA_PROBE_BEGIN peer_candidate=%u/%u "
                       "npu_candidate=%zu npu_source=%s bytes=%llu\n",
                       probe_prefix, peer_candidate, candidate_count, npu_index, candidate.source.c_str(),
                       static_cast<unsigned long long>(probe_bytes));
          last = impl_->runtime_core->BatchPut(
              BatchPutRequest{{BatchPutGroup{
                  {std::numeric_limits<KeyId>::max()},
                  ConstBufferView{impl_->probe_buffer, probe_bytes,
                                  DeviceKind::kNpu},
                  {0U}, {probe_bytes}}}, std::nullopt},
              BatchPutExecutionOptions{stream});
          if (last.ok()) last = impl_->runtime_core->BatchFree(BatchFreeRequest{{std::numeric_limits<KeyId>::max()}});
          if (last.ok()) {
            std::fprintf(stderr,
                         "%s PEER_LINK_EID_PAIR_RMA_PROBE_READY peer_candidate=%u/%u "
                         "npu_candidate=%zu npu_source=%s bytes=%llu\n",
                         probe_prefix, peer_candidate, candidate_count, npu_index, candidate.source.c_str(),
                         static_cast<unsigned long long>(probe_bytes));
          }
        }
        if (stream != nullptr && aclrtDestroyStream(stream) != ACL_SUCCESS && last.ok()) {
          last = Internal("failed to destroy peer candidate probe stream");
        }
      }
      if (last.ok()) {
        std::fprintf(stderr,
                     "%s PEER_LINK_EID_PAIR_SELECTED peer_candidate=%u/%u npu_candidate=%zu/%zu "
                     "npu_source=%s validation=%s\n",
                     probe_prefix, peer_candidate, candidate_count, npu_index, npu_candidates.size(), candidate.source.c_str(),
                     impl_->config.aicpu_urma_kernel_json_path.empty() ? "aiv_write_cq" : "aicpu_write_cq");
        return Status::OK();
      }
      std::fprintf(stderr,
                   "[KVAF_AIV] PEER_LINK_EID_PAIR_FAILED peer_candidate=%u npu_candidate=%zu npu_source=%s "
                   "reason=%s\n",
                   peer_candidate, npu_index, candidate.source.c_str(), last.message().c_str());
      if (impl_->runtime_core->initialized()) {
        const Status cleanup = impl_->runtime_core->Shutdown();
        if (!cleanup.ok()) return cleanup;
      }
      if (impl_->peer_allocator->has_allocation()) {
        return Status::Internal("failed peer candidate left DDR allocated");
      }
    }
  }
  // The failed probe's source buffer is no longer referenced after the
  // transport teardown above. Keep it only for a successfully selected Link.
  impl_->ReleaseProbe();
  return last;
}

Status NpuRuntime::InitializeHbmService() {
#if defined(KVAF_BUILD_NPU_SERVICE)
  if (!impl_->hbm_mode || impl_->hbm_service == nullptr) {
    return Status::InvalidState("InitializeHbmService requires HBM provider mode");
  }
  return impl_->hbm_service->Initialize(impl_->hbm_options);
#else
  return Status::NotImplemented("NpuRuntime HBM provider was not built");
#endif
}

Status NpuRuntime::GatherRead(const CpuGatherReadRequest& request) {
  if (!initialized() || impl_->hbm_mode || impl_->peer_link == nullptr ||
      impl_->config.aicpu_urma_kernel_json_path.empty())
    return Status::NotInitialized("GatherRead requires an initialized AICPU URMA NPU Runtime");
  if (request.output.device != DeviceKind::kNpu || !request.output.valid() ||
      request.stream == nullptr)
    return Status::InvalidArgument("GatherRead requires a caller-owned HBM output and ACL stream");
  const auto& region = impl_->runtime_core->remote_region();
  std::uint32_t total_bytes = 0U;
  if (!peer::ValidateGather(region.base, region.length, request.staging_address,
                            request.spans, &total_bytes))
    return Status::InvalidArgument("invalid GatherRead source or staging layout");
  const std::uint64_t scratch_offset =
      (static_cast<std::uint64_t>(total_bytes) + 63U) & ~std::uint64_t{63U};
  if (request.output.bytes < scratch_offset + 64U)
    return Status::InvalidArgument("GatherRead output needs 64-byte aligned scratch after the payload");
  peer::GatherReady ready{};
  Status status = impl_->peer_link->ConfigureGather(region.base, request.staging_address,
                                                    request.spans, &ready);
  if (!status.ok()) return status;
  return impl_->access_operator->ExecuteCpuGatherRead(CpuGatherReadExecRequest{
      request.output, request.staging_address, ready.command_address,
      ready.notify_address, ready.ack_address, ready.sequence, ready.total_bytes,
      request.stream});
}

RuntimeCore& NpuRuntime::core() {
  if (impl_->hbm_mode || impl_->runtime_core == nullptr) {
    throw std::logic_error("NpuRuntime requires LinkTo before core access");
  }
  return *impl_->runtime_core;
}
const RuntimeCore& NpuRuntime::core() const {
  if (impl_->hbm_mode || impl_->runtime_core == nullptr) {
    throw std::logic_error("NpuRuntime requires LinkTo before core access");
  }
  return *impl_->runtime_core;
}
std::uint32_t NpuRuntime::device_id() const { return impl_->options.device_id; }
std::uint32_t NpuRuntime::npu_core_count() const { return impl_->options.npu_core_count; }
std::uint32_t NpuRuntime::jettys_per_core() const { return impl_->options.jettys_per_core; }
std::uint32_t NpuRuntime::batch_get_token_bytes() const {
  return impl_->options.batch_get_token_bytes;
}
std::uint32_t NpuRuntime::batch_get_tokens_per_block() const {
  return impl_->options.batch_get_tokens_per_block;
}

Status NpuRuntime::AcceptLink(const LinkTarget& peer) {
#if defined(KVAF_BUILD_NPU_SERVICE)
  if (impl_->hbm_mode) return impl_->hbm_service->AcceptLink(peer);
#else
  (void)peer;
#endif
  return Status::NotImplemented("NpuRuntime does not accept links in the CPU--NPU deployment");
}

Status NpuRuntime::ReleaseLink(LinkId link_id) {
#if defined(KVAF_BUILD_NPU_SERVICE)
  if (impl_->hbm_mode) return impl_->hbm_service->ReleaseLink(link_id);
#endif
  if (link_id != 1U || impl_->peer_link == nullptr) return Status::NotFound("PeerLink was not found");
  const Status shutdown = Shutdown();
  if (!shutdown.ok()) return shutdown;
  impl_->peer_link.reset();
  impl_->config.peer_link.reset();
  impl_->config.peer_local_eid = {};
  impl_->peer_allocator = nullptr;
  impl_->allocator.reset();
  impl_->runtime_core.reset();
  impl_->access_operator.reset();
  return Status::OK();
}

Status NpuRuntime::ServeConnection(int connection_fd) {
#if defined(KVAF_BUILD_NPU_SERVICE)
  if (impl_->hbm_mode) return impl_->hbm_service->ServeConnection(connection_fd);
#else
  (void)connection_fd;
#endif
  return Status::NotImplemented("NpuRuntime has no incoming control service in this deployment");
}

Status NpuRuntime::Shutdown() {
  if (impl_ == nullptr) return Status::OK();
#if defined(KVAF_BUILD_NPU_SERVICE)
  if (impl_->hbm_mode) {
    return impl_->hbm_service != nullptr ? impl_->hbm_service->Shutdown() : Status::OK();
  }
#endif
  const Status status = impl_->runtime_core != nullptr && impl_->runtime_core->initialized()
                            ? impl_->runtime_core->Shutdown()
                            : Status::OK();
  impl_->ReleaseProbe();
  return status;
}

bool NpuRuntime::initialized() const {
  return impl_ != nullptr &&
#if defined(KVAF_BUILD_NPU_SERVICE)
         (impl_->hbm_mode
              ? (impl_->hbm_service != nullptr && impl_->hbm_service->initialized())
              : (impl_->runtime_core != nullptr && impl_->runtime_core->initialized()));
#else
         !impl_->hbm_mode && impl_->runtime_core != nullptr && impl_->runtime_core->initialized();
#endif
}

RuntimeCapabilities NpuRuntime::capabilities() const {
#if defined(KVAF_BUILD_NPU_SERVICE)
  if (impl_->hbm_mode) {
    return impl_->hbm_service != nullptr ? impl_->hbm_service->capabilities()
                                         : RuntimeCapabilities{};
  }
#endif
  return RuntimeCapabilities{true, !impl_->config.batch_put_kernel_object_path.empty() ||
      !impl_->config.aicpu_urma_kernel_json_path.empty(), false, true};
}

}  // namespace kvaf
