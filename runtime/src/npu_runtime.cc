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
#include <unistd.h>
#include <utility>
#include <vector>

#include "batch_get_abi.h"
#include "access_operator.h"
#include "peer_link.h"
#include "meta_manager.h"

namespace kvaf {
namespace {

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

constexpr char kUbDevIdPath[] = "/proc/asdrv_ub/dev_id";
constexpr char kUbPairInfoPath[] = "/proc/asdrv_ub/pair_info";
constexpr char kUbSelectorLockPath[] = "/tmp/kvaf_npu_runtime_asdrv_ub.lock";

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

bool ParseJsonUnsigned(const std::string& line, const char* field, std::uint32_t* value) {
  if (value == nullptr) return false;
  const std::string marker = std::string{"\""} + field + "\"";
  const std::size_t field_position = line.find(marker);
  const std::size_t colon = field_position == std::string::npos
                                ? std::string::npos
                                : line.find(':', field_position + marker.size());
  const std::size_t first = colon == std::string::npos
                                ? std::string::npos
                                : line.find_first_of("0123456789", colon + 1U);
  if (first == std::string::npos) return false;
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(line.c_str() + first, &end, 10);
  if (end == line.c_str() + first || parsed > std::numeric_limits<std::uint32_t>::max()) return false;
  *value = static_cast<std::uint32_t>(parsed);
  return true;
}

bool ParseJsonString(const std::string& line, const char* field, std::string* value) {
  if (value == nullptr) return false;
  const std::string marker = std::string{"\""} + field + "\"";
  const std::size_t field_position = line.find(marker);
  const std::size_t colon = field_position == std::string::npos
                                ? std::string::npos
                                : line.find(':', field_position + marker.size());
  const std::size_t first_quote = colon == std::string::npos
                                      ? std::string::npos
                                      : line.find('"', colon + 1U);
  const std::size_t second_quote = first_quote == std::string::npos
                                       ? std::string::npos
                                       : line.find('"', first_quote + 1U);
  if (second_quote == std::string::npos) return false;
  *value = line.substr(first_quote + 1U, second_quote - first_quote - 1U);
  return true;
}

bool ParseEid(const std::string& text,
              std::array<std::uint8_t, peer::kEidBytes>* eid) {
  if (eid == nullptr || text.size() != eid->size() * 2U) return false;
  const auto hex = [](char value) -> int {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
  };
  for (std::size_t index = 0; index < eid->size(); ++index) {
    const int high = hex(text[index * 2U]);
    const int low = hex(text[index * 2U + 1U]);
    if (high < 0 || low < 0) return false;
    (*eid)[index] = static_cast<std::uint8_t>((high << 4U) | low);
  }
  return true;
}

bool ParsePairInfoEid(const std::string& text,
                      std::array<std::uint8_t, peer::kEidBytes>* eid) {
  if (eid == nullptr) return false;
  std::string compact;
  compact.reserve(eid->size() * 2U);
  for (const unsigned char value : text) {
    if (std::isxdigit(value)) {
      compact.push_back(static_cast<char>(value));
    } else if (value != ':') {
      return false;
    }
  }
  return ParseEid(compact, eid);
}

std::string Trim(std::string text) {
  const std::size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  const std::size_t last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1U);
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

bool ReadHcclClosEids(const std::string& path, std::uint32_t physical_device_id,
                      std::vector<std::array<std::uint8_t, peer::kEidBytes>>* eids,
                      std::string* error) {
  std::ifstream input(path);
  if (!input.is_open()) {
    *error = "cannot open " + path;
    return false;
  }
  bool selected_rank = false;
  bool found_rank = false;
  bool clos_level_one = false;
  bool clos_type = false;
  bool expect_eid_address = false;
  std::string line;
  while (std::getline(input, line)) {
    std::uint32_t value = 0U;
    if (ParseJsonUnsigned(line, "device_id", &value)) {
      selected_rank = value == physical_device_id;
      found_rank = found_rank || selected_rank;
      clos_level_one = clos_type = expect_eid_address = false;
      continue;
    }
    if (!selected_rank) continue;
    if (ParseJsonUnsigned(line, "net_layer", &value)) {
      clos_level_one = value == 1U;
      clos_type = expect_eid_address = false;
      continue;
    }
    if (!clos_level_one) continue;
    if (line.find("\"net_type\"") != std::string::npos) {
      clos_type = line.find("\"CLOS\"") != std::string::npos;
      continue;
    }
    if (clos_type && line.find("\"addr_type\"") != std::string::npos) {
      expect_eid_address = line.find("\"EID\"") != std::string::npos;
      continue;
    }
    if (!expect_eid_address) continue;
    std::string eid_text;
    if (!ParseJsonString(line, "addr", &eid_text)) continue;
    expect_eid_address = false;
    std::array<std::uint8_t, peer::kEidBytes> eid{};
    if (!ParseEid(eid_text, &eid)) {
      *error = "invalid CLOS EID for device_id=" + std::to_string(physical_device_id);
      return false;
    }
    if (std::find(eids->begin(), eids->end(), eid) == eids->end()) eids->push_back(eid);
  }
  if (!found_rank || eids->empty()) {
    *error = found_rank ? "no CLOS EID at net_layer=1 for physical device_id=" +
                              std::to_string(physical_device_id)
                        : "no rank for physical device_id=" + std::to_string(physical_device_id);
    return false;
  }
  return true;
}

bool ReadPairInfoDeviceEid(std::uint32_t physical_device_id,
                           std::array<std::uint8_t, peer::kEidBytes>* eid,
                           std::string* error) {
  if (eid == nullptr || error == nullptr) return false;
  const int lock_fd = open(kUbSelectorLockPath, O_CREAT | O_RDWR, 0600);
  if (lock_fd < 0 || flock(lock_fd, LOCK_EX) != 0) {
    *error = "cannot lock " + std::string(kUbSelectorLockPath) + ": " + std::strerror(errno);
    if (lock_fd >= 0) close(lock_fd);
    return false;
  }
  bool success = false;
  {
    std::ofstream selector(kUbDevIdPath);
    if (!selector.is_open()) {
      *error = "cannot open " + std::string(kUbDevIdPath);
    } else {
      selector << physical_device_id << '\n';
      selector.close();
      std::ifstream pair_info(kUbPairInfoPath);
      if (!pair_info.is_open()) {
        *error = "cannot open " + std::string(kUbPairInfoPath);
      } else {
        const std::string selected_marker = "dev_id=" + std::to_string(physical_device_id) + ";";
        bool selected_device = false;
        bool active_channel = false;
        std::uint32_t active_channels = 0U;
        std::uint32_t device_eids = 0U;
        std::string line;
        while (std::getline(pair_info, line)) {
          if (line.find("Dev id info.") != std::string::npos) {
            selected_device = line.find(selected_marker) != std::string::npos;
            active_channel = false;
            continue;
          }
          if (!selected_device) continue;
          if (line.find("chan_id=") != std::string::npos) {
            active_channel = line.find("flag=1") != std::string::npos;
            if (active_channel) ++active_channels;
            continue;
          }
          constexpr char kRemotePrefix[] = "remote_eid:";
          const std::size_t remote_pos = line.find(kRemotePrefix);
          if (active_channel && remote_pos != std::string::npos &&
              ParsePairInfoEid(Trim(line.substr(remote_pos + sizeof(kRemotePrefix) - 1U)), eid)) {
            ++device_eids;
          }
        }
        success = selected_device && active_channels == 1U && device_eids == 1U;
        if (!success) {
          *error = "expected one active pair_info device EID for physical device_id=" +
                   std::to_string(physical_device_id) + "; active=" +
                   std::to_string(active_channels) + " device_eids=" + std::to_string(device_eids);
        }
      }
    }
  }
  (void)flock(lock_fd, LOCK_UN);
  close(lock_fd);
  return success;
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

bool ParseConfiguredNpuLocalEids(
    std::uint32_t logical_device_id, const std::vector<std::string>& configured,
    std::vector<NpuLocalEidCandidate>* candidates, std::string* error) {
  if (candidates == nullptr || error == nullptr || configured.empty()) return false;
  candidates->clear();
  std::uint32_t physical_device_id = 0U;
  if (!ResolvePhysicalDeviceId(logical_device_id, &physical_device_id)) {
    *error = "cannot resolve physical device from logical device " +
             std::to_string(logical_device_id);
    return false;
  }
  for (std::size_t index = 0U; index < configured.size(); ++index) {
    std::array<std::uint8_t, peer::kEidBytes> eid{};
    if (!ParseEid(configured[index], &eid) ||
        std::all_of(eid.begin(), eid.end(),
                    [](std::uint8_t byte) { return byte == 0U; })) {
      *error = "npu_eids[" + std::to_string(index) +
               "] must be a non-zero 32-digit hexadecimal EID";
      return false;
    }
    if (std::find_if(candidates->begin(), candidates->end(),
                     [&eid](const NpuLocalEidCandidate& candidate) {
                       return candidate.eid == eid;
                     }) != candidates->end()) {
      *error = "npu_eids contains duplicate EID " + configured[index];
      return false;
    }
    AddNpuLocalEidCandidate(physical_device_id, eid,
                            "explicit_npu_eid_" + std::to_string(index),
                            candidates);
  }
  return true;
}

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
    config.batch_get_stream_count = options.batch_get_stream_count;
    config.batch_get_transport_count = options.batch_get_transport_count;
    // BatchPut is intentionally unchanged and keeps its existing two-Jetty
    // maximum when BatchGet selects three to sixty-four lanes.
    config.batch_put_jettys_per_core =
        std::min(options.jettys_per_core, std::uint32_t{2U});
  }

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
  void* probe_buffer{nullptr};
};

NpuRuntime::NpuRuntime(NpuRuntimeOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {
  if (impl_->config.batch_get_kernel_object_path.empty() || impl_->options.npu_core_count == 0U ||
      impl_->options.npu_core_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES ||
      impl_->options.jettys_per_core == 0U ||
      impl_->options.jettys_per_core > KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE ||
      (impl_->options.batch_get_stream_count != 1U &&
       impl_->options.batch_get_stream_count != 2U) ||
      (impl_->options.batch_get_transport_count != 1U &&
       impl_->options.batch_get_transport_count != 2U) ||
      impl_->options.batch_get_transport_count > impl_->options.batch_get_stream_count ||
      static_cast<std::uint64_t>(impl_->options.npu_core_count) *
              impl_->options.batch_get_stream_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES ||
      impl_->options.batch_get_token_bytes == 0U ||
      impl_->options.batch_get_tokens_per_block == 0U) {
    throw std::invalid_argument("invalid NPU Runtime configuration");
  }
  if (aclrtSetDevice(impl_->options.device_id) != ACL_SUCCESS) {
    throw std::runtime_error("aclrtSetDevice failed for NpuRuntime");
  }
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
  if (target.address.empty()) return Status::InvalidArgument("NpuRuntime LinkTo requires peer host:port");
  if (initialized()) return Status::InvalidState("NpuRuntime LinkTo must run before initialize");
  if (impl_->peer_link != nullptr) return Status::InvalidState("NpuRuntime already has a PeerLink");
  if (impl_->config.batch_put_kernel_object_path.empty()) {
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
  const RuntimeInitRequest request{ddr_bytes, slot_size, DeviceKind::kCpu};
  if (initialized()) return Status::InvalidState("NpuRuntime has already been initialized");
  if (impl_->peer_link == nullptr || impl_->peer_allocator == nullptr ||
      impl_->runtime_core == nullptr) {
    return Status::InvalidState("NpuRuntime requires LinkTo before Initialize");
  }

  std::uint32_t candidate_count = 0U;
  Status last = impl_->peer_link->GetCandidateCount(&candidate_count);
  if (!last.ok()) return last;
  if (candidate_count == 0U) {
    return Status::NotFound("peer Runtime returned no UMDK/EID candidates");
  }
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
      impl_->options.batch_get_stream_count * impl_->options.jettys_per_core /
      impl_->options.batch_get_transport_count;
  if (explicit_eids && npu_candidates.size() > worker_count) {
    return Status::InvalidArgument(
        "npu_eids count exceeds BatchGet workers per transport");
  }
  // Automatic discovery retains the established first-working-candidate
  // behavior. An explicit list is one atomic multi-EID group: every path must
  // initialize and pass the all-lane RMA/CQ probe, with no silent EID fallback.
  const std::size_t npu_group_count = explicit_eids ? 1U : npu_candidates.size();
  for (std::uint32_t peer_candidate = 0U; peer_candidate < candidate_count; ++peer_candidate) {
    impl_->peer_allocator->SetCandidate(peer_candidate);
    for (std::size_t npu_group = 0U; npu_group < npu_group_count; ++npu_group) {
      const std::size_t group_begin = explicit_eids ? 0U : npu_group;
      const std::size_t group_size = explicit_eids ? npu_candidates.size() : 1U;
      impl_->runtime_core.reset();
      impl_->access_operator.reset();
      impl_->ReleaseProbe();
      impl_->config.peer_local_eids.clear();
      impl_->config.peer_local_eids.reserve(group_size);
      std::string group_sources;
      std::string group_eids;
      for (std::size_t offset = 0U; offset < group_size; ++offset) {
        const NpuLocalEidCandidate& candidate = npu_candidates[group_begin + offset];
        impl_->config.peer_local_eids.push_back(candidate.eid);
        if (offset != 0U) {
          group_sources += ',';
          group_eids += ',';
        }
        group_sources += candidate.source;
        group_eids += EidToHex(candidate.eid);
      }
      // BatchPut intentionally remains on the established single-EID path.
      // Multi-EID striping applies only to BatchGet and uses the vector above.
      impl_->config.peer_local_eid = impl_->config.peer_local_eids.front();
      impl_->CreateCore();
      if (group_size == 1U) {
        std::fprintf(stderr,
                     "[KVAF_AIV] PEER_LINK_EID_PAIR_ATTEMPT peer_candidate=%u/%u "
                     "npu_candidate=%zu/%zu npu_source=%s npu_eid=%s strict=%u\n",
                     peer_candidate, candidate_count, group_begin,
                     npu_candidates.size(), group_sources.c_str(), group_eids.c_str(),
                     explicit_eids ? 1U : 0U);
      } else {
        std::fprintf(stderr,
                     "[KVAF_AIV] PEER_LINK_EID_GROUP_ATTEMPT peer_candidate=%u/%u "
                     "npu_group=%zu/%zu eid_paths=%zu npu_sources=%s npu_eids=%s "
                     "distribution=jetty_major_worker_round_robin strict=1\n",
                     peer_candidate, candidate_count, npu_group, npu_group_count,
                     group_size, group_sources.c_str(), group_eids.c_str());
      }
      last = impl_->runtime_core->Initialize(request);
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
            probe_keys.push_back(std::numeric_limits<KeyId>::max() - row);
            probe_offsets.push_back(row * probe_bytes);
            probe_lengths.push_back(probe_bytes);
          }
          if (group_size == 1U) {
            std::fprintf(stderr,
                         "[KVAF_AIV] PEER_LINK_EID_PAIR_RMA_PROBE_BEGIN "
                         "peer_candidate=%u/%u npu_candidate=%zu npu_source=%s "
                         "rows=%llu bytes_per_row=%llu source_bytes=%llu cores=%u "
                         "jettys_per_core=%u\n",
                         peer_candidate, candidate_count, group_begin,
                         group_sources.c_str(),
                         static_cast<unsigned long long>(probe_rows),
                         static_cast<unsigned long long>(probe_bytes),
                         static_cast<unsigned long long>(probe_total_bytes),
                         impl_->options.npu_core_count,
                         impl_->config.batch_put_jettys_per_core);
          } else {
            std::fprintf(stderr,
                         "[KVAF_AIV] PEER_LINK_EID_GROUP_RMA_PROBE_BEGIN "
                         "peer_candidate=%u/%u npu_group=%zu eid_paths=%zu "
                         "npu_sources=%s rows=%llu bytes_per_row=%llu "
                         "source_bytes=%llu cores=%u jettys_per_core=%u\n",
                         peer_candidate, candidate_count, npu_group, group_size,
                         group_sources.c_str(),
                         static_cast<unsigned long long>(probe_rows),
                         static_cast<unsigned long long>(probe_bytes),
                         static_cast<unsigned long long>(probe_total_bytes),
                         impl_->options.npu_core_count,
                         impl_->config.batch_put_jettys_per_core);
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
          if (last.ok()) {
            if (group_size == 1U) {
              std::fprintf(stderr,
                           "[KVAF_AIV] PEER_LINK_EID_PAIR_RMA_PROBE_READY "
                           "peer_candidate=%u/%u npu_candidate=%zu npu_source=%s "
                           "rows=%llu source_bytes=%llu\n",
                           peer_candidate, candidate_count, group_begin,
                           group_sources.c_str(),
                           static_cast<unsigned long long>(probe_rows),
                           static_cast<unsigned long long>(probe_total_bytes));
            } else {
              std::fprintf(stderr,
                           "[KVAF_AIV] PEER_LINK_EID_GROUP_RMA_PROBE_READY "
                           "peer_candidate=%u/%u npu_group=%zu eid_paths=%zu "
                           "npu_sources=%s rows=%llu source_bytes=%llu\n",
                           peer_candidate, candidate_count, npu_group, group_size,
                           group_sources.c_str(),
                           static_cast<unsigned long long>(probe_rows),
                           static_cast<unsigned long long>(probe_total_bytes));
            }
          }
        }
        if (stream != nullptr && aclrtDestroyStream(stream) != ACL_SUCCESS && last.ok()) {
          last = Internal("failed to destroy peer candidate probe stream");
        }
      }
      if (last.ok()) {
        if (group_size == 1U) {
          std::fprintf(stderr,
                       "[KVAF_AIV] PEER_LINK_EID_PAIR_SELECTED peer_candidate=%u/%u "
                       "npu_candidate=%zu/%zu npu_source=%s npu_eid=%s "
                       "validation=aiv_write_cq_all_lanes\n",
                       peer_candidate, candidate_count, group_begin,
                       npu_candidates.size(), group_sources.c_str(),
                       group_eids.c_str());
        } else {
          std::fprintf(stderr,
                       "[KVAF_AIV] PEER_LINK_EID_GROUP_SELECTED peer_candidate=%u/%u "
                       "npu_group=%zu/%zu eid_paths=%zu npu_sources=%s npu_eids=%s "
                       "distribution=jetty_major_worker_round_robin "
                       "validation=batch_get_channels_plus_primary_eid_aiv_write_cq\n",
                       peer_candidate, candidate_count, npu_group, npu_group_count,
                       group_size, group_sources.c_str(), group_eids.c_str());
        }
        return Status::OK();
      }
      if (group_size == 1U) {
        std::fprintf(stderr,
                     "[KVAF_AIV] PEER_LINK_EID_PAIR_FAILED peer_candidate=%u "
                     "npu_candidate=%zu npu_source=%s reason=%s\n",
                     peer_candidate, group_begin, group_sources.c_str(),
                     last.message().c_str());
      } else {
        std::fprintf(stderr,
                     "[KVAF_AIV] PEER_LINK_EID_GROUP_FAILED peer_candidate=%u "
                     "npu_group=%zu eid_paths=%zu npu_sources=%s npu_eids=%s "
                     "reason=%s\n",
                     peer_candidate, npu_group, group_size, group_sources.c_str(),
                     group_eids.c_str(), last.message().c_str());
      }
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
std::uint32_t NpuRuntime::batch_get_stream_count() const {
  return impl_->options.batch_get_stream_count;
}
std::uint32_t NpuRuntime::batch_get_transport_count() const {
  return impl_->options.batch_get_transport_count;
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

Status NpuRuntime::SetBatchGetKernelProfiling(bool enabled) {
  if (impl_->access_operator == nullptr) {
    return Status::NotInitialized(
        "NpuRuntime requires LinkTo before configuring BatchGet profiling");
  }
  return impl_->access_operator->SetBatchGetKernelProfiling(enabled);
}

Status NpuRuntime::CopyBatchGetKernelProfiles(
    std::vector<BatchGetKernelProfile>* profiles) const {
  if (impl_->access_operator == nullptr) {
    return Status::NotInitialized(
        "NpuRuntime requires LinkTo before reading BatchGet profiling");
  }
  return impl_->access_operator->CopyBatchGetKernelProfiles(profiles);
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
  impl_->access_operator.reset();
  return Status::OK();
}

Status NpuRuntime::ServeConnection(int) {
  return Status::NotImplemented("NpuRuntime has no incoming control service in this deployment");
}

Status NpuRuntime::Shutdown() {
  if (impl_ == nullptr) return Status::OK();
  const Status status = impl_->runtime_core != nullptr && impl_->runtime_core->initialized()
                            ? impl_->runtime_core->Shutdown()
                            : Status::OK();
  impl_->ReleaseProbe();
  return status;
}

bool NpuRuntime::initialized() const {
  return impl_ != nullptr && impl_->runtime_core != nullptr && impl_->runtime_core->initialized();
}

RuntimeCapabilities NpuRuntime::capabilities() const {
  return RuntimeCapabilities{true, !impl_->config.batch_put_kernel_object_path.empty(), false, true};
}

}  // namespace kvaf
