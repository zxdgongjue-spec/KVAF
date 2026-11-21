#include "npu_runtime.h"
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
#include <mutex>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
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
    if (get && (!options.batch_get_token_bytes || !options.batch_get_tokens_per_block))
      throw std::invalid_argument("ordinary cache Get requires a token layout");
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
  void EnsureCacheEids() {
    std::lock_guard<std::mutex> lock(cache_eid_mutex);
    if(cache_eids_ready)return;
    std::vector<NpuLocalEidCandidate> candidates;
    std::string error;
    if (!options.npu_eids.empty()) {
      if (!ParseConfiguredNpuLocalEids(options.device_id, options.npu_eids,
                                      &candidates, &error))
        throw std::invalid_argument(error);
    } else if (!DiscoverNpuLocalEidCandidates(options.device_id, &candidates, &error)) {
      throw std::invalid_argument(error);
    }
    for (const auto& candidate : candidates) local_eids.push_back(candidate.eid);
    cache_eids_ready=true;
  }
  std::vector<detail::NpuEid> ProviderEids(Transport transport, Executor executor, bool get,
                                         const ExecutionResources& budget) {
    if(transport==Transport::Urma)EnsureCacheEids();
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
  std::uint32_t selected_candidate = 0;
  std::vector<detail::NpuEid> local_eids;
  std::mutex cache_eid_mutex;
  bool cache_eids_ready{false};
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
#endif
};

NpuRuntime::NpuRuntime(NpuRuntimeOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {
  const auto& config = impl_->options;
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
  impl_->cache_eids_ready=false;
  impl_->selected_candidate = 0;
  if (impl_->options.transport_mode != TransportMode::Ubmem) {
    std::uint32_t candidate_count = 0U;
    std::uint32_t device_count = 0U;
    bool device_filter_applied = false;
    auto status = impl_->peer_link->GetCandidateCount(
        &candidate_count, &device_count, &device_filter_applied);
    if (!status.ok()) return status;
    if (candidate_count == 0U)
      return Status::NotFound("peer Runtime returned no UMDK/EID candidates");
    if (!impl_->options.peer_ub_device.empty()) {
      bool found = false;
      for (std::uint32_t candidate = 0; candidate < candidate_count; ++candidate) {
        peer::PeerCandidateInfo info{};
        status = impl_->peer_link->GetCandidateInfo(candidate, &info);
        if (!status.ok()) return status;
        if (info.candidate_id != candidate || info.device_id >= device_count)
          return Status::Internal("peer Runtime returned invalid candidate metadata");
        if (info.device_name == impl_->options.peer_ub_device) {
          impl_->selected_candidate = candidate;
          found = true;
          break;
        }
      }
      if (!found) return Status::NotFound(
          "peer Runtime did not advertise fixed UB device " + impl_->options.peer_ub_device);
    }

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

Status NpuRuntime::GetOffloadRegion(OffloadRegionTicket* out) {
#if defined(KVAF_ENABLE_CPU_OFFLOAD)
  if(!out)return Status::InvalidArgument("missing CPU Pull region ticket");
  if(!ddr_initialized())return Status::NotInitialized("initialize DDR before CPU Pull");
  if(impl_->closing)return Status::InvalidState("retry Runtime shutdown");
  if(impl_->options.transport_mode==TransportMode::Ubmem)
    return Status::InvalidState("CPU Pull requires DDR URMA capability");
  std::lock_guard<std::mutex> lock(impl_->cpu_pull_prepare_mutex);
  if(impl_->ddr_closing)return Status::InvalidState("retry DDR shutdown before exporting a ticket");
  if(impl_->cpu_pull && impl_->cpu_pull->ticket().region_id) {
    *out=impl_->cpu_pull->ticket();return Status::OK();
  }
  if(!impl_->cpu_pull)impl_->cpu_pull=std::make_unique<RemotePullOperator>(impl_->peer_link,impl_->selected_candidate);
  auto status=core().SetCpuPullExecutor(impl_->cpu_pull.get());
  if(!status.ok())return status;
  status=impl_->cpu_pull->Initialize(core().remote_region());
  if(!status.ok())return status;
  *out=impl_->cpu_pull->ticket();return Status::OK();
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
        o.max_inflight_total,kernel,o.max_host_inflight,o.max_output_bindings,o.diagnostics);
  }
  return impl_->hbm_reader->Listen(address,advertised,local_eid,receiver);
#else
  return Status::NotImplemented("NPU Pull was not built");
#endif
}
Status NpuRuntime::BatchGetRawAddr(const ReadyToken& ready,NpuInt64ArrayView src,NpuInt64ArrayView dst,
    NpuInt64ArrayView lengths,MutableBufferView output,void* stream,HbmReadProfile* profile,
    bool validate_ranges,bool synchronize,std::uint64_t graph_lease,RawReadLifetime lifetime) {
#if defined(KVAF_ENABLE_NPU_PULL)
  auto reader=impl_->Reader();
  if(!initialized() || !reader)return Status::NotInitialized("listen and register source before raw READ");
  if(impl_->closing)return Status::InvalidState("retry Runtime shutdown");
  return reader->Read(ready,src,dst,lengths,output,stream,profile,validate_ranges,
                                 synchronize,graph_lease,std::move(lifetime));
#else
  return Status::NotImplemented("NPU Pull was not built");
#endif
}
Status NpuRuntime::DrainRawAddr(const ReadyToken& ready) {
#if defined(KVAF_ENABLE_NPU_PULL)
  auto reader=impl_->Reader();
  if(reader)return reader->Drain(ready);
#endif
  return Status::NotInitialized("NPU Pull is not initialized");
}
Status NpuRuntime::BeginRawGraph(const ReadyToken& ready,NpuInt64ArrayView src,NpuInt64ArrayView dst,
    NpuInt64ArrayView lengths,MutableBufferView output,void* stream,std::uint64_t* lease,RawReadLifetime lifetime) {
#if defined(KVAF_ENABLE_NPU_PULL)
  auto reader=impl_->Reader();
  if(!initialized() || !reader)return Status::NotInitialized("NPU Pull is not initialized");
  if(impl_->closing)return Status::InvalidState("retry Runtime shutdown");
  return reader->BeginGraph(ready,src,dst,lengths,output,stream,lease,std::move(lifetime));
#else
  return Status::NotImplemented("NPU Pull was not built");
#endif
}
Status NpuRuntime::EndRawGraph(std::uint64_t lease) {
#if defined(KVAF_ENABLE_NPU_PULL)
  auto reader=impl_->Reader();
  if(reader)return reader->EndGraph(lease);
#endif
  return Status::NotInitialized("NPU Pull is not initialized");
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
bool NpuRuntime::raw_inflight() const {
#if defined(KVAF_ENABLE_NPU_PULL)
  auto reader=impl_->Reader();
  return reader && reader->inflight();
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
