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

#ifndef KVAF_ENABLE_URMA
#define KVAF_ENABLE_URMA 1
#endif
#ifndef KVAF_ENABLE_UBMEM
#define KVAF_ENABLE_UBMEM 0
#endif

#include "access_operator.h"
#include "meta_manager.h"
#if KVAF_ENABLE_URMA
#include "batch_get_abi.h"
#include "direct_hccp/direct_hccp_session.h"
#include "peer_link.h"
#include "startup_bandwidth_probe.h"
#include "startup_selection.h"
#endif
#if KVAF_ENABLE_UBMEM
#include "ubmem/host_arena.h"
#include "ubmem/access_operator.h"
#endif

namespace kvaf {
namespace {
#if KVAF_ENABLE_URMA

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
// Same PID-scoped HDC service used by the BatchGet and BatchPut transports.
constexpr std::uint32_t kPidHdcType = 18U;

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

#endif
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
    if (options.transport == Transport::kUbmem) {
#if KVAF_ENABLE_UBMEM
      access_operator = CreateUbmemAccessOperator(config);
#else
      throw std::invalid_argument("KVAF was built without UBmem support");
#endif
    } else {
#if KVAF_ENABLE_URMA
      auto op = std::make_unique<DefaultAccessOperator>(config);
      urma_operator = op.get();
      access_operator = std::move(op);
#else
      throw std::invalid_argument("KVAF was built without URMA support");
#endif
    }
    runtime_core = std::make_unique<RuntimeCore>(*meta_manager, *access_operator, *allocator);
  }

  void ReleaseProbe() {
#if KVAF_ENABLE_URMA
    bandwidth_probe.reset();
    if (probe_buffer != nullptr) {
      (void)aclrtFree(probe_buffer);
      probe_buffer = nullptr;
    }
#endif
  }

  NpuRuntimeOptions options;
  AccessOperatorConfig config;
  std::unique_ptr<MetaManager> meta_manager;
  std::unique_ptr<DdrMemoryAllocator> allocator;
  std::unique_ptr<AccessOperator> access_operator;
  std::unique_ptr<RuntimeCore> runtime_core;
#if KVAF_ENABLE_UBMEM
  UbmemFusedContext ubmem_fused;
  void* ubmem_fused_stream{nullptr};
  std::uint64_t next_ubmem_lease{1};
  std::vector<AccessTableHandle> ubmem_fused_tables;

  Status CheckUbmemDevice() const {
    std::int32_t current = -1;
    const aclError error = aclrtGetDevice(&current);
    if (error != ACL_SUCCESS || current < 0 ||
        static_cast<std::uint32_t>(current) != options.device_id) {
      return Status::InvalidState("UBmem fused context must run on its Runtime device");
    }
    return Status::OK();
  }

  Status CheckUbmemOwner(const UbmemFusedContext& owner) const {
    const Status ready = runtime_core->CheckFusedReadLease();
    if (!ready.ok()) return ready;
    const auto& live = ubmem_fused;
    if (live.lease_id == 0) return Status::NotFound("Runtime has no UBmem fused lease");
    if (owner.abi_version != live.abi_version || owner.lease_id != live.lease_id ||
        owner.device_id != live.device_id || owner.region_base != live.region_base ||
        owner.region_bytes != live.region_bytes || owner.slot_bytes != live.slot_bytes ||
        owner.output_address != live.output_address || owner.output_bytes != live.output_bytes ||
        owner.token_bytes != live.token_bytes || owner.tokens_per_block != live.tokens_per_block) {
      return Status::InvalidArgument("pass the original unmodified UBmem fused owner context");
    }
    return CheckUbmemDevice();
  }

  Status CheckUbmemOutput(void* output, std::uint64_t bytes) const {
    const auto base = reinterpret_cast<std::uintptr_t>(output);
    const auto token_bytes = options.batch_get_token_bytes;
    if (base == 0 || base % 32 != 0 || bytes == 0 ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - base ||
        (token_bytes != 656 && token_bytes != 2624) || bytes % token_bytes != 0) {
      return Status::InvalidArgument("UBmem fused output requires aligned complete 656/2624-byte rows");
    }
    return Status::OK();
  }

  Status FenceUbmemReads() {
    Status status = CheckUbmemDevice();
    if (!status.ok()) return status;
    const aclError error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
      return Status::CompletionUnknown("UBmem fused device fence failed; retain arena, output, tables and graphs; ACL=" +
                                        std::to_string(error));
    }
    return Status::OK();
  }
#endif
#if KVAF_ENABLE_URMA
  PeerDdrAllocator* peer_allocator{nullptr};
  std::shared_ptr<peer::PeerLink> peer_link;
  DefaultAccessOperator* urma_operator{nullptr};
  void* probe_buffer{nullptr};
  std::unique_ptr<detail::StartupBandwidthProbe> bandwidth_probe;
#endif
};

NpuRuntime::NpuRuntime(NpuRuntimeOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {
  const auto& o = impl_->options;
  if (o.batch_get_kernel_object_path.empty() || o.npu_core_count == 0U ||
      o.batch_get_token_bytes == 0U || o.batch_get_tokens_per_block == 0U ||
      (o.transport != Transport::kUrma && o.transport != Transport::kUbmem)) {
    throw std::invalid_argument("invalid NPU Runtime configuration");
  }
  if (o.ubmem_page_mode != UbmemPageMode::kHuge &&
      o.ubmem_page_mode != UbmemPageMode::kBase && o.ubmem_page_mode != UbmemPageMode::kThp) {
    throw std::invalid_argument("UBmem page mode must be huge, base, or thp");
  }
  if (o.ubmem_numa_node < -1 || o.ubmem_numa_node > 4095 ||
      (o.ubmem_numa_node >= 0 &&
       (o.transport != Transport::kUbmem || o.ubmem_page_mode != UbmemPageMode::kThp))) {
    throw std::invalid_argument("ubmem_numa_node requires UBmem THP and must be -1 or 0..4095");
  }
  if (o.transport == Transport::kUrma) {
    if (o.ubmem_page_mode != UbmemPageMode::kHuge) {
      throw std::invalid_argument("non-default ubmem_page_mode requires transport=ubmem");
    }
#if KVAF_ENABLE_URMA
    if (o.npu_core_count > KVAF_AIV_BATCH_GET_MAX_AIV_CORES || o.jettys_per_core == 0U ||
        o.jettys_per_core > KVAF_AIV_BATCH_GET_MAX_JETTYS_PER_CORE ||
        (o.batch_get_max_inflight_total != 0U && o.batch_get_max_inflight_total < 2U * o.npu_core_count) ||
        (o.adaptive_ub_device && !o.peer_ub_device.empty()) ||
        (o.adaptive_ub_device && (o.adaptive_ub_miss_count == 0U ||
         o.adaptive_ub_miss_count > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())))) {
      throw std::invalid_argument("invalid URMA Runtime configuration");
    }
#else
    throw std::invalid_argument("KVAF was built without URMA support; select transport=ubmem");
#endif
  } else {
#if KVAF_ENABLE_UBMEM
    if (o.batch_put_kernel_object_path.empty() || o.reuse_device_ra || o.adaptive_ub_device ||
        !o.npu_eids.empty() || !o.peer_ub_device.empty() || !o.adaptive_peer_ub_devices.empty()) {
      throw std::invalid_argument("UBmem requires its Put kernel and does not accept URMA routing options");
    }
#else
    throw std::invalid_argument("KVAF was built without UBmem support");
#endif
  }
  if (aclrtSetDevice(o.device_id) != ACL_SUCCESS) {
    throw std::runtime_error("aclrtSetDevice failed for NpuRuntime");
  }
#if KVAF_ENABLE_UBMEM
  if (o.transport == Transport::kUbmem) {
    impl_->allocator = CreateUbmemDdrMemoryAllocator(o.device_id, o.ubmem_page_mode, o.ubmem_numa_node);
    impl_->CreateCore();
  }
#endif
}

NpuRuntime::~NpuRuntime() {
  if (impl_ != nullptr) {
    if (!Shutdown().ok()) {
      std::fprintf(stderr, "[KVAF_AIV] FUSED_RUNTIME_RETAINED_ON_DESTRUCTION "
                           "reason=teardown_not_safe\n");
      (void)impl_.release();
      return;
    }
    impl_->runtime_core.reset();
    impl_->access_operator.reset();
    impl_->ReleaseProbe();
  }
}

Status NpuRuntime::LinkTo(const LinkTarget& target) {
  if (impl_->options.transport == Transport::kUbmem) {
    return Status::NotImplemented("UBmem uses worker-owned local DDR and has no peer link");
  }
#if KVAF_ENABLE_URMA
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
#else
  return Status::NotImplemented("URMA support is not built");
#endif
}

Status NpuRuntime::Initialize(std::uint64_t ddr_bytes, std::uint64_t slot_size) {
  if (impl_->options.transport == Transport::kUbmem) {
    return impl_->runtime_core->Initialize(RuntimeInitRequest{ddr_bytes, slot_size, DeviceKind::kCpu});
  }
#if KVAF_ENABLE_URMA
  const RuntimeInitRequest request{ddr_bytes, slot_size, DeviceKind::kCpu};
  if (initialized()) return Status::InvalidState("NpuRuntime has already been initialized");
  if (impl_->peer_link == nullptr || impl_->peer_allocator == nullptr) {
    return Status::InvalidState("NpuRuntime requires LinkTo before Initialize");
  }
  if (ddr_bytes == 0U || slot_size == 0U || ddr_bytes % slot_size != 0U) {
    return Status::InvalidArgument("DDR size must be a nonzero multiple of slot_size");
  }
  if (impl_->options.reuse_device_ra) {
    std::uint32_t physical_device_id = 0U;
    if (!ResolvePhysicalDeviceId(impl_->options.device_id, &physical_device_id)) {
      return Internal("cannot resolve physical NPU for caller-initialized device RA");
    }
    direct_hccp::DirectHccpSession* session = nullptr;
    if (!direct_hccp::AcquireDirectHccpSession(
            physical_device_id, kPidHdcType, &session, true)) {
      return Internal("failed to borrow caller-initialized device RA");
    }
    // The shared session remains cached at refs=0 for the process lifetime.
    // Every candidate's BatchGet/BatchPut transport acquires this same session.
    direct_hccp::ReleaseDirectHccpSession(session);
  }

  std::uint32_t candidate_count = 0U;
  std::uint32_t device_count = 0U;
  bool device_filter_applied = false;
  Status last = impl_->peer_link->GetCandidateCount(
      &candidate_count, &device_count, &device_filter_applied);
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
      impl_->options.jettys_per_core;
  if (explicit_eids && npu_candidates.size() > worker_count) {
    return Status::InvalidArgument(
        "npu_eids count exceeds npu_core_count * jettys_per_core workers");
  }
  const bool fixed_peer_device = !impl_->options.peer_ub_device.empty();
  const bool has_n_side_allowlist =
      !impl_->options.adaptive_peer_ub_devices.empty();
  const bool tune_candidates = impl_->options.adaptive_ub_device &&
                               !device_filter_applied && has_n_side_allowlist;
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
        last = impl_->bandwidth_probe->Run(*impl_->runtime_core, *impl_->urma_operator, cycles);
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
    impl_->access_operator.reset();
    impl_->ReleaseProbe();
    return Status::OK();
  };
  if (!tune_candidates) {
    for (std::uint32_t peer_candidate = 0U; peer_candidate < candidate_count;
         ++peer_candidate) {
      if (fixed_peer_device && !peer_candidate_allowed[peer_candidate]) continue;
      for (std::size_t npu_group = 0U; npu_group < npu_group_count; ++npu_group) {
        last = initialize_candidate(
            detail::StartupCandidate{peer_candidate, npu_group}, false, nullptr);
        if (last.ok()) return last;
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
  return Status::NotImplemented("URMA support is not built");
#endif
}

RuntimeCore& NpuRuntime::core() {
  if (impl_->runtime_core == nullptr) {
    throw std::logic_error("Runtime core is unavailable; URMA requires LinkTo first");
  }
  return *impl_->runtime_core;
}
const RuntimeCore& NpuRuntime::core() const {
  if (impl_->runtime_core == nullptr) {
    throw std::logic_error("Runtime core is unavailable; URMA requires LinkTo first");
  }
  return *impl_->runtime_core;
}
Transport NpuRuntime::transport() const { return impl_->options.transport; }
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
  if (impl_->options.transport == Transport::kUbmem) {
    return Status::NotImplemented("this interface requires URMA; UBmem uses a separate completion/lease ABI");
  }
#if KVAF_ENABLE_URMA
  if (impl_->access_operator == nullptr) {
    return Status::NotInitialized(
        "NpuRuntime requires LinkTo before configuring BatchGet profiling");
  }
  return impl_->urma_operator->SetBatchGetKernelProfiling(enabled);
#else
  return Status::NotImplemented("URMA support is not built");
#endif
}

Status NpuRuntime::CopyBatchGetKernelProfiles(
    std::vector<BatchGetKernelProfile>* profiles) const {
  if (impl_->options.transport == Transport::kUbmem) {
    return Status::NotImplemented("this interface requires URMA; UBmem uses a separate completion/lease ABI");
  }
#if KVAF_ENABLE_URMA
  if (impl_->access_operator == nullptr) {
    return Status::NotInitialized(
        "NpuRuntime requires LinkTo before reading BatchGet profiling");
  }
  return impl_->urma_operator->CopyBatchGetKernelProfiles(profiles);
#else
  return Status::NotImplemented("URMA support is not built");
#endif
}

Status NpuRuntime::AcceptLink(const LinkTarget&) {
  return Status::NotImplemented("NpuRuntime does not accept links in the CPU--NPU deployment");
}

Status NpuRuntime::AcquireFusedBatchGetContext(
    void* output, std::uint64_t output_bytes, FusedBatchGetContext* context) {
  if (impl_->options.transport == Transport::kUbmem) {
    return Status::NotImplemented("this interface requires URMA; UBmem uses a separate completion/lease ABI");
  }
#if KVAF_ENABLE_URMA
  if (!initialized() || impl_->access_operator == nullptr) {
    return Status::NotInitialized("initialize the linked Runtime before acquiring a fused context");
  }
  return impl_->urma_operator->AcquireFusedBatchGetContext(
      output, output_bytes, impl_->options.batch_get_token_bytes,
      impl_->options.batch_get_tokens_per_block, context);
#else
  return Status::NotImplemented("URMA support is not built");
#endif
}

Status NpuRuntime::RegisterFusedBatchGetOutput(
    const FusedBatchGetContext& owner, void* output,
    std::uint64_t output_bytes, FusedBatchGetContext* view) {
  if (impl_->options.transport == Transport::kUbmem) {
    return Status::NotImplemented("this interface requires URMA; UBmem uses a separate completion/lease ABI");
  }
#if KVAF_ENABLE_URMA
  if (!initialized() || impl_->access_operator == nullptr) {
    return Status::NotInitialized("Runtime has no fused BatchGet context");
  }
  return impl_->urma_operator->RegisterFusedBatchGetOutput(
      owner, output, output_bytes, view);
#else
  return Status::NotImplemented("URMA support is not built");
#endif
}

Status NpuRuntime::ReleaseFusedBatchGetContext(
    const FusedBatchGetContext& context, void* stream) {
  if (impl_->options.transport == Transport::kUbmem) {
    return Status::NotImplemented("this interface requires URMA; UBmem uses a separate completion/lease ABI");
  }
#if KVAF_ENABLE_URMA
  if (!initialized() || impl_->access_operator == nullptr) {
    return Status::NotInitialized("Runtime has no fused BatchGet context");
  }
  return impl_->urma_operator->ReleaseFusedBatchGetContext(context, stream);
#else
  return Status::NotImplemented("URMA support is not built");
#endif
}

Status NpuRuntime::AcquireUbmemFusedContext(
    void* output, std::uint64_t output_bytes, void* stream,
    UbmemFusedContext* context) {
  if (impl_->options.transport != Transport::kUbmem) {
    return Status::NotImplemented("UBmem fused context requires transport=ubmem");
  }
#if KVAF_ENABLE_UBMEM
  if (context == nullptr || stream == nullptr) {
    return Status::InvalidArgument("UBmem fused acquire requires output context and caller stream");
  }
  *context = {};
  if (!initialized()) return Status::NotInitialized("initialize Runtime before acquiring a UBmem fused lease");
  if (impl_->ubmem_fused.lease_id != 0) return Status::Busy("Runtime already has a UBmem fused lease");
  Status status = impl_->CheckUbmemDevice();
  if (!status.ok()) return status;
  status = impl_->CheckUbmemOutput(output, output_bytes);
  if (!status.ok()) return status;
  if (impl_->next_ubmem_lease == 0) return Status::InvalidState("UBmem fused lease ID space exhausted");
  const auto& region = core().remote_region();
  const std::uint64_t block_bytes =
      static_cast<std::uint64_t>(impl_->options.batch_get_token_bytes) *
      impl_->options.batch_get_tokens_per_block;
  if (region.slot_size < block_bytes) {
    return Status::InvalidArgument("UBmem fused block layout exceeds the DDR slot size");
  }
  status = core().AcquireFusedReadLease([state = impl_.get()] {
    return state->FenceUbmemReads();
  });
  if (!status.ok()) return status;
  impl_->ubmem_fused = UbmemFusedContext{
      1, impl_->options.device_id, impl_->next_ubmem_lease++,
      region.base, region.length, region.slot_size,
      reinterpret_cast<std::uintptr_t>(output), output_bytes,
      impl_->options.batch_get_token_bytes, impl_->options.batch_get_tokens_per_block};
  impl_->ubmem_fused_stream = stream;
  // Drain previous ordinary operations once. A failed fence deliberately keeps
  // the lease and native owner intact even though acquire cannot return it.
  status = core().FenceFusedReads();
  if (!status.ok()) return status;
  *context = impl_->ubmem_fused;
  return Status::OK();
#else
  return Status::NotImplemented("KVAF was built without UBmem support");
#endif
}

Status NpuRuntime::RegisterUbmemFusedOutput(
    const UbmemFusedContext& owner, void* output, std::uint64_t output_bytes,
    UbmemFusedContext* view) {
  if (impl_->options.transport != Transport::kUbmem) {
    return Status::NotImplemented("UBmem fused context requires transport=ubmem");
  }
#if KVAF_ENABLE_UBMEM
  if (view == nullptr) return Status::InvalidArgument("UBmem fused output view must not be null");
  *view = {};
  Status status = impl_->CheckUbmemOwner(owner);
  if (!status.ok()) return status;
  status = impl_->CheckUbmemOutput(output, output_bytes);
  if (!status.ok()) return status;
  *view = owner;
  view->output_address = reinterpret_cast<std::uintptr_t>(output);
  view->output_bytes = output_bytes;
  return Status::OK();
#else
  return Status::NotImplemented("KVAF was built without UBmem support");
#endif
}

Status NpuRuntime::RegisterUbmemFusedAccessTable(
    const UbmemFusedContext& owner, const AccessTableHandle& table) {
  if (impl_->options.transport != Transport::kUbmem) {
    return Status::NotImplemented("UBmem fused context requires transport=ubmem");
  }
#if KVAF_ENABLE_UBMEM
  Status status = impl_->CheckUbmemOwner(owner);
  if (!status.ok()) return status;
  for (const auto& registered : impl_->ubmem_fused_tables) {
    if (registered.opaque == table.opaque) return Status::OK();
  }
  // Reserve bookkeeping before taking the extra reader, so host allocation
  // failure cannot strand a pin absent from the release list.
  impl_->ubmem_fused_tables.push_back(table);
  status = core().RetainFusedAccessTable(
      table, static_cast<std::uint64_t>(owner.token_bytes) * owner.tokens_per_block);
  if (!status.ok()) impl_->ubmem_fused_tables.pop_back();
  return status;
#else
  return Status::NotImplemented("KVAF was built without UBmem support");
#endif
}

Status NpuRuntime::ReleaseUbmemFusedContext(
    const UbmemFusedContext& owner, void* stream) {
  if (impl_->options.transport != Transport::kUbmem) {
    return Status::NotImplemented("UBmem fused context requires transport=ubmem");
  }
#if KVAF_ENABLE_UBMEM
  Status status = impl_->CheckUbmemOwner(owner);
  if (!status.ok()) return status;
  if (stream == nullptr || stream != impl_->ubmem_fused_stream) {
    return Status::InvalidArgument("release UBmem fused context with its acquire stream");
  }
  status = core().FenceFusedReads();
  if (!status.ok()) return status;
  while (!impl_->ubmem_fused_tables.empty()) {
    status = core().ReleaseFusedAccessTable(impl_->ubmem_fused_tables.back());
    if (!status.ok()) return status;
    impl_->ubmem_fused_tables.pop_back();
  }
  status = core().ReleaseFusedReadLease();
  if (!status.ok()) return status;
  impl_->ubmem_fused = {};
  impl_->ubmem_fused_stream = nullptr;
  return Status::OK();
#else
  return Status::NotImplemented("KVAF was built without UBmem support");
#endif
}

Status NpuRuntime::ReleaseLink(LinkId link_id) {
  if (impl_->options.transport == Transport::kUbmem) {
    return Status::NotImplemented("UBmem uses worker-owned local DDR and has no peer link");
  }
#if KVAF_ENABLE_URMA
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
#else
  return Status::NotImplemented("URMA support is not built");
#endif
}

Status NpuRuntime::ServeConnection(int) {
  return Status::NotImplemented("NpuRuntime has no incoming control service in this deployment");
}

Status NpuRuntime::Shutdown() {
  if (impl_ == nullptr) return Status::OK();
  const Status status = impl_->runtime_core != nullptr && impl_->runtime_core->initialized()
                            ? impl_->runtime_core->Shutdown()
                            : Status::OK();
  if (!status.ok()) return status;
  impl_->ReleaseProbe();
  return status;
}

bool NpuRuntime::initialized() const {
  return impl_ != nullptr && impl_->runtime_core != nullptr && impl_->runtime_core->initialized();
}

RuntimeCapabilities NpuRuntime::capabilities() const {
  return RuntimeCapabilities{true, !impl_->config.batch_put_kernel_object_path.empty(), false,
                             impl_->options.transport == Transport::kUrma};
}

}  // namespace kvaf
