#include "cpu_runtime.h"
#include "peer_protocol.h"

#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdarg>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <urma_api.h>

namespace kvaf {
namespace {

using kvaf::peer::AllocateRegionRequest;
using kvaf::peer::AllocateRegionResponse;
using kvaf::peer::CandidateCountRequest;
using kvaf::peer::CandidateCountResponse;
using kvaf::peer::Command;
using kvaf::peer::CreateChannelRequest;
using kvaf::peer::CreateChannelResponse;
using kvaf::peer::IsValidHeader;
using kvaf::peer::MakeHeader;
using kvaf::peer::MessageHeader;
using kvaf::peer::ReleaseRegionRequest;
using kvaf::peer::ResponseCommand;
using kvaf::peer::Result;
using kvaf::peer::StatusResponse;
using kvaf::peer::kEidBytes;
using kvaf::peer::kQpKeyBytes;
using kvaf::peer::kSegKeyBytes;

constexpr uint32_t kJettyDepth = 8192U;
constexpr uint32_t kJfcDepth = 16384U;
constexpr uint32_t kJettyKeyBytes = sizeof(urma_jetty_id_t) + sizeof(urma_transport_mode_t);
// HCCP reports the NPU's USER_CTL Jetty key as a 32-byte record on the
// deployed CANN release.  Its first 28 bytes are the public UMDK Jetty id
// and transport mode; the trailing four bytes are retained on the wire.
constexpr uint32_t kNpuJettyKeyBytes = 32U;
constexpr uint32_t kSegBytes = sizeof(urma_seg_t);
constexpr uint32_t kPageBytes = 4096U;

static_assert(sizeof(urma_jetty_id_t) == 24U, "UMDK Jetty id ABI changed");
static_assert(kJettyKeyBytes == 28U, "HCCP UMDK Jetty key ABI changed");
static_assert(kSegBytes == 48U, "HCCP UMDK segment ABI changed");
static_assert(kJettyKeyBytes <= kQpKeyBytes, "QpKey protocol buffer is too small");
static_assert(kSegBytes <= kSegKeyBytes, "segment protocol buffer is too small");

// These two calls are exported by the deployed liburma but are not declared
// by the public UMDK header used to build this standalone CPU service.  The
// layouts below come from the matching HCCP/UMDK source ABI.  Keeping them
// private here avoids importing any CANN or HCCP header/library into CPU.
struct UrmaExtImportJettyFlag {
  uint64_t value;
  uint64_t stag;
  uint64_t dtag;
};

struct UrmaExtRjetty {
  urma_jetty_id_t jetty_id;
  uint32_t trans_mode;
  uint32_t policy;
  uint32_t type;
  uint32_t reserved_before_flag;
  UrmaExtImportJettyFlag flag;
  uint32_t tp_type;
  uint32_t reserved_after_tp_type;
};

struct UrmaExtGetTpCfg {
  uint32_t flag;
  uint32_t trans_mode;
  uint8_t local_eid[kEidBytes];
  uint8_t peer_eid[kEidBytes];
};

struct UrmaExtTpInfo {
  uint64_t tp_handle;
};

struct UrmaExtActiveTpCfg {
  uint64_t tp_handle;
  uint64_t peer_tp_handle;
  uint64_t tag;
  uint32_t tx_psn;
  uint32_t rx_psn;
  uint64_t reserved;
};

static_assert(sizeof(UrmaExtImportJettyFlag) == 24U, "UMDK expanded Jetty flag ABI changed");
static_assert(sizeof(UrmaExtRjetty) == 72U, "UMDK expanded remote Jetty ABI changed");
static_assert(offsetof(UrmaExtRjetty, trans_mode) == 24U, "UMDK remote Jetty mode offset changed");
static_assert(offsetof(UrmaExtRjetty, flag) == 40U, "UMDK remote Jetty flag offset changed");
static_assert(offsetof(UrmaExtRjetty, tp_type) == 64U, "UMDK remote Jetty TP type offset changed");
static_assert(sizeof(UrmaExtGetTpCfg) == 40U, "UMDK CTP query ABI changed");
static_assert(sizeof(UrmaExtTpInfo) == 8U, "UMDK TP info ABI changed");
static_assert(sizeof(UrmaExtActiveTpCfg) == 40U, "UMDK expanded Jetty config ABI changed");

using UrmaGetTpListFn = int (*)(urma_context_t*, UrmaExtGetTpCfg*, uint32_t*, UrmaExtTpInfo*);
using UrmaImportJettyExFn = urma_target_jetty_t* (*)(urma_context_t*, UrmaExtRjetty*, urma_token_t*,
                                                      UrmaExtActiveTpCfg*);

void Log(const char* format, ...) {
  va_list arguments;
  va_start(arguments, format);
  std::vfprintf(stderr, format, arguments);
  va_end(arguments);
}

bool ReadAll(int fd, void* data, size_t bytes) {
  auto* cursor = static_cast<uint8_t*>(data);
  size_t completed = 0U;
  while (completed < bytes) {
    const ssize_t result = recv(fd, cursor + completed, bytes - completed, 0);
    if (result <= 0) {
      return false;
    }
    completed += static_cast<size_t>(result);
  }
  return true;
}

bool WriteAll(int fd, const void* data, size_t bytes) {
  const auto* cursor = static_cast<const uint8_t*>(data);
  size_t completed = 0U;
  while (completed < bytes) {
    const ssize_t result = send(fd, cursor + completed, bytes - completed, MSG_NOSIGNAL);
    if (result <= 0) {
      return false;
    }
    completed += static_cast<size_t>(result);
  }
  return true;
}

class UrmaResources {
 public:
  ~UrmaResources() { Shutdown(); }

  bool Initialize(const std::string& device_name, uint32_t eid_index) {
    selected_device_name_ = device_name;
    device_ = urma_get_device_by_name(const_cast<char*>(device_name.c_str()));
    if (device_ == nullptr || device_->type != URMA_TRANSPORT_UB) {
      Log("[KVAF_CPU] unusable UB device=%s\n", device_name.c_str());
      return false;
    }
    context_ = urma_create_context(device_, eid_index);
    if (context_ == nullptr) {
      Log("[KVAF_CPU] urma_create_context failed device=%s eid_index=%u errno=%d\n",
          selected_device_name_.c_str(), eid_index, errno);
      return false;
    }
    get_tp_list_ = reinterpret_cast<UrmaGetTpListFn>(dlsym(RTLD_DEFAULT, "urma_get_tp_list"));
    import_jetty_ex_ = reinterpret_cast<UrmaImportJettyExFn>(dlsym(RTLD_DEFAULT, "urma_import_jetty_ex"));
    if (get_tp_list_ == nullptr || import_jetty_ex_ == nullptr) {
      Log("[KVAF_CPU] liburma lacks required expanded CTP APIs get_tp_list=%p import_jetty_ex=%p\n",
          reinterpret_cast<void*>(get_tp_list_), reinterpret_cast<void*>(import_jetty_ex_));
      return false;
    }
    eid_index_ = eid_index;
    Log("[KVAF_CPU] CANDIDATE_READY ub_device=%s eid_index=%u eid=", selected_device_name_.c_str(), eid_index);
    for (uint8_t byte : context_->eid.raw) {
      std::fprintf(stderr, "%02x", byte);
    }
    std::fprintf(stderr, "\n");
    return true;
  }

  void Shutdown() {
    if (context_ != nullptr) {
      (void)urma_delete_context(context_);
      context_ = nullptr;
    }
  }

  urma_context_t* context() const { return context_; }
  const std::string& device_name() const { return selected_device_name_; }
  uint32_t eid_index() const { return eid_index_; }

  bool GetCtpForPeer(const uint8_t peer_eid[kEidBytes], uint64_t* tp_handle) const {
    if (peer_eid == nullptr || tp_handle == nullptr || get_tp_list_ == nullptr || context_ == nullptr) {
      return false;
    }
    UrmaExtGetTpCfg config{};
    config.flag = 1U;  // CTP flag bit.
    config.trans_mode = static_cast<uint32_t>(URMA_TM_RM);
    std::memcpy(config.local_eid, context_->eid.raw, kEidBytes);
    std::memcpy(config.peer_eid, peer_eid, kEidBytes);
    UrmaExtTpInfo info{};
    uint32_t count = 1U;
    const int status = get_tp_list_(context_, &config, &count, &info);
    if (status != 0 || count == 0U || info.tp_handle == 0U) {
      Log("[KVAF_CPU] urma_get_tp_list failed status=%d count=%u tp=0x%llx errno=%d\n", status, count,
          static_cast<unsigned long long>(info.tp_handle), errno);
      return false;
    }
    *tp_handle = info.tp_handle;
    return true;
  }

  urma_target_jetty_t* ImportJettyEx(const uint8_t key[kQpKeyBytes], uint32_t token_value,
                                     uint64_t local_tp, uint64_t peer_tp, uint32_t local_psn,
                                     uint32_t peer_psn) const {
    if (key == nullptr || token_value == 0U || local_tp == 0U || peer_tp == 0U || local_psn == 0U ||
        peer_psn == 0U || import_jetty_ex_ == nullptr || context_ == nullptr) {
      return nullptr;
    }
    UrmaExtRjetty remote{};
    std::memcpy(&remote.jetty_id, key, sizeof(remote.jetty_id));
    std::memcpy(&remote.trans_mode, key + sizeof(remote.jetty_id), sizeof(remote.trans_mode));
    remote.policy = static_cast<uint32_t>(URMA_JETTY_GRP_POLICY_RR);
    remote.type = static_cast<uint32_t>(URMA_JETTY);
    urma_import_jetty_flag_t flag{};
    flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    remote.flag.value = flag.value;
    remote.tp_type = 1U;  // URMA_CTP in the expanded UMDK ABI.
    UrmaExtActiveTpCfg config{};
    config.tp_handle = local_tp;
    config.peer_tp_handle = peer_tp;
    config.tx_psn = local_psn;
    config.rx_psn = peer_psn;
    urma_token_t token{};
    token.token = token_value;
    urma_target_jetty_t* target = import_jetty_ex_(context_, &remote, &token, &config);
    if (target == nullptr) {
      Log("[KVAF_CPU] urma_import_jetty_ex failed remote_jetty=%u remote_uasid=%u token=0x%x mode=%u "
          "local_tp=0x%llx peer_tp=0x%llx tx_psn=0x%x rx_psn=0x%x errno=%d\n",
          remote.jetty_id.id, remote.jetty_id.uasid, token_value, remote.trans_mode,
          static_cast<unsigned long long>(local_tp), static_cast<unsigned long long>(peer_tp), local_psn,
          peer_psn, errno);
    }
    return target;
  }

 private:
  std::string selected_device_name_;
  uint32_t eid_index_ = 0U;
  urma_device_t* device_ = nullptr;
  urma_context_t* context_ = nullptr;
  UrmaGetTpListFn get_tp_list_ = nullptr;
  UrmaImportJettyExFn import_jetty_ex_ = nullptr;
};

struct UrmaCandidate {
  uint32_t id = 0U;
  std::string device_name;
  uint32_t eid_index = 0U;
  std::array<uint8_t, kEidBytes> eid{};
  bool unavailable = false;
  std::unique_ptr<UrmaResources> resources;
};

class UrmaResourcePool {
 public:
  ~UrmaResourcePool() { Shutdown(); }

  bool Initialize(const std::string& requested_device, bool eid_index_requested,
                  uint32_t requested_eid_index) {
    urma_init_attr_t init{};
    if (urma_init(&init) != URMA_SUCCESS) {
      Log("[KVAF_CPU] urma_init failed\n");
      return false;
    }
    initialized_ = true;

    int device_count = 0;
    urma_device_t** devices = urma_get_device_list(&device_count);
    if (devices == nullptr) {
      Log("[KVAF_CPU] urma_get_device_list returned no devices\n");
      return false;
    }
    for (int device_index = 0; device_index < device_count; ++device_index) {
      urma_device_t* device = devices[device_index];
      if (device == nullptr || device->type != URMA_TRANSPORT_UB ||
          (!requested_device.empty() && requested_device != device->name)) {
        continue;
      }
      uint32_t eid_count = 0U;
      urma_eid_info_t* eids = urma_get_eid_list(device, &eid_count);
      if (eids == nullptr) {
        Log("[KVAF_CPU] CANDIDATE_ENUM_FAILED ub_device=%s errno=%d\n", device->name, errno);
        continue;
      }
      for (uint32_t eid_position = 0U; eid_position < eid_count; ++eid_position) {
        const urma_eid_info_t& info = eids[eid_position];
        if ((eid_index_requested && info.eid_index != requested_eid_index) ||
            std::all_of(info.eid.raw, info.eid.raw + kEidBytes,
                        [](uint8_t byte) { return byte == 0U; })) {
          continue;
        }
        UrmaCandidate candidate{};
        candidate.device_name = device->name;
        candidate.eid_index = info.eid_index;
        std::memcpy(candidate.eid.data(), info.eid.raw, candidate.eid.size());
        candidates_.push_back(std::move(candidate));
      }
      urma_free_eid_list(eids);
    }
    urma_free_device_list(devices);

    std::sort(candidates_.begin(), candidates_.end(),
              [](const UrmaCandidate& left, const UrmaCandidate& right) {
                return left.device_name == right.device_name
                           ? left.eid_index < right.eid_index
                           : left.device_name < right.device_name;
              });
    for (uint32_t index = 0U; index < candidates_.size(); ++index) {
      UrmaCandidate& candidate = candidates_[index];
      candidate.id = index;
      Log("[KVAF_CPU] CANDIDATE_DISCOVERED id=%u ub_device=%s eid_index=%u eid=", candidate.id,
          candidate.device_name.c_str(), candidate.eid_index);
      for (uint8_t byte : candidate.eid) {
        std::fprintf(stderr, "%02x", byte);
      }
      std::fprintf(stderr, "\n");
    }
    if (candidates_.empty()) {
      Log("[KVAF_CPU] no usable UB EID candidates");
      if (!requested_device.empty()) {
        Log(" for ub_device=%s", requested_device.c_str());
      }
      if (eid_index_requested) {
        Log(" eid_index=%u", requested_eid_index);
      }
      Log("\n");
      return false;
    }
    return true;
  }

  uint32_t CandidateCount() const {
    return static_cast<uint32_t>(candidates_.size());
  }

  UrmaResources* Get(uint32_t candidate_id) {
    if (candidate_id >= candidates_.size()) {
      Log("[KVAF_CPU] CANDIDATE_REJECTED id=%u reason=out_of_range count=%zu\n", candidate_id,
          candidates_.size());
      return nullptr;
    }
    UrmaCandidate& candidate = candidates_[candidate_id];
    if (candidate.resources != nullptr) {
      return candidate.resources.get();
    }
    if (candidate.unavailable) {
      Log("[KVAF_CPU] CANDIDATE_REJECTED id=%u reason=initialization_failed\n", candidate_id);
      return nullptr;
    }
    auto resources = std::make_unique<UrmaResources>();
    if (!resources->Initialize(candidate.device_name, candidate.eid_index)) {
      candidate.unavailable = true;
      Log("[KVAF_CPU] CANDIDATE_UNAVAILABLE id=%u ub_device=%s eid_index=%u\n", candidate.id,
          candidate.device_name.c_str(), candidate.eid_index);
      return nullptr;
    }
    if (std::memcmp(resources->context()->eid.raw, candidate.eid.data(), candidate.eid.size()) != 0) {
      candidate.unavailable = true;
      Log("[KVAF_CPU] CANDIDATE_UNAVAILABLE id=%u reason=eid_changed ub_device=%s eid_index=%u\n",
          candidate.id, candidate.device_name.c_str(), candidate.eid_index);
      return nullptr;
    }
    candidate.resources = std::move(resources);
    return candidate.resources.get();
  }

 private:
  void Shutdown() {
    candidates_.clear();
    if (initialized_) {
      (void)urma_uninit();
      initialized_ = false;
    }
  }

  bool initialized_ = false;
  std::vector<UrmaCandidate> candidates_;
};

struct Channel {
  urma_jfc_t* local_jfc = nullptr;
  urma_jfr_t* local_jfr = nullptr;
  urma_jetty_t* local_jetty = nullptr;
  urma_target_jetty_t* remote_jetty = nullptr;
};

// A remote NPU uses each exported CPU Jetty as an independent target.  Keep
// its completion resources equally independent: sharing one never-polled JFC
// across all target JFRs allows unrelated AIV lanes to consume each other's
// completion credits.  The order is the UMDK ownership order.
void ReleaseChannel(Channel* channel) {
  if (channel == nullptr) {
    return;
  }
  if (channel->remote_jetty != nullptr) {
    (void)urma_unimport_jetty(channel->remote_jetty);
    channel->remote_jetty = nullptr;
  }
  if (channel->local_jetty != nullptr) {
    (void)urma_delete_jetty(channel->local_jetty);
    channel->local_jetty = nullptr;
  }
  if (channel->local_jfr != nullptr) {
    (void)urma_delete_jfr(channel->local_jfr);
    channel->local_jfr = nullptr;
  }
  if (channel->local_jfc != nullptr) {
    (void)urma_delete_jfc(channel->local_jfc);
    channel->local_jfc = nullptr;
  }
}

struct Region {
  uint64_t id = 0U;
  uint32_t candidate_id = 0U;
  UrmaResources* resources = nullptr;
  void* memory = nullptr;
  uint64_t bytes = 0U;
  urma_token_id_t* token_id = nullptr;
  urma_target_seg_t* segment = nullptr;
  uint32_t token_value = 0U;
  std::vector<Channel> channels;
};

class CpuService {
 public:
  explicit CpuService(UrmaResourcePool* resource_pool, CpuMemoryPageMode mode)
      : resource_pool_(resource_pool), memory_page_mode_(mode) {}
  ~CpuService() { ReleaseAll(); }

  uint32_t CandidateCount() const {
    return resource_pool_ == nullptr ? 0U : resource_pool_->CandidateCount();
  }

  Result Allocate(uint64_t bytes, uint64_t slot_size, uint32_t candidate_id,
                  AllocateRegionResponse* response) {
    if (response == nullptr || bytes == 0U || slot_size == 0U || bytes % slot_size != 0U ||
        bytes > std::numeric_limits<size_t>::max() || resource_pool_ == nullptr) {
      return Result::kInvalidRequest;
    }
    UrmaResources* resources = resource_pool_->Get(candidate_id);
    if (resources == nullptr) {
      return Result::kResourceError;
    }
    auto region = std::make_unique<Region>();
    region->candidate_id = candidate_id;
    region->resources = resources;
    const size_t alignment = memory_page_mode_ == CpuMemoryPageMode::kTransparentHugePage
        ? 2U * 1024U * 1024U : kPageBytes;
    const int allocation_status = posix_memalign(&region->memory, alignment, static_cast<size_t>(bytes));
    if (allocation_status != 0 || region->memory == nullptr) {
      Log("[KVAF_CPU] REGION_ALLOCATE_FAILED stage=posix_memalign bytes=%" PRIu64
          " alignment=%" PRIu64 " status=%d memory=%p\n",
          bytes, static_cast<uint64_t>(alignment), allocation_status, region->memory);
      return Result::kResourceError;
    }
    if (memory_page_mode_ != CpuMemoryPageMode::kDefault) {
      const int advice = memory_page_mode_ == CpuMemoryPageMode::kTransparentHugePage
          ? MADV_HUGEPAGE : MADV_NOHUGEPAGE;
      if (madvise(region->memory, static_cast<size_t>(bytes), advice) != 0) {
        Log("[KVAF_CPU] REGION_ALLOCATE_FAILED stage=madvise errno=%d\n", errno);
        free(region->memory);
        region->memory = nullptr;
        return Result::kResourceError;
      }
      // Populate under the service NUMA policy BEFORE registering the region.
      std::memset(region->memory, 0, static_cast<size_t>(bytes));
    }
    region->bytes = bytes;
    region->token_value = NextToken();
    region->token_id = urma_alloc_token_id(resources->context());
    if (region->token_id == nullptr) {
      Log("[KVAF_CPU] REGION_ALLOCATE_FAILED stage=urma_alloc_token_id errno=%d\n", errno);
      ReleaseRegion(region.get());
      return Result::kResourceError;
    }
    urma_reg_seg_flag_t flag{};
    flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    flag.bs.cacheable = URMA_NON_CACHEABLE;
    // Keep the CPU DDR registration equivalent to the existing HCCP host-memory
    // path. LOCAL_WRITE is deliberately absent: it was rejected by the deployed
    // UDMA driver, whereas remote atomic and non-pinned registration are part of
    // the established host-DDR contract.
    flag.bs.access = URMA_ACCESS_REMOTE_READ | URMA_ACCESS_REMOTE_WRITE |
                     URMA_ACCESS_REMOTE_ATOMIC;
    flag.bs.non_pin = 1U;
    flag.bs.token_id_valid = 1U;
    urma_seg_cfg_t config{};
    config.va = reinterpret_cast<uint64_t>(region->memory);
    config.len = bytes;
    config.token_id = region->token_id;
    config.token_value.token = region->token_value;
    config.flag = flag;
    region->segment = urma_register_seg(resources->context(), &config);
    if (region->segment == nullptr) {
      Log("[KVAF_CPU] REGION_ALLOCATE_FAILED stage=urma_register_seg base=%p bytes=%" PRIu64
          " token_id=0x%x token_value=0x%x reg_flags=0x%x errno=%d\n",
          region->memory, bytes, region->token_id->token_id, region->token_value, flag.value, errno);
      ReleaseRegion(region.get());
      return Result::kResourceError;
    }
    const uint64_t id = next_region_id_++;
    region->id = id;
    response->session_id = id;
    response->peer_base = region->segment->seg.ubva.va;
    response->peer_length = bytes;
    // HCCP's UB resource-service forwards the public UMDK token object value
    // (rather than transforming it through its private TokenId wrapper).
    response->token_id = region->token_id->token_id;
    response->token_value = region->token_value;
    response->segment_key_bytes = kSegBytes;
    std::memcpy(response->peer_eid, region->segment->seg.ubva.eid.raw, kEidBytes);
    std::memcpy(response->segment_key, &region->segment->seg, kSegBytes);
    if (response->peer_base == 0U || response->token_id == 0U) {
      Log("[KVAF_CPU] REGION_ALLOCATE_FAILED stage=region_descriptor base=0x%" PRIx64
          " token_id=0x%x\n",
        response->peer_base, response->token_id);
      ReleaseRegion(region.get());
      return Result::kResourceError;
    }
    regions_.emplace(id, std::move(region));
    Log("[KVAF_CPU] REGION_ALLOCATED session=%" PRIu64 " candidate=%u ub_device=%s eid_index=%u "
        "bytes=%" PRIu64 " base=0x%" PRIx64 " token_id=0x%x reg_flags=0x%x page_mode=%s\n",
        id, candidate_id, resources->device_name().c_str(), resources->eid_index(), bytes,
        response->peer_base, response->token_id, flag.value,
        memory_page_mode_ == CpuMemoryPageMode::kTransparentHugePage ? "thp" :
        (memory_page_mode_ == CpuMemoryPageMode::kBasePage ? "base" : "default"));
    return Result::kOk;
  }

  Result CreateChannel(const CreateChannelRequest& request, CreateChannelResponse* response) {
    if (response == nullptr || request.local_qp_key_bytes != kNpuJettyKeyBytes ||
        std::all_of(request.local_eid, request.local_eid + kEidBytes,
                    [](uint8_t byte) { return byte == 0U; }) ||
        request.peer_token_value == 0U || request.local_tp_handle == 0U || request.local_psn == 0U) {
      Log("[KVAF_CPU] CHANNEL_REJECTED key_bytes=%u expected_npu_key_bytes=%u token=0x%x npu_tp=0x%llx "
          "npu_psn=0x%x\n",
          request.local_qp_key_bytes, kNpuJettyKeyBytes, request.peer_token_value,
          static_cast<unsigned long long>(request.local_tp_handle), request.local_psn);
      return Result::kInvalidRequest;
    }
    const auto found = regions_.find(request.session_id);
    if (found == regions_.end()) {
      return Result::kNotFound;
    }
    Region& region = *found->second;
    Log("[KVAF_CPU] CHANNEL_REQUEST session=%" PRIu64 " key_bytes=%u token=0x%x npu_tp=0x%llx npu_psn=0x%x npu_eid=",
        region.id, request.local_qp_key_bytes, request.peer_token_value,
        static_cast<unsigned long long>(request.local_tp_handle), request.local_psn);
    for (uint8_t byte : request.local_eid) {
      std::fprintf(stderr, "%02x", byte);
    }
    std::fprintf(stderr, "\n");
    Channel channel{};
    uint64_t cpu_tp_handle = 0U;
    const uint32_t cpu_psn = NextPsn();
    if (!CreateLocalJetty(region.resources, region.token_value, &channel.local_jfc,
                          &channel.local_jfr, &channel.local_jetty) ||
        !region.resources->GetCtpForPeer(request.local_eid, &cpu_tp_handle) ||
        !ImportRemoteJetty(region.resources, request.local_qp_key, request.peer_token_value, cpu_tp_handle,
                           request.local_tp_handle, cpu_psn, request.local_psn, &channel.remote_jetty)) {
      ReleaseChannel(&channel);
      return Result::kResourceError;
    }
    std::memcpy(response->peer_qp_key, &channel.local_jetty->jetty_id,
                sizeof(channel.local_jetty->jetty_id));
    const urma_transport_mode_t mode = channel.local_jetty->jetty_cfg.jfs_cfg.trans_mode;
    std::memcpy(response->peer_qp_key + sizeof(channel.local_jetty->jetty_id), &mode, sizeof(mode));
    response->peer_qp_key_bytes = kJettyKeyBytes;
    response->peer_tp_handle = cpu_tp_handle;
    response->peer_psn = cpu_psn;
    region.channels.push_back(channel);
    Log("[KVAF_CPU] CHANNEL_JFC_READY session=%" PRIu64 " jfc_id=%u depth=%u\n",
        region.id, channel.local_jfc->jfc_id.id, kJfcDepth);
    Log("[KVAF_CPU] CHANNEL_CREATED session=%" PRIu64 " local_jetty=%u remote_jetty=%u cpu_tp=0x%llx "
        "cpu_psn=0x%x\n",
        region.id, channel.local_jetty->jetty_id.id, channel.remote_jetty->id.id,
        static_cast<unsigned long long>(cpu_tp_handle), cpu_psn);
    return Result::kOk;
  }

  Result Release(uint64_t id) {
    const auto found = regions_.find(id);
    if (found == regions_.end()) {
      return Result::kNotFound;
    }
    Log("[KVAF_CPU] REGION_RELEASE session=%" PRIu64 " channels=%zu\n", id,
        found->second->channels.size());
    ReleaseRegion(found->second.get());
    regions_.erase(found);
    return Result::kOk;
  }

 private:
  uint32_t NextToken() {
    ++next_token_;
    if (next_token_ == 0U) {
      ++next_token_;
    }
    return next_token_;
  }

  uint32_t NextPsn() {
    ++next_psn_;
    if (next_psn_ == 0U) {
      ++next_psn_;
    }
    return next_psn_;
  }

  bool CreateLocalJetty(UrmaResources* resources, uint32_t token_value, urma_jfc_t** jfc,
                        urma_jfr_t** jfr, urma_jetty_t** jetty) {
    if (resources == nullptr || jfc == nullptr || jfr == nullptr || jetty == nullptr) {
      return false;
    }
    *jfc = nullptr;
    *jfr = nullptr;
    *jetty = nullptr;
    urma_jfc_cfg_t jfc_config{};
    jfc_config.depth = kJfcDepth;
    *jfc = urma_create_jfc(resources->context(), &jfc_config);
    if (*jfc == nullptr) {
      Log("[KVAF_CPU] urma_create_jfc failed scope=channel depth=%u errno=%d\n",
          kJfcDepth, errno);
      return false;
    }
    urma_jfs_cfg_t jfs{};
    jfs.depth = kJettyDepth;
    jfs.trans_mode = URMA_TM_RM;
    jfs.max_sge = 1U;
    jfs.jfc = *jfc;
    urma_jfr_cfg_t jfr_config{};
    jfr_config.depth = kJettyDepth;
    jfr_config.trans_mode = URMA_TM_RM;
    jfr_config.max_sge = 1U;
    jfr_config.jfc = *jfc;
    jfr_config.token_value.token = token_value;
    jfr_config.flag.bs.tag_matching = URMA_NO_TAG_MATCHING;
    *jfr = urma_create_jfr(resources->context(), &jfr_config);
    if (*jfr == nullptr) {
      Log("[KVAF_CPU] urma_create_jfr failed depth=%u token=0x%x errno=%d\n", kJettyDepth,
          token_value, errno);
      (void)urma_delete_jfc(*jfc);
      *jfc = nullptr;
      return false;
    }
    urma_jetty_cfg_t config{};
    config.flag.bs.share_jfr = URMA_SHARE_JFR;
    config.jfs_cfg = jfs;
    config.shared.jfr = *jfr;
    config.shared.jfc = *jfc;
    *jetty = urma_create_jetty(resources->context(), &config);
    if (*jetty == nullptr) {
      Log("[KVAF_CPU] urma_create_jetty failed depth=%u token=0x%x errno=%d\n", kJettyDepth,
          token_value, errno);
      (void)urma_delete_jfr(*jfr);
      *jfr = nullptr;
      (void)urma_delete_jfc(*jfc);
      *jfc = nullptr;
      return false;
    }
    return true;
  }

  bool ImportRemoteJetty(UrmaResources* resources, const uint8_t key[kQpKeyBytes], uint32_t token_value,
                         uint64_t cpu_tp_handle,
                         uint64_t npu_tp_handle, uint32_t cpu_psn, uint32_t npu_psn,
                         urma_target_jetty_t** target) {
    if (resources == nullptr || key == nullptr || target == nullptr) {
      return false;
    }
    *target = resources->ImportJettyEx(key, token_value, cpu_tp_handle, npu_tp_handle, cpu_psn, npu_psn);
    if (*target == nullptr) {
      return false;
    }
    return true;
  }

  void ReleaseRegion(Region* region) {
    if (region == nullptr) {
      return;
    }
    for (auto iterator = region->channels.rbegin(); iterator != region->channels.rend(); ++iterator) {
      ReleaseChannel(&*iterator);
    }
    region->channels.clear();
    if (region->segment != nullptr) {
      (void)urma_unregister_seg(region->segment);
      region->segment = nullptr;
    }
    if (region->token_id != nullptr) {
      (void)urma_free_token_id(region->token_id);
      region->token_id = nullptr;
    }
    if (region->memory != nullptr) {
      free(region->memory);
      region->memory = nullptr;
    }
  }

  void ReleaseAll() {
    for (auto& entry : regions_) {
      ReleaseRegion(entry.second.get());
    }
    regions_.clear();
  }

  UrmaResourcePool* resource_pool_ = nullptr;
  CpuMemoryPageMode memory_page_mode_;
  uint64_t next_region_id_ = 1U;
  uint32_t next_token_ = 1U;
  uint32_t next_psn_ = 1U;
  std::unordered_map<uint64_t, std::unique_ptr<Region>> regions_;
};

void SendStatus(int fd, Command command, Result result) {
  StatusResponse response{};
  response.header = MakeHeader(command, sizeof(response));
  response.header.command = ResponseCommand(command);
  response.result = static_cast<int32_t>(result);
  (void)WriteAll(fd, &response, sizeof(response));
}

void HandleClient(int fd, CpuService* service) {
  MessageHeader header{};
  if (!ReadAll(fd, &header, sizeof(header)) || header.magic != kvaf::peer::kProtocolMagic ||
      header.version != kvaf::peer::kProtocolVersion) {
    return;
  }
  const Command command = static_cast<Command>(header.command);
  switch (command) {
    case Command::kAllocateRegion: {
      AllocateRegionRequest request{};
      request.header = header;
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<uint8_t*>(&request) + sizeof(header),
                   sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      AllocateRegionResponse response{};
      response.header = MakeHeader(command, sizeof(response));
      response.header.command = ResponseCommand(command);
      response.result = static_cast<int32_t>(
          service->Allocate(request.ddr_bytes, request.slot_size, request.candidate_id, &response));
      (void)WriteAll(fd, &response, sizeof(response));
      return;
    }
    case Command::kGetCandidateCount: {
      if (!IsValidHeader(header, command, sizeof(CandidateCountRequest))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      CandidateCountResponse response{};
      response.header = MakeHeader(command, sizeof(response));
      response.header.command = ResponseCommand(command);
      response.candidate_count = service == nullptr ? 0U : service->CandidateCount();
      response.result = static_cast<int32_t>(response.candidate_count == 0U
                                                  ? Result::kResourceError
                                                  : Result::kOk);
      (void)WriteAll(fd, &response, sizeof(response));
      return;
    }
    case Command::kCreateChannel: {
      CreateChannelRequest request{};
      request.header = header;
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<uint8_t*>(&request) + sizeof(header),
                   sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      CreateChannelResponse response{};
      response.header = MakeHeader(command, sizeof(response));
      response.header.command = ResponseCommand(command);
      response.result = static_cast<int32_t>(service->CreateChannel(request, &response));
      (void)WriteAll(fd, &response, sizeof(response));
      return;
    }
    case Command::kReleaseRegion: {
      ReleaseRegionRequest request{};
      request.header = header;
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<uint8_t*>(&request) + sizeof(header),
                   sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      SendStatus(fd, command, service->Release(request.session_id));
      return;
    }
  }
  SendStatus(fd, command, Result::kInvalidRequest);
}

}  // namespace

struct CpuRuntime::Impl {
  UrmaResourcePool resources;
  std::unique_ptr<CpuService> service;
};

CpuRuntime::CpuRuntime() = default;

CpuRuntime::~CpuRuntime() { (void)Shutdown(); }

Status CpuRuntime::Initialize(const CpuRuntimeOptions& options) {
  if (impl_ != nullptr) return Status::InvalidState("CpuRuntime has already been initialized");
  if (options.has_eid_index && options.ub_device.empty()) {
    return Status::InvalidArgument("CpuRuntime eid_index requires ub_device");
  }
  auto impl = std::make_unique<Impl>();
  if (!impl->resources.Initialize(options.ub_device, options.has_eid_index, options.eid_index)) {
    return Status::Internal("CpuRuntime failed to initialize UMDK resources");
  }
  impl->service = std::make_unique<CpuService>(&impl->resources, options.memory_page_mode);
  impl_ = std::move(impl);
  return Status::OK();
}

Status CpuRuntime::LinkTo(const LinkTarget&) {
  return Status::NotImplemented("CpuRuntime does not initiate links in the CPU--NPU deployment");
}

Status CpuRuntime::AcceptLink(const LinkTarget&) {
  return initialized() ? Status::OK() : Status::NotInitialized("CpuRuntime has not been initialized");
}

Status CpuRuntime::ReleaseLink(LinkId link_id) {
  if (!initialized()) return Status::NotInitialized("CpuRuntime has not been initialized");
  const Result result = impl_->service->Release(link_id);
  if (result == Result::kOk) return Status::OK();
  if (result == Result::kNotFound) return Status::NotFound("CpuRuntime Link was not found");
  return Status::Internal("CpuRuntime failed to release Link");
}

Status CpuRuntime::ServeConnection(int connection_fd) {
  if (!initialized()) return Status::NotInitialized("CpuRuntime has not been initialized");
  if (connection_fd < 0) return Status::InvalidArgument("CpuRuntime connection fd is invalid");
  HandleClient(connection_fd, impl_->service.get());
  return Status::OK();
}

Status CpuRuntime::Shutdown() {
  if (impl_ == nullptr) return Status::OK();
  impl_.reset();
  return Status::OK();
}

bool CpuRuntime::initialized() const { return impl_ != nullptr; }

RuntimeCapabilities CpuRuntime::capabilities() const {
  return RuntimeCapabilities{false, false, true, false};
}

}  // namespace kvaf
