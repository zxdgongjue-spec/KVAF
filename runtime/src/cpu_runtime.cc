#include "cpu_runtime.h"
#include "cpu_urma_channel.h"
#include "urma_transport.h"
#include "peer_protocol.h"
#include "cpu_urma_discovery.h"
#include "peer_link.h"
#include "cpu_gather.h"
#include "cpu_gather_executor.h"
#include "gather_workspace.h"
#include "cpu_push_get_worker.h"

#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>
#if defined(__linux__)
#include <time.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdarg>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <thread>
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
using kvaf::peer::CandidateInfoRequest;
using kvaf::peer::CandidateInfoResponse;
using kvaf::peer::Command;
using kvaf::peer::CreateChannelRequest;
using kvaf::peer::CreateChannelResponse;
using kvaf::peer::EnableDirectGatherRequest;
using kvaf::peer::EnableGatherLanesRequest;
using kvaf::peer::GatherSpan;
using kvaf::peer::IsValidHeader;
using kvaf::peer::MakeHeader;
using kvaf::peer::MessageHeader;
using kvaf::peer::ReleaseRegionRequest;
using kvaf::peer::ResponseCommand;
using kvaf::peer::Result;
using kvaf::peer::StatusResponse;
using kvaf::peer::kDeviceNameBytes;
using kvaf::peer::kEidBytes;
using kvaf::peer::kQpKeyBytes;
using kvaf::peer::kSegKeyBytes;

// Match the Host STANDARD Jetty profile used by the legacy in-process HCCP
// transport. Keep the send and receive depths separate: the old QP used an
// 8192-entry SQ but only a 256-entry RQ.
constexpr uint32_t kHostJfsDepth = 8192U;
constexpr uint32_t kHostJfrDepth = 256U;
constexpr uint32_t kHostJfcDepth = 16384U;
constexpr uint8_t kHostJettyPriority = 15U;
constexpr uint8_t kHostJettyRnrRetry = 7U;
constexpr uint8_t kHostJettyErrTimeout = 16U;
constexpr uint32_t kJettyKeyBytes = sizeof(urma_jetty_id_t) + sizeof(urma_transport_mode_t);
// HCCP reports the NPU's USER_CTL Jetty key as a 32-byte record on the
// deployed CANN release.  Its first 28 bytes are the public UMDK Jetty id
// and transport mode; the trailing four bytes are retained on the wire.
constexpr uint32_t kNpuJettyKeyBytes = 32U;
constexpr uint32_t kSegBytes = sizeof(urma_seg_t);
constexpr std::size_t kBasePageBytes = 4096U;
constexpr std::size_t kHugePage2MBytes = 2U * 1024U * 1024U;

static_assert(sizeof(urma_jetty_id_t) == 24U, "UMDK Jetty id ABI changed");
static_assert(kJettyKeyBytes == 28U, "HCCP UMDK Jetty key ABI changed");
static_assert(kSegBytes == 48U, "HCCP UMDK segment ABI changed");
static_assert(kJettyKeyBytes <= kQpKeyBytes, "QpKey protocol buffer is too small");
static_assert(kSegBytes <= kSegKeyBytes, "segment protocol buffer is too small");

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
    get_tp_list_ = reinterpret_cast<urma::GetTpList>(dlsym(RTLD_DEFAULT, "urma_get_tp_list"));
    import_jetty_ex_ = reinterpret_cast<urma::ImportJettyEx>(dlsym(RTLD_DEFAULT, "urma_import_jetty_ex"));
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
  urma::GetTpList tp_query() const { return get_tp_list_; }
  urma::ImportJettyEx jetty_import() const { return import_jetty_ex_; }

  bool GetCtpForPeer(const uint8_t peer_eid[kEidBytes], uint64_t* tp_handle) const {
    if (peer_eid == nullptr || tp_handle == nullptr || get_tp_list_ == nullptr || context_ == nullptr) {
      return false;
    }
    return urma::QueryTp(get_tp_list_, context_, peer_eid, tp_handle);
  }

  urma_target_jetty_t* ImportJettyEx(const uint8_t key[kQpKeyBytes], uint32_t token_value,
                                     uint64_t local_tp, uint64_t peer_tp, uint32_t local_psn,
                                     uint32_t peer_psn) const {
    if (key == nullptr || token_value == 0U || local_tp == 0U || peer_tp == 0U || local_psn == 0U ||
        peer_psn == 0U || import_jetty_ex_ == nullptr || context_ == nullptr) {
      return nullptr;
    }
    auto* target = urma::ImportJetty(import_jetty_ex_, context_, key, token_value,
                                     local_tp, peer_tp, local_psn, peer_psn);
    if (!target) Log("[KVAF_CPU] urma_import_jetty_ex failed errno=%d\n", errno);
    return target;
  }

 private:
  std::string selected_device_name_;
  uint32_t eid_index_ = 0U;
  urma_device_t* device_ = nullptr;
  urma_context_t* context_ = nullptr;
  urma::GetTpList get_tp_list_ = nullptr;
  urma::ImportJettyEx import_jetty_ex_ = nullptr;
};

struct UrmaCandidate : CpuUrmaCandidate {
  uint32_t id = 0U;
  uint32_t device_id = 0U;
  bool unavailable = false;
  std::unique_ptr<UrmaResources> resources;
};

class UrmaResourcePool {
 public:
  ~UrmaResourcePool() { Shutdown(); }

  bool Initialize(const std::string& requested_device, bool eid_index_requested,
                  uint32_t requested_eid_index) {
    const Status status = discovery_.Initialize(requested_device, eid_index_requested, requested_eid_index);
    if (!status.ok()) {
      Log("[KVAF_CPU] candidate discovery failed: %s\n", status.message().c_str());
      return false;
    }
    for (const auto& discovered : discovery_.candidates()) {
      UrmaCandidate candidate;
      static_cast<CpuUrmaCandidate&>(candidate) = discovered;
      candidates_.push_back(std::move(candidate));
    }
    uint32_t device_id = 0U;
    std::string previous_device;
    for (uint32_t index = 0U; index < candidates_.size(); ++index) {
      UrmaCandidate& candidate = candidates_[index];
      candidate.id = index;
      if (index != 0U && candidate.device_name != previous_device) ++device_id;
      candidate.device_id = device_id;
      previous_device = candidate.device_name;
      Log("[KVAF_CPU] CANDIDATE_DISCOVERED id=%u device_id=%u ub_device=%s "
          "eid_index=%u eid=", candidate.id, candidate.device_id,
          candidate.device_name.c_str(), candidate.eid_index);
      for (uint8_t byte : candidate.eid) {
        std::fprintf(stderr, "%02x", byte);
      }
      std::fprintf(stderr, "\n");
    }
    return true;
  }

  uint32_t DeviceCount() const {
    uint32_t count = 0U;
    std::string previous;
    for (const auto& candidate : candidates_) {
      if (candidate.device_name != previous) ++count;
      previous = candidate.device_name;
    }
    return count;
  }

  uint32_t CandidateCount() const {
    return static_cast<uint32_t>(candidates_.size());
  }

  const std::vector<UrmaCandidate>& candidates() const { return candidates_; }
  bool GetInfo(uint32_t candidate_id, UrmaCandidate* info) const {
    if (info == nullptr || candidate_id >= candidates_.size()) return false;
    *info = UrmaCandidate{};
    info->id = candidates_[candidate_id].id;
    info->device_id = candidates_[candidate_id].device_id;
    info->device_name = candidates_[candidate_id].device_name;
    info->eid_index = candidates_[candidate_id].eid_index;
    info->eid = candidates_[candidate_id].eid;
    return true;
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
    discovery_.Shutdown();
  }

  CpuUrmaDiscovery discovery_;
  std::vector<UrmaCandidate> candidates_;
};

using Channel = cpu_runtime_detail::UrmaChannel;

peer::GatherWaitMode ToGatherWaitMode(CpuGatherMode mode) {
  switch (mode) {
    case CpuGatherMode::kAdaptive: return peer::GatherWaitMode::kAdaptive;
    case CpuGatherMode::kActive: return peer::GatherWaitMode::kActive;
    case CpuGatherMode::kPassive: return peer::GatherWaitMode::kPassive;
  }
  return peer::GatherWaitMode::kPassive;
}

class CpuGatherWorker {
 public:
  CpuGatherWorker(void* memory, uint64_t base, uint64_t bytes,
                  std::size_t threads, std::size_t min_bytes, bool affinity,
                  CpuGatherMode mode, uint64_t control_address = 0U,
                  std::size_t cpu_offset = 0U)
      : memory_(static_cast<std::uint8_t*>(memory)), base_(base), bytes_(bytes),
        control_(reinterpret_cast<peer::GatherControl*>(memory_ +
            (control_address ? control_address - base : peer::GatherControlOffset(bytes)))),
        executor_(memory, base, threads, min_bytes, ToGatherWaitMode(mode), affinity, cpu_offset), mode_(mode) {
    std::memset(control_, 0, sizeof(*control_));
    peer::GatherBarrier();
    worker_ = std::thread(&CpuGatherWorker::Run, this);
  }
  ~CpuGatherWorker() {
    stop_.store(true, std::memory_order_release);
    worker_.join();
  }
 private:
  void Run() {
    executor_.BindWorker(0U);
    std::uint64_t last = 0U;
    while (!stop_.load(std::memory_order_acquire)) {
      const auto sequence = __atomic_load_n(&control_->notify, __ATOMIC_ACQUIRE);
      if (!sequence || sequence == last || (sequence & peer::kGatherErrorBit)) {
        if (mode_ == CpuGatherMode::kPassive)
          std::this_thread::sleep_for(std::chrono::microseconds(20));
        else
          peer::GatherSpinPause();
        continue;
      }
      last = sequence;
      __atomic_store_n(&control_->ack, peer::GatherStartedAck(sequence), __ATOMIC_RELEASE);
      peer::GatherBarrier();
      const auto address = __atomic_load_n(&control_->command, __ATOMIC_ACQUIRE);
      bool valid = false;
      try {
        valid = executor_.PrepareDirect(bytes_, address, sequence, peer::kMaxGatherSpans) != 0U;
        if (valid) executor_.Copy();
      } catch (const std::exception& error) {
        Log("[KVAF_CPU] GatherRead failed: %s\n", error.what());
        valid = false;
      }
      if (!valid && address % 64U == 0U &&
          peer::GatherRange(base_, peer::GatherControlOffset(bytes_), address,
                            sizeof(peer::GatherDescriptorHeader))) {
        auto* progress = reinterpret_cast<std::uint64_t*>(
            memory_ + address - base_ + offsetof(peer::GatherDescriptorHeader, ready_bytes));
        __atomic_store_n(progress, peer::kGatherErrorBit, __ATOMIC_RELEASE);
      }
      __atomic_store_n(&control_->ack,
                      sequence | (valid ? 0U : peer::kGatherErrorBit), __ATOMIC_RELEASE);
      peer::GatherBarrier();
    }
  }
  std::uint8_t* memory_;
  std::uint64_t base_, bytes_;
  peer::GatherControl* control_;
  peer::CpuGatherExecutor executor_;
  CpuGatherMode mode_;
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

struct Region {
  uint64_t id = 0U;
  uint32_t candidate_id = 0U;
  UrmaResources* resources = nullptr;
  void* memory = nullptr;
  uint64_t bytes = 0U;
  uint64_t base = 0U;
  urma_token_id_t* token_id = nullptr;
  urma_target_seg_t* segment = nullptr;
  uint32_t token_value = 0U;
  std::vector<Channel> channels;
  std::unique_ptr<CpuGatherWorker> gather;
  std::vector<std::unique_ptr<CpuGatherWorker>> gather_lanes;
  std::unique_ptr<cpu_push::Worker> push;
};

class CpuService {
 public:
  CpuService(UrmaResourcePool* resource_pool, std::size_t gather_threads,
             std::size_t min_bytes_per_thread, bool gather_affinity,
             CpuGatherMode gather_mode,
             CpuMemoryPageMode memory_page_mode = CpuMemoryPageMode::kTransparentHugePage,
             bool device_filter_applied = false)
      : resource_pool_(resource_pool), gather_threads_(gather_threads),
        min_bytes_per_thread_(min_bytes_per_thread),
        gather_affinity_(gather_affinity), gather_mode_(gather_mode),
        memory_page_mode_(memory_page_mode), device_filter_applied_(device_filter_applied) {}
  ~CpuService() { ReleaseAll(); }

  uint32_t CandidateCount() const {
    return resource_pool_ == nullptr ? 0U : resource_pool_->CandidateCount();
  }

  uint32_t DeviceCount() const {
    return resource_pool_ == nullptr ? 0U : resource_pool_->DeviceCount();
  }

  bool DeviceFilterApplied() const { return device_filter_applied_; }

  Result GetCandidateInfo(uint32_t candidate_id,
                          CandidateInfoResponse* response) const {
    if (response == nullptr || resource_pool_ == nullptr) {
      return Result::kInvalidRequest;
    }
    UrmaCandidate candidate{};
    if (!resource_pool_->GetInfo(candidate_id, &candidate)) {
      return Result::kNotFound;
    }
    if (candidate.device_name.empty() ||
        candidate.device_name.size() >= kDeviceNameBytes) {
      return Result::kResourceError;
    }
    response->candidate_id = candidate.id;
    response->device_id = candidate.device_id;
    response->eid_index = candidate.eid_index;
    std::memcpy(response->device_name, candidate.device_name.c_str(),
                candidate.device_name.size());
    return Result::kOk;
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
    region->bytes = bytes;
    if (!AllocateMemory(region.get())) {
      return Result::kResourceError;
    }
    region->bytes = bytes;
    if (bytes >= 64U)
      std::memset(static_cast<uint8_t*>(region->memory) + peer::GatherControlOffset(bytes), 0, 64U);
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
    region->base = region->segment->seg.ubva.va;
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
        response->peer_base, response->token_id, flag.value, MemoryPageModeName());
    return Result::kOk;
  }

  Result CreateChannel(const CreateChannelRequest& request, CreateChannelResponse* response) {
    if (response == nullptr ||
        (request.local_qp_key_bytes != kJettyKeyBytes &&
         request.local_qp_key_bytes != kNpuJettyKeyBytes) ||
        std::all_of(request.local_eid, request.local_eid + kEidBytes,
                    [](uint8_t byte) { return byte == 0U; }) ||
        request.peer_token_value == 0U || request.local_tp_handle == 0U || request.local_psn == 0U ||
        (request.reserved & ~peer::kChannelSharedTp) != 0U) {
      Log("[KVAF_CPU] CHANNEL_REJECTED key_bytes=%u expected_key_bytes=%u/%u token=0x%x peer_tp=0x%llx "
          "peer_psn=0x%x\n",
          request.local_qp_key_bytes, kJettyKeyBytes, kNpuJettyKeyBytes, request.peer_token_value,
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
    const bool share_tp = (request.reserved & peer::kChannelSharedTp) != 0U;
    const bool created = share_tp
        ? channel.CreateLocal(region.resources->context(), region.token_value)
        : CreateLocalJetty(region.resources, region.token_value, &channel.local_jfc,
                            &channel.local_jfr, &channel.local_jetty);
    if (!created)
      return Result::kResourceError;
    const auto* shared = share_tp ? cpu_runtime_detail::FindSharedChannelTp(
        region.channels, request.local_eid, request.local_tp_handle) : nullptr;
    if (shared != nullptr) {
      if (shared->peer_psn != request.local_psn) return Result::kInvalidRequest;
      channel.tp = *shared;
    } else {
      channel.tp.shared = share_tp;
      std::copy_n(request.local_eid, kEidBytes, channel.tp.peer_eid.begin());
      channel.tp.peer = request.local_tp_handle;
      channel.tp.peer_psn = request.local_psn;
      channel.tp.local_psn = NextPsn();
      if (!region.resources->GetCtpForPeer(request.local_eid, &channel.tp.local))
        return Result::kResourceError;
    }
    if (!ImportRemoteJetty(region.resources, request.local_qp_key, request.peer_token_value,
                           channel.tp.local, channel.tp.peer, channel.tp.local_psn,
                           channel.tp.peer_psn, &channel.remote_jetty))
      return Result::kResourceError;
    std::memcpy(response->peer_qp_key, &channel.local_jetty->jetty_id,
                sizeof(channel.local_jetty->jetty_id));
    const urma_transport_mode_t mode = channel.local_jetty->jetty_cfg.jfs_cfg.trans_mode;
    std::memcpy(response->peer_qp_key + sizeof(channel.local_jetty->jetty_id), &mode, sizeof(mode));
    response->peer_qp_key_bytes = kJettyKeyBytes;
    response->peer_tp_handle = channel.tp.local;
    response->peer_psn = channel.tp.local_psn;
    response->reserved = share_tp ? peer::kChannelSharedTp : 0U;
    const bool tp_reused = shared != nullptr;
    region.channels.push_back(std::move(channel));
    const auto& connected = region.channels.back();
    Log("[KVAF_CPU] CHANNEL_CREATED session=%" PRIu64 " local_jetty=%u remote_jetty=%u cpu_tp=0x%llx "
        "cpu_psn=0x%x cpu_cq_id=%u cpu_jfr_id=%u peer_tp=0x%llx tp_shared=%u tp_reused=%u\n",
        region.id, connected.local_jetty->jetty_id.id, connected.remote_jetty->id.id,
        static_cast<unsigned long long>(connected.tp.local), connected.tp.local_psn,
        connected.local_jfc->jfc_id.id, connected.local_jfr->jfr_id.id,
        static_cast<unsigned long long>(connected.tp.peer), share_tp, tp_reused);
    return Result::kOk;
  }

  Result Release(uint64_t id) {
    const auto found = regions_.find(id);
    if (found == regions_.end()) {
      return Result::kNotFound;
    }
    if (StopCpuPushGet(id) != Result::kOk) return Result::kResourceError;
    Log("[KVAF_CPU] REGION_RELEASE session=%" PRIu64 " channels=%zu\n", id,
        found->second->channels.size());
    ReleaseRegion(found->second.get());
    regions_.erase(found);
    return Result::kOk;
  }





  Result EnableDirectGather(uint64_t session_id) {
    const auto found = regions_.find(session_id);
    if (found == regions_.end()) return Result::kNotFound;
    Region& region = *found->second;
    if (region.push || region.gather != nullptr || !region.gather_lanes.empty()) return Result::kResourceError;
    try {
      region.gather = std::make_unique<CpuGatherWorker>(
          region.memory, region.base, region.bytes, gather_threads_,
          min_bytes_per_thread_, gather_affinity_, gather_mode_);
    } catch (const std::exception& error) {
      Log("[KVAF_CPU] GATHER_DIRECT_ENABLE_FAILED reason=%s\n", error.what());
      return Result::kResourceError;
    }
    return Result::kOk;
  }

  Result EnableGatherLanes(uint64_t session_id, uint64_t control_address, uint32_t lanes) {
    const auto found = regions_.find(session_id);
    if (found == regions_.end()) return Result::kNotFound;
    auto& region = *found->second;
    if (!lanes || lanes > kGatherMaxWorkers || lanes > gather_threads_ || control_address % 64U ||
        !peer::GatherRange(region.base, peer::GatherControlOffset(region.bytes), control_address,
                           lanes * sizeof(peer::GatherControl)))
      return Result::kInvalidRequest;
    if (region.push || region.gather || !region.gather_lanes.empty()) return Result::kResourceError;
    std::vector<std::unique_ptr<CpuGatherWorker>> created;
    std::size_t cpu_offset = 0U;
    try {
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        const auto budget = gather_threads_ / lanes + (lane < gather_threads_ % lanes ? 1U : 0U);
        created.push_back(std::make_unique<CpuGatherWorker>(
            region.memory, region.base, region.bytes, budget, min_bytes_per_thread_,
            gather_affinity_, gather_mode_, control_address + lane * sizeof(peer::GatherControl),
            cpu_offset));
        cpu_offset += budget;
      }
    } catch (const std::exception& error) {
      Log("[KVAF_CPU] GATHER_LANE_SETUP_FAILED reason=%s\n", error.what());
      return Result::kResourceError;
    }
    region.gather_lanes = std::move(created);
    return Result::kOk;
  }

  Result StopGatherLanes(uint64_t session_id) {
    const auto found = regions_.find(session_id);
    if (found == regions_.end()) return Result::kOk;
    // Destruction joins the workers; DDR descriptor/staging can then be freed.
    found->second->gather_lanes.clear();
    return Result::kOk;
  }

  Result OpenCpuPushGet(const peer::OpenCpuPushGetRequest& request,
                        peer::OpenCpuPushGetResponse* response) {
    const auto found = regions_.find(request.session_id);
    if (found == regions_.end()) return Result::kNotFound;
    auto& region = *found->second;
    if (region.push || region.gather || !region.gather_lanes.empty()) return Result::kResourceError;
    if (!request.source_bytes || request.source_bytes > region.bytes)
      return Result::kInvalidRequest;
    Log("[KVAF_CPU] CPU_PUSH_OPEN session=%llu workers=%u token_bytes=%u hbm_bytes=%llu\n",
        static_cast<unsigned long long>(request.session_id), request.workers, request.token_bytes,
        static_cast<unsigned long long>(request.hbm_bytes));
    auto transport = std::make_unique<cpu_push::Transport>();
    const auto status = transport->Initialize(region.resources->context(), region.segment,
        request, region.resources->tp_query(), region.resources->jetty_import(),
        NextPsn(), response);
    if (!status.ok()) {
      Log("[KVAF_CPU] CPU_PUSH_OPEN_FAILED reason=%s\n", status.message().c_str());
      const auto stopped = transport->Shutdown();
      if (!stopped.ok()) {
        // Retain the failed setup in the region so STOP can retry teardown.
        region.push = std::make_unique<cpu_push::Worker>(std::move(transport));
      }
      return Result::kResourceError;
    }
    region.push = std::make_unique<cpu_push::Worker>(std::move(transport));
    return Result::kOk;
  }
  Result StartCpuPushGet(uint64_t session) {
    const auto found = regions_.find(session);
    if (found == regions_.end() || !found->second->push) return Result::kNotFound;
    Log("[KVAF_CPU] CPU_PUSH_START session=%llu\n", static_cast<unsigned long long>(session));
    const auto status = found->second->push->Start();
    if (!status.ok()) {
      Log("[KVAF_CPU] CPU_PUSH_START_FAILED session=%llu reason=%s\n",
          static_cast<unsigned long long>(session), status.message().c_str());
      return Result::kResourceError;
    }
    Log("[KVAF_CPU] CPU_PUSH_STARTED session=%llu\n", static_cast<unsigned long long>(session));
    return Result::kOk;
  }
  Result StopCpuPushGet(uint64_t session) {
    const auto found = regions_.find(session);
    if (found == regions_.end()) return Result::kOk;
    auto& push = found->second->push;
    if (!push) return Result::kOk;
    const auto status = push->Stop();
    if (!status.ok()) {
      Log("[KVAF_CPU] CPU_PUSH_STOP_FAILED session=%llu reason=%s\n",
          static_cast<unsigned long long>(session), status.message().c_str());
      return Result::kResourceError;
    }
    push.reset();
    return Result::kOk;
  }
  Status StopAllPush() {
    for (auto& entry : regions_)
      if (StopCpuPushGet(entry.first) != Result::kOk)
        return Status::Internal("CPU push queues still live; runtime resources retained");
    return Status::OK();
  }

 private:
  const char* MemoryPageModeName() const {
    switch (memory_page_mode_) {
      case CpuMemoryPageMode::kDefault:
        return "default";
      case CpuMemoryPageMode::kTransparentHugePage:
        return "thp";
    }
    return "unknown";
  }

  bool AllocateMemory(Region* region) {
    if (region == nullptr || region->bytes == 0U ||
        region->bytes > std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(region->bytes);
    const std::size_t alignment =
        memory_page_mode_ == CpuMemoryPageMode::kTransparentHugePage
            ? kHugePage2MBytes
            : kBasePageBytes;
    const int allocation_status = posix_memalign(&region->memory, alignment, bytes);
    if (allocation_status != 0 || region->memory == nullptr) {
      Log("[KVAF_CPU] REGION_ALLOCATE_FAILED stage=posix_memalign bytes=%" PRIu64
          " alignment=%zu status=%d memory=%p page_mode=%s\n",
          region->bytes, alignment, allocation_status, region->memory,
          MemoryPageModeName());
      return false;
    }
    if (memory_page_mode_ == CpuMemoryPageMode::kTransparentHugePage) {
      if (madvise(region->memory, bytes, MADV_HUGEPAGE) != 0) {
        Log("[KVAF_CPU] REGION_ALLOCATE_FAILED stage=madvise_hugepage bytes=%" PRIu64
            " errno=%d\n",
            region->bytes, errno);
        free(region->memory);
        region->memory = nullptr;
        return false;
      }
      // Fault the range while huge-page advice is active. Launching the service
      // under a NUMA memory policy makes placement deterministic as well.
      std::memset(region->memory, 0, bytes);
    }
    return true;
  }

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
    jfc_config.depth = kHostJfcDepth;
    *jfc = urma_create_jfc(resources->context(), &jfc_config);
    if (*jfc == nullptr) {
      Log("[KVAF_CPU] urma_create_jfc failed scope=channel depth=%u errno=%d\n",
          kHostJfcDepth, errno);
      return false;
    }
    urma_jfs_cfg_t jfs{};
    jfs.depth = kHostJfsDepth;
    jfs.flag.bs.error_suspend = 1U;
    jfs.trans_mode = URMA_TM_RM;
    jfs.priority = kHostJettyPriority;
    jfs.max_sge = 1U;
    jfs.rnr_retry = kHostJettyRnrRetry;
    jfs.err_timeout = kHostJettyErrTimeout;
    jfs.jfc = *jfc;
    urma_jfr_cfg_t jfr_config{};
    jfr_config.depth = kHostJfrDepth;
    jfr_config.trans_mode = URMA_TM_RM;
    jfr_config.max_sge = 1U;
    jfr_config.jfc = *jfc;
    jfr_config.token_value.token = token_value;
    jfr_config.flag.bs.tag_matching = URMA_NO_TAG_MATCHING;
    *jfr = urma_create_jfr(resources->context(), &jfr_config);
    if (*jfr == nullptr) {
      Log("[KVAF_CPU] urma_create_jfr failed depth=%u token=0x%x errno=%d\n", kHostJfrDepth,
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
      Log("[KVAF_CPU] urma_create_jetty failed jfs_depth=%u jfr_depth=%u token=0x%x errno=%d\n",
          kHostJfsDepth, kHostJfrDepth, token_value, errno);
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
    if (region->push) {
      if (!region->push->Stop().ok()) {
        // The process must keep source DDR registered until all DMA stops.
        Log("[KVAF_CPU] CPU_PUSH_RELEASE_FAILED source region retained\n");
        return;
      }
      region->push.reset();
    }
    region->gather.reset();
    region->gather_lanes.clear();
    while (!region->channels.empty()) region->channels.pop_back();
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
  std::size_t gather_threads_{1U};
  std::size_t min_bytes_per_thread_{256U * 1024U};
  bool gather_affinity_{true};
  CpuGatherMode gather_mode_{CpuGatherMode::kAdaptive};
  CpuMemoryPageMode memory_page_mode_{CpuMemoryPageMode::kTransparentHugePage};
  bool device_filter_applied_{false};
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
      response.device_count = service == nullptr ? 0U : service->DeviceCount();
      response.device_filter_applied =
          service != nullptr && service->DeviceFilterApplied() ? 1U : 0U;
      response.result = static_cast<int32_t>(response.candidate_count == 0U
                                                  ? Result::kResourceError
                                                  : Result::kOk);
      (void)WriteAll(fd, &response, sizeof(response));
      return;
    }
    case Command::kGetCandidateInfo: {
      CandidateInfoRequest request{};
      std::memcpy(&request, &header, sizeof(header));
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<uint8_t*>(&request) + sizeof(header),
                   sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      CandidateInfoResponse response{};
      response.header = MakeHeader(command, sizeof(response));
      response.header.command = ResponseCommand(command);
      response.result = static_cast<int32_t>(
          service == nullptr
              ? Result::kResourceError
              : service->GetCandidateInfo(request.candidate_id, &response));
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
    case Command::kEnableDirectGather: {
      EnableDirectGatherRequest request{};
      request.header = header;
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<uint8_t*>(&request) + sizeof(header),
                   sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      SendStatus(fd, command, service->EnableDirectGather(request.session_id));
      return;
    }
    case Command::kEnableGatherLanes: {
      EnableGatherLanesRequest request{};
      request.header = header;
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<uint8_t*>(&request) + sizeof(header),
                   sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      SendStatus(fd, command,
                 service->EnableGatherLanes(request.session_id, request.control_address, request.lanes));
      return;
    }

    case Command::kOpenCpuPushGet: {
      peer::OpenCpuPushGetRequest request{};
      request.header = header;
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<uint8_t*>(&request) + sizeof(header), sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      peer::OpenCpuPushGetResponse response{};
      response.header = MakeHeader(command, sizeof(response));
      response.header.command = ResponseCommand(command);
      response.result = static_cast<int32_t>(service->OpenCpuPushGet(request, &response));
      (void)WriteAll(fd, &response, sizeof(response));
      return;
    }
    case Command::kStopGatherLanes:
    case Command::kStartCpuPushGet:
    case Command::kStopCpuPushGet: {
      peer::CpuPushGetSessionRequest request{};
      request.header = header;
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<uint8_t*>(&request) + sizeof(header), sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      SendStatus(fd, command, command == Command::kStopGatherLanes ?
          service->StopGatherLanes(request.session_id) : command == Command::kStartCpuPushGet
          ? service->StartCpuPushGet(request.session_id) : service->StopCpuPushGet(request.session_id));
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

CpuRuntime::~CpuRuntime() { if (!Shutdown().ok()) (void)impl_.release(); }

Status CpuRuntime::Initialize(const CpuRuntimeOptions& options) {
  if (impl_ != nullptr) return Status::InvalidState("CpuRuntime has already been initialized");
  if (options.has_eid_index && options.ub_device.empty()) {
    return Status::InvalidArgument("CpuRuntime eid_index requires ub_device");
  }
  if (options.gather_threads > 64U)
    return Status::InvalidArgument("CpuRuntime gather_threads must be in [1,64] or auto");
  const auto detected = std::thread::hardware_concurrency();
  const std::size_t gather_threads = options.gather_threads == 0U
      ? std::min<std::size_t>(detected == 0U ? 1U : detected, 32U)
      : options.gather_threads;
  auto impl = std::make_unique<Impl>();
  if (!impl->resources.Initialize(options.ub_device, options.has_eid_index,
                                  options.eid_index)) {
    return Status::Internal("CpuRuntime failed to initialize UMDK resources");
  }
  const std::size_t min_bytes_per_thread = options.gather_threads == 0U
      ? 256U * 1024U : 64U * 1024U;
  impl->service = std::make_unique<CpuService>(
      &impl->resources, gather_threads, min_bytes_per_thread,
      options.gather_affinity, options.gather_mode,
      options.memory_page_mode, !options.ub_device.empty());
  impl_ = std::move(impl);
  return Status::OK();
}

Status CpuRuntime::AcceptLink(const LinkTarget&) {
  return initialized() ? Status::OK() : Status::NotInitialized("CpuRuntime has not been initialized");
}

Status CpuRuntime::ServeConnection(int connection_fd) {
  if (!initialized()) return Status::NotInitialized("CpuRuntime has not been initialized");
  if (connection_fd < 0) return Status::InvalidArgument("CpuRuntime connection fd is invalid");
  HandleClient(connection_fd, impl_->service.get());
  return Status::OK();
}

Status CpuRuntime::LinkTo(const LinkTarget&) {
  return Status::NotImplemented("CpuRuntime does not initiate links in the CPU--NPU deployment");
}

Status CpuRuntime::ReleaseLink(LinkId link_id) {
  if (!initialized()) return Status::NotInitialized("CpuRuntime has not been initialized");
  const Result result = impl_->service->Release(link_id);
  if (result == Result::kOk) return Status::OK();
  if (result == Result::kNotFound) return Status::NotFound("CpuRuntime Link was not found");
  return Status::Internal("CpuRuntime failed to release Link");
}

Status CpuRuntime::Shutdown() {
  if (impl_ == nullptr) return Status::OK();
  auto stopped = impl_->service->StopAllPush();
  if (!stopped.ok()) return stopped;
  impl_.reset();
  return Status::OK();
}

bool CpuRuntime::initialized() const { return impl_ != nullptr; }

RuntimeCapabilities CpuRuntime::capabilities() const {
  return RuntimeCapabilities{false, false, true, false};
}

}  // namespace kvaf
