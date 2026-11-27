#include "roce_npu_transport.h"

#include <acl/acl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "roce_connection.h"
#include "meta_manager.h"

extern "C" int halMemRegUbSegment(std::uint32_t device, std::uint64_t address, std::uint64_t bytes);
extern "C" int halMemUnRegUbSegment(std::uint32_t device, std::uint64_t address);

namespace kvaf {
namespace {

constexpr auto kCompletionTimeout = std::chrono::seconds(10);
constexpr std::size_t kMaxGetRequests = 1U << 20;
constexpr std::size_t kAclCopyAlignment = 64;
constexpr std::size_t kMaxStagingBytes = 32U << 20;
constexpr std::size_t kMaxPendingGets = 64;
constexpr std::size_t kMaxCachedMrs = 64;
constexpr std::size_t kMaxCachedMrBytes = 1U << 30;

using MrClock = std::chrono::steady_clock;
std::atomic<std::uint64_t> next_mr_timing_id{1};

bool MrTimingEnabled() {
  const char* value = std::getenv("KVAF_ROCE_MR_TIMING");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

MrClock::time_point MrTimingStart(bool enabled) {
  return enabled ? MrClock::now() : MrClock::time_point{};
}

double MrElapsedUs(bool enabled, MrClock::time_point start) {
  return enabled ? std::chrono::duration<double, std::micro>(MrClock::now() - start).count() : 0.0;
}

bool RoceLatencyEnabled() {
  const char* value = std::getenv("KVAF_ROCE_LATENCY");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

struct ScopedLatency {
  explicit ScopedLatency(double* elapsed)
      : elapsed(elapsed), start(MrTimingStart(elapsed != nullptr)) {}
  ~ScopedLatency() { Finish(); }

  void Finish() {
    if (!elapsed) return;
    *elapsed += MrElapsedUs(true, start);
    elapsed = nullptr;
  }

  double* elapsed;
  MrClock::time_point start;
};

struct RdmaLatency {
  double queue_lock_us{};
  double post_send_us{};
  double cq_wait_us{};
  std::uint64_t posted_bytes{};
  std::uint64_t posted_wrs{};
  std::uint64_t completed_wrs{};
};

struct RoceCallLatency {
  RoceCallLatency(bool enabled, const char* operation, std::uint32_t device)
      : id(enabled ? next_id.fetch_add(1, std::memory_order_relaxed) : 0),
        operation(operation), device(device), start(MrTimingStart(enabled)) {}

  ~RoceCallLatency() {
    if (!id) return;
    std::fprintf(stderr,
                 "==============[KVAF_ROCE_LATENCY]=========== pid=%ld id=%llu device=%u op=%s phase=call ok=%d "
                 "total_us=%.3f requests_or_capacity=%zu wait_prev_get_us=%.3f "
                 "stream_sync_us=%.3f mr_us=%.3f mr_mode=%s staging_us=%.3f "
                 "enqueue_us=%.3f wait_current_get_us=%.3f queue_lock_us=%.3f "
                 "post_send_us=%.3f cq_wait_us=%.3f posted_bytes=%llu posted_wrs=%llu "
                 "completed_wrs=%llu\n",
                 static_cast<long>(getpid()), static_cast<unsigned long long>(id), device,
                 operation, ok, MrElapsedUs(true, start), requests_or_capacity,
                 wait_prev_get_us, stream_sync_us, mr_us, mr_mode, staging_us,
                 enqueue_us, wait_current_get_us, rdma.queue_lock_us, rdma.post_send_us,
                 rdma.cq_wait_us, static_cast<unsigned long long>(rdma.posted_bytes),
                 static_cast<unsigned long long>(rdma.posted_wrs),
                 static_cast<unsigned long long>(rdma.completed_wrs));
  }

  static std::atomic<std::uint64_t> next_id;
  const std::uint64_t id;
  const char* operation;
  std::uint32_t device;
  MrClock::time_point start;
  bool ok{};
  std::size_t requests_or_capacity{};
  const char* mr_mode{"none"};
  double wait_prev_get_us{};
  double stream_sync_us{};
  double mr_us{};
  double staging_us{};
  double enqueue_us{};
  double wait_current_get_us{};
  RdmaLatency rdma;
};

std::atomic<std::uint64_t> RoceCallLatency::next_id{1};

// Serialize HAL registration lifetimes across Runtime instances. The HAL
// segment is device-wide, while each verbs registration is local to its PD.
std::mutex hbm_registration_mutex;
using SegmentKey = std::pair<std::uint32_t, std::uint64_t>;
struct SegmentUsers {
  std::size_t bytes;
  std::size_t refs;
};
std::map<SegmentKey, SegmentUsers> hbm_segments;

bool IsOnNpu(const void* address, std::uint32_t device) {
  aclrtPtrAttributes attributes{};
  return aclrtPointerGetAttributes(address, &attributes) == ACL_SUCCESS &&
         attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE &&
         attributes.location.id == device;
}

[[noreturn]] void UnsafeTeardown(const char* reason) {
  std::fprintf(stderr, "==============[KVAF_ROCE]=========== unsafe teardown: %s\n", reason);
  // The C++ API borrows the Tensor. Returning to the caller could free it
  // while the provider still owns the registration or may DMA into it.
  std::abort();
}

struct HbmRange {
  void* base{};
  std::size_t bytes{};
};

Status FindHbmRange(std::uint32_t device, const void* buffer, std::size_t bytes,
                    HbmRange* range) {
  if (!IsOnNpu(buffer, device)) {
    return Status::InvalidArgument("RoCE buffer must reside on the selected NPU");
  }
  if (aclrtMemGetAddressRange(const_cast<void*>(buffer), &range->base, &range->bytes) !=
          ACL_SUCCESS || range->base == nullptr || range->bytes == 0) {
    return Status::InvalidArgument("cannot resolve the RoCE buffer allocation");
  }
  const auto address = reinterpret_cast<std::uintptr_t>(buffer);
  const auto base = reinterpret_cast<std::uintptr_t>(range->base);
  if (address < base || bytes > range->bytes || address - base > range->bytes - bytes) {
    return Status::InvalidArgument("RoCE buffer range exceeds its NPU allocation");
  }
  return Status::OK();
}

class HbmRegistration {
 public:
  explicit HbmRegistration(bool timing)
      : timing_id_(timing ? next_mr_timing_id.fetch_add(1, std::memory_order_relaxed) : 0) {}

  ~HbmRegistration() {
    const bool timing = timing_id_ != 0;
    const auto start = MrTimingStart(timing);
    double verbs_us = 0.0;
    const bool verbs_called = mr != nullptr;
    if (verbs_called) {
      const auto verbs_start = MrTimingStart(timing);
      if (ibv_dereg_mr(mr) != 0) UnsafeTeardown("ibv_dereg_mr failed");
      verbs_us = MrElapsedUs(timing, verbs_start);
    }
    double hal_us = 0.0;
    bool hal_called = false;
    if (registered_) {
      std::lock_guard<std::mutex> lock(hbm_registration_mutex);
      const auto segment = hbm_segments.find(key_);
      if (--segment->second.refs == 0) {
        hal_called = true;
        const auto hal_start = MrTimingStart(timing);
        if (halMemUnRegUbSegment(key_.first, key_.second) != 0)
          UnsafeTeardown("halMemUnRegUbSegment failed");
        hal_us = MrElapsedUs(timing, hal_start);
        hbm_segments.erase(segment);
      }
    }
    if (timing && (verbs_called || registered_))
      LogTiming("deregister", 0, hal_called, hal_us, verbs_called, verbs_us, start);
  }

  Status Register(ibv_pd* pd, std::uint32_t device, HbmRange range, const void* buffer,
                  std::size_t bytes, int access, std::shared_ptr<void> owner) {
    const bool timing = timing_id_ != 0;
    const auto start = MrTimingStart(timing);
    owner_ = std::move(owner);
    key_ = {device, reinterpret_cast<std::uintptr_t>(range.base)};
    bytes_ = bytes;
    double hal_us = 0.0;
    bool hal_called = false;
    {
      std::lock_guard<std::mutex> lock(hbm_registration_mutex);
      auto [segment, inserted] = hbm_segments.try_emplace(key_, SegmentUsers{range.bytes, 0});
      if (inserted) {
        hal_called = true;
        const auto hal_start = MrTimingStart(timing);
        const int rc = halMemRegUbSegment(device, key_.second, range.bytes);
        hal_us = MrElapsedUs(timing, hal_start);
        if (rc != 0) {
          hbm_segments.erase(segment);
          return Status::Internal("halMemRegUbSegment failed: " + std::to_string(rc));
        }
      } else if (segment->second.bytes != range.bytes) {
        return Status::InvalidState("RoCE HBM registration has a different allocation size");
      }
      ++segment->second.refs;
      registered_ = true;
    }
    const auto verbs_start = MrTimingStart(timing);
    mr = ibv_reg_mr(pd, const_cast<void*>(buffer), bytes, access);
    if (timing) LogTiming("register", mr == nullptr ? 1 : 0, hal_called, hal_us, true,
                          MrElapsedUs(timing, verbs_start), start);
    if (mr == nullptr) return Status::Internal("ibv_reg_mr failed for NPU HBM");
    return Status::OK();
  }

  bool Matches(const HbmRange& range, const std::shared_ptr<void>& owner) const {
    return mr->addr == range.base && mr->length == range.bytes &&
           !owner.owner_before(owner_) && !owner_.owner_before(owner);
  }

  ibv_mr* mr{};

 private:
  void LogTiming(const char* phase, int failed, bool hal_called, double hal_us, bool verbs_called,
                 double verbs_us, MrClock::time_point start) const {
    std::fprintf(stderr, "==============[KVAF_ROCE_MR]=========== id=%llu device=%u phase=%s failed=%d bytes=%zu "
                         "hal_called=%d hal_us=%.3f verbs_called=%d verbs_us=%.3f total_us=%.3f\n",
                 static_cast<unsigned long long>(timing_id_), key_.first, phase, failed, bytes_,
                 hal_called, hal_us, verbs_called, verbs_us, MrElapsedUs(true, start));
  }

  std::shared_ptr<void> owner_;
  SegmentKey key_{};
  bool registered_{false};
  std::size_t bytes_{};
  const std::uint64_t timing_id_;
};

class RoceHbmRegion final : public HbmRegion,
                           public std::enable_shared_from_this<RoceHbmRegion> {
 public:
  RoceHbmRegion(std::shared_ptr<const void> session_identity, std::uint32_t device,
                std::uintptr_t base,
                std::size_t bytes, std::shared_ptr<HbmRegistration> registration)
      : session_identity_(std::move(session_identity)), device_(device), base_(base), bytes_(bytes),
        registration_(std::move(registration)) {}

  ~RoceHbmRegion() override { (void)Close(); }

  Status Pin(const void* address, std::size_t bytes,
             std::shared_ptr<void>* lease) override {
    if (address == nullptr || bytes == 0 || lease == nullptr) {
      return Status::InvalidArgument("invalid HBM region pin request");
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(address);
    if (begin < base_ || bytes > bytes_ || begin - base_ > bytes_ - bytes) {
      return Status::InvalidArgument("buffer range is outside the registered HBM region");
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closing_ || closed_) return Status::InvalidState("HBM region is closing or closed");
      ++active_;
    }
    try {
      *lease = std::make_shared<PinToken>(shared_from_this());
    } catch (const std::bad_alloc&) {
      Unpin();
      return Status::OutOfMemory("cannot pin registered HBM region");
    }
    return Status::OK();
  }

  Status Close() override {
    std::shared_ptr<HbmRegistration> registration;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (closed_) return Status::OK();
      closing_ = true;
      changed_.wait(lock, [this] { return active_ == 0; });
      closed_ = true;
      registration.swap(registration_);
    }
    registration.reset();
    return Status::OK();
  }

  bool closed() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_ || closing_;
  }
  std::uint32_t device_id() const override { return device_; }
  std::uintptr_t base() const override { return base_; }
  std::size_t bytes() const override { return bytes_; }
  const void* transport_identity() const override { return session_identity_.get(); }

  std::shared_ptr<HbmRegistration> registration() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return registration_;
  }

 private:
  struct PinToken {
    explicit PinToken(std::shared_ptr<RoceHbmRegion> region) : region_(std::move(region)) {}
    ~PinToken() { region_->Unpin(); }
    std::shared_ptr<RoceHbmRegion> region_;
  };

  void Unpin() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (--active_ == 0) changed_.notify_all();
  }

  std::shared_ptr<const void> session_identity_;
  std::uint32_t device_{};
  std::uintptr_t base_{};
  std::size_t bytes_{};
  mutable std::mutex mutex_;
  std::condition_variable changed_;
  std::size_t active_{};
  bool closing_{false};
  bool closed_{false};
  std::shared_ptr<HbmRegistration> registration_;
};

struct FreeNpuProbe {
  void operator()(void* buffer) const { (void)aclrtFree(buffer); }
};

struct Transfer {
  std::uint64_t local;
  std::uint64_t remote;
  std::uint32_t bytes;
};

Status CheckStream(void* stream, std::uint32_t device) {
  if (stream == nullptr) return Status::InvalidArgument("RoCE requires a caller ACL stream");
  std::int32_t current = -1;
  if (aclrtGetDevice(&current) != ACL_SUCCESS || current != static_cast<int>(device)) {
    return Status::InvalidState("RoCE requires the Runtime NPU to be current");
  }
  aclmdlRICaptureStatus capture{};
  aclmdlRI graph{};
  if (aclmdlRICaptureGetInfo(static_cast<aclrtStream>(stream), &capture, &graph) != ACL_SUCCESS) {
    return Status::Internal("cannot query ACL stream capture state");
  }
  if (capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE) {
    return Status::NotImplemented("RoCE Host transport requires eager execution outside ACL Graph");
  }
  return Status::OK();
}

Status Synchronize(aclrtStream stream) {
  if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
    return Status::Internal("RoCE stream synchronization failed");
  }
  return Status::OK();
}

bool Contains(std::uint64_t base, std::uint64_t bytes,
              std::uint64_t address, std::uint64_t length) {
  return address >= base && length <= bytes && address - base <= bytes - length;
}

}  // namespace

class RoceSession {
 public:
  RoceSession(std::string address, RoceOptions options, std::uint32_t device_id)
      : address_(std::move(address)), options_(std::move(options)), device_id_(device_id) {}

  ~RoceSession() {
    if (fd_ >= 0 && !Stop().ok()) UnsafeTeardown("cannot stop RoCE session");
    (void)Release();
  }

  Status Allocate(const RuntimeInitRequest& request, RemoteRegionDesc* region) {
    if (connection_ != nullptr || fd_ >= 0) {
      return Status::InvalidState("RoCE session already has resources");
    }
    connection_ = std::make_unique<roce::Connection>();
    Status status = connection_->Open(options_);
    if (!status.ok()) {
      connection_.reset();
      return status;
    }
    fd_ = roce::ConnectControl(address_);
    if (fd_ < 0) {
      status = Status::Internal("RoCE TCP control connection failed to " + address_);
    } else if (!roce::SendAllocation(fd_, request.ddr_bytes, request.slot_size)) {
      status = Status::Internal("RoCE v7 Allocate request could not be sent to " + address_);
    } else if (!roce::ReceiveRegion(fd_, &region_)) {
      status = Status::Internal("RoCE v7 Allocate: no valid DDR region response from " + address_ +
                                " (requested " + std::to_string(request.ddr_bytes) + " bytes)");
    } else if (region_.base == 0 || region_.bytes != request.ddr_bytes ||
               region_.bytes > std::numeric_limits<std::uint64_t>::max() - region_.base) {
      status = Status::Internal("RoCE peer returned an invalid DDR region from " + address_ +
                                " (requested " + std::to_string(request.ddr_bytes) +
                                " bytes, received " + std::to_string(region_.bytes) + " bytes)");
    }
    if (!status.ok()) {
      (void)Release();
      return status;
    }
    {
      std::lock_guard<std::mutex> lock(hbm_regions_mutex_);
      hbm_regions_closing_ = false;
    }
    *region = {region_.base, region_.bytes, request.slot_size, DeviceKind::kCpu};
    return Status::OK();
  }

  Status Connect() {
    roce::Endpoint remote;
    if (!roce::SendEndpoint(fd_, false, connection_->local()) ||
        !roce::ReceiveEndpoint(fd_, true, &remote)) {
      return Status::Internal("RoCE peer QP exchange failed");
    }
    auto status = connection_->Connect(remote);
    connected_ = status.ok();
    queue_ready_.store(connected_, std::memory_order_release);
    return status;
  }

  Status Stop() {
    if (connection_ == nullptr) return Status::OK();
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_ready_.store(false, std::memory_order_release);
    return connection_->Stop();
  }

  Status Release() {
    auto status = Stop();
    if (!status.ok()) return status;
    if (fd_ >= 0) {
      if (connected_ && (!roce::SendRelease(fd_, false) || !roce::ReceiveRelease(fd_, true))) {
        status = Status::Internal("RoCE peer did not acknowledge region release");
      }
      close(fd_);
      fd_ = -1;
    }
    connected_ = false;
    cache_.clear();
    cached_bytes_ = 0;
    connection_.reset();
    region_ = {};
    return status;
  }

  std::uint32_t device_id() const { return device_id_; }
  const roce::Region& region() const { return region_; }

  Status Register(const void* buffer, std::size_t bytes, bool read,
                  const std::shared_ptr<void>& owner,
                  std::shared_ptr<HbmRegistration>* result, bool timing = false,
                  ConstBufferView storage = {}, const char** mr_mode = nullptr) {
    HbmRange range;
    auto status = FindHbmRange(device_id_, buffer, bytes, &range);
    if (!status.ok()) return status;
    if (!Usable()) return Status::InvalidState("RoCE session is unavailable");
    if (bytes > connection_->max_mr_size()) {
      return Status::InvalidArgument("NPU buffer exceeds RDMA device MR limit");
    }

    HbmRange mr_range = range;
    bool storage_covered = true;
    if (owner && !storage.empty()) {
      const auto base = reinterpret_cast<std::uintptr_t>(storage.data);
      const auto address = reinterpret_cast<std::uintptr_t>(buffer);
      storage_covered = storage.valid() &&
          Contains(reinterpret_cast<std::uintptr_t>(range.base), range.bytes, base, storage.bytes) &&
          Contains(base, storage.bytes, address, bytes);
      if (storage_covered) {
        mr_range = {const_cast<void*>(storage.data), storage.bytes};
      }
    }
    bool cacheable = owner && storage_covered && mr_range.bytes <= std::min<std::uint64_t>(
        kMaxCachedMrBytes, connection_->max_mr_size());
    if (cacheable) {
      auto cached = std::find_if(cache_.begin(), cache_.end(), [&](const auto& entry) {
        return entry->Matches(mr_range, owner);
      });
      if (cached != cache_.end()) {
        *result = *cached;
        cache_.splice(cache_.end(), cache_, cached);
        if (mr_mode) *mr_mode = "cache_hit";
        if (timing) std::fprintf(stderr, "==============[KVAF_ROCE_MR]=========== device=%u phase=cache_hit bytes=%zu\n",
                                 device_id_, bytes);
        return Status::OK();
      }
      while (cache_.size() >= kMaxCachedMrs || cached_bytes_ > kMaxCachedMrBytes - mr_range.bytes) {
        const auto idle = std::find_if(cache_.begin(), cache_.end(), [](const auto& entry) {
          return entry.use_count() == 1;
        });
        if (idle == cache_.end()) {
          cacheable = false;
          break;
        }
        cached_bytes_ -= (*idle)->mr->length;
        cache_.erase(idle);
      }
    }

    if (mr_mode) *mr_mode = cacheable ? "cache_register" : "temporary_register";
    auto mr = std::make_shared<HbmRegistration>(timing);
    status = mr->Register(connection_->pd(), device_id_, range, cacheable ? mr_range.base : buffer,
                          cacheable ? mr_range.bytes : bytes,
                          (cacheable || read) ? IBV_ACCESS_LOCAL_WRITE : 0,
                          cacheable ? owner : std::shared_ptr<void>{});
    if (!status.ok()) return status;
    if (cacheable) {
      cache_.push_back(mr);
      cached_bytes_ += mr_range.bytes;
    }
    *result = std::move(mr);
    return Status::OK();
  }

  Status RegisterHbm(const ConstBufferView& buffer,
                     const std::shared_ptr<void>& owner,
                     ConstBufferView storage,
                     std::shared_ptr<HbmRegion>* result) {
    if (result == nullptr || buffer.data == nullptr || buffer.bytes == 0 ||
        buffer.device != DeviceKind::kNpu || !owner || !storage.valid() ||
        storage.device != DeviceKind::kNpu) {
      return Status::InvalidArgument("explicit HBM registration requires an owned NPU storage range");
    }
    HbmRange allocation;
    auto status = FindHbmRange(device_id_, buffer.data, buffer.bytes, &allocation);
    if (!status.ok()) return status;
    const auto storage_base = reinterpret_cast<std::uintptr_t>(storage.data);
    const auto buffer_base = reinterpret_cast<std::uintptr_t>(buffer.data);
    if (!Contains(reinterpret_cast<std::uintptr_t>(allocation.base), allocation.bytes,
                  storage_base, storage.bytes) ||
        !Contains(storage_base, storage.bytes, buffer_base, buffer.bytes)) {
      return Status::InvalidArgument("Tensor storage is outside its HBM allocation or does not cover the Tensor");
    }
    if (!Usable()) return Status::InvalidState("RoCE session is unavailable");
    if (storage.bytes > connection_->max_mr_size()) {
      return Status::InvalidArgument("Tensor storage exceeds RDMA device MR limit");
    }

    auto registration = std::make_shared<HbmRegistration>(MrTimingEnabled());
    status = registration->Register(connection_->pd(), device_id_, allocation,
                                    storage.data, storage.bytes,
                                    IBV_ACCESS_LOCAL_WRITE, owner);
    if (!status.ok()) return status;
    auto region = std::make_shared<RoceHbmRegion>(session_identity_, device_id_, storage_base,
                                                 storage.bytes, std::move(registration));
    {
      std::lock_guard<std::mutex> lock(hbm_regions_mutex_);
      if (hbm_regions_closing_) {
        return Status::InvalidState("RoCE session is closing HBM registrations");
      }
      hbm_regions_.erase(std::remove_if(hbm_regions_.begin(), hbm_regions_.end(),
          [](const auto& entry) { return entry.expired(); }), hbm_regions_.end());
      hbm_regions_.push_back(region);
    }
    *result = std::move(region);
    return Status::OK();
  }

  bool Owns(const std::shared_ptr<HbmRegion>& region) const {
    return region && region->transport_identity() == session_identity_.get() &&
           region->device_id() == device_id_;
  }

  Status CloseHbmRegions() {
    std::vector<std::shared_ptr<HbmRegion>> regions;
    {
      std::lock_guard<std::mutex> lock(hbm_regions_mutex_);
      hbm_regions_closing_ = true;
      for (const auto& weak : hbm_regions_) {
        if (auto region = weak.lock()) regions.push_back(std::move(region));
      }
      hbm_regions_.clear();
    }
    for (const auto& region : regions) {
      const auto status = region->Close();
      if (!status.ok()) return status;
    }
    return Status::OK();
  }

  Status Execute(const void* buffer, std::size_t buffer_bytes,
                 const std::vector<Transfer>& transfers, bool read,
                 const std::shared_ptr<void>& owner = {}, ConstBufferView storage = {},
                 RoceCallLatency* latency = nullptr) {
    if (transfers.empty()) return Status::OK();
    std::shared_ptr<HbmRegistration> registration;
    ScopedLatency mr_time(latency ? &latency->mr_us : nullptr);
    auto status = Register(buffer, buffer_bytes, read, owner, &registration, false, storage,
                           latency ? &latency->mr_mode : nullptr);
    mr_time.Finish();
    if (!status.ok()) return status;
    return ExecuteRegistered(*registration, transfers, read, latency ? &latency->rdma : nullptr);
  }

  Status ExecuteRegistered(const HbmRegistration& registration,
                           const std::vector<Transfer>& transfers, bool read,
                           RdmaLatency* latency = nullptr) {
    if (transfers.empty()) return Status::OK();
    ScopedLatency queue_wait(latency ? &latency->queue_lock_us : nullptr);
    std::lock_guard<std::mutex> lock(queue_mutex_);
    queue_wait.Finish();
    if (!Usable()) return Status::InvalidState("RoCE session is unavailable");

    const std::size_t window = read ? read_window() : connection_->depth();
    std::vector<ibv_send_wr> requests(window);
    std::vector<ibv_sge> entries(window);
    std::vector<bool> seen(window);
    for (std::size_t begin = 0; begin < transfers.size();) {
      const std::size_t count = std::min(window, transfers.size() - begin);
      std::fill(seen.begin(), seen.end(), false);
      for (std::size_t i = 0; i < count; ++i) {
        const auto& transfer = transfers[begin + i];
        entries[i] = {transfer.local, transfer.bytes, registration.mr->lkey};
        auto& request = requests[i];
        request = {};
        request.wr_id = begin + i;
        request.opcode = read ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
        request.sg_list = &entries[i];
        request.num_sge = 1;
        request.send_flags = IBV_SEND_SIGNALED;
        request.next = i + 1 == count ? nullptr : &requests[i + 1];
        request.wr.rdma.remote_addr = transfer.remote;
        request.wr.rdma.rkey = region_.rkey;
      }
      ibv_send_wr* bad = nullptr;
      ScopedLatency post(latency ? &latency->post_send_us : nullptr);
      const int result = ibv_post_send(connection_->qp(), requests.data(), &bad);
      post.Finish();
      if (result != 0) {
        // A prefix may already be in flight. Stop the QP before any MR,
        // buffer or slot can be reused, even when bad_wr is unavailable.
        StopFailedQueue();
        return Status::Internal("ibv_post_send failed: " + std::to_string(result) +
                                ", completed before window=" + std::to_string(begin) +
                                ", window size=" + std::to_string(count));
      }
      if (latency) {
        latency->posted_wrs += count;
        for (std::size_t i = 0; i < count; ++i) latency->posted_bytes += entries[i].length;
      }
      ScopedLatency wait(latency ? &latency->cq_wait_us : nullptr);
      const auto deadline = std::chrono::steady_clock::now() + kCompletionTimeout;
      std::size_t completed = 0;
      while (completed < count) {
        ibv_wc completions[16];
        const int polled = ibv_poll_cq(connection_->cq(), 16, completions);
        if (polled < 0 || std::chrono::steady_clock::now() >= deadline) {
          wait.Finish();
          StopFailedQueue();
          return Status::Internal(std::string(polled < 0 ? "ibv_poll_cq failed" :
                                             "RoCE completion timed out") +
                                  ", completed=" + std::to_string(begin + completed) +
                                  ", submitted=" + std::to_string(begin + count));
        }
        for (int i = 0; i < polled; ++i) {
          const auto& wc = completions[i];
          if (wc.status != IBV_WC_SUCCESS ||
              wc.opcode != (read ? IBV_WC_RDMA_READ : IBV_WC_RDMA_WRITE) || wc.wr_id < begin ||
              wc.wr_id - begin >= count || seen[wc.wr_id - begin]) {
            wait.Finish();
            StopFailedQueue();
            return Status::Internal("RoCE completion failed: wr=" + std::to_string(wc.wr_id) +
                                    " status=" + std::to_string(wc.status) +
                                    " vendor=" + std::to_string(wc.vendor_err));
          }
          seen[wc.wr_id - begin] = true;
          ++completed;
          if (latency) ++latency->completed_wrs;
        }
        if (polled == 0) std::this_thread::yield();
      }
      wait.Finish();
      begin += count;
    }
    return Status::OK();
  }

  std::size_t read_window() const {
    return std::min(connection_->depth(), connection_->read_depth());
  }

 private:
  bool Usable() const { return queue_ready_.load(std::memory_order_acquire); }

  void StopFailedQueue() {
    queue_ready_.store(false, std::memory_order_release);
    if (!connection_->Stop().ok()) UnsafeTeardown("cannot stop QP with outstanding DMA");
  }

  std::string address_;
  RoceOptions options_;
  std::uint32_t device_id_;
  // Shared with every explicit region so an old handle keeps its generation
  // identity alive and its address cannot be reused by a later session.
  const std::shared_ptr<const void> session_identity_{std::make_shared<std::uint8_t>(0)};
  int fd_{-1};
  bool connected_{false};
  std::atomic<bool> queue_ready_{false};
  roce::Region region_{};
  std::unique_ptr<roce::Connection> connection_;
  std::mutex queue_mutex_;
  std::list<std::shared_ptr<HbmRegistration>> cache_;
  std::size_t cached_bytes_{};
  mutable std::mutex hbm_regions_mutex_;
  bool hbm_regions_closing_{false};
  std::vector<std::weak_ptr<HbmRegion>> hbm_regions_;
};

namespace {

class RoceAllocator final : public DdrMemoryAllocator {
 public:
  explicit RoceAllocator(std::shared_ptr<RoceSession> session) : session_(std::move(session)) {}

  Status Allocate(const RuntimeInitRequest& request, RemoteRegionDesc* region) override {
    return session_->Allocate(request, region);
  }

  Status Release(const RemoteRegionDesc&) override { return session_->Release(); }

 private:
  std::shared_ptr<RoceSession> session_;
};

class RoceOperator final : public AccessOperator {
 public:
  explicit RoceOperator(std::shared_ptr<RoceSession> session) : session_(std::move(session)) {}

  ~RoceOperator() override { (void)Shutdown(); }

  Status Initialize(const RemoteRegionDesc& region) override {
    auto status = session_->Connect();
    if (!status.ok()) return status;

    // Probe using an NPU allocation before publishing any application slots.
    const auto bytes = static_cast<std::uint32_t>(std::min<std::uint64_t>(region.length, 64));
    std::array<std::uint8_t, 64> expected{};
    std::array<std::uint8_t, 64> result{};
    std::fill_n(expected.begin(), bytes, 0xa5);
    std::vector<Transfer> probe{{0, region.base, bytes}};
    void* buffer = nullptr;
    if (aclrtMalloc(&buffer, bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) {
      return Status::Internal("cannot allocate RoCE probe buffer");
    }
    const std::unique_ptr<void, FreeNpuProbe> owned_buffer(buffer);
    const auto address = reinterpret_cast<std::uintptr_t>(buffer);
    probe.front().local = address;
    if (aclrtMemcpy(buffer, bytes, expected.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
      status = Status::Internal("cannot upload RoCE probe bytes");
    } else {
      status = session_->Execute(buffer, bytes, probe, false);
    }
    // Clear the local buffer so a successful comparison requires a real READ.
    if (status.ok() &&
        aclrtMemcpy(buffer, bytes, result.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
      status = Status::Internal("cannot clear RoCE probe buffer");
    }
    if (status.ok()) status = session_->Execute(buffer, bytes, probe, true);
    if (status.ok() &&
        (aclrtMemcpy(result.data(), bytes, buffer, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
         result != expected)) {
      status = Status::Internal("RoCE WRITE/READ probe payload mismatch");
    }
    if (!status.ok()) return status;
    try {
      worker_ = std::thread([this] { RunWorker(); });
    } catch (const std::system_error& error) {
      return Status::Internal("cannot start RoCE worker: " + std::string(error.what()));
    }
    return Status::OK();
  }

  Status Shutdown() override {
    for (const auto& job : pending_) WaitForGet(*job);
    pending_range_ = nullptr;
    range_stream_ = nullptr;
    range_status_ = Status::OK();
    pending_.clear();
    staging_bytes_ = 0;
    StopWorker();
    auto status = session_->CloseHbmRegions();
    if (!status.ok()) return status;
    return session_->Stop();
  }

  Status RegisterHbm(const ConstBufferView& buffer,
                     const std::shared_ptr<void>& owner,
                     ConstBufferView storage,
                     std::shared_ptr<HbmRegion>* region) override {
    return session_->RegisterHbm(buffer, owner, storage, region);
  }

  Status UnregisterHbm(std::shared_ptr<HbmRegion>* region) override {
    if (region == nullptr || !*region) {
      return Status::InvalidArgument("HBM region handle must not be null");
    }
    if (!session_->Owns(*region)) {
      return Status::InvalidArgument("HBM region belongs to a different RoCE session or NPU");
    }
    auto status = (*region)->Close();
    if (status.ok()) region->reset();
    return status;
  }

  Status WaitForAccessTable(AccessTableHandle handle) override {
    for (const auto& job : pending_) {
      if (job->handle.opaque == handle.opaque) WaitForGet(*job);
    }
    ReapCompleted();
    return Status::OK();
  }

  Status ExecuteBatchPut(const BatchPutExecRequest& exec) override try {
    RoceCallLatency latency(latency_enabled_, "PUT", session_->device_id());
    latency.requests_or_capacity = exec.request->keys.size();
    ScopedLatency wait_prev(latency.id ? &latency.wait_prev_get_us : nullptr);
    auto status = FinishPendingRange();
    wait_prev.Finish();
    if (!status.ok()) return status;
    ReapCompleted();
    const auto& request = *exec.request;
    const auto base = reinterpret_cast<std::uintptr_t>(request.input.data);
    const auto& region = session_->region();
    if (request.input.device != DeviceKind::kNpu ||
        request.input.bytes > std::numeric_limits<std::uint64_t>::max() - base) {
      return Status::InvalidArgument("RoCE Put requires a valid NPU source");
    }

    std::vector<Transfer> transfers;
    transfers.reserve(request.keys.size());
    for (std::size_t i = 0; i < request.keys.size(); ++i) {
      const auto bytes = request.lengths[i];
      const auto target = exec.put_desc->entries[i].address;
      if (bytes > std::numeric_limits<std::uint32_t>::max() ||
          !Contains(region.base, region.bytes, target, bytes)) {
        return Status::InvalidArgument("RoCE Put request exceeds region or WR size");
      }
      transfers.push_back({base + request.input_offsets[i], target,
                           static_cast<std::uint32_t>(bytes)});
    }
    status = CheckStream(exec.execution.stream, session_->device_id());
    if (!status.ok()) return status;
    ScopedLatency stream_sync(latency.id ? &latency.stream_sync_us : nullptr);
    status = Synchronize(static_cast<aclrtStream>(exec.execution.stream));
    stream_sync.Finish();
    if (!status.ok()) return status;
    if (exec.execution.hbm_region) {
      latency.mr_mode = "explicit";
      ScopedLatency mr_time(latency.id ? &latency.mr_us : nullptr);
      if (!session_->Owns(exec.execution.hbm_region)) {
        return Status::InvalidArgument("BatchPut HBM region belongs to a different RoCE session or NPU");
      }
      std::shared_ptr<void> lease;
      status = exec.execution.hbm_region->Pin(request.input.data, request.input.bytes, &lease);
      if (!status.ok()) return status;
      auto region = std::dynamic_pointer_cast<RoceHbmRegion>(exec.execution.hbm_region);
      if (!region) return Status::InvalidArgument("BatchPut HBM region has an unsupported implementation");
      auto registration = region->registration();
      if (!registration) return Status::InvalidState("BatchPut HBM region is closed");
      mr_time.Finish();
      status = session_->ExecuteRegistered(*registration, transfers, false,
                                           latency.id ? &latency.rdma : nullptr);
    } else {
      status = session_->Execute(request.input.data, request.input.bytes, transfers, false,
                                 exec.execution.hbm_owner, exec.execution.hbm_storage,
                                 latency.id ? &latency : nullptr);
    }
    latency.ok = status.ok();
    return status;
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("cannot stage RoCE BatchPut");
  }

  Status ExecuteBatchGet(const BatchGetExecRequest& exec) override try {
    const auto& request = *exec.request;
    const auto token = request.output_block_bytes;
    if (token == 0 || token > std::numeric_limits<std::uint32_t>::max()) {
      return Status::InvalidArgument("RoCE Get WR size exceeded");
    }
    auto job = std::make_unique<GetJob>();
    job->handle = request.access_table;
    job->capacity = request.indices.bytes / sizeof(std::int32_t);
    job->table_size = request.device_access_table.valid.bytes;
    job->rows = request.output.bytes / token;
    job->token = static_cast<std::uint32_t>(token);
    job->block_bytes = token * request.tokens_per_block;
    const auto& table = request.device_access_table;
    return EnqueueGet(std::move(job), request.output, exec.execution,
                      {table.addresses, request.miss_count, request.indices, request.offsets,
                       request.output_indices, table.valid});
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("cannot stage RoCE BatchGet");
  }

  Status ExecuteBatchGetRanges(const BatchGetRangesExecRequest& exec) override try {
    const auto& request = *exec.request;
    auto job = std::make_unique<GetJob>();
    job->handle = request.access_table;
    job->ranges = true;
    job->capacity = request.indices.bytes / sizeof(std::int32_t);
    job->table_size = exec.host_access_table.size;
    job->chunk_bytes = request.read_chunk_bytes;
    const auto& table = request.device_access_table;
    return EnqueueGet(std::move(job), request.output, exec.execution,
                      {table.addresses, request.request_count, request.indices, request.source_offsets,
                       request.output_offsets, table.valid, request.lengths, table.lengths});
  } catch (const std::bad_alloc&) {
    return Status::OutOfMemory("cannot stage RoCE range Get");
  }

  Status CheckBatchGetRangesStatus(void* stream) override {
    if (!range_stream_) return Status::InvalidState("no RoCE range Get has been submitted");
    if (stream != range_stream_)
      return Status::InvalidArgument("range Get status requires its execution stream");
    (void)FinishPendingRange();
    return range_status_;
  }

 private:
  struct GetJob {
    ~GetJob() {
      if (staging != nullptr && aclrtFreeHost(staging) != ACL_SUCCESS)
        UnsafeTeardown("cannot release RoCE Get staging");
    }

    RoceOperator* owner{};
    aclrtStream stream{};
    AccessTableHandle handle;
    bool synchronous{};
    bool ranges{};
    std::size_t capacity{};
    std::size_t table_size{};
    std::uintptr_t output{};
    std::size_t output_bytes{};
    std::size_t rows{};
    std::uint32_t token{};
    std::uint32_t chunk_bytes{};
    std::uint64_t block_bytes{};
    std::size_t staging_bytes{};
    void* staging{};
    std::array<void*, 8> data{};
    std::shared_ptr<void> hbm_owner;
    std::shared_ptr<void> hbm_region_lease;
    std::shared_ptr<HbmRegistration> registration;
    std::uint64_t latency_id{};
    MrClock::time_point enqueue_start;
    MrClock::time_point callback_start;
    RdmaLatency rdma;
    double parse_us{};
    std::int32_t active_requests{};
    Status result;
    GetJob* next{nullptr};
    bool worker_done{false};
    std::atomic<bool> done{false};
  };

  Status EnqueueGet(std::unique_ptr<GetJob> job, MutableBufferView output,
                    const BatchGetExecutionOptions& execution,
                    const std::array<ConstBufferView, 8>& inputs) {
    RoceCallLatency latency(latency_enabled_, job->ranges ? "GET_RANGES" : "GET",
                            session_->device_id());
    latency.requests_or_capacity = job->capacity;
    job->latency_id = latency.id;
    if (output.device != DeviceKind::kNpu) {
      return Status::InvalidArgument("RoCE Get requires NPU tensors");
    }
    auto status = CheckStream(execution.stream, session_->device_id());
    if (!status.ok()) return status;
    ScopedLatency wait_prev(latency.id ? &latency.wait_prev_get_us : nullptr);
    status = FinishPendingRange();
    wait_prev.Finish();
    if (!status.ok()) return status;
    ReapCompleted();
    if (pending_.size() == kMaxPendingGets) {
      WaitForGet(*pending_.front());
      ReapCompleted();
    }

    struct MetadataCopy {
      const void* source{};
      std::size_t bytes{};
      std::size_t prefix{};
      std::size_t stride{};
    };
    std::array<MetadataCopy, 8> copies{};
    std::size_t staging_bytes = kAclCopyAlignment - 1;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      if (input.bytes == 0) continue;
      HbmRange range;
      status = FindHbmRange(session_->device_id(), input.data, input.bytes, &range);
      if (!status.ok()) return status;
      const auto address = reinterpret_cast<std::uintptr_t>(input.data);
      const auto prefix = address % kAclCopyAlignment;
      if (address - prefix < reinterpret_cast<std::uintptr_t>(range.base)) {
        return Status::InvalidArgument("RoCE metadata has no aligned copy range");
      }
      if (input.bytes > kMaxStagingBytes) {
        return Status::OutOfMemory("RoCE Get metadata exceeds the host staging budget");
      }
      const auto bytes = input.bytes + prefix;
      const auto stride = (bytes + kAclCopyAlignment - 1) / kAclCopyAlignment * kAclCopyAlignment;
      if (staging_bytes > kMaxStagingBytes - staging_bytes_ ||
          stride > kMaxStagingBytes - staging_bytes_ - staging_bytes) {
        return Status::OutOfMemory("RoCE Get host staging budget exhausted");
      }
      copies[i] = {reinterpret_cast<const void*>(address - prefix), bytes, prefix, stride};
      staging_bytes += stride;
    }
    job->output = reinterpret_cast<std::uintptr_t>(output.data);
    if (job->capacity > kMaxGetRequests ||
        output.bytes > std::numeric_limits<std::uint64_t>::max() - job->output) {
      return Status::InvalidArgument("RoCE Get capacity or output range exceeded");
    }

    job->owner = this;
    job->stream = static_cast<aclrtStream>(execution.stream);
    job->synchronous = execution.synchronize;
    job->output_bytes = output.bytes;
    job->staging_bytes = staging_bytes;
    // Keep the owner through completion even when cache pressure requires a
    // temporary MR. Reaping on the caller thread also keeps Python destruction
    // off the worker while the submitting thread may be waiting with the GIL.
    job->hbm_owner = execution.hbm_owner;
    ScopedLatency mr_time(latency.id ? &latency.mr_us : nullptr);
    if (execution.hbm_region) {
      latency.mr_mode = "explicit";
      if (!session_->Owns(execution.hbm_region)) {
        return Status::InvalidArgument("BatchGet HBM region belongs to a different RoCE session or NPU");
      }
      status = execution.hbm_region->Pin(output.data, output.bytes, &job->hbm_region_lease);
      if (status.ok()) {
        auto region = std::dynamic_pointer_cast<RoceHbmRegion>(execution.hbm_region);
        if (!region) return Status::InvalidArgument("BatchGet HBM region has an unsupported implementation");
        job->registration = region->registration();
        if (!job->registration) return Status::InvalidState("BatchGet HBM region is closed");
      }
    } else {
      status = session_->Register(output.data, output.bytes, true,
                                  job->hbm_owner, &job->registration, mr_timing_,
                                  execution.hbm_storage, latency.id ? &latency.mr_mode : nullptr);
    }
    mr_time.Finish();
    if (!status.ok()) return status;
    ScopedLatency staging_time(latency.id ? &latency.staging_us : nullptr);
    if (aclrtMallocHost(&job->staging, staging_bytes) != ACL_SUCCESS)
      return Status::OutOfMemory("cannot allocate pinned RoCE Get staging");
    std::memset(job->staging, 0, staging_bytes);
    staging_time.Finish();

    pending_.push_back(std::move(job));
    staging_bytes_ += staging_bytes;
    auto& queued = *pending_.back();
    const auto staging = reinterpret_cast<std::uintptr_t>(queued.staging);
    auto* to = reinterpret_cast<std::uint8_t*>(
        (staging + kAclCopyAlignment - 1) / kAclCopyAlignment * kAclCopyAlignment);
    queued.enqueue_start = MrTimingStart(latency.id != 0);
    ScopedLatency enqueue_time(latency.id ? &latency.enqueue_us : nullptr);
    for (std::size_t i = 0; i < copies.size(); ++i) {
      if (i == 1) std::memset(to, 0xff, copies[i].stride);  // Invalid count until copied.
      const auto& copy = copies[i];
      queued.data[i] = to + copy.prefix;
      if (copy.bytes != 0 &&
          aclrtMemcpyAsync(to, copy.stride, copy.source, copy.bytes,
                           ACL_MEMCPY_DEVICE_TO_HOST, queued.stream) != ACL_SUCCESS) {
        return CancelEnqueue(queued, "cannot enqueue RoCE Get descriptor copy");
      }
      to += copy.stride;
    }
    if (aclrtLaunchHostFunc(queued.stream, RunGetOnStream, &queued) != ACL_SUCCESS) {
      return CancelEnqueue(queued, "cannot enqueue RoCE Get host task");
    }
    enqueue_time.Finish();
    if (queued.ranges) {
      pending_range_ = &queued;
      range_stream_ = queued.stream;
      if (queued.synchronous) {
        ScopedLatency wait_current(latency.id ? &latency.wait_current_get_us : nullptr);
        status = FinishPendingRange();
      }
      latency.ok = status.ok();
      return status;
    }
    if (!queued.synchronous) {
      latency.ok = true;
      return Status::OK();
    }

    ScopedLatency wait_current(latency.id ? &latency.wait_current_get_us : nullptr);
    status = Synchronize(queued.stream);
    wait_current.Finish();
    if (!queued.done.load(std::memory_order_acquire)) {
      UnsafeTeardown("RoCE Get host task did not finish after stream synchronization");
    }
    if (status.ok()) status = queued.result;
    ReapCompleted();
    latency.ok = status.ok();
    return status;
  }

  static void RunGetOnStream(void* data) {
    auto& job = *static_cast<GetJob*>(data);
    job.callback_start = MrTimingStart(job.latency_id != 0);
    auto& op = *job.owner;
    {
      // CANN host functions cannot call ACL. The worker owns parsing and verbs.
      std::unique_lock<std::mutex> lock(op.worker_mutex_);
      *op.work_tail_ = &job;
      op.work_tail_ = &job.next;
      op.worker_changed_.notify_all();
      op.worker_changed_.wait(lock, [&job] { return job.worker_done; });
    }
    if (!job.result.ok() && !job.synchronous && !job.ranges) {
      std::fprintf(stderr, "==============[KVAF_ROCE]=========== asynchronous BatchGet failed: %s\n",
                   job.result.message().c_str());
      // This callback has no error return to the stream. Letting it return
      // would permit downstream consumption of a partial or invalid result.
      std::abort();
    }
    job.done.store(true, std::memory_order_release);
  }

  void RunWorker() {
    std::unique_lock<std::mutex> lock(worker_mutex_);
    while (true) {
      worker_changed_.wait(lock, [this] { return work_head_ != nullptr || stop_worker_; });
      if (stop_worker_ && work_head_ == nullptr) return;
      auto& job = *work_head_;
      work_head_ = job.next;
      if (work_head_ == nullptr) work_tail_ = &work_head_;
      lock.unlock();
      const bool timing = job.latency_id != 0;
      char timing_line[1024];
      try {
        const auto worker_start = MrTimingStart(timing);
        job.result = job.ranges ? ExecuteRangeGet(job) : ExecuteGet(job);
        // A borrowed output may be freed as soon as the caller stream completes.
        // Release temporary MR/HAL references before the host task unblocks it.
        const auto release_start = MrTimingStart(timing);
        job.registration.reset();
        job.hbm_region_lease.reset();
        if (timing) {
          const auto finished = MrClock::now();
          std::snprintf(timing_line, sizeof(timing_line),
                       "==============[KVAF_ROCE_LATENCY]=========== pid=%ld id=%llu device=%u op=%s phase=worker ok=%d "
                       "active_requests=%d enqueue_to_callback_us=%.3f worker_queue_us=%.3f "
                       "worker_us=%.3f parse_us=%.3f mr_release_us=%.3f queue_lock_us=%.3f "
                       "post_send_us=%.3f cq_wait_us=%.3f posted_bytes=%llu posted_wrs=%llu "
                       "completed_wrs=%llu\n",
                       static_cast<long>(getpid()),
                       static_cast<unsigned long long>(job.latency_id), session_->device_id(),
                       job.ranges ? "GET_RANGES" : "GET", job.result.ok(), job.active_requests,
                       std::chrono::duration<double, std::micro>(
                           job.callback_start - job.enqueue_start).count(),
                       std::chrono::duration<double, std::micro>(
                           worker_start - job.callback_start).count(),
                       std::chrono::duration<double, std::micro>(finished - worker_start).count(),
                       job.parse_us,
                       std::chrono::duration<double, std::micro>(finished - release_start).count(),
                       job.rdma.queue_lock_us, job.rdma.post_send_us, job.rdma.cq_wait_us,
                       static_cast<unsigned long long>(job.rdma.posted_bytes),
                       static_cast<unsigned long long>(job.rdma.posted_wrs),
                       static_cast<unsigned long long>(job.rdma.completed_wrs));
        }
      } catch (...) {
        // An exception in the posting/completion path may leave DMA outstanding.
        UnsafeTeardown("exception in RoCE Get worker");
      }
      lock.lock();
      job.worker_done = true;
      worker_changed_.notify_all();
      if (timing) {
        // The callback may release the job after notification. Only the
        // local formatted line is accessed after dropping the worker lock.
        lock.unlock();
        std::fputs(timing_line, stderr);
        lock.lock();
      }
    }
  }

  void StopWorker() {
    if (!worker_.joinable()) return;
    {
      std::lock_guard<std::mutex> lock(worker_mutex_);
      stop_worker_ = true;
    }
    worker_changed_.notify_all();
    worker_.join();
    stop_worker_ = false;
  }

  template <typename T>
  static T ReadMetadata(const void* data, std::size_t index) {
    T value;
    std::memcpy(&value, static_cast<const std::uint8_t*>(data) + index * sizeof(T), sizeof(T));
    return value;
  }

  Status ExecuteGet(GetJob& job) {
    ScopedLatency parse(job.latency_id ? &job.parse_us : nullptr);
    const auto count = ReadMetadata<std::int32_t>(job.data[1], 0);
    job.active_requests = count;
    const auto* valid = static_cast<const std::uint8_t*>(job.data[5]);
    if (count < 0 || static_cast<std::size_t>(count) > job.capacity) {
      return Status::InvalidArgument("RoCE Get miss_count exceeds request capacity");
    }
    const auto& region = session_->region();
    std::vector<Transfer> transfers;
    transfers.reserve(count);
    for (std::int32_t i = 0; i < count; ++i) {
      const auto index = ReadMetadata<std::int32_t>(job.data[2], i);
      const auto offset = ReadMetadata<std::int32_t>(job.data[3], i);
      const auto row = ReadMetadata<std::int32_t>(job.data[4], i);
      if (index < 0 || static_cast<std::size_t>(index) >= job.table_size || !valid[index] ||
          offset < 0 || row < 0 || static_cast<std::size_t>(row) >= job.rows ||
          !Contains(0, job.block_bytes, offset, job.token)) {
        return Status::InvalidArgument("invalid RoCE Get descriptor at index " + std::to_string(i));
      }
      const auto address = ReadMetadata<Address>(job.data[0], index);
      if (!Contains(region.base, region.bytes, address, job.block_bytes)) {
        return Status::InvalidArgument("RoCE Get table entry is outside the current DDR region");
      }
      transfers.push_back({job.output + static_cast<std::uint64_t>(row) * job.token,
                           address + static_cast<std::uint64_t>(offset), job.token});
    }
    parse.Finish();
    return session_->ExecuteRegistered(*job.registration, transfers, true,
                                       job.latency_id ? &job.rdma : nullptr);
  }

  Status ExecuteRangeGet(GetJob& job) {
    ScopedLatency parse(job.latency_id ? &job.parse_us : nullptr);
    const auto count = ReadMetadata<std::int32_t>(job.data[1], 0);
    job.active_requests = count;
    if (count < 0 || static_cast<std::size_t>(count) > job.capacity)
      return Status::InvalidArgument("RoCE range Get request_count exceeds capacity");
    if (count == 0) return Status::OK();

    struct Range {
      std::uint64_t local;
      std::uint64_t remote;
      std::uint64_t bytes;
    };
    const auto& region = session_->region();
    const auto* valid = static_cast<const std::uint8_t*>(job.data[5]);
    const auto window = session_->read_window();
    std::vector<Range> ranges;
    std::vector<Transfer> transfers;
    try {
      ranges.reserve(count);
      transfers.reserve(window);
    } catch (const std::bad_alloc&) {
      return Status::OutOfMemory("cannot prepare RoCE range READs");
    }
    // Validate the active prefix before issuing any DMA. Padding rows in the
    // device table are outside the live Host table.
    for (std::int32_t i = 0; i < count; ++i) {
      const auto index = ReadMetadata<std::int32_t>(job.data[2], i);
      const auto source = ReadMetadata<std::int64_t>(job.data[3], i);
      const auto destination = ReadMetadata<std::int64_t>(job.data[4], i);
      const auto bytes = ReadMetadata<std::int64_t>(job.data[6], i);
      if (index < 0 || static_cast<std::size_t>(index) >= job.table_size || !valid[index] ||
          source < 0 || destination < 0 || bytes <= 0 ||
          (!job.chunk_bytes && static_cast<std::uint64_t>(bytes) > UINT32_MAX)) {
        return Status::InvalidArgument("invalid RoCE range Get descriptor at index " +
                                       std::to_string(i));
      }
      const auto address = ReadMetadata<Address>(job.data[0], index);
      const auto package_bytes = ReadMetadata<std::uint64_t>(job.data[7], index);
      if (!Contains(region.base, region.bytes, address, package_bytes) ||
          !Contains(0, package_bytes, source, bytes) ||
          !Contains(0, job.output_bytes, destination, bytes)) {
        return Status::InvalidArgument("RoCE range Get source/output out of bounds at index " +
                                       std::to_string(i));
      }
      ranges.push_back({job.output + static_cast<std::uint64_t>(destination),
                        address + static_cast<std::uint64_t>(source),
                        static_cast<std::uint64_t>(bytes)});
    }

    parse.Finish();
    auto* latency = job.latency_id ? &job.rdma : nullptr;
    const std::uint64_t limit = job.chunk_bytes ? job.chunk_bytes : UINT32_MAX;
    for (const auto& range : ranges) {
      for (std::uint64_t offset = 0; offset < range.bytes;) {
        const auto bytes = std::min(range.bytes - offset, limit);
        transfers.push_back({range.local + offset, range.remote + offset,
                             static_cast<std::uint32_t>(bytes)});
        offset += bytes;
        if (transfers.size() == window) {
          auto status = session_->ExecuteRegistered(*job.registration, transfers, true, latency);
          if (!status.ok()) return status;
          transfers.clear();
        }
      }
    }
    return session_->ExecuteRegistered(*job.registration, transfers, true, latency);
  }

  Status FinishPendingRange() {
    if (!pending_range_) return Status::OK();
    auto& job = *pending_range_;
    range_status_ = Synchronize(job.stream);
    if (!job.done.load(std::memory_order_acquire))
      UnsafeTeardown("cannot drain RoCE range Get host task");
    if (range_status_.ok()) range_status_ = std::move(job.result);
    pending_range_ = nullptr;
    ReapCompleted();
    return range_status_;
  }

  static void WaitForGet(const GetJob& job) {
    if (job.done.load(std::memory_order_acquire)) return;
    (void)Synchronize(job.stream);
    if (!job.done.load(std::memory_order_acquire))
      UnsafeTeardown("cannot drain RoCE Get host task");
  }

  void ReapCompleted() {
    pending_.remove_if([this](const auto& job) {
      if (job.get() == pending_range_ || !job->done.load(std::memory_order_acquire)) return false;
      staging_bytes_ -= job->staging_bytes;
      return true;
    });
  }

  Status CancelEnqueue(const GetJob& job, const char* message) {
    if (!Synchronize(job.stream).ok()) UnsafeTeardown("cannot drain partial RoCE Get enqueue");
    staging_bytes_ -= job.staging_bytes;
    pending_.pop_back();
    return Status::Internal(message);
  }

  std::shared_ptr<RoceSession> session_;
  const bool mr_timing_{MrTimingEnabled()};
  const bool latency_enabled_{RoceLatencyEnabled()};
  std::list<std::unique_ptr<GetJob>> pending_;
  GetJob* pending_range_{nullptr};
  aclrtStream range_stream_{};
  Status range_status_;
  std::size_t staging_bytes_{};
  std::mutex worker_mutex_;
  std::condition_variable worker_changed_;
  GetJob* work_head_{nullptr};
  GetJob** work_tail_{&work_head_};
  bool stop_worker_{false};
  std::thread worker_;
};

}  // namespace

std::shared_ptr<RoceSession> CreateRoceSession(
    std::string address, RoceOptions options, std::uint32_t device_id) {
  return std::make_shared<RoceSession>(std::move(address), std::move(options), device_id);
}

std::unique_ptr<DdrMemoryAllocator> CreateRoceAllocator(std::shared_ptr<RoceSession> session) {
  return std::make_unique<RoceAllocator>(std::move(session));
}

std::unique_ptr<AccessOperator> CreateRoceOperator(std::shared_ptr<RoceSession> session) {
  return std::make_unique<RoceOperator>(std::move(session));
}

}  // namespace kvaf
