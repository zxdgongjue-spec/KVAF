#include "ubmem/local_access_operator.h"

#include <acl/acl.h>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <utility>
#include <vector>

#include "kernel_loader.h"
#include "meta_manager.h"
#include "ubmem/transfer_abi.h"

namespace kvaf {
namespace {
using namespace kvaf_ubmem;

uint64_t Pointer(const void* address) { return reinterpret_cast<uintptr_t>(address); }
bool U32(uint64_t value) { return value <= std::numeric_limits<uint32_t>::max(); }
Status AclFailure(const char* operation, aclError error) {
  return Status::Internal(std::string("UBmem ") + operation + ": ACL=" + std::to_string(error));
}

class UbmemAccessOperator final : public AccessOperator {
 public:
  explicit UbmemAccessOperator(AccessOperatorConfig config)
      : state_(std::make_unique<State>(std::move(config))) {}
  ~UbmemAccessOperator() override {
    if (!Shutdown().ok()) {
      std::fprintf(stderr, "[KVAF_UBMEM] RETAINED_ON_DESTRUCTION completion_or_release_unconfirmed\n");
      (void)state_.release();
    }
  }

  Status Initialize(const RemoteRegionDesc& region) override {
    std::unique_lock<std::shared_mutex> lifetime(state_->lifecycle);
    auto& s = *state_;
    if (s.initialized) return Status::InvalidState("UBmem operator already initialized");
    if (s.poisoned) return Status::CompletionUnknown("UBmem operator is poisoned");
    if (!Contains(region.base, region.length, region.base, region.length) ||
        region.slot_size == 0 || region.length % region.slot_size != 0 ||
        region.device != DeviceKind::kCpu || s.config.npu_core_count == 0 ||
        s.config.npu_core_count > kMaxWorkers ||
        s.config.batch_get_kernel_object_path.empty() ||
        s.config.batch_put_kernel_object_path.empty()) {
      return Status::InvalidArgument("invalid UBmem region, worker count, or kernel paths");
    }
    aclError error = aclrtGetDevice(&s.device);
    if (error != ACL_SUCCESS) return AclFailure("get current device", error);
    auto status = s.get.Load(s.config.batch_get_kernel_object_path, "KvafUbmemBatchGet");
    if (!status.ok()) return status;
    status = s.put.Load(s.config.batch_put_kernel_object_path, "KvafUbmemBatchPut");
    if (!status.ok()) { (void)s.get.Unload(); return status; }
    error = aclrtMalloc(&s.status, kMaxWorkers * sizeof(CoreStatus), ACL_MEM_MALLOC_NORMAL_ONLY);
    if (error != ACL_SUCCESS) {
      (void)s.put.Unload(); (void)s.get.Unload();
      return AclFailure("allocate status", error);
    }
    s.region = region;
    s.initialized = true;
    return Status::OK();
  }

  Status Shutdown() override {
    std::unique_lock<std::shared_mutex> lifetime(state_->lifecycle);
    auto& s = *state_;
    if (s.poisoned) return Status::CompletionUnknown("UBmem operator retains resources after unknown completion");
    if (!s.initialized) {
      auto status = s.put.Unload();
      if (!status.ok()) return status;
      return s.get.Unload();
    }
    auto ready = CheckDevice();
    if (!ready.ok()) return ready;
    // Also covers asynchronous Get on streams which the caller has destroyed
    // after completion. Graph owners must not replay after Runtime shutdown.
    const aclError error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) return Poison("shutdown synchronization", error);
    if (s.metadata_device != nullptr) {
      const aclError release = aclrtFree(s.metadata_device);
      if (release != ACL_SUCCESS) return AclFailure("free Put metadata", release);
      s.metadata_device = nullptr;
    }
    if (s.metadata_host != nullptr) {
      const aclError release = aclrtFreeHost(s.metadata_host);
      if (release != ACL_SUCCESS) return AclFailure("free pinned Put metadata", release);
      s.metadata_host = nullptr;
    }
    if (s.status_host != nullptr) {
      const aclError release = aclrtFreeHost(s.status_host);
      if (release != ACL_SUCCESS) return AclFailure("free pinned status", release);
      s.status_host = nullptr;
    }
    if (s.status != nullptr) {
      const aclError release = aclrtFree(s.status);
      if (release != ACL_SUCCESS) return AclFailure("free status", release);
      s.status = nullptr;
    }
    auto status = s.put.Unload();
    if (!status.ok()) return status;
    status = s.get.Unload();
    if (!status.ok()) return status;
    s.initialized = false;
    s.metadata_capacity = 0;
    s.region = {};
    return Status::OK();
  }

  Status ExecuteBatchGet(const BatchGetExecRequest& exec) override {
    std::shared_lock<std::shared_mutex> lifetime(state_->lifecycle);
    // Only synchronous diagnostics share Put's status workspace. Graph/async
    // Get submission must not wait for a background Put to drain its stream.
    std::unique_lock<std::mutex> lock(state_->mutex, std::defer_lock);
    if (exec.execution.synchronize) lock.lock();
    auto ready = CheckReady();
    if (!ready.ok()) return ready;
    if (exec.request == nullptr || exec.execution.stream == nullptr) return Status::InvalidArgument("UBmem Get requires request and caller stream");
    const auto& r = *exec.request;
    const auto& table = r.device_access_table;
    if (!r.indices.has_elements<int32_t>() || !r.offsets.has_elements<int32_t>() ||
        !r.output_indices.has_elements<int32_t>() || !r.miss_count.has_elements<int32_t>() ||
        r.indices.bytes != r.offsets.bytes || r.indices.bytes != r.output_indices.bytes ||
        r.miss_count.bytes != sizeof(int32_t) || !r.output.valid() ||
        (r.output_block_bytes != 656 && r.output_block_bytes != 2624) ||
        r.tokens_per_block == 0 || !U32(r.tokens_per_block) ||
        r.tokens_per_block > state_->region.slot_size / r.output_block_bytes ||
        r.output.bytes % r.output_block_bytes != 0 || Pointer(r.output.data) % 32 != 0 ||
        !Contains(Pointer(r.output.data), r.output.bytes, Pointer(r.output.data), r.output.bytes) ||
        !table.addresses.has_elements<Address>() || !table.valid.has_elements<uint8_t>() ||
        table.valid.bytes != table.addresses.bytes / sizeof(Address) ||
        r.indices.device != DeviceKind::kNpu || r.offsets.device != DeviceKind::kNpu ||
        r.output_indices.device != DeviceKind::kNpu || r.miss_count.device != DeviceKind::kNpu ||
        r.output.device != DeviceKind::kNpu || table.addresses.device != DeviceKind::kNpu ||
        table.valid.device != DeviceKind::kNpu || !U32(r.indices.bytes / 4) ||
        !U32(table.valid.bytes) || !U32(r.output.bytes / r.output_block_bytes)) {
      return Status::InvalidArgument("UBmem Get requires NPU device tables/requests and aligned 656/2624-byte output rows");
    }
    GetArgs args{};
    args.address_table = Pointer(table.addresses.data);
    args.valid_table = Pointer(table.valid.data);
    args.indices = Pointer(r.indices.data);
    args.offsets = Pointer(r.offsets.data);
    args.miss_count = Pointer(r.miss_count.data);
    args.output_indices = Pointer(r.output_indices.data);
    args.output = Pointer(r.output.data);
    // Async/graph launches use no shared diagnostic workspace. Synchronous
    // validation serializes on this mutex and may safely inspect the status.
    args.status = exec.execution.synchronize ? Pointer(state_->status) : 0;
    args.region_base = state_->region.base;
    args.region_bytes = state_->region.length;
    args.slot_bytes = state_->region.slot_size;
    args.table_capacity = static_cast<uint32_t>(table.valid.bytes);
    args.request_capacity = static_cast<uint32_t>(r.indices.bytes / 4);
    args.output_rows = static_cast<uint32_t>(r.output.bytes / r.output_block_bytes);
    args.token_bytes = static_cast<uint32_t>(r.output_block_bytes);
    args.tokens_per_block = static_cast<uint32_t>(r.tokens_per_block);
    args.workers = state_->config.npu_core_count;
    args.abi_version = kAbiVersion;
    const auto stream = static_cast<aclrtStream>(exec.execution.stream);
    aclError error;
    if (exec.execution.synchronize) {
      const auto ready_status = EnsureHostStatus();
      if (!ready_status.ok()) return ready_status;
      error = aclrtMemsetAsync(state_->status, kMaxWorkers * sizeof(CoreStatus), 0,
                              args.workers * sizeof(CoreStatus), stream);
      if (error != ACL_SUCCESS) return DrainFailure(stream, "clear Get status", error);
    }
    error = state_->get.Launch(&args, sizeof(args), args.workers, stream);
    if (error != ACL_SUCCESS) return DrainFailure(stream, "launch Get", error);
    if (!exec.execution.synchronize) return Status::OK();
    return Complete(stream, args.workers, "Get");
  }

  Status ExecuteBatchPut(const BatchPutExecRequest& exec) override {
    std::shared_lock<std::shared_mutex> lifetime(state_->lifecycle);
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto ready = CheckReady();
    if (!ready.ok()) return ready;
    if (exec.request == nullptr || exec.put_desc == nullptr || exec.execution.stream == nullptr) return Status::InvalidArgument("UBmem Put requires request, reservations, and caller stream");
    const auto& r = *exec.request;
    const auto& entries = exec.put_desc->entries;
    if (r.keys.empty() || !U32(r.keys.size()) || entries.size() != r.keys.size() ||
        r.input_offsets.size() != r.keys.size() || r.lengths.size() != r.keys.size() ||
        r.input.device != DeviceKind::kNpu || !r.input.valid() ||
        !Contains(Pointer(r.input.data), r.input.bytes, Pointer(r.input.data), r.input.bytes)) {
      return Status::InvalidArgument("UBmem Put requires packed NPU input and one reservation per key");
    }
    for (size_t i = 0; i < entries.size(); ++i) {
      if (entries[i].key != r.keys[i] || entries[i].reserved_length != r.lengths[i] ||
          !ValidOffset(r.input.bytes, r.input_offsets[i], r.lengths[i]) ||
          !Contains(state_->region.base, state_->region.length, entries[i].address, r.lengths[i]) ||
          r.lengths[i] > state_->region.slot_size ||
          (entries[i].address - state_->region.base) % state_->region.slot_size != 0) {
        return Status::InvalidArgument("UBmem Put source/reservation is outside the valid payload range");
      }
    }
    auto status = EnsureHostStatus();
    if (!status.ok()) return status;
    status = EnsureMetadata(entries.size() * sizeof(PutEntry));
    if (!status.ok()) return status;
    auto* metadata = static_cast<PutEntry*>(state_->metadata_host);
    for (size_t i = 0; i < entries.size(); ++i) metadata[i] = {r.input_offsets[i], r.lengths[i], entries[i].address};
    const auto stream = static_cast<aclrtStream>(exec.execution.stream);
    aclError error = aclrtMemcpyAsync(state_->metadata_device, state_->metadata_capacity,
        state_->metadata_host, entries.size() * sizeof(PutEntry), ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (error != ACL_SUCCESS) return DrainFailure(stream, "copy Put metadata", error);
    const uint32_t workers = state_->region.base % 32 == 0 && state_->region.slot_size % 32 == 0
        ? state_->config.npu_core_count : 1;
    error = aclrtMemsetAsync(state_->status, kMaxWorkers * sizeof(CoreStatus), 0,
                            workers * sizeof(CoreStatus), stream);
    if (error != ACL_SUCCESS) return DrainFailure(stream, "clear Put status", error);
    PutArgs args{Pointer(r.input.data), r.input.bytes, Pointer(state_->metadata_device),
        Pointer(state_->status), state_->region.base, state_->region.length,
        state_->region.slot_size, static_cast<uint32_t>(entries.size()), kAbiVersion};
    error = state_->put.Launch(&args, sizeof(args), workers, stream);
    if (error != ACL_SUCCESS) return DrainFailure(stream, "launch Put", error);
    // Only a completed status copy and successful validation permit CommitPut.
    return Complete(stream, workers, "Put", entries.size());
  }

 private:
  struct State {
    explicit State(AccessOperatorConfig value) : config(std::move(value)) {}
    AccessOperatorConfig config;
    RemoteRegionDesc region;
    ubmem::Kernel get, put;
    std::shared_mutex lifecycle;
    std::mutex mutex;
    void* status{nullptr};
    void* status_host{nullptr};
    void* metadata_host{nullptr};
    void* metadata_device{nullptr};
    size_t metadata_capacity{0};
    int32_t device{-1};
    bool initialized{false};
    std::atomic<bool> poisoned{false};
  };

  Status CheckDevice() const {
    int32_t device = -1;
    const aclError error = aclrtGetDevice(&device);
    if (error != ACL_SUCCESS) return AclFailure("get current device", error);
    if (device != state_->device) return Status::InvalidState("UBmem operator must run on its creating NPU device");
    return Status::OK();
  }
  Status CheckReady() const {
    if (state_->poisoned) return Status::CompletionUnknown("UBmem operator is poisoned; resources remain reserved");
    if (!state_->initialized) return Status::NotInitialized("UBmem operator is not initialized");
    return CheckDevice();
  }
  Status Poison(const char* operation, aclError error) {
    state_->poisoned = true;
    return Status::CompletionUnknown(std::string("UBmem ") + operation +
        ": completion unknown, retain arena and metadata; ACL=" + std::to_string(error));
  }
  Status DrainFailure(aclrtStream stream, const char* operation, aclError error) {
    const aclError drain = aclrtSynchronizeStream(stream);
    if (drain != ACL_SUCCESS) return Poison(operation, drain);
    return AclFailure(operation, error);
  }
  Status EnsureHostStatus() {
    if (state_->status_host != nullptr) return Status::OK();
    const aclError error = aclrtMallocHost(&state_->status_host, kMaxWorkers * sizeof(CoreStatus));
    if (error != ACL_SUCCESS) return AclFailure("allocate pinned status", error);
    return Status::OK();
  }
  Status Complete(aclrtStream stream, uint32_t workers, const char* operation,
                  uint64_t expected = UINT64_MAX) {
    // Queue the readback behind the kernel, then drain once. Reusing pinned
    // memory avoids both per-call heap allocation and a separate blocking
    // D2H submission after the first stream synchronization.
    const aclError copy = aclrtMemcpyAsync(state_->status_host, kMaxWorkers * sizeof(CoreStatus),
        state_->status, workers * sizeof(CoreStatus), ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (copy != ACL_SUCCESS) return DrainFailure(stream, "enqueue status readback", copy);
    const aclError error = aclrtSynchronizeStream(stream);
    if (error != ACL_SUCCESS) return Poison("transfer/status synchronization", error);
    const auto* statuses = static_cast<const CoreStatus*>(state_->status_host);
    uint64_t total = 0;
    for (uint32_t worker = 0; worker < workers; ++worker) {
      const auto& status = statuses[worker];
      total += status.copied_requests;
      if (status.magic != kStatusMagic || status.result != kSuccess) {
        return Status::Internal(std::string("UBmem ") + operation + " completed with device error: magic=" +
            std::to_string(status.magic) + " result=" + std::to_string(status.result) +
            " request=" + std::to_string(status.failed_request) +
            " copied=" + std::to_string(status.copied_requests));
      }
    }
    if (expected != UINT64_MAX && total != expected)
      return Status::Internal("UBmem Put completed count differs from reserved entries");
    return Status::OK();
  }
  Status EnsureMetadata(size_t bytes) {
    auto& s = *state_;
    if (bytes <= s.metadata_capacity) return Status::OK();
    // Previous Put returned only after synchronization. Async Get never uses
    // these allocations, so resizing cannot invalidate captured Get graphs.
    // Preserve any pointer whose release fails, so Shutdown can retry it.
    // A partial release invalidates the old capacity immediately.
    s.metadata_capacity = 0;
    if (s.metadata_device != nullptr) {
      const aclError error = aclrtFree(s.metadata_device);
      if (error != ACL_SUCCESS) return AclFailure("resize/free Put metadata", error);
      s.metadata_device = nullptr;
    }
    if (s.metadata_host != nullptr) {
      const aclError error = aclrtFreeHost(s.metadata_host);
      if (error != ACL_SUCCESS) return AclFailure("resize/free pinned Put metadata", error);
      s.metadata_host = nullptr;
    }
    aclError error = aclrtMallocHost(&s.metadata_host, bytes);
    if (error != ACL_SUCCESS) return AclFailure("allocate pinned Put metadata", error);
    error = aclrtMalloc(&s.metadata_device, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (error != ACL_SUCCESS) return AclFailure("allocate Put metadata", error);
    s.metadata_capacity = bytes;
    return Status::OK();
  }
  std::unique_ptr<State> state_;
};
}  // namespace

std::unique_ptr<AccessOperator> CreateUbmemAccessOperator(const AccessOperatorConfig& config) {
  return std::make_unique<UbmemAccessOperator>(config);
}
}  // namespace kvaf
