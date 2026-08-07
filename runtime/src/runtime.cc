#include "kvaf/runtime/runtime.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kvaf {

namespace {

bool RuntimeTimingEnabled() {
  const char* value = std::getenv("KVAF_RUNTIME_TIMING_LOG");
  return value != nullptr && value[0] == '1';
}

// Runtime 外层计时与 MetaManager 内部计时分开，避免普通分析扰动哈希查询热路径。
bool MetaManagerTimingEnabled() {
  const char* value = std::getenv("KVAF_META_MANAGER_TIMING_LOG");
  return value != nullptr && value[0] == '1';
}

class ScopedRuntimeTimer {
 public:
  ScopedRuntimeTimer(const char* operation,
                     const char* phase,
                     std::size_t key_count)
      : operation_(operation),
        phase_(phase),
        key_count_(key_count),
        active_(RuntimeTimingEnabled()) {
    if (active_) {
      start_ = std::chrono::steady_clock::now();
    }
  }

  ~ScopedRuntimeTimer() {
    if (!active_) {
      return;
    }

    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_)
            .count();
    std::cerr << "RUNTIME_PERF_DETAIL operation=" << operation_
              << " phase=" << phase_ << " keys=" << key_count_
              << " elapsed_ns=" << elapsed_ns << '\n';
  }

 private:
  const char* operation_;
  const char* phase_;
  std::size_t key_count_;
  bool active_{false};
  std::chrono::steady_clock::time_point start_;
};

class ScopedMetaManagerTiming {
 public:
  explicit ScopedMetaManagerTiming(MetaManager& meta_manager)
      : meta_manager_(meta_manager), active_(MetaManagerTimingEnabled()) {
    if (active_) {
      meta_manager_.SetTimingLogEnabled(true);
    }
  }

  ~ScopedMetaManagerTiming() {
    if (!active_) {
      return;
    }

    meta_manager_.SetTimingLogEnabled(false);
    FlushMetaManagerTimingSummary();
  }

 private:
  MetaManager& meta_manager_;
  bool active_{false};
};

class HostDdrMemoryAllocator final : public DdrMemoryAllocator {
 public:
  Status Allocate(const RuntimeInitRequest& request,
                  RemoteRegionDesc* remote_region) override {
    if (remote_region == nullptr) {
      return Status::InvalidArgument("remote_region must not be null");
    }
    if (request.ddr_device != DeviceKind::kCpu) {
      return Status::InvalidArgument(
          "HostDdrMemoryAllocator supports only CPU host DDR");
    }
    if (request.ddr_bytes == 0 || request.short_slot_size == 0 ||
        request.long_slot_size == 0 ||
        request.slot_count_per_class == 0 ||
        request.short_slot_size >= request.long_slot_size) {
      return Status::InvalidArgument(
          "invalid dual-slot DDR layout");
    }
    if (request.short_slot_size >
            std::numeric_limits<std::uint64_t>::max() -
                request.long_slot_size ||
        request.slot_count_per_class >
            std::numeric_limits<std::uint64_t>::max() /
                (request.short_slot_size + request.long_slot_size) ||
        request.slot_count_per_class *
                (request.short_slot_size + request.long_slot_size) !=
            request.ddr_bytes) {
      return Status::InvalidArgument(
          "ddr_bytes does not match the dual-slot layout");
    }

    auto bytes = std::unique_ptr<std::uint8_t[]>(
        new (std::nothrow) std::uint8_t[request.ddr_bytes]());
    if (!bytes) {
      return Status::OutOfMemory("failed to allocate host DDR");
    }

    const Address base = reinterpret_cast<Address>(bytes.get());
    allocations_.emplace(base, std::move(bytes));
    *remote_region = RemoteRegionDesc{
        base,
        request.ddr_bytes,
        request.short_slot_size,
        request.long_slot_size,
        request.slot_count_per_class,
        request.ddr_device};
    return Status::OK();
  }

  Status Release(const RemoteRegionDesc& remote_region) override {
    const auto it = allocations_.find(remote_region.base);
    if (it == allocations_.end()) {
      return Status::NotFound("host DDR allocation was not found");
    }
    allocations_.erase(it);
    return Status::OK();
  }

 private:
  std::unordered_map<Address, std::unique_ptr<std::uint8_t[]>> allocations_;
};

}  // namespace

std::unique_ptr<DdrMemoryAllocator> CreateHostDdrMemoryAllocator() {
  return std::make_unique<HostDdrMemoryAllocator>();
}

Runtime::Runtime(MetaManager& meta_manager, AccessOperator& access_operator,
                 DdrMemoryAllocator& ddr_allocator)
    : meta_manager_(meta_manager),
      access_operator_(access_operator),
      ddr_allocator_(ddr_allocator) {}

Runtime::~Runtime() {
  if (initialized_) {
    (void)Shutdown();
  }
}

Status Runtime::Initialize(const RuntimeInitRequest& request) {
  if (initialized_) {
    return Status::InvalidState("Runtime has already been initialized");
  }
  if (!access_tables_.empty()) {
    return Status::InvalidState("Runtime has unreleased AccessTables");
  }

  RemoteRegionDesc remote_region;
  Status status = ddr_allocator_.Allocate(request, &remote_region);
  if (!status.ok()) {
    return status;
  }
  if (remote_region.length == 0 || remote_region.short_slot_size == 0 ||
      remote_region.long_slot_size == 0 ||
      remote_region.slot_count_per_class == 0 ||
      remote_region.short_slot_size >= remote_region.long_slot_size ||
      remote_region.short_slot_size >
          std::numeric_limits<std::uint64_t>::max() -
              remote_region.long_slot_size ||
      remote_region.slot_count_per_class >
          std::numeric_limits<std::uint64_t>::max() /
              (remote_region.short_slot_size + remote_region.long_slot_size) ||
      remote_region.slot_count_per_class *
              (remote_region.short_slot_size + remote_region.long_slot_size) !=
          remote_region.length) {
    (void)ddr_allocator_.Release(remote_region);
    return Status::InvalidArgument(
        "DDR allocator returned an invalid dual-slot region");
  }

  status = access_operator_.Initialize(remote_region);
  if (!status.ok()) {
    (void)ddr_allocator_.Release(remote_region);
    return status;
  }

  status = meta_manager_.RegisterRemoteRegion(remote_region);
  if (!status.ok()) {
    (void)access_operator_.Shutdown();
    (void)ddr_allocator_.Release(remote_region);
    return status;
  }

  remote_region_ = remote_region;
  initialized_ = true;
  return Status::OK();
}

Status Runtime::Shutdown() {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }

  Status first_error = Status::OK();
  for (const auto& entry : access_tables_) {
    const Status status = meta_manager_.UnlockKeys(entry.second->lock_context);
    if (!status.ok() && first_error.ok()) {
      first_error = status;
    }
  }
  access_tables_.clear();

  const Status operator_status = access_operator_.Shutdown();
  if (!operator_status.ok() && first_error.ok()) {
    first_error = operator_status;
  }
  const Status release_status = ddr_allocator_.Release(remote_region_);
  if (!release_status.ok() && first_error.ok()) {
    first_error = release_status;
  }

  initialized_ = false;
  remote_region_ = RemoteRegionDesc{};
  return first_error;
}

Status Runtime::PrepareAccessTable(const AccessTablePrepareRequest& request,
                                   AccessTableHandle* handle) {
  ScopedRuntimeTimer total_timer("prepare_access_table", "total",
                                 request.flat_keys.size());
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (handle == nullptr) {
    return Status::InvalidArgument("handle must not be null");
  }
  if (request.flat_keys.empty()) {
    return Status::InvalidArgument("PrepareAccessTable requires at least one key");
  }

  ScopedMetaManagerTiming meta_manager_timing(meta_manager_);

  LockContext lock_context;
  Status status;
  {
    ScopedRuntimeTimer phase_timer("prepare_access_table", "lock_keys",
                                   request.flat_keys.size());
    status = meta_manager_.LockKeys(
        request.flat_keys, MetaManagerLockMode::kStableReadAllowAppend,
        &lock_context);
  }
  if (!status.ok()) {
    return status;
  }

  FlatAccessDesc lookup;
  {
    ScopedRuntimeTimer phase_timer("prepare_access_table", "batch_lookup",
                                   request.flat_keys.size());
    status = meta_manager_.BatchLookup(request.flat_keys, lock_context, &lookup);
  }
  if (!status.ok()) {
    (void)meta_manager_.UnlockKeys(lock_context);
    return status;
  }
  {
    ScopedRuntimeTimer phase_timer("prepare_access_table", "validate_layout",
                                   request.flat_keys.size());
    if (lookup.addresses.size() != request.flat_keys.size() ||
        lookup.keys.size() != request.flat_keys.size() ||
        lookup.lengths.size() != request.flat_keys.size() ||
        lookup.valid.size() != request.flat_keys.size()) {
      (void)meta_manager_.UnlockKeys(lock_context);
      return Status::InvalidState(
          "MetaManager returned an invalid AccessTable lookup");
    }
  }

  bool has_valid_entries = false;
  bool has_uniform_valid_length = true;
  std::uint64_t uniform_valid_length = 0;
  {
    ScopedRuntimeTimer phase_timer("prepare_access_table", "scan_lengths",
                                   request.flat_keys.size());
    for (std::size_t index = 0; index < lookup.addresses.size(); ++index) {
      if (lookup.valid[index] == 0) {
        continue;
      }
      if (!has_valid_entries) {
        has_valid_entries = true;
        uniform_valid_length = lookup.lengths[index];
      } else if (lookup.lengths[index] != uniform_valid_length) {
        has_uniform_valid_length = false;
        break;
      }
    }
  }

  {
    ScopedRuntimeTimer phase_timer("prepare_access_table", "publish_table",
                                   request.flat_keys.size());
    const std::uint64_t id = next_access_table_id_++;
    if (id == 0) {
      (void)meta_manager_.UnlockKeys(lock_context);
      return Status::InvalidState("AccessTable handle space is exhausted");
    }
    auto table = std::make_unique<AccessTableState>();
    table->lock_context = lock_context;
    table->addresses = std::move(lookup.addresses);
    table->lengths = std::move(lookup.lengths);
    table->valid = std::move(lookup.valid);
    table->has_valid_entries = has_valid_entries;
    table->has_uniform_valid_length = has_uniform_valid_length;
    table->uniform_valid_length = uniform_valid_length;
    access_tables_.emplace(id, std::move(table));
    *handle = AccessTableHandle{id};
  }
  return Status::OK();
}

Status Runtime::GetAccessTableView(const AccessTableHandle& handle,
                                   AccessTableView* view) const {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (view == nullptr) {
    return Status::InvalidArgument("view must not be null");
  }
  const AccessTableState* table = FindAccessTable(handle);
  if (table == nullptr) {
    return Status::NotFound("AccessTable handle was not found");
  }
  *view = AccessTableView{table->addresses.data(), table->lengths.data(),
                          table->valid.data(), table->addresses.size()};
  return Status::OK();
}

Status Runtime::ReleaseAccessTable(const AccessTableHandle& handle) {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  AccessTableState* table = FindAccessTable(handle);
  if (table == nullptr) {
    return Status::NotFound("AccessTable handle was not found");
  }
  const Status status = meta_manager_.UnlockKeys(table->lock_context);
  if (!status.ok()) {
    return status;
  }
  access_tables_.erase(handle.opaque);
  return Status::OK();
}

Status Runtime::BatchGet(const BatchGetRequest& request,
                         const BatchGetExecutionOptions& execution) {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  const AccessTableState* table = FindAccessTable(request.access_table);
  if (table == nullptr) {
    return Status::NotFound("AccessTable handle was not found");
  }
  if (!request.indices.has_elements<std::int32_t>() ||
      !request.output_indices.has_elements<std::int32_t>() ||
      request.indices.bytes != request.output_indices.bytes) {
    return Status::InvalidArgument(
        "indices and output_indices must be equally sized int32 buffers");
  }
  const std::size_t request_count =
      request.indices.bytes / sizeof(std::int32_t);
  if (!request.mask.empty() &&
      (!request.mask.valid() || request.mask.bytes != request_count ||
       request.mask.device != request.indices.device)) {
    return Status::InvalidArgument("mask must match indices when provided");
  }
  if (!request.output.valid() ||
      request.output_block_bytes == 0 ||
      request.output.bytes % request.output_block_bytes != 0) {
    return Status::InvalidArgument("output must contain whole non-empty output blocks");
  }
  if (request.indices.device != request.output_indices.device ||
      request.indices.device != request.output.device) {
    return Status::InvalidArgument("BatchGet input and output buffers must share a device");
  }

  if (request.indices.device == DeviceKind::kNpu) {
    if (request.device_access_table.empty() ||
        request.device_access_table.addresses.data == nullptr ||
        request.device_access_table.valid.data == nullptr ||
        request.device_access_table.addresses.bytes == 0 ||
        request.device_access_table.addresses.bytes % sizeof(Address) != 0 ||
        request.device_access_table.valid.bytes !=
            request.device_access_table.addresses.bytes / sizeof(Address) ||
        request.device_access_table.addresses.device != DeviceKind::kNpu ||
        request.device_access_table.valid.device != DeviceKind::kNpu) {
      return Status::InvalidArgument(
          "NPU BatchGet requires matching caller-provided device AccessTable buffers");
    }
    if (!table->has_uniform_valid_length ||
        (table->has_valid_entries &&
         table->uniform_valid_length != request.output_block_bytes)) {
      return Status::InvalidArgument(
          "NPU BatchGet requires one fixed block length in its AccessTable");
    }
  } else if (!request.device_access_table.empty()) {
    return Status::InvalidArgument("CPU BatchGet must not provide device AccessTable buffers");
  }

  const AccessTableView view{table->addresses.data(), table->lengths.data(),
                             table->valid.data(), table->addresses.size()};
  return access_operator_.ExecuteBatchGet(
      BatchGetExecRequest{view, &request, execution});
}

Status Runtime::BatchPut(const BatchPutRequest& request,
                         BatchPutExecutionOptions execution) {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (request.keys.empty() || request.input.data == nullptr ||
      request.input.bytes == 0 ||
      request.keys.size() != request.input_offsets.size() ||
      request.keys.size() != request.lengths.size()) {
    return Status::InvalidArgument(
        "BatchPut keys, input_offsets, and lengths must be non-empty and equal in size");
  }

  for (std::size_t index = 0; index < request.keys.size(); ++index) {
    const std::uint64_t offset = request.input_offsets[index];
    const std::uint64_t length = request.lengths[index];
    if (length == 0 || offset > request.input.bytes ||
        length > request.input.bytes - offset) {
      return Status::InvalidArgument(
          "BatchPut input_offsets and lengths must describe bytes inside input");
    }
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  Status status = meta_manager_.PreparePut(request.keys, request.lengths,
                                           &put_context, &put_desc);
  if (!status.ok()) {
    return status;
  }

  status = access_operator_.ExecuteBatchPut(
      BatchPutExecRequest{&request, &put_desc, execution});
  if (!status.ok()) {
    (void)meta_manager_.AbortPut(put_context);
    return status;
  }

  return meta_manager_.CommitPut(put_context);
}

Status Runtime::BatchFree(const BatchFreeRequest& request) {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (request.keys.empty()) {
    return Status::InvalidArgument("BatchFree requires at least one key");
  }
  return meta_manager_.BatchFree(request.keys);
}

Runtime::AccessTableState* Runtime::FindAccessTable(
    const AccessTableHandle& handle) {
  const auto it = access_tables_.find(handle.opaque);
  return it == access_tables_.end() ? nullptr : it->second.get();
}

const Runtime::AccessTableState* Runtime::FindAccessTable(
    const AccessTableHandle& handle) const {
  const auto it = access_tables_.find(handle.opaque);
  return it == access_tables_.end() ? nullptr : it->second.get();
}

}  // namespace kvaf
