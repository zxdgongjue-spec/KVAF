#include "runtime.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kvaf {

namespace {

using HostClock = std::chrono::steady_clock;

std::uint64_t ElapsedNs(HostClock::time_point begin,
                        HostClock::time_point end) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count());
}

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
    if (request.ddr_bytes == 0 || request.slot_size == 0 ||
        request.ddr_bytes % request.slot_size != 0) {
      return Status::InvalidArgument(
          "ddr_bytes must be a non-zero multiple of slot_size");
    }

    auto bytes = std::unique_ptr<std::uint8_t[]>(
        new (std::nothrow) std::uint8_t[request.ddr_bytes]());
    if (!bytes) {
      return Status::OutOfMemory("failed to allocate host DDR");
    }

    const Address base = reinterpret_cast<Address>(bytes.get());
    allocations_.emplace(base, std::move(bytes));
    *remote_region = RemoteRegionDesc{base, request.ddr_bytes,
                                      request.slot_size, request.ddr_device};
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

RuntimeCore::RuntimeCore(MetaManager& meta_manager, AccessOperator& access_operator,
                         DdrMemoryAllocator& ddr_allocator)
    : meta_manager_(meta_manager),
      access_operator_(access_operator),
      ddr_allocator_(ddr_allocator) {}

RuntimeCore::~RuntimeCore() {
  if (initialized_) {
    (void)Shutdown();
  }
}

Status RuntimeCore::Initialize(const RuntimeInitRequest& request) {
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
  if (remote_region.length == 0 || remote_region.slot_size == 0 ||
      remote_region.length % remote_region.slot_size != 0) {
    (void)ddr_allocator_.Release(remote_region);
    return Status::InvalidArgument("DDR allocator returned an invalid remote region");
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

Status RuntimeCore::Shutdown() {
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
  batch_put_host_profiling_enabled_ = false;
  last_batch_put_host_profile_ = BatchPutHostProfile{};
  return first_error;
}

Status RuntimeCore::PrepareAccessTable(const AccessTablePrepareRequest& request,
                                       AccessTableHandle* handle) {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (handle == nullptr) {
    return Status::InvalidArgument("handle must not be null");
  }
  if (request.flat_keys.empty()) {
    return Status::InvalidArgument("PrepareAccessTable requires at least one key");
  }

  LockContext lock_context;
  Status status = meta_manager_.LockKeys(
      request.flat_keys, MetaManagerLockMode::kStableReadAllowAppend,
      &lock_context);
  if (!status.ok()) {
    return status;
  }

  FlatAccessDesc lookup;
  status = meta_manager_.BatchLookup(request.flat_keys, lock_context, &lookup);
  if (!status.ok()) {
    (void)meta_manager_.UnlockKeys(lock_context);
    return status;
  }
  if (lookup.addresses.size() != request.flat_keys.size() ||
      lookup.lengths.size() != request.flat_keys.size() ||
      lookup.valid.size() != request.flat_keys.size()) {
    (void)meta_manager_.UnlockKeys(lock_context);
    return Status::InvalidState("MetaManager returned an invalid AccessTable lookup");
  }

  bool has_valid_entries = false;
  bool has_uniform_valid_length = true;
  std::uint64_t uniform_valid_length = 0;
  for (std::size_t index = 0; index < lookup.valid.size(); ++index) {
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
  return Status::OK();
}

Status RuntimeCore::GetAccessTableView(const AccessTableHandle& handle,
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

Status RuntimeCore::ReleaseAccessTable(const AccessTableHandle& handle) {
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

Status RuntimeCore::BatchGet(const BatchGetRequest& request,
                             const BatchGetExecutionOptions& execution) {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  const AccessTableState* table = nullptr;
  if (request.access_table.opaque != 0) {
    table = FindAccessTable(request.access_table);
    if (table == nullptr) {
      return Status::NotFound("AccessTable handle was not found");
    }
  }
  if (!request.indices.has_elements<std::int32_t>() ||
      !request.offsets.has_elements<std::int32_t>() ||
      !request.output_indices.has_elements<std::int32_t>() ||
      request.indices.bytes != request.offsets.bytes ||
      request.indices.bytes != request.output_indices.bytes) {
    return Status::InvalidArgument(
        "indices, offsets, and output_indices must be equally sized int32 buffers");
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
  if (request.indices.device != request.offsets.device ||
      request.indices.device != request.output_indices.device ||
      request.indices.device != request.output.device) {
    return Status::InvalidArgument("BatchGet input and output buffers must share a device");
  }

  if (request.indices.device == DeviceKind::kNpu) {
    if (request.tokens_per_block == 0 ||
        request.output_block_bytes >
            std::numeric_limits<std::uint64_t>::max() /
                request.tokens_per_block) {
      return Status::InvalidArgument(
          "NPU BatchGet requires a valid tokens_per_block layout");
    }
    const std::uint64_t address_table_entry_bytes =
        request.tokens_per_block * request.output_block_bytes;
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
    if (table != nullptr &&
        (!table->has_uniform_valid_length ||
         (table->has_valid_entries &&
          table->uniform_valid_length != address_table_entry_bytes))) {
      return Status::InvalidArgument(
          "NPU BatchGet AccessTable entries do not match the configured layout");
    }
  } else {
    if (table == nullptr) {
      return Status::InvalidArgument("CPU BatchGet requires an AccessTable handle");
    }
    if (!request.device_access_table.empty()) {
      return Status::InvalidArgument(
          "CPU BatchGet must not provide device AccessTable buffers");
    }
  }

  AccessTableView view{};
  if (table != nullptr) {
    view = AccessTableView{table->addresses.data(), table->lengths.data(),
                           table->valid.data(), table->addresses.size()};
  }
  return access_operator_.ExecuteBatchGet(
      BatchGetExecRequest{view, &request, execution});
}

Status RuntimeCore::BatchPut(const BatchPutRequest& request,
                             BatchPutExecutionOptions execution) {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  const bool host_profiling = batch_put_host_profiling_enabled_;
  BatchPutHostProfile host_profile{};
  const HostClock::time_point runtime_begin =
      host_profiling ? HostClock::now() : HostClock::time_point{};
  const HostClock::time_point validation_begin =
      host_profiling ? HostClock::now() : HostClock::time_point{};
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
  if (host_profiling) {
    host_profile.row_count = request.keys.size();
    host_profile.payload_bytes = request.input.bytes;
    host_profile.request_validation_ns =
        ElapsedNs(validation_begin, HostClock::now());
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  const HostClock::time_point prepare_begin =
      host_profiling ? HostClock::now() : HostClock::time_point{};
  Status status = meta_manager_.PreparePut(request.keys, request.lengths,
                                           &put_context, &put_desc);
  if (host_profiling) {
    host_profile.prepare_put_ns = ElapsedNs(prepare_begin, HostClock::now());
  }
  if (!status.ok()) {
    return status;
  }

  status = access_operator_.ExecuteBatchPut(
      BatchPutExecRequest{&request, &put_desc, execution});
  if (!status.ok()) {
    (void)meta_manager_.AbortPut(put_context);
    return status;
  }
  if (host_profiling) {
    BatchPutHostProfile data_plane_profile{};
    if (access_operator_.CopyBatchPutHostProfile(&data_plane_profile).ok()) {
      host_profile.metadata_bytes = data_plane_profile.metadata_bytes;
      host_profile.metadata_reserve_ns = data_plane_profile.metadata_reserve_ns;
      host_profile.metadata_pack_ns = data_plane_profile.metadata_pack_ns;
      host_profile.metadata_h2d_ns = data_plane_profile.metadata_h2d_ns;
      host_profile.local_buffer_acquire_ns =
          data_plane_profile.local_buffer_acquire_ns;
      host_profile.execute_sync_ns = data_plane_profile.execute_sync_ns;
      host_profile.local_buffer_release_ns =
          data_plane_profile.local_buffer_release_ns;
    }
  }

  const HostClock::time_point commit_begin =
      host_profiling ? HostClock::now() : HostClock::time_point{};
  status = meta_manager_.CommitPut(put_context);
  if (host_profiling) {
    host_profile.commit_put_ns = ElapsedNs(commit_begin, HostClock::now());
  }
  if (!status.ok()) {
    return status;
  }
  if (host_profiling) {
    host_profile.runtime_e2e_ns = ElapsedNs(runtime_begin, HostClock::now());
    host_profile.runtime_non_execute_ns =
        host_profile.runtime_e2e_ns >= host_profile.execute_sync_ns
            ? host_profile.runtime_e2e_ns - host_profile.execute_sync_ns
            : 0U;
    last_batch_put_host_profile_ = host_profile;
  }
  return Status::OK();
}

Status RuntimeCore::SetBatchPutHostProfiling(bool enabled) {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  const Status status = access_operator_.SetBatchPutHostProfiling(enabled);
  if (!status.ok()) {
    return status;
  }
  batch_put_host_profiling_enabled_ = enabled;
  last_batch_put_host_profile_ = BatchPutHostProfile{};
  return Status::OK();
}

Status RuntimeCore::CopyBatchPutHostProfile(BatchPutHostProfile* profile) const {
  if (profile == nullptr) {
    return Status::InvalidArgument("BatchPut host profile output must not be null");
  }
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (!batch_put_host_profiling_enabled_) {
    return Status::InvalidState("BatchPut host profiling is not enabled");
  }
  *profile = last_batch_put_host_profile_;
  return Status::OK();
}

Status RuntimeCore::BatchFree(const BatchFreeRequest& request) {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (request.keys.empty()) {
    return Status::InvalidArgument("BatchFree requires at least one key");
  }
  return meta_manager_.BatchFree(request.keys);
}

RuntimeCore::AccessTableState* RuntimeCore::FindAccessTable(
    const AccessTableHandle& handle) {
  const auto it = access_tables_.find(handle.opaque);
  return it == access_tables_.end() ? nullptr : it->second.get();
}

const RuntimeCore::AccessTableState* RuntimeCore::FindAccessTable(
    const AccessTableHandle& handle) const {
  const auto it = access_tables_.find(handle.opaque);
  return it == access_tables_.end() ? nullptr : it->second.get();
}

}  // namespace kvaf
