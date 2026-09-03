#include "kvaf/runtime/runtime.h"

#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kvaf {

namespace {

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
  access_tables_.emplace(id, std::move(table));
  *handle = AccessTableHandle{id};
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
  if (!request.miss_count.has_elements<std::int32_t>() ||
      request.miss_count.bytes != sizeof(std::int32_t)) {
    return Status::InvalidArgument(
        "miss_count must be a single int32 on the BatchGet device");
  }
  if (!request.output.valid() ||
      request.output_block_bytes == 0 ||
      request.output.bytes % request.output_block_bytes != 0) {
    return Status::InvalidArgument("output must contain whole non-empty output blocks");
  }
  if (request.output.device == DeviceKind::kCpu) {
    const AccessTableState* table = FindAccessTable(request.access_table);
    if (table == nullptr) {
      return Status::NotFound("CPU BatchGet AccessTable handle was not found");
    }
    if (!request.indices.has_elements<std::int32_t>() ||
        !request.offsets.has_elements<std::int32_t>() ||
        !request.output_indices.has_elements<std::int32_t>() ||
        request.indices.bytes != request.offsets.bytes ||
        request.indices.bytes != request.output_indices.bytes ||
        request.indices.device != DeviceKind::kCpu ||
        request.offsets.device != DeviceKind::kCpu ||
        request.output_indices.device != DeviceKind::kCpu ||
        request.miss_count.device != DeviceKind::kCpu ||
        !request.remote_addresses.empty() ||
        !request.local_addresses.empty()) {
      return Status::InvalidArgument(
          "CPU BatchGet requires equally sized int32 lookup buffers");
    }
    const std::size_t request_count =
        request.indices.bytes / sizeof(std::int32_t);
    const std::int32_t miss_count =
        *static_cast<const std::int32_t*>(request.miss_count.data);
    if (miss_count < 0 || static_cast<std::size_t>(miss_count) > request_count) {
      return Status::InvalidArgument(
          "miss_count must be between zero and the BatchGet request capacity");
    }
    const AccessTableView view{table->addresses.data(), table->lengths.data(),
                               table->valid.data(), table->addresses.size()};
    return access_operator_.ExecuteBatchGet(
        BatchGetExecRequest{view, &request, execution});
  }

  if (request.output.device == DeviceKind::kNpu) {
    if (request.access_table.opaque != 0 || !request.indices.empty() ||
        !request.offsets.empty() || !request.output_indices.empty() ||
        !request.remote_addresses.has_elements<Address>() ||
        !request.local_addresses.has_elements<Address>() ||
        request.remote_addresses.bytes != request.local_addresses.bytes ||
        request.remote_addresses.device != DeviceKind::kNpu ||
        request.local_addresses.device != DeviceKind::kNpu ||
        request.miss_count.device != DeviceKind::kNpu) {
      return Status::InvalidArgument(
          "NPU BatchGet requires equally sized uint64 remote/local address buffers");
    }
    if (request.output_block_bytes > kBatchGetMaxTransferBytes) {
      return Status::InvalidArgument(
          "NPU BatchGet output row size exceeds the device ABI");
    }
    return access_operator_.ExecuteBatchGet(
        BatchGetExecRequest{AccessTableView{}, &request, execution});
  }

  return Status::InvalidArgument("BatchGet output device is not supported");
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
