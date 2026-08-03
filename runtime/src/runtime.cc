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
  } else if (!request.device_access_table.empty()) {
    return Status::InvalidArgument("CPU BatchGet must not provide device AccessTable buffers");
  }

  const AccessTableView view{table->addresses.data(), table->lengths.data(),
                             table->valid.data(), table->addresses.size()};
  return access_operator_.ExecuteBatchGet(
      BatchGetExecRequest{view, &request, execution});
}

Status Runtime::BatchPut(const BatchPutRequest& request) {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (request.keys.empty() || request.keys.size() != request.blocks.size()) {
    return Status::InvalidArgument("BatchPut keys and blocks must be non-empty and equal in size");
  }

  const DeviceKind device = request.blocks.front().device;
  std::vector<std::uint64_t> lengths;
  lengths.reserve(request.blocks.size());
  for (const ConstBufferView& block : request.blocks) {
    if (block.data == nullptr || block.bytes == 0) {
      return Status::InvalidArgument("BatchPut blocks must be non-empty");
    }
    if (block.device != device) {
      return Status::InvalidArgument("BatchPut blocks must share a device");
    }
    lengths.push_back(block.bytes);
  }

  PutContext put_context;
  PutAccessDesc put_desc;
  Status status = meta_manager_.PreparePut(request.keys, lengths, &put_context,
                                           &put_desc);
  if (!status.ok()) {
    return status;
  }

  status = access_operator_.ExecuteBatchPut(
      BatchPutExecRequest{&request, &put_desc});
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
