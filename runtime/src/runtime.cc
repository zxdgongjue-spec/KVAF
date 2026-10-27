#include "runtime.h"

#include <cstdint>
#include <atomic>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kvaf {

namespace {

std::atomic<std::uint64_t> next_table_id{1};

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
  if (pool_manager_) {
    // Stop every operator before dropping pins or any region allocation.
    // A backend that cannot stop must retain its transport resources.
    for (const auto pool : pool_manager_->Handles()) {
      if (!pool_manager_->Stop(pool).ok()) {
        (void)pool_manager_.release();
        return;
      }
    }
    for (const auto& table : access_tables_)
      (void)meta_manager_.pools().UnlockKeys(table.second->lock_context);
    access_tables_.clear();
  }
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

Status RuntimeCore::InitializePools(AccessOperatorFactory factory) {
  if (initialized_) return Status::InvalidState("Runtime has already been initialized");
  if (!factory) return Status::InvalidArgument("pool operator factory is required");
  pool_manager_ = std::make_unique<PoolManager>(ddr_allocator_, std::move(factory));
  initialized_ = true;
  return Status::OK();
}
Status RuntimeCore::CreatePool(const PoolSpec& spec, PoolHandle* output) {
  if (!pool_manager_) return Status::InvalidState("Runtime is not in pool mode");
  return pool_manager_->Create(spec, output);
}
Status RuntimeCore::GetPoolInfo(PoolHandle pool, PoolInfo* output) const {
  if (!pool_manager_) return Status::InvalidState("Runtime is not in pool mode");
  return pool_manager_->Info(pool, output);
}
Status RuntimeCore::BindLayer(std::uint64_t ns, std::uint64_t layer, PoolHandle pool, LayerHandle* output) {
  if (!pool_manager_) return Status::InvalidState("Runtime is not in pool mode");
  return pool_manager_->Bind(ns, layer, pool, output);
}
Status RuntimeCore::GetLayer(std::uint64_t ns, std::uint64_t layer, LayerHandle* output) const {
  if (!pool_manager_) return Status::InvalidState("Runtime is not in pool mode");
  return pool_manager_->FindLayer(ns, layer, output);
}
Status RuntimeCore::GetLayerLayout(LayerHandle layer, LayerLayout* output) const {
  if (!pool_manager_) return Status::InvalidState("Runtime is not in pool mode");
  return pool_manager_->Resolve(layer, output);
}
Status RuntimeCore::UnbindLayer(LayerHandle layer) {
  if (!pool_manager_) return Status::InvalidState("Runtime is not in pool mode");
  LayerLayout layout;
  auto status = pool_manager_->Resolve(layer, &layout, true);
  if (!status.ok()) return status;
  if (meta_manager_.pools().CountLayer(layer)) return Status::Busy("layer still owns keys");
  return pool_manager_->Unbind(layer);
}
Status RuntimeCore::DestroyPool(PoolHandle pool) {
  if (!pool_manager_) return Status::InvalidState("Runtime is not in pool mode");
  PoolInfo info;
  auto status = pool_manager_->Info(pool, &info);
  if (!status.ok()) return status;
  if (meta_manager_.pools().CountPool(pool)) return Status::Busy("pool still owns keys");
  return pool_manager_->Destroy(pool);
}

Status RuntimeCore::Shutdown() {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }

  if (pool_manager_) {
    if (!access_tables_.empty()) return Status::Busy("release AccessTables before pool shutdown");
    Status first;
    for (const auto pool : pool_manager_->Handles()) {
      auto status = pool_manager_->Stop(pool);
      if (status.ok()) {
        status = meta_manager_.pools().ErasePool(pool);
        if (status.ok()) {
          pool_manager_->UnbindPool(pool);
          status = pool_manager_->Destroy(pool);
        }
      }
      if (!status.ok() && first.ok()) first = status;
    }
    if (pool_manager_->Handles().empty()) {
      pool_manager_.reset();
      initialized_ = false;
    }
    return first;
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

  LayerLayout layout;
  if (pool_manager_) {
    auto status = pool_manager_->Resolve(request.layer, &layout);
    if (!status.ok()) return status;
  } else if (!request.layer.empty()) return Status::InvalidArgument("layer supplied to legacy Runtime");
  const bool scoped = pool_manager_ != nullptr;
  LockContext lock_context;
  Status status = scoped ? meta_manager_.pools().LockKeys(layout, request.flat_keys, &lock_context)
      : meta_manager_.LockKeys(request.flat_keys, MetaManagerLockMode::kStableReadAllowAppend, &lock_context);
  if (!status.ok()) {
    return status;
  }

  FlatAccessDesc lookup;
  auto unlock = [&] { return scoped ? meta_manager_.pools().UnlockKeys(lock_context) : meta_manager_.UnlockKeys(lock_context); };
  status = scoped ? meta_manager_.pools().Lookup(request.flat_keys, lock_context, &lookup)
                  : meta_manager_.BatchLookup(request.flat_keys, lock_context, &lookup);
  if (!status.ok()) {
    (void)unlock();
    return status;
  }
  if (lookup.addresses.size() != request.flat_keys.size() ||
      lookup.lengths.size() != request.flat_keys.size() ||
      lookup.valid.size() != request.flat_keys.size()) {
    (void)unlock();
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

  const std::uint64_t id = next_table_id.fetch_add(1);
  if (id == 0) {
    (void)unlock();
    return Status::InvalidState("AccessTable handle space is exhausted");
  }
  try {
    auto table = std::make_unique<AccessTableState>();
    table->layout = layout;
    table->lock_context = lock_context;
    table->addresses = std::move(lookup.addresses);
    table->lengths = std::move(lookup.lengths);
    table->valid = std::move(lookup.valid);
    table->has_valid_entries = has_valid_entries;
    table->has_uniform_valid_length = has_uniform_valid_length;
    table->uniform_valid_length = uniform_valid_length;
    access_tables_.emplace(id, std::move(table));
  } catch (const std::bad_alloc&) {
    (void)unlock();
    return Status::OutOfMemory("AccessTable allocation failed");
  }
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
  view->layout = table->layout;
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
  const Status status = pool_manager_ ? meta_manager_.pools().UnlockKeys(table->lock_context)
                                      : meta_manager_.UnlockKeys(table->lock_context);
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
  AccessOperator* op = &access_operator_;
  if (pool_manager_) {
    if (!table) return Status::InvalidArgument("pool BatchGet requires a managed AccessTable");
    if (!request.layer.empty() &&
        (request.layer.owner != table->layout.layer.owner || request.layer.id != table->layout.layer.id))
      return Status::InvalidArgument("AccessTable belongs to another layer");
    LayerLayout layout;
    auto status = pool_manager_->Resolve(table->layout.layer, &layout);
    if (!status.ok()) return status;
    if (request.output_block_bytes != layout.spec.token_bytes ||
        (request.indices.device == DeviceKind::kNpu && request.tokens_per_block != layout.spec.tokens_per_block))
      return Status::InvalidArgument("BatchGet layout does not match pool");
    op = pool_manager_->Operator(layout.pool);
  } else if (!request.layer.empty()) return Status::InvalidArgument("layer supplied to legacy Runtime");
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
  if (!request.miss_count.has_elements<std::int32_t>() ||
      request.miss_count.bytes != sizeof(std::int32_t) ||
      request.miss_count.device != request.indices.device) {
    return Status::InvalidArgument(
        "miss_count must be a single int32 on the BatchGet device");
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

  if (request.indices.device == DeviceKind::kCpu) {
    const std::int32_t miss_count =
        *static_cast<const std::int32_t*>(request.miss_count.data);
    if (miss_count < 0 || static_cast<std::size_t>(miss_count) > request_count) {
      return Status::InvalidArgument(
          "miss_count must be between zero and the BatchGet request capacity");
    }
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
    view.layout = table->layout;
  }
  auto status = op->ExecuteBatchGet(BatchGetExecRequest{view, &request, execution});
  if (!status.ok() && pool_manager_ && request.indices.device == DeviceKind::kNpu)
    pool_manager_->Quarantine(table->layout.pool);
  return status;
}

Status RuntimeCore::BatchPut(const BatchPutRequest& request,
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
  if (pool_manager_) {
    LayerLayout layout;
    auto status = pool_manager_->Resolve(request.layer, &layout);
    if (!status.ok()) return status;
    const auto block_bytes = std::uint64_t{layout.spec.token_bytes} * layout.spec.tokens_per_block;
    for (auto length : request.lengths)
      if (length != block_bytes) return Status::InvalidArgument("pool Put requires complete layout blocks");
    status = meta_manager_.pools().PreparePut(*pool_manager_, layout, request.keys, request.lengths,
                                             &put_context, &put_desc);
    if (!status.ok()) return status;
    auto* op = pool_manager_->Operator(layout.pool);
    status = op->ExecuteBatchPut(BatchPutExecRequest{&request, &put_desc, execution});
    if (!status.ok()) {
      if (op->FailedPutIsQuiescent()) {
        auto rollback = meta_manager_.pools().AbortPut(*pool_manager_, put_context);
        if (!rollback.ok()) pool_manager_->Quarantine(layout.pool);
      } else pool_manager_->Quarantine(layout.pool);
      return status;
    }
    status = meta_manager_.pools().CommitPut(*pool_manager_, put_context);
    if (!status.ok()) pool_manager_->Quarantine(layout.pool);
    return status;
  }
  if (!request.layer.empty()) return Status::InvalidArgument("layer supplied to legacy Runtime");
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
  status = meta_manager_.CommitPut(put_context);
  if (!status.ok()) {
    return status;
  }
  return Status::OK();
}

Status RuntimeCore::BatchFree(const BatchFreeRequest& request) {
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (request.keys.empty()) {
    return Status::InvalidArgument("BatchFree requires at least one key");
  }
  if (pool_manager_) {
    LayerLayout layout;
    auto status = pool_manager_->Resolve(request.layer, &layout);
    if (!status.ok()) return status;
    return meta_manager_.pools().Free(*pool_manager_, layout, request.keys);
  }
  if (!request.layer.empty()) return Status::InvalidArgument("layer supplied to legacy Runtime");
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
