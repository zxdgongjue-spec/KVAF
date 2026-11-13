#include "runtime.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kvaf {

namespace {

std::uint64_t LegacyGetTraceNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
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
      ddr_allocator_(ddr_allocator) {
  if (const char* path = std::getenv("KVAF_PUT_DISPATCH_TRACE"); path && *path) {
    put_trace_path_ = path;
    put_trace_rows_.reserve(256);
  }
  if (!access_operator_.TracksGetCompletion()) {
    if (const char* path = std::getenv("KVAF_GET_DISPATCH_TRACE"); path && *path)
      legacy_get_trace_path_ = path;
  }
}

RuntimeCore::~RuntimeCore() {
  if (needs_cleanup_ && !Shutdown().ok()) {
    // The allocation owner must also retain its resources. Do not unpin keys
    // as an incidental consequence of destroying the public handle registry.
    auto* retained = new decltype(access_tables_)(std::move(access_tables_));
    (void)retained;
    if (pending_get_table_) (void)new std::shared_ptr<AccessTableState>(std::move(pending_get_table_));
    if (!graph_tables_.empty()) (void)new decltype(graph_tables_)(std::move(graph_tables_));
  }
  FlushLegacyGetTrace();
  FlushPutTrace();
}

void RuntimeCore::FlushLegacyGetTrace() {
  if (legacy_get_trace_rows_.empty()) return;
  std::ofstream output(legacy_get_trace_path_, std::ios::app);
  if (!output) {
    std::fprintf(stderr, "[KVAF] cannot write legacy Get trace: %s\n", legacy_get_trace_path_.c_str());
    return;
  }
  output.seekp(0, std::ios::end);
  if (output.tellp() == 0)
    output << "event,index,algorithm,path,total_ns,provider_ns,provider_prepare_ns,"
              "provider_submit_ns,provider_completion_ns,provider_cleanup_ns,"
              "aiv_capture_prepare_ns,aiv_capture_submit_ns,aiv_status_reset_ns,aiv_kernel_launch_ns\n";
  std::size_t index = 0;
  for (const auto& row : legacy_get_trace_rows_)
    output << "submit," << ++index << ",legacy,legacy," << row.total_ns << ',' << row.provider_ns
           << ',' << row.provider_detail.prepare_ns << ',' << row.provider_detail.submit_ns
           << ',' << row.provider_detail.completion_ns << ',' << row.provider_detail.cleanup_ns
           << ',' << row.provider_detail.aiv_capture_prepare_ns << ',' << row.provider_detail.aiv_capture_submit_ns
           << ',' << row.provider_detail.aiv_status_reset_ns << ',' << row.provider_detail.aiv_kernel_launch_ns << '\n';
  legacy_get_trace_rows_.clear();
}

void RuntimeCore::FlushPutTrace() {
  if (put_trace_rows_.empty()) return;
  std::ofstream output(put_trace_path_, std::ios::app);
  if (!output) {
    std::fprintf(stderr, "[KVAF] cannot write Put dispatch trace: %s\n", put_trace_path_.c_str());
    return;
  }
  output.seekp(0, std::ios::end);
  if (output.tellp() == 0)
    output << "event,index,transport,executor,path,result,total_ns,lock_ns,validation_ns,prepare_ns,"
              "metadata_prepare_ns,execute_ns,metadata_finalize_ns,acknowledge_ns,resolve_ns,"
              "dispatch_lock_ns,dispatch_resolve_ns,context_ns,describe_ns,acquire_ns,bind_ns,provider_ns,"
              "provider_prepare_ns,provider_registration_ns,provider_metadata_ns,provider_submit_ns,"
              "provider_completion_ns,provider_cleanup_ns,provider_other_ns\n";
  for (const auto& row : put_trace_rows_) {
    const auto& d = row.dispatch;
    const auto& p = row.provider;
    const char* transport = d.config ? (d.config->transport == Transport::Ubmem ? "ubmem" : "urma") : p.transport;
    const char* executor = d.config ? (d.config->executor == Executor::Aicpu ? "aicpu" : "aiv") : p.executor;
    const auto phases = p.prepare_ns + p.metadata_ns + p.submit_ns + p.completion_ns + p.cleanup_ns;
    output << row.event << ',' << ++put_trace_index_ << ',' << transport << ',' << executor << ','
           << row.path << ',' << row.result << ',' << row.total_ns << ',' << row.lock_ns << ','
           << row.validation_ns << ',' << row.prepare_ns << ',' << row.metadata_prepare_ns << ','
           << row.execute_ns << ',' << row.metadata_finalize_ns << ',' << row.acknowledge_ns << ','
           << row.resolve_ns << ',' << d.lock_ns << ',' << d.resolve_ns << ',' << d.context_ns << ','
           << d.describe_ns << ',' << d.acquire_ns << ',' << d.bind_ns << ',' << d.provider_ns << ','
           << p.prepare_ns << ',' << p.registration_ns << ',' << p.metadata_ns << ',' << p.submit_ns << ','
           << p.completion_ns << ',' << p.cleanup_ns << ','
           << (d.provider_ns > phases ? d.provider_ns - phases : 0) << '\n';
  }
  put_trace_rows_.clear();
}

Status RuntimeCore::Initialize(const RuntimeInitRequest& request) {
  std::unique_lock<std::shared_mutex> lifecycle(lifecycle_mutex_);
  if (initialized_) {
    return Status::InvalidState("Runtime has already been initialized");
  }
  if (needs_cleanup_) {
    return Status::InvalidState("Runtime still has resources to clean up");
  }
  if (!access_tables_.empty()) {
    return Status::InvalidState("Runtime has unreleased AccessTables");
  }

  RemoteRegionDesc remote_region;
  Status status = ddr_allocator_.Allocate(request, &remote_region);
  if (!status.ok()) {
    return status;
  }
  remote_region_ = remote_region;
  needs_cleanup_ = true;
  if (remote_region.length == 0 || remote_region.slot_size == 0 ||
      remote_region.length % remote_region.slot_size != 0) {
    lifecycle.unlock();
    (void)Shutdown();
    return Status::InvalidArgument("DDR allocator returned an invalid remote region");
  }

  status = access_operator_.Initialize(remote_region);
  if (!status.ok()) {
    lifecycle.unlock();
    (void)Shutdown();
    return status;
  }

  status = meta_manager_.RegisterRemoteRegion(remote_region);
  if (!status.ok()) {
    lifecycle.unlock();
    (void)Shutdown();
    return status;
  }

  initialized_ = true;
  return Status::OK();
}

Status RuntimeCore::Shutdown() {
  std::unique_lock<std::shared_mutex> lifecycle(lifecycle_mutex_);
  if (!needs_cleanup_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (!graph_tables_.empty()) return Status::Busy("release graph leases before shutdown");
  // Stop accepting work, but keep ownership until cleanup has succeeded.
  initialized_ = false;

  // Pending transfers may still reference the region and access tables.
  // Keep ownership intact if the operator cannot drain or close its session.
  const Status operator_status = access_operator_.Shutdown();
  FlushLegacyGetTrace();
  FlushPutTrace();
  if (!operator_status.ok()) return operator_status;

  pending_get_table_.reset();
  access_tables_.clear();
  if (pending_put_) {
    if (!pending_put_->metadata_finalized) {
      const auto aborted = meta_manager_.AbortPut(pending_put_->transaction);
      if (!aborted.ok()) return aborted;
    }
    pending_put_.reset();
  }
  Status first_error = Status::OK();

  const Status release_status = ddr_allocator_.Release(remote_region_);
  if (!release_status.ok()) {
    return first_error.ok() ? release_status : first_error;
  }

  needs_cleanup_ = false;
  remote_region_ = RemoteRegionDesc{};
  return first_error;
}

Status RuntimeCore::PrepareAccessTable(const AccessTablePrepareRequest& request,
                                       AccessTableHandle* handle) {
  std::shared_lock<std::shared_mutex> lifecycle(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(table_mutex_);
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
  // Public handles and in-flight requests independently own this pin.
  auto table = std::shared_ptr<AccessTableState>(new AccessTableState,
      [this](AccessTableState* state) {
        (void)meta_manager_.UnlockKeys(state->lock_context);
        delete state;
      });
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
  std::shared_lock<std::shared_mutex> lifecycle(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(table_mutex_);
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
  std::shared_lock<std::shared_mutex> lifecycle(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(table_mutex_);
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  AccessTableState* table = FindAccessTable(handle);
  if (table == nullptr) {
    return Status::NotFound("AccessTable handle was not found");
  }
  access_tables_.erase(handle.opaque);
  return Status::OK();
}

Status RuntimeCore::BatchGet(const BatchGetRequest& request,
                             const BatchGetExecutionOptions& execution) {
  const auto trace_started = legacy_get_trace_path_.empty() ? 0 : LegacyGetTraceNowNs();
  std::shared_lock<std::shared_mutex> lifecycle(lifecycle_mutex_);
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (execution.config && !access_operator_.TracksGetCompletion())
    return Status::NotImplemented("per-call execution requires an execution policy at construction");
  std::shared_ptr<AccessTableState> table_ref;
  {
    std::lock_guard<std::mutex> lock(table_mutex_);
    const auto it = access_tables_.find(request.access_table.opaque);
    if (it != access_tables_.end()) table_ref = it->second;
  }
  const AccessTableState* table = table_ref.get();
  if (request.access_table.opaque != 0 && !table)
    return Status::NotFound("AccessTable handle was not found");
  // Pin before any resource wait. This lock serializes only Get submissions.
  std::lock_guard<std::mutex> direction(get_mutex_);
  if (execution.graph_lease && !graph_tables_.count(execution.graph_lease))
    return Status::NotFound("Get graph lease not found");
  if (access_operator_.TracksGetCompletion() && !execution.graph_lease) {
    const auto drained = DrainGetLocked();
    if (!drained.ok()) return drained;
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
  }
  if (execution.graph_lease) {
    auto& pinned = graph_tables_.at(execution.graph_lease);
    if (pinned && pinned != table_ref)
      return Status::InvalidState("graph lease already pins another table");
    pinned = table_ref;
  } else if (access_operator_.TracksGetCompletion())
    pending_get_table_ = table_ref;
  GetProviderTiming provider_detail{};
  std::uint64_t provider_ns = 0;
  Status status;
  if (trace_started) {
    const auto provider_started = LegacyGetTraceNowNs();
    status = access_operator_.ExecuteBatchGet({view, &request, execution, &provider_detail});
    provider_ns = LegacyGetTraceNowNs() - provider_started;
  } else {
    status = access_operator_.ExecuteBatchGet({view, &request, execution});
  }
  if (status.ok() && execution.synchronize && !execution.graph_lease) pending_get_table_.reset();
  if (!status.ok() && access_operator_.TracksGetCompletion() && !execution.graph_lease) {
    // Preparation failures have no outstanding DMA. Failed submissions retain
    // their pin unless the provider can positively establish completion.
    (void)DrainGetLocked();
  }
  if (trace_started && status.ok())
    legacy_get_trace_rows_.push_back({LegacyGetTraceNowNs() - trace_started, provider_ns, provider_detail});
  return status;
}

Status RuntimeCore::BeginGetGraph(GetExecutionConfig config, const GetGraphOptions& request,
                                  std::uint64_t* id) {
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!initialized_) return Status::NotInitialized("Runtime is not initialized");
  auto s = DrainGetLocked();
  if (!s.ok()) return s;
  s = access_operator_.BeginGetGraph(config, request, id);
  if (s.ok()) graph_tables_.emplace(*id, nullptr);
  return s;
}
Status RuntimeCore::BeginGetOutput(GetExecutionConfig config, const GetOutputOptions& request,
                                    std::uint64_t* id) {
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!initialized_) return Status::NotInitialized("Runtime is not initialized");
  auto s = DrainGetLocked();
  return s.ok() ? access_operator_.BeginGetOutput(config, request, id) : s;
}
Status RuntimeCore::EndGetOutput(std::uint64_t id) {
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!initialized_) return Status::NotInitialized("Runtime is not initialized");
  auto s = access_operator_.EndGetOutput(id);
  if (s.ok()) pending_get_table_.reset();
  return s;
}
Status RuntimeCore::EndGetGraph(std::uint64_t id, void* stream, bool discard) {
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!graph_tables_.count(id)) return Status::NotFound("Get graph lease not found");
  auto s = access_operator_.EndGetGraph(id, stream, discard);
  if (s.ok()) graph_tables_.erase(id);
  return s;
}

Status RuntimeCore::BeginPutInput(PutExecutionConfig config, const PutInputOptions& request,
                                   std::uint64_t* id) {
  if (id) *id = 0;
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!initialized_) return Status::NotInitialized("Runtime is not initialized");
  if (pending_put_) return Status::Busy("Put transaction awaits completion recovery");
  return access_operator_.BeginPutInput(config, request, id);
}
Status RuntimeCore::EndPutInput(std::uint64_t id) {
  std::shared_lock<std::shared_mutex> life(lifecycle_mutex_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!initialized_) return Status::NotInitialized("Runtime is not initialized");
  if (pending_put_) return Status::Busy("Put transaction awaits completion recovery");
  return access_operator_.EndPutInput(id);
}

Status RuntimeCore::BatchPut(const BatchPutRequest& request,
                             BatchPutExecutionOptions execution) {
  const bool tracing = !put_trace_path_.empty();
  const auto started = tracing ? LegacyGetTraceNowNs() : 0;
  std::shared_lock<std::shared_mutex> lifecycle(lifecycle_mutex_);
  std::lock_guard<std::mutex> direction(put_mutex_);
  PutTraceRow row;
  row.path = !access_operator_.TracksGetCompletion() ? "legacy" : execution.input_lease ? "input" : "ordinary";
  row.dispatch.config = execution.config;
  if (tracing) {
    row.lock_ns = LegacyGetTraceNowNs() - started;
    execution.host_timing = &row.provider;
    execution.dispatch_timing = &row.dispatch;
  }
  // The trace is buffered only after the operation has finished. No CSV I/O
  // occurs in the measured path. Each phase also records early failures.
  auto measure = [&](std::uint64_t& elapsed, auto&& work) {
    if (!tracing) return work();
    const auto phase = LegacyGetTraceNowNs();
    auto result = work();
    elapsed += LegacyGetTraceNowNs() - phase;
    return result;
  };
  auto run = [&]() -> Status {
    auto validated = measure(row.validation_ns, [&]() -> Status {
      if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
      if (request.keys.empty() || request.input.data == nullptr || request.input.bytes == 0 ||
          request.keys.size() != request.input_offsets.size() || request.keys.size() != request.lengths.size())
        return Status::InvalidArgument("BatchPut keys, input_offsets, and lengths must be non-empty and equal in size");
      for (std::size_t index = 0; index < request.keys.size(); ++index) {
        const std::uint64_t offset = request.input_offsets[index];
        const std::uint64_t length = request.lengths[index];
        if (length == 0 || offset > request.input.bytes || length > request.input.bytes - offset)
          return Status::InvalidArgument("BatchPut input_offsets and lengths must describe bytes inside input");
      }
      if ((execution.config || execution.input_lease) && !access_operator_.TracksGetCompletion())
        return Status::NotImplemented("per-call execution requires an execution policy at construction");
      if (pending_put_) return Status::Busy("Put transaction awaits completion recovery");
      return Status::OK();
    });
    if (!validated.ok()) return validated;
    auto prepared = measure(row.prepare_ns, [&] { return access_operator_.PreparePut(request, execution); });
    if (!prepared.ok()) return prepared;
    // Allocate the completion record before reserving metadata; retain it until
    // data-plane completion, metadata finalization and provider acknowledgement.
    PendingPut pending;
    PutContext put_context;
    PutAccessDesc put_desc;
    auto status = measure(row.metadata_prepare_ns, [&] {
      return meta_manager_.PreparePut(request.keys, request.lengths, &put_context, &put_desc);
    });
    if (!status.ok()) return status;
    pending.transaction = put_context;
    pending.input_bound = execution.input_lease != 0;
    pending_put_.emplace(std::move(pending));
    status = measure(row.execute_ns, [&]() -> Status {
      const auto provider_started = tracing ? LegacyGetTraceNowNs() : 0;
      Status result;
      try {
        result = access_operator_.ExecuteBatchPut({&request, &put_desc, execution});
      } catch (const std::exception& error) {
        result = Status::CompletionUnknown(std::string("Put execution threw after preparation: ") + error.what());
      } catch (...) {
        result = Status::CompletionUnknown("Put execution threw after preparation");
      }
      if (tracing && !access_operator_.TracksGetCompletion())
        row.dispatch.provider_ns += LegacyGetTraceNowNs() - provider_started;
      return result;
    });
    if (!status.ok()) {
      if (status.code() == StatusCode::kCompletionUnknown) return status;
      pending_put_->completion = {PutOutcome::Failed, status};
    } else pending_put_->completion = {PutOutcome::Succeeded, Status::OK()};
    return FinalizePutLocked(tracing ? &row : nullptr);
  };
  const auto result = run();
  if (tracing) {
    row.total_ns = LegacyGetTraceNowNs() - started;
    row.result = static_cast<int>(result.code());
    put_trace_rows_.push_back(row);
  }
  return result;
}

Status RuntimeCore::FinalizePutLocked(PutTraceRow* timing) {
  auto& pending = *pending_put_;
  if (pending.completion.outcome == PutOutcome::Unknown)
    return Status::CompletionUnknown("Put data-plane completion remains unresolved");
  if (!pending.metadata_finalized) {
    const auto started = timing ? LegacyGetTraceNowNs() : 0;
    const auto status = pending.completion.outcome == PutOutcome::Succeeded
                            ? meta_manager_.CommitPut(pending.transaction)
                            : meta_manager_.AbortPut(pending.transaction);
    if (timing) timing->metadata_finalize_ns += LegacyGetTraceNowNs() - started;
    if (!status.ok()) return status;
    pending.metadata_finalized = true;
  }
  // An acknowledgement retry must not repeat CommitPut/AbortPut. Input leases
  // also remain pinned if either metadata finalization or acknowledgement fails.
  const auto started = timing ? LegacyGetTraceNowNs() : 0;
  const auto acknowledged = access_operator_.AcknowledgePut();
  if (timing) timing->acknowledge_ns += LegacyGetTraceNowNs() - started;
  if (!acknowledged.ok()) return acknowledged;
  const auto result = pending.completion.result;
  pending_put_.reset();
  return result;
}

Status RuntimeCore::RecoverPut(PutExecutionConfig config) {
  const bool tracing = !put_trace_path_.empty();
  const auto started = tracing ? LegacyGetTraceNowNs() : 0;
  std::shared_lock<std::shared_mutex> lifecycle(lifecycle_mutex_);
  std::lock_guard<std::mutex> direction(put_mutex_);
  PutTraceRow row;
  row.event = "recover";
  row.path = !access_operator_.TracksGetCompletion() ? "legacy"
             : pending_put_ && pending_put_->input_bound ? "input" : "ordinary";
  row.dispatch.config = config;
  if (tracing) row.lock_ns = LegacyGetTraceNowNs() - started;
  auto run = [&]() -> Status {
    if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
    const auto phase = tracing ? LegacyGetTraceNowNs() : 0;
    PutCompletion completion;
    // Never replay the Put, or recover a route different from the original.
    const auto resolved = pending_put_ ? access_operator_.ResolvePut(config, &completion)
                                       : access_operator_.Recover(config);
    if (tracing) row.resolve_ns = LegacyGetTraceNowNs() - phase;
    if (!resolved.ok() || !pending_put_) return resolved;
    if (completion.outcome == PutOutcome::Unknown)
      return Status::CompletionUnknown("Put data-plane completion remains unresolved");
    if ((completion.outcome == PutOutcome::Succeeded) != completion.result.ok())
      return Status::Internal("inconsistent Put completion result");
    if (pending_put_->completion.outcome == PutOutcome::Unknown)
      pending_put_->completion = completion;
    else if (pending_put_->completion.outcome != completion.outcome)
      return Status::Internal("Put completion changed after it was resolved");
    return FinalizePutLocked(tracing ? &row : nullptr);
  };
  auto result = run();
  if (tracing) {
    row.total_ns = LegacyGetTraceNowNs() - started;
    row.result = static_cast<int>(result.code());
    put_trace_rows_.push_back(row);
  }
  return result;
}

bool RuntimeCore::HasPendingPut() const {
  std::shared_lock<std::shared_mutex> lifecycle(lifecycle_mutex_);
  std::lock_guard<std::mutex> direction(put_mutex_);
  return pending_put_.has_value();
}

Status RuntimeCore::BatchFree(const BatchFreeRequest& request) {
  std::shared_lock<std::shared_mutex> lifecycle(lifecycle_mutex_);
  if (!initialized_) {
    return Status::NotInitialized("Runtime has not been initialized");
  }
  if (request.keys.empty()) {
    return Status::InvalidArgument("BatchFree requires at least one key");
  }
  return meta_manager_.BatchFree(request.keys);
}

Status RuntimeCore::DrainGetLocked() {
  auto status = access_operator_.DrainGet();
  if (status.ok()) pending_get_table_.reset();
  return status;
}
Status RuntimeCore::DrainGet() {
  std::shared_lock<std::shared_mutex> lifecycle(lifecycle_mutex_);
  std::lock_guard<std::mutex> direction(get_mutex_);
  if (!initialized_) return Status::NotInitialized("Runtime has not been initialized");
  return DrainGetLocked();
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
