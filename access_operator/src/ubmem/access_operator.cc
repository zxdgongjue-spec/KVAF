#include "ubmem/access_operator.h"
#include "meta_manager.h"
#include "get_host_timing.h"
#include <cstdio>

namespace kvaf::ubmem {
namespace {
std::uint64_t AddressOf(const void* ptr) { return reinterpret_cast<std::uintptr_t>(ptr); }
BatchArgs Args(Operation operation, void* local, std::size_t bytes, const MappedExecutionView& view) {
  BatchArgs a{};
  a.operation = operation;
  a.remote_base = view.common.region().canonical_base;
  a.remote_bytes = view.common.region().logical_bytes;
  a.mapped_base = view.memory.base;
  a.local_base = AddressOf(local);
  a.local_bytes = bytes;
  a.statuses = AddressOf(view.slot->statuses);
  a.blocks = view.kernel.cores;
  return a;
}
Status LaunchKernel(BatchArgs& args, aclrtStream stream, const MappedExecutionView& view,
                  UbmemInvocation& call, bool synchronize, GetProviderTiming* host_timing = nullptr) {
  const auto submit_started = GetHostTimingStart(host_timing);
  auto& state = *view.slot;

  args.profile = view.profiling && args.operation == Operation::kGet ? (view.profile_detail ? 2 : 1) : 0;
  call.profile_stream = nullptr;
  aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI graph = nullptr;
  if (!synchronize || view.managed_completion) {
    auto status = Acl("aclmdlRICaptureGetInfo",
        aclmdlRICaptureGetInfo(stream, &capture, &graph));
    if (!status.ok()) return status;
  }
  if (view.managed_completion && !call.graph_active &&
      capture == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE)
    return Status::InvalidState("call begin_get_graph before capturing a request-level Get");
  const auto size = view.kernel.cores * sizeof(CoreStatus);
  if (synchronize || view.managed_completion) {
    // Same ordering as KvafExecuteBatchPutSync. An unwritten report must
    // not look like a successful zero-row kernel.
    auto status = Acl("aclrtMemsetAsync(status)", aclrtMemsetAsync(state.statuses, size, 0xFF, size, stream));
    if (!status.ok()) return status;
    if (!call.graph_active) call.pending[stream] = ExecutionSlot::Pending::kStream;
  }
  aclrtLaunchKernelAttr attrs[4]{};
  attrs[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
  attrs[0].value.schemMode = 1;
  attrs[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US;
  attrs[1].value.timeoutUs.timeoutLow = 180000000U;
  attrs[2].id = ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE;
  attrs[2].value.engineType = ACL_RT_ENGINE_TYPE_AIV;
  attrs[3].id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
  attrs[3].value.dynUBufSize = kMteBufferBytes;
  aclrtLaunchKernelCfg cfg{}; cfg.numAttrs = 4; cfg.attrs = attrs;
  if (args.operation == Operation::kPut) call.put_issued = true;
  auto status = Acl("aclrtLaunchKernelWithHostArgs", aclrtLaunchKernelWithHostArgs(
      view.kernel.function, view.kernel.cores, stream, &cfg, &args, sizeof(args), nullptr, 0));
  if (!status.ok()) return status;
  GetHostTimingRecord(host_timing, &GetProviderTiming::submit_ns, submit_started);
  const auto completion_started = GetHostTimingStart(host_timing);
  if (call.graph_active) {
    call.graph_recorded = true;
    return status;
  }
  call.pending[stream] = ExecutionSlot::Pending::kStream;
  if (args.profile) call.profile_stream = stream;
  if (!synchronize) {
    // Ordinary events cannot be recorded during ACL Graph capture. The
    // graph owner completes replay on this stream before releasing it;
    // Shutdown retains the stream drain for captured work.
    if (capture == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) return Status::OK();
    status = Acl("aclrtRecordEvent", aclrtRecordEvent(state.event, stream));
    if (!status.ok()) {
      view.health->failed = true;
      // No event: drain now or leave Shutdown responsible for retrying.
      if (aclrtSynchronizeStream(stream) == ACL_SUCCESS) call.pending[stream] = ExecutionSlot::Pending::kNone;
      return status;
    }
    call.pending[stream] = ExecutionSlot::Pending::kEvent;
    return Status::OK();
  }
  // Keep the kernel, status D2H and completion wait on the caller's stream,
  // exactly as the existing synchronous BatchPut path does.
  status = Acl("aclrtMemcpyAsync(status)", aclrtMemcpyAsync(state.host_statuses, size,
      state.statuses, size, ACL_MEMCPY_DEVICE_TO_HOST, stream));
  if (status.ok()) status = Acl("aclrtSynchronizeStream", aclrtSynchronizeStream(stream));
  if (!status.ok()) {
    std::fprintf(stderr, "[KVAF_UBMEM] FAILED_LAUNCH op=%u rows=%u local_base=0x%llx local_bytes=%llu "
        "mapped_base=0x%llx remote_bytes=%llu metadata=0x%llx\n",
        static_cast<unsigned>(args.operation), args.capacity,
        static_cast<unsigned long long>(args.local_base),
        static_cast<unsigned long long>(args.local_bytes),
        static_cast<unsigned long long>(args.mapped_base),
        static_cast<unsigned long long>(args.remote_bytes),
        static_cast<unsigned long long>(args.put_offsets));
    view.health->failed = true;
    return status;
  }
  call.pending[stream] = ExecutionSlot::Pending::kNone;
  const auto* reports = static_cast<const CoreStatus*>(state.host_statuses);
  for (std::size_t i = 0; i < view.kernel.cores; ++i) {
    if (reports[i].result == UINT64_MAX) {
      return Status::Internal("UBmem kernel status was not written (still 0xFF)");
    }
    if (reports[i].result != 0) {
      std::fprintf(stderr, "[KVAF_UBMEM] KERNEL_ERROR core=%zu code=%llu request=%llu\n", i,
          static_cast<unsigned long long>(reports[i].result),
          static_cast<unsigned long long>(reports[i].failed_request));
      return Status::InvalidArgument("UBmem kernel rejected a request; see KERNEL_ERROR");
    }
    if (args.operation == Operation::kPut &&
        reports[i].completed != static_cast<std::uint64_t>(args.capacity) * (i + 1) / args.blocks -
            static_cast<std::uint64_t>(args.capacity) * i / args.blocks) {
      return Status::Internal("UBmem Put completion report does not cover the submitted rows");
    }
  }
  GetHostTimingRecord(host_timing, &GetProviderTiming::completion_ns, completion_started);
  return Status::OK();
}
}  // namespace

AccessOperator::AccessOperator(OperatorOptions options)
    : owned_resources_(std::make_unique<UbmemResources>(std::move(options))),
      resources_(owned_resources_.get()) {}
AccessOperator::~AccessOperator() = default;
std::unique_ptr<ExecutionResourceOwner> AccessOperator::TakeResources() { return std::move(owned_resources_); }
std::unique_ptr<ExecutionState> AccessOperator::CreateExecutionState() { return resources_->CreateState(); }
Status AccessOperator::BindExecutionState(ExecutionState* state) { return resources_->Bind(state); }
Status AccessOperator::Initialize(const RemoteRegionDesc& region) { return resources_->Initialize(region); }
Status AccessOperator::Shutdown() { return resources_->Close(); }
Status AccessOperator::Warmup(const ExecutionWarmup& request) { return resources_->Warmup(request); }
Status AccessOperator::PrepareGetGraph(const GetGraphOptions& options) {
  std::lock_guard<std::mutex> lock(resources_->mutex);
  if (resources_->invocation->graph_active) return Status::Busy("UBMEM graph lease already held");
  if (!options.stream || !options.output.valid() || options.output.device != DeviceKind::kNpu ||
      !options.request_capacity)
    return Status::InvalidArgument("graph preparation requires output, capacity and stream");
  aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI graph = nullptr;
  auto s = Acl("aclmdlRICaptureGetInfo",
               aclmdlRICaptureGetInfo(static_cast<aclrtStream>(options.stream), &capture, &graph));
  if (!s.ok()) return s;
  if (capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE)
    return Status::InvalidState("prepare graph before capture");
  ExecutionSlot* state = nullptr;
  s = resources_->Slot(static_cast<aclrtStream>(options.stream), &state);
  if (!s.ok()) return s;
  if (resources_->invocation->pending[static_cast<aclrtStream>(options.stream)] != ExecutionSlot::Pending::kNone)
    return Status::Busy("UBMEM work is still pending");
  resources_->invocation->graph_options = options;
  resources_->invocation->graph_active = true;
  resources_->invocation->graph_recorded = false;
  return Status::OK();
}
Status AccessOperator::FinishGetGraph(void* stream, bool discard) {
  std::lock_guard<std::mutex> lock(resources_->mutex);
  if (!resources_->invocation->graph_active) return Status::NotFound("UBMEM graph lease not found");
  if (!stream) return Status::InvalidArgument("graph release requires its last replay stream");
  aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI graph = nullptr;
  auto s = Acl("aclmdlRICaptureGetInfo",
               aclmdlRICaptureGetInfo(static_cast<aclrtStream>(stream), &capture, &graph));
  if (!s.ok()) return s;
  if (capture != ACL_MODEL_RI_CAPTURE_STATUS_NONE)
    return Status::InvalidState("finish capture before releasing graph");
  s = Acl("aclrtSynchronizeStream(graph)",
          aclrtSynchronizeStream(static_cast<aclrtStream>(stream)));
  if (!s.ok()) return s;
  if (resources_->invocation->graph_recorded && !discard) {
    auto& state = resources_->streams.at(static_cast<aclrtStream>(resources_->invocation->graph_options.stream));
    const auto bytes = resources_->options.cores * sizeof(CoreStatus);
    s = Acl("aclrtMemcpy(graph status)", aclrtMemcpy(state.host_statuses, bytes, state.statuses,
                                                     bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!s.ok()) return s;
    const auto* reports = static_cast<const CoreStatus*>(state.host_statuses);
    for (std::uint32_t i = 0; i < resources_->options.cores; ++i)
      if (reports[i].result)
        return Status::Internal(
            "graph result failed or was never replayed; inspect then release with "
            "discard_result=True");
  }
  resources_->invocation->graph_active = false;
  resources_->invocation->graph_recorded = false;
  resources_->invocation->graph_options = {};
  return Status::OK();
}
Status AccessOperator::DrainGet() {
  std::lock_guard<std::mutex> lock(resources_->mutex);
  for (auto& entry : resources_->streams) {
    auto& state = entry.second;
    if (resources_->invocation->pending[entry.first] == ExecutionSlot::Pending::kNone) continue;
    auto status = resources_->invocation->pending[entry.first] == ExecutionSlot::Pending::kEvent
        ? Acl("aclrtSynchronizeEvent", aclrtSynchronizeEvent(state.event))
        : Acl("aclrtSynchronizeStream", aclrtSynchronizeStream(entry.first));
    if (!status.ok()) return status;
    const auto size = resources_->options.cores * sizeof(CoreStatus);
    status = Acl("aclrtMemcpy(completion)", aclrtMemcpy(state.host_statuses, size,
        state.statuses, size, ACL_MEMCPY_DEVICE_TO_HOST));
    if (!status.ok()) return status;
    const auto* reports = static_cast<const CoreStatus*>(state.host_statuses);
    for (std::size_t i = 0; i < resources_->options.cores; ++i)
      if (reports[i].result != 0) return Status::Internal("UBMEM Get kernel reported an error");
    resources_->invocation->pending[entry.first] = ExecutionSlot::Pending::kNone;
  }
  return Status::OK();
}
Status AccessOperator::SetBatchGetKernelProfiling(bool enabled, bool detailed) {
  std::lock_guard<std::mutex> lock(resources_->mutex);
  if (!resources_->ready || resources_->health.failed) return Status::NotInitialized("UBmem operator is not initialized");
  resources_->profiling = enabled;
  resources_->profile_detail = enabled && detailed;
  resources_->invocation->profile_stream = nullptr;
  return Status::OK();
}
Status AccessOperator::CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>* profiles) const {
  std::lock_guard<std::mutex> lock(resources_->mutex);
  if (!profiles) return Status::InvalidArgument("profile output is null");
  if (!resources_->ready || resources_->health.failed || !resources_->profiling || !resources_->invocation->profile_stream)
    return Status::InvalidState("no profiled UBmem BatchGet launch");
  const auto& state = resources_->streams.at(resources_->invocation->profile_stream);
  std::vector<CoreStatus> reports(resources_->options.cores);
  // As for URMA profiling, the caller completes the stream before this snapshot.
  auto status = Acl("aclrtMemcpy(profile)", aclrtMemcpy(reports.data(), reports.size() * sizeof(CoreStatus),
      state.statuses, reports.size() * sizeof(CoreStatus), ACL_MEMCPY_DEVICE_TO_HOST));
  if (!status.ok()) return status;
  profiles->clear();
  for (const auto& report : reports) {
    if (report.result != 0 || report.profile_magic != kProfileMagic || report.finish_cycle < report.start_cycle) {
      profiles->clear();
      return Status::Internal("invalid UBmem BatchGet profile");
    }
    BatchGetKernelProfile profile{};
    profile.kernel_start_cycle = report.start_cycle;
    profile.kernel_finish_cycle = report.finish_cycle;
    profile.active_requests = report.completed;
    profile.ubmem_profile_detail = report.profile_detail != 0;
    if (profile.ubmem_profile_detail) {
      profile.ubmem_setup_cycles = report.setup_cycles;
      profile.ubmem_address_cycles = report.address_cycles;
      profile.ubmem_submit_cycles = report.submit_cycles;
      profile.ubmem_drain_cycles = report.drain_cycles;
    }
    profiles->push_back(profile);
  }
  return Status::OK();
}
Status AccessOperator::ResolvePut(PutCompletion* out) {
  if (!out) return Status::InvalidArgument("null Put completion output");
  std::lock_guard<std::mutex> lock(resources_->mutex);
  auto& call = *resources_->invocation;
  if (call.put_completion.outcome == PutOutcome::Unknown && call.put_stream) {
    if (aclrtSynchronizeStream(call.put_stream) != ACL_SUCCESS)
      return Status::CompletionUnknown("UBMEM Put stream still pending");
    bool success = false;
    if (call.put_issued) {
      auto& workspace = resources_->streams.at(call.put_stream);
      const auto bytes = resources_->options.cores * sizeof(CoreStatus);
      if (aclrtMemcpy(workspace.host_statuses, bytes, workspace.statuses, bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)
        return Status::CompletionUnknown("UBMEM Put completion report unavailable");
      const auto* reports = static_cast<const CoreStatus*>(workspace.host_statuses);
      success = true;
      for (std::uint32_t i = 0; i < resources_->options.cores; ++i) {
        const auto expected = call.put_rows * (i + 1) / resources_->options.cores -
                              call.put_rows * i / resources_->options.cores;
        success &= reports[i].result == 0 && reports[i].completed == expected;
      }
    }
    // UBMEM has no posted remote WRs: the completed stream fences local MTE.
    call.pending[call.put_stream] = ExecutionSlot::Pending::kNone;
    auto failure = call.put_completion.result;
    if (failure.ok()) failure = Status::Internal("UBMEM Put did not complete all rows");
    call.put_completion = {success ? PutOutcome::Succeeded : PutOutcome::Failed,
        success ? Status::OK() : failure};
    resources_->ready = true;
    resources_->health.failed = false;
    resources_->views.Initialize(resources_->remote);
  }
  *out = call.put_completion;
  return Status::OK();
}
Status AccessOperator::AcknowledgePut() {
  if (resources_->invocation->put_completion.outcome == PutOutcome::Unknown)
    return Status::CompletionUnknown("cannot acknowledge unresolved UBMEM Put");
  resources_->invocation->put_completion = {};
  return Status::OK();
}
Status UbmemGetOperator::Launch(const BatchGetExecRequest& exec, const UbmemGetView& view,
                                 UbmemInvocation& call) {
  if (!view.execution_.common.Validate(&call, ExecutionDirection::Get).ok() ||
      !view.execution_.health || view.execution_.health->failed || !view.execution_.slot || !view.execution_.kernel.function || !exec.request ||
      call.prepared_stream != static_cast<aclrtStream>(exec.execution.stream))
    return Status::InvalidArgument("invalid UBMEM Get resource view or invocation");

  // RuntimeCore owns buffer/layout validation. This adapter only restricts
  // the device and checks narrowing to its kernel ABI.
  const auto& r = *exec.request;
  if (call.graph_active) {
    if (r.output.data != call.graph_options.output.data ||
        r.output.bytes != call.graph_options.output.bytes ||
        exec.execution.stream != call.graph_options.stream ||
        r.indices.bytes / 4 > call.graph_options.request_capacity || exec.execution.synchronize)
      return Status::InvalidArgument("captured request differs from prepared graph lease");
    aclmdlRICaptureStatus capture = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI graph = nullptr;
    auto s = Acl(
        "aclmdlRICaptureGetInfo",
        aclmdlRICaptureGetInfo(static_cast<aclrtStream>(exec.execution.stream), &capture, &graph));
    if (!s.ok()) return s;
    if (capture != ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE)
      return Status::InvalidState("graph submission requires active capture");
  }
  if (r.output.device != DeviceKind::kNpu ||
      r.output_block_bytes > UINT32_MAX || r.tokens_per_block > UINT32_MAX ||
      r.indices.bytes / 4 > INT32_MAX || r.device_access_table.valid.bytes > UINT32_MAX) {
    return Status::InvalidArgument("invalid UBmem BatchGet device buffers/layout");
  }
  auto stream = static_cast<aclrtStream>(exec.execution.stream);
  auto a = Args(Operation::kGet, r.output.data, r.output.bytes, view.execution_);
  a.addresses = AddressOf(r.device_access_table.addresses.data);
  a.valid = AddressOf(r.device_access_table.valid.data);
  a.indices = AddressOf(r.indices.data);
  a.offsets = AddressOf(r.offsets.data);
  a.output_indices = AddressOf(r.output_indices.data);
  a.miss_count = AddressOf(r.miss_count.data);
  a.capacity = r.indices.bytes / 4;
  a.table_entries = r.device_access_table.valid.bytes;
  a.token_bytes = r.output_block_bytes;
  a.tokens_per_block = r.tokens_per_block;
  return LaunchKernel(a, stream, view.execution_, call, exec.execution.synchronize, exec.host_timing);
}
Status AccessOperator::ExecuteBatchGet(const BatchGetExecRequest& exec) {
  std::lock_guard<std::mutex> lock(resources_->mutex);
  UbmemGetView view;
  const auto prepare_started = GetHostTimingStart(exec.host_timing);
  auto s = resources_->PrepareGetResources(static_cast<aclrtStream>(exec.execution.stream), &view);
  GetHostTimingRecord(exec.host_timing, &GetProviderTiming::prepare_ns, prepare_started);
  return s.ok() ? UbmemGetOperator::Launch(exec, view, *resources_->invocation) : s;
}
Status UbmemPutOperator::Launch(const BatchPutExecRequest& exec, const UbmemPutView& view,
                                 UbmemInvocation& call) {
  if (!view.execution_.common.Validate(&call, ExecutionDirection::Put).ok() ||
      !view.execution_.health || view.execution_.health->failed || !view.execution_.slot || !view.execution_.kernel.function || !exec.request ||
      !exec.put_desc ||
      call.prepared_stream != static_cast<aclrtStream>(exec.execution.stream))
    return Status::InvalidArgument("invalid UBMEM Put resource view or invocation");

  const auto& r = *exec.request;
  const auto rows = r.keys.size();
  if (r.input.device != DeviceKind::kNpu || rows > UINT32_MAX ||
      r.input_offsets.size() != rows || r.lengths.size() != rows ||
      exec.put_desc->entries.size() != rows) {
    return Status::InvalidArgument("invalid UBmem BatchPut metadata");
  }
  // RuntimeCore checked source ranges; MetaManager allocated these destination
  // ranges in the region supplied to Initialize, in request order.
  std::vector<std::uint64_t> metadata(rows * 3);
  for (std::size_t i = 0; i < rows; ++i) {
    metadata[i] = r.input_offsets[i]; metadata[rows + i] = r.lengths[i];
    metadata[2 * rows + i] = exec.put_desc->entries[i].address;
  }
  auto* state = view.execution_.slot;
  auto stream = static_cast<aclrtStream>(exec.execution.stream);
  const auto bytes = metadata.size() * sizeof(std::uint64_t);
  if (state->capacity < bytes) return Status::InvalidArgument("UBMEM Put resource capacity exceeded");
  auto status = Acl("aclrtMemcpy(metadata)", aclrtMemcpy(state->metadata, bytes, metadata.data(), bytes,
                                                  ACL_MEMCPY_HOST_TO_DEVICE));
  if (!status.ok()) return status;
  auto a = Args(Operation::kPut, const_cast<void*>(r.input.data), r.input.bytes, view.execution_);
  a.capacity = rows;
  call.put_rows = rows;
  call.put_stream = stream;
  a.put_offsets = AddressOf(state->metadata);
  a.put_lengths = a.put_offsets + rows * 8;
  a.put_destinations = a.put_lengths + rows * 8;
  return LaunchKernel(a, stream, view.execution_, call, true);
}
Status AccessOperator::ExecuteBatchPut(const BatchPutExecRequest& exec) {
  std::lock_guard<std::mutex> lock(resources_->mutex);
  auto& call = *resources_->invocation;
  call.put_completion = {};
  call.put_stream = nullptr;
  call.put_issued = false;
  UbmemPutView view;
  auto result = !exec.request || exec.request->keys.size() > UINT32_MAX
      ? Status::InvalidArgument("invalid UBMEM Put request capacity")
      : resources_->PreparePutResources(static_cast<aclrtStream>(exec.execution.stream),
                               static_cast<std::uint32_t>(exec.request->keys.size()), &view);
  if (result.ok()) result = UbmemPutOperator::Launch(exec, view, call);
  if (result.ok()) call.put_completion = {PutOutcome::Succeeded, result};
  else if (!call.put_stream) call.put_completion = {PutOutcome::Failed, result};
  else call.put_completion.result = result;
  return result;
}
}  // namespace kvaf::ubmem
