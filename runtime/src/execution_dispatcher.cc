#include "execution_dispatcher.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <vector>

namespace kvaf {
class GetDispatchTrace {
 public:
  struct Row {
    const char* event{};
    const char* algorithm{};
    const char* path{};
    std::uint64_t lock_ns{}, resolve_ns{}, previous_drain_ns{}, context_ns{};
    std::uint64_t output_check_ns{};
    std::uint64_t describe_ns{}, acquire_ns{}, bind_ns{};
    GetLaunchTiming launch;
    std::uint64_t context_release_ns{}, drain_provider_ns{}, drain_release_ns{}, total_ns{};
  };
  explicit GetDispatchTrace(std::string path) : path_(std::move(path)) { rows_.reserve(256); }
  ~GetDispatchTrace() { Flush(); }
  void Flush() {
    if (rows_.empty()) return;
    std::ofstream output(path_, std::ios::app);
    if (!output) {
      std::fprintf(stderr, "[KVAF] cannot write Get dispatch trace: %s\n", path_.c_str());
      return;
    }
    output.seekp(0, std::ios::end);
    if (output.tellp() == 0)
      output << "event,index,algorithm,path,lock_ns,resolve_ns,output_check_ns,previous_drain_ns,"
                "context_ns,describe_ns,acquire_ns,bind_ns,requirements_ns,warmup_ns,provider_ns,"
                "provider_prepare_ns,provider_submit_ns,provider_completion_ns,provider_cleanup_ns,"
                "aiv_capture_prepare_ns,aiv_capture_submit_ns,aiv_status_reset_ns,aiv_kernel_launch_ns,context_release_ns,"
                "drain_provider_ns,drain_release_ns,total_ns\n";
    for (const auto& row : rows_)
      output << row.event << ',' << ++next_index_ << ',' << (row.algorithm ? row.algorithm : "")
             << ',' << (row.path ? row.path : "") << ',' << row.lock_ns << ',' << row.resolve_ns
             << ',' << row.output_check_ns << ',' << row.previous_drain_ns << ',' << row.context_ns << ',' << row.describe_ns
             << ',' << row.acquire_ns << ',' << row.bind_ns << ',' << row.launch.requirements_ns
             << ',' << row.launch.warmup_ns << ',' << row.launch.provider_ns << ','
             << row.launch.provider_detail.prepare_ns << ',' << row.launch.provider_detail.submit_ns
             << ',' << row.launch.provider_detail.completion_ns << ',' << row.launch.provider_detail.cleanup_ns << ','
             << row.launch.provider_detail.aiv_capture_prepare_ns << ',' << row.launch.provider_detail.aiv_capture_submit_ns
             << ',' << row.launch.provider_detail.aiv_status_reset_ns << ',' << row.launch.provider_detail.aiv_kernel_launch_ns << ','
             << row.context_release_ns << ',' << row.drain_provider_ns << ','
             << row.drain_release_ns << ',' << row.total_ns << '\n';
    rows_.clear();
  }
  void Record(Row row) { rows_.push_back(row); }
 private:
  std::string path_;
  std::vector<Row> rows_;
  std::uint64_t next_index_{};
};
namespace {
const char* GetAlgorithmLabel(GetExecutionConfig c) {
  if (c.transport == Transport::Ubmem) return "ubmem";
  if (c.executor == Executor::Aicpu)
    return c.algorithm == GetAlgorithm::Scatter ? "scatter" : "gather";
  return "aiv_direct";
}
std::uint64_t TraceNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
Status LaunchGet(GetExecutionContext& context, GetLaunchTiming* timing = nullptr) {
  try {
    return context.lease.view().Launch(context.invocation(), timing);
  } catch (const std::exception& e) {
    return Status::CompletionUnknown(std::string("provider Get threw after submission: ") + e.what());
  } catch (...) {
    return Status::CompletionUnknown("provider Get threw after submission");
  }
}
}  // namespace
ExecutionDispatcher::ExecutionDispatcher(ExecutionPolicy policy) : policy_(std::move(policy)) {
  policy_status_ = NormalizeExecutionPolicy(&policy_);
  if (const char* path = std::getenv("KVAF_GET_DISPATCH_TRACE"); path && *path)
    get_trace_ = std::make_unique<GetDispatchTrace>(path);
}
ExecutionDispatcher::~ExecutionDispatcher() {
  if (!Shutdown().ok()) {
    // Retain invocation state as well as the resource lease after unsafe close.
    (void)pending_put_.release();
    (void)pending_get_.release();
    (void)graph_context_.release();
    (void)output_context_.release();
  }
}
Status ExecutionDispatcher::RegisterPut(PutExecutionConfig c, Factory f) {
  std::unique_lock<std::shared_mutex> lock(lifecycle_);
  if (ready_ || region_.base) return Status::InvalidState("register providers before initialization");
  auto s = ValidateExecution(policy_.transport_mode, c);
  return s.ok() ? resources_.Register(c, std::move(f)) : s;
}
Status ExecutionDispatcher::RegisterGet(GetExecutionConfig c, Factory f) {
  std::unique_lock<std::shared_mutex> lock(lifecycle_);
  if (ready_ || region_.base) return Status::InvalidState("register providers before initialization");
  auto s = ValidateExecution(policy_.transport_mode, c);
  return s.ok() ? resources_.Register(c, std::move(f)) : s;
}
Status ExecutionDispatcher::Initialize(const RemoteRegionDesc& region) {
  std::unique_lock<std::shared_mutex> lock(lifecycle_);
  if (!policy_status_.ok()) return policy_status_;
  if (ready_ || region_.base) return Status::InvalidState("dispatcher requires shutdown before initialize");
  for (auto c : *policy_.enabled_put_configs)
    if (!resources_.Contains(c)) return Status::NotImplemented("enabled Put provider is not built or registered");
  for (auto c : *policy_.enabled_get_configs)
    if (!resources_.Contains(c))
      return Status::NotImplemented("enabled Get provider is not built or registered (AIV Gather/Scatter reserved)");
  auto s = resources_.Initialize(region);
  if (s.ok()) { region_ = region; ready_ = true; }
  return s;
}
Status ExecutionDispatcher::PreparePut(const BatchPutExecutionOptions& options) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (pending_put_) return Status::InvalidState("Put awaits completion/transaction acknowledgement");
  PutExecutionConfig c;
  auto s = ResolveExecution(policy_, options.config, &c);
  PutResourceLease lease;
  return s.ok() ? resources_.AcquirePut({c}, &lease) : s;
}
Status ExecutionDispatcher::PreparePut(const BatchPutRequest& request,
                                       const BatchPutExecutionOptions& options) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (pending_put_) return Status::InvalidState("Put awaits completion/transaction acknowledgement");
  PutExecutionConfig c;
  auto s = ResolveExecution(policy_, options.config, &c);
  PutResourceLease lease;
  PutResourceRequirements requirements;
  if (s.ok()) s = resources_.Describe(c, request, &requirements);
  if (s.ok()) s = resources_.AcquirePut(requirements, &lease);
  return s.ok() ? lease.view().Prepare(request, options) : s;
}
Status ExecutionDispatcher::Recover(PutExecutionConfig c) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (pending_put_) return Status::Busy("resolve pending Put through Runtime before resource recovery");
  PutExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  return s.ok() ? resources_.Recover(resolved) : s;
}
Status ExecutionDispatcher::Recover(GetExecutionConfig c) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (output_id_) return Status::Busy("release Get output lease before recovery");
  GetExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  if (s.ok()) s = DrainGetLocked();
  return s.ok() ? resources_.Recover(resolved) : s;
}
Status ExecutionDispatcher::ExecuteBatchPut(const BatchPutExecRequest& request) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (pending_put_) return Status::InvalidState("Put awaits completion/transaction acknowledgement");
  if (!request.request || !request.put_desc) return Status::InvalidArgument("missing Put request");
  auto context = std::make_unique<PutExecutionContext>();
  auto s = ResolveExecution(policy_, request.execution.config, &context->config);
  if (!s.ok()) return s;
  try {
    context->request = *request.request;
    context->destination = *request.put_desc;
    context->options = request.execution;
    context->options.config = context->config;
    PutResourceRequirements requirements;
    s = resources_.Describe(context->config, context->request, &requirements);
    if (s.ok()) s = resources_.AcquirePut(requirements, &context->lease);
    if (s.ok()) s = context->Bind();
  } catch (const std::exception& e) {
    s = Status::Internal(std::string("Put preparation failed before submission: ") + e.what());
  } catch (...) {
    s = Status::Internal("Put preparation failed before submission");
  }
  if (!s.ok()) {
    context->completion = {PutOutcome::Failed, s};
    pending_put_ = std::move(context);
    return s;
  }
  pending_put_ = std::move(context);
  pending_put_->launched = true;
  try {
    s = pending_put_->lease.view().Launch(pending_put_->invocation());
  } catch (const std::exception& e) {
    s = Status::CompletionUnknown(std::string("provider Put threw after submission: ") + e.what());
  } catch (...) {
    s = Status::CompletionUnknown("provider Put threw after submission");
  }
  if (s.ok()) pending_put_->completion = {PutOutcome::Succeeded, Status::OK()};
  else pending_put_->completion.result = s;
  return s.ok() ? s : Status::CompletionUnknown(s.message());
}
Status ExecutionDispatcher::ResolvePut(PutExecutionConfig c, PutCompletion* out) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!out) return Status::InvalidArgument("null Put completion output");
  if (!pending_put_) return Status::NotFound("no pending Put execution");
  if (!(pending_put_->config == c)) return Status::InvalidArgument("recovery configuration differs from original Put");
  if (pending_put_->completion.outcome == PutOutcome::Unknown) {
    PutCompletion result;
    auto s = pending_put_->lease.view().Resolve(&result);
    if (!s.ok()) return s;
    if ((result.outcome == PutOutcome::Succeeded && !result.result.ok()) ||
        (result.outcome == PutOutcome::Failed && result.result.ok()))
      return Status::Internal("inconsistent provider Put completion");
    pending_put_->completion = result;
  }
  *out = pending_put_->completion;
  return Status::OK();
}
Status ExecutionDispatcher::AcknowledgePut() {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!pending_put_) return Status::OK();
  if (pending_put_->completion.outcome == PutOutcome::Unknown)
    return Status::Busy("cannot acknowledge unresolved Put");
  auto s = pending_put_->launched ? pending_put_->lease.view().Acknowledge() : Status::OK();
  if (s.ok()) pending_put_.reset();
  return s;
}
Status ExecutionDispatcher::DrainGetLocked() {
  if (graph_id_) return Status::Busy("Get graph lease owns this direction");
  if (output_context_) return output_context_->lease.view().Drain();
  if (!pending_get_) return Status::OK();
  GetDispatchTrace::Row row{};
  if (get_trace_) {
    row.algorithm = GetAlgorithmLabel(pending_get_->config);
    row.path = "ordinary";
  }
  const auto started = get_trace_ ? TraceNowNs() : 0;
  auto s = pending_get_->cleanup_only ? resources_.Close(pending_get_->lease)
                                      : pending_get_->lease.view().Drain();
  const auto drained = get_trace_ ? TraceNowNs() : 0;
  if (s.ok()) pending_get_.reset();
  if (get_trace_) {
    row.event = "drain";
    row.drain_provider_ns = drained - started;
    row.drain_release_ns = TraceNowNs() - drained;
    row.total_ns = TraceNowNs() - started;
    get_trace_->Record(row);
  }
  return s;
}
Status ExecutionDispatcher::ExecuteBatchGet(const BatchGetExecRequest& request) {
  const auto started = get_trace_ ? TraceNowNs() : 0;
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  const auto locked = get_trace_ ? TraceNowNs() : 0;
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (!request.request) return Status::InvalidArgument("missing Get request");
  GetExecutionConfig c;
  auto s = ResolveExecution(policy_, request.execution.config, &c);
  if (!s.ok()) return s;
  const auto resolved = get_trace_ ? TraceNowNs() : 0;
  if (output_id_ || request.execution.output_lease) {
    if (!output_id_ || request.execution.output_lease != output_id_ ||
        request.execution.graph_lease || !(c == output_context_->config))
      return Status::Busy("Get output lease/configuration mismatch");
    const auto& r = *request.request;
    if (r.output.data != output_options_.output.data || r.output.bytes != output_options_.output.bytes ||
        r.output.device != output_options_.output.device ||
        r.indices.bytes % sizeof(std::int32_t) ||
        r.indices.bytes / sizeof(std::int32_t) > output_options_.request_capacity)
      return Status::InvalidArgument("request differs from bound Get output/capacity");
    GetDispatchTrace::Row row{};
    if (get_trace_) {
      row.event = "submit";
      row.algorithm = GetAlgorithmLabel(c);
      row.path = "output";
      row.lock_ns = locked - started;
      row.resolve_ns = resolved - locked;
      row.output_check_ns = TraceNowNs() - resolved;
    }
    auto phase = get_trace_ ? TraceNowNs() : 0;
    s = DrainGetLocked();
    if (!s.ok()) return s;
    if (get_trace_) row.previous_drain_ns = TraceNowNs() - phase;
    phase = get_trace_ ? TraceNowNs() : 0;
    output_context_->request = r;
    output_context_->table = request.host_access_table;
    output_context_->options = request.execution;
    output_context_->options.config = c;
    if (get_trace_) row.context_ns = TraceNowNs() - phase;
    s = LaunchGet(*output_context_, get_trace_ ? &row.launch : nullptr);
    if (get_trace_) {
      row.total_ns = TraceNowNs() - started;
      get_trace_->Record(row);
    }
    return s;
  }
  if (graph_id_ || request.execution.graph_lease) {
    if (!graph_id_ || request.execution.graph_lease != graph_id_ || !(c == graph_context_->config))
      return Status::Busy("Get graph lease/configuration mismatch");
    if (graph_captured_ || request.execution.synchronize || request.execution.profile)
      return Status::InvalidState("one asynchronous Get capture is allowed per lease");
    graph_context_->request = *request.request;
    graph_context_->table = request.host_access_table;
    graph_context_->options = request.execution;
    graph_context_->options.config = c;
    graph_captured_ = true;
    return LaunchGet(*graph_context_);
  }
  GetDispatchTrace::Row row{};
  if (get_trace_) {
    row.event = "submit";
    row.algorithm = GetAlgorithmLabel(c);
    row.path = "ordinary";
    row.lock_ns = locked - started;
    row.resolve_ns = resolved - locked;
  }
  auto phase = get_trace_ ? TraceNowNs() : 0;
  s = DrainGetLocked();
  if (!s.ok()) return s;
  if (get_trace_) row.previous_drain_ns = TraceNowNs() - phase;
  phase = get_trace_ ? TraceNowNs() : 0;
  auto context = std::make_unique<GetExecutionContext>();
  context->config = c;
  context->request = *request.request;
  context->table = request.host_access_table;
  context->options = request.execution;
  context->options.config = c;
  if (get_trace_) row.context_ns = TraceNowNs() - phase;
  GetResourceRequirements requirements;
  phase = get_trace_ ? TraceNowNs() : 0;
  s = resources_.Describe(c, context->request, &requirements);
  if (get_trace_) row.describe_ns = TraceNowNs() - phase;
  phase = get_trace_ ? TraceNowNs() : 0;
  if (s.ok()) s = resources_.AcquireGet(requirements, &context->lease);
  if (get_trace_) row.acquire_ns = TraceNowNs() - phase;
  phase = get_trace_ ? TraceNowNs() : 0;
  if (s.ok()) s = context->Bind();
  if (get_trace_) row.bind_ns = TraceNowNs() - phase;
  if (!s.ok()) return s;
  pending_get_ = std::move(context);
  s = LaunchGet(*pending_get_, get_trace_ ? &row.launch : nullptr);
  phase = get_trace_ ? TraceNowNs() : 0;
  if (s.ok() && request.execution.synchronize) pending_get_.reset();
  if (get_trace_) {
    row.context_release_ns = TraceNowNs() - phase;
    row.total_ns = TraceNowNs() - started;
    get_trace_->Record(row);
  }
  return s;
}
Status ExecutionDispatcher::DrainGet() {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  return DrainGetLocked();
}
Status ExecutionDispatcher::Warmup(PutExecutionConfig c, const ExecutionWarmup& r) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (pending_put_) return Status::Busy("Put awaits completion/transaction acknowledgement");
  PutExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  PutResourceLease lease;
  if (s.ok()) s = resources_.AcquirePut({resolved, r.request_capacity}, &lease);
  return s.ok() ? lease.view().Warmup(r) : s;
}
Status ExecutionDispatcher::Warmup(GetExecutionConfig c, const ExecutionWarmup& r) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (output_id_) return Status::Busy("release Get output lease before warmup");
  GetExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  if (s.ok()) s = DrainGetLocked();
  GetResourceLease lease;
  if (s.ok()) s = resources_.AcquireGet({resolved, r.request_capacity}, &lease);
  return s.ok() ? lease.view().Warmup(r) : s;
}
Status ExecutionDispatcher::BeginGetGraph(GetExecutionConfig c, const GetGraphOptions& r, std::uint64_t* id) {
  if (!id) return Status::InvalidArgument("null graph lease output");
  *id = 0;
  if (!r.request_capacity) return Status::InvalidArgument("graph capacity must be positive");
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (output_id_) return Status::Busy("release Get output lease before graph capture");
  if (c.executor != Executor::Aiv || c.algorithm != GetAlgorithm::Direct)
    return Status::NotImplemented("only AIV Direct Get supports graph leases");
  auto context = std::make_unique<GetExecutionContext>();
  auto s = ResolveExecution(policy_, c, &context->config);
  if (s.ok()) s = DrainGetLocked();
  if (s.ok()) s = resources_.AcquireGet({c, r.request_capacity}, &context->lease);
  if (s.ok()) s = context->Bind();
  if (!s.ok()) return s;
  if (!next_graph_id_) return Status::InvalidState("graph lease ID exhausted");
  try {
    s = context->lease.view().PrepareGraph(r);
  } catch (const std::exception& e) {
    s = Status::Internal(std::string("graph preparation failed: ") + e.what());
  } catch (...) {
    s = Status::Internal("graph preparation failed");
  }
  if (!s.ok()) {
    context->cleanup_only = true;
    const auto closed = resources_.Close(context->lease);
    if (!closed.ok()) pending_get_ = std::move(context);
    return s;
  }
  graph_id_ = next_graph_id_++;
  graph_context_ = std::move(context);
  graph_captured_ = false;
  *id = graph_id_;
  return s;
}
Status ExecutionDispatcher::EndGetGraph(std::uint64_t id, void* stream, bool discard) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!id || id != graph_id_) return Status::NotFound("Get graph lease not found");
  auto s = graph_context_->lease.view().FinishGraph(stream, discard);
  if (s.ok()) { graph_id_ = 0; graph_context_.reset(); graph_captured_ = false; }
  return s;
}
Status ExecutionDispatcher::BeginGetOutput(GetExecutionConfig c, const GetOutputOptions& r,
                                            std::uint64_t* id) {
  if (!id) return Status::InvalidArgument("null output lease ID");
  *id = 0;
  if (!r.output.valid() || r.output.device != DeviceKind::kNpu || !r.stream || !r.request_capacity)
    return Status::InvalidArgument("output binding requires NPU output, stream and capacity");
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (output_id_) return Status::Busy("Get output lease already exists");
  if (c.transport != Transport::Urma ||
      !((c.executor == Executor::Aiv && c.algorithm == GetAlgorithm::Direct) ||
        (c.executor == Executor::Aicpu && c.algorithm == GetAlgorithm::Gather)))
    return Status::NotImplemented("output binding supports URMA AIV Direct and AICPU Gather");
  auto context = std::make_unique<GetExecutionContext>();
  auto s = ResolveExecution(policy_, c, &context->config);
  if (s.ok()) s = DrainGetLocked();
  if (s.ok()) s = resources_.AcquireGet({c, r.request_capacity}, &context->lease);
  if (s.ok()) s = context->Bind();
  if (!s.ok()) return s;
  if (!next_output_id_) return Status::InvalidState("output lease ID exhausted");
  try {
    s = context->lease.view().PrepareOutput(r);
  } catch (const std::exception& e) {
    s = Status::Internal(std::string("output preparation failed: ") + e.what());
  } catch (...) {
    s = Status::Internal("output preparation failed");
  }
  if (!s.ok()) {
    context->cleanup_only = true;
    if (!resources_.Close(context->lease).ok()) pending_get_ = std::move(context);
    return s;
  }
  output_options_ = r;
  output_id_ = next_output_id_++;
  output_context_ = std::move(context);
  *id = output_id_;
  return Status::OK();
}
Status ExecutionDispatcher::FinishOutputLocked() {
  if (!output_context_) return Status::OK();
  auto s = output_context_->lease.view().Drain();
  if (s.ok()) s = output_context_->lease.view().FinishOutput();
  if (s.ok()) { output_context_.reset(); output_id_ = 0; output_options_ = {}; }
  return s;
}
Status ExecutionDispatcher::EndGetOutput(std::uint64_t id) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!id || id != output_id_) return Status::NotFound("Get output lease not found");
  return FinishOutputLocked();
}
Status ExecutionDispatcher::Shutdown() {
  std::unique_lock<std::shared_mutex> life(lifecycle_);
  if (graph_id_) return Status::Busy("release Get graph lease before shutdown");
  ready_ = false;
  Status first;
  auto close = [&](auto& context) {
    if (!context) return;
    auto s = resources_.Close(context->lease);
    if (s.ok()) context.reset();
    else if (first.ok()) first = s;
  };
  // Queue teardown can recover a failed completion report/unregistration.
  // Keep the output lease (and the caller's storage) if teardown also fails.
  close(output_context_);
  if (!output_context_) { output_id_ = 0; output_options_ = {}; }
  close(pending_put_);
  close(pending_get_);
  auto s = resources_.Shutdown();
  if (!s.ok() && first.ok()) first = s;
  if (first.ok()) region_ = {};
  if (get_trace_) get_trace_->Flush();
  return first;
}
}  // namespace kvaf
