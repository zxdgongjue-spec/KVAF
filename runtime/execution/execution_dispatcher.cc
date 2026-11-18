#include "execution_dispatcher.h"
#include <algorithm>
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
class PutDispatchPhase {
 public:
  PutDispatchPhase(PutDispatchTiming* timing, std::uint64_t PutDispatchTiming::*field)
      : timing_(timing), field_(field), started_(timing ? TraceNowNs() : 0) {}
  ~PutDispatchPhase() { Finish(); }
  void Finish() {
    if (timing_) { timing_->*field_ += TraceNowNs() - started_; timing_ = nullptr; }
  }
 private:
  PutDispatchTiming* timing_;
  std::uint64_t PutDispatchTiming::*field_;
  std::uint64_t started_;
};
Status LaunchGet(GetExecutionContext& context, GetLaunchTiming* timing = nullptr) {
  try {
    return context.view().LaunchPrepared(context.invocation(), timing);
  } catch (const std::exception& e) {
    return Status::CompletionUnknown(std::string("provider Get threw after submission: ") + e.what());
  } catch (...) {
    return Status::CompletionUnknown("provider Get threw after submission");
  }
}

// The owner is bound only while preparing, capturing or inspecting a node;
// replay uses captured device addresses and never consults the bound state.
template <class Fn>
Status WithGraphNode(GetGraphNode& node, Fn&& fn) {
  auto view = node.instance->view();
  auto s = view.Bind(node.state.get());
  if (!s.ok()) return s;
  try {
    s = fn(view);
  } catch (...) {
    (void)view.Bind(node.instance->state.get());
    throw;
  }
  auto restored = view.Bind(node.instance->state.get());
  return restored.ok() ? s : restored;
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
    for (auto& instance : put_instances_) (void)instance.release();
    for (auto& instance : get_instances_) (void)instance.release();
    for (auto& entry : graph_nodes_) (void)entry.second.state.release();
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
  auto s = resources_.Initialize(region);
  if (s.ok()) { region_ = region; ready_ = true; }
  return s;
}
Status ExecutionDispatcher::ValidateInputLocked(const BatchPutRequest& request,
                                                 const BatchPutExecutionOptions& options,
                                                 PutExecutionConfig c) const {
  if (failed_input_context_) return Status::Busy("failed Put input setup requires shutdown");
  if (!input_id_ && !options.input_lease) return Status::OK();
  if (!input_id_ || options.input_lease != input_id_ || !input_context_ ||
      !(input_context_->config == c))
    return Status::Busy("Put input lease/configuration mismatch");
  if (request.input.data != input_options_.input.data ||
      request.input.bytes != input_options_.input.bytes ||
      request.input.device != input_options_.input.device ||
      options.stream != input_options_.stream || request.keys.size() > input_options_.request_capacity)
    return Status::InvalidArgument("request differs from bound Put input/stream/capacity");
  return Status::OK();
}
PutExecutionContext* ExecutionDispatcher::FindPutInstanceLocked(PutExecutionConfig c) const {
  for (const auto& instance : put_instances_)
    if (instance->config == c) return instance.get();
  return nullptr;
}
Status ExecutionDispatcher::ClosePutInstanceLocked(PutExecutionContext* instance) {
  auto s = resources_.Close(instance->resources);
  if (!s.ok()) { instance->cleanup_only = true; return s; }
  if (pending_put_ == instance) pending_put_ = nullptr;
  if (prepared_put_ == instance) {
    prepared_put_ = nullptr;
    prepared_request_ = nullptr;
    prepared_stream_ = nullptr;
  }
  if (input_context_ == instance) {
    input_context_ = nullptr;
    input_id_ = 0;
    input_options_ = {};
  }
  if (failed_input_context_ == instance) failed_input_context_ = nullptr;
  put_instances_.erase(std::remove_if(put_instances_.begin(), put_instances_.end(),
      [instance](const auto& entry) { return entry.get() == instance; }), put_instances_.end());
  return Status::OK();
}
Status ExecutionDispatcher::PreparePutInstanceLocked(PutExecutionConfig config,
    const BatchPutRequest* request, const ExecutionWarmup& warmup,
    PutExecutionContext** out, PutDispatchTiming* timing) {
  PutExecutionContext* instance;
  {
    PutDispatchPhase phase(timing, &PutDispatchTiming::context_ns);
    instance = FindPutInstanceLocked(config);
  }
  if (instance && instance->cleanup_only) return Status::Busy("Put instance requires recovery or shutdown");
  const bool prepare = !instance || !instance->warmed ||
      instance->warmed_stream != warmup.stream ||
      instance->warmed_request_capacity < warmup.request_capacity;
  std::uint32_t capacity = warmup.request_capacity;
  Status s;
  if (prepare) {
    PutDispatchPhase phase(timing, &PutDispatchTiming::describe_ns);
    if (request) {
      if (instance) s = instance->view().Describe(*request, &capacity);
      else {
        PutResourceRequirements requirements;
        s = resources_.Describe(config, *request, &requirements);
        if (s.ok()) capacity = requirements.capacity;
      }
    } else if (instance) s = instance->view().ValidateCapacity(capacity);
    if (!s.ok()) return s;
  }
  if (!instance) {
    auto created = std::make_unique<PutExecutionContext>();
    created->config = config;
    {
      PutDispatchPhase phase(timing, &PutDispatchTiming::acquire_ns);
      s = resources_.PinPut({config, capacity}, &created->resources);
    }
    if (!s.ok()) return s;
    instance = created.get();
    put_instances_.push_back(std::move(created));
    {
      PutDispatchPhase phase(timing, &PutDispatchTiming::bind_ns);
      try { s = instance->Bind(); }
      catch (const std::exception& e) { s = Status::Internal(std::string("Put state creation failed: ") + e.what()); }
      catch (...) { s = Status::Internal("Put state creation failed"); }
    }
    if (!s.ok()) { (void)ClosePutInstanceLocked(instance); return s; }
  }
  {
    PutDispatchPhase phase(timing, &PutDispatchTiming::acquire_ns);
    s = instance->resources.Acquire(&instance->lease);
  }
  if (!s.ok()) return s;
  if (prepare) {
    capacity = std::max(capacity, instance->warmed_capacity);
    // A failed warmup can have changed internal staging, so a retry must warm
    // again even when it returns to the previously successful capacity/stream.
    instance->warmed = false;
    try { s = instance->view().Warmup({capacity, warmup.stream}); }
    catch (const std::exception& e) { s = Status::Internal(std::string("Put warmup failed: ") + e.what()); }
    catch (...) { s = Status::Internal("Put warmup failed"); }
    if (!s.ok()) { instance->ClearExecution(); return s; }
    instance->warmed = true;
    instance->warmed_capacity = capacity;
    instance->warmed_request_capacity = std::max(instance->warmed_request_capacity, warmup.request_capacity);
    instance->warmed_stream = warmup.stream;
  }
  *out = instance;
  return Status::OK();
}
Status ExecutionDispatcher::PreparePutLocked(PutExecutionConfig config,
    const BatchPutRequest& request, const BatchPutExecutionOptions& options) {
  if (request.keys.size() > std::numeric_limits<std::uint32_t>::max())
    return Status::InvalidArgument("invalid Put request capacity");
  PutExecutionContext* instance = nullptr;
  Status s;
  try {
    if (input_id_) {
      instance = input_context_;
      PutDispatchPhase phase(options.dispatch_timing, &PutDispatchTiming::acquire_ns);
      s = instance->resources.Acquire(&instance->lease);
    } else {
      s = PreparePutInstanceLocked(config, &request,
          {static_cast<std::uint32_t>(request.keys.size()), options.stream}, &instance,
          options.dispatch_timing);
    }
    // Per-request provider validation remains separate from cached capacity and
    // state preparation. This must not submit work or retain application input.
    if (s.ok()) s = instance->view().PrepareBound(request, options);
  } catch (const std::exception& e) {
    s = Status::Internal(std::string("Put preparation failed before submission: ") + e.what());
  } catch (...) { s = Status::Internal("Put preparation failed before submission"); }
  if (!s.ok()) {
    if (instance) instance->ClearExecution();
    return s;
  }
  prepared_put_ = instance;
  prepared_request_ = &request;
  prepared_stream_ = options.stream;
  return Status::OK();
}
Status ExecutionDispatcher::PreparePut(const BatchPutExecutionOptions& options) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (pending_put_ || prepared_put_) return Status::Busy("Put is prepared or awaits transaction acknowledgement");
  if (input_id_ || options.input_lease || failed_input_context_)
    return Status::Busy("bound Put requires a matching request");
  PutExecutionConfig c;
  auto s = ResolveExecution(policy_, options.config, &c);
  PutExecutionContext* instance = nullptr;
  if (s.ok()) s = PreparePutInstanceLocked(c, nullptr, {0, options.stream}, &instance);
  if (s.ok()) instance->ClearExecution();
  return s;
}
Status ExecutionDispatcher::PreparePut(const BatchPutRequest& request,
                                       const BatchPutExecutionOptions& options) {
  auto* timing = options.dispatch_timing;
  PutDispatchPhase locking(timing, &PutDispatchTiming::lock_ns);
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  locking.Finish();
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (pending_put_ || prepared_put_) return Status::Busy("Put is prepared or awaits transaction acknowledgement");
  PutExecutionConfig c;
  Status s;
  {
    PutDispatchPhase phase(timing, &PutDispatchTiming::resolve_ns);
    s = ResolveExecution(policy_, options.config, &c);
    if (s.ok() && timing) timing->config = c;
  }
  if (s.ok()) s = ValidateInputLocked(request, options, c);
  return s.ok() ? PreparePutLocked(c, request, options) : s;
}
Status ExecutionDispatcher::CancelPreparedPut() {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (prepared_put_) prepared_put_->ClearExecution();
  prepared_put_ = nullptr;
  prepared_request_ = nullptr;
  prepared_stream_ = nullptr;
  return Status::OK();
}
Status ExecutionDispatcher::BeginPutInput(PutExecutionConfig c, const PutInputOptions& r,
                                          std::uint64_t* id) {
  if (!id) return Status::InvalidArgument("null input lease ID");
  *id = 0;
  if (!r.input.valid() || r.input.device != DeviceKind::kNpu || !r.stream || !r.request_capacity)
    return Status::InvalidArgument("input binding requires NPU input, stream and capacity");
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (pending_put_ || prepared_put_ || input_id_ || failed_input_context_)
    return Status::Busy("Put input binding or unresolved execution already exists");
  if (c.transport != Transport::Urma)
    return Status::NotImplemented("input binding supports URMA Put");
  if (!next_input_id_) return Status::InvalidState("input lease ID exhausted");
  PutExecutionConfig config;
  auto s = ResolveExecution(policy_, c, &config);
  PutExecutionContext* context = nullptr;
  try {
    if (s.ok()) s = PreparePutInstanceLocked(config, nullptr, {r.request_capacity, r.stream}, &context);
    if (s.ok()) s = context->view().PrepareInput(r);
  } catch (const std::exception& e) {
    s = Status::Internal(std::string("input preparation failed: ") + e.what());
  } catch (...) { s = Status::Internal("input preparation failed"); }
  if (!s.ok()) {
    if (context && !ClosePutInstanceLocked(context).ok()) failed_input_context_ = context;
    return s;
  }
  context->ClearExecution();
  input_options_ = r;
  input_id_ = next_input_id_++;
  input_context_ = context;
  *id = input_id_;
  return Status::OK();
}
Status ExecutionDispatcher::EndPutInput(std::uint64_t id) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!id || id != input_id_) return Status::NotFound("Put input lease not found");
  if (pending_put_ || prepared_put_) return Status::Busy("Put must be cancelled or acknowledged before releasing input");
  auto s = input_context_->view().FinishInput();
  if (s.ok()) { input_context_ = nullptr; input_id_ = 0; input_options_ = {}; }
  return s;
}
Status ExecutionDispatcher::Recover(PutExecutionConfig c) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (pending_put_ || prepared_put_) return Status::Busy("cancel preparation or resolve pending Put before resource recovery");
  if (input_id_ || failed_input_context_) return Status::Busy("release Put input lease before resource recovery");
  PutExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  if (!s.ok()) return s;
  if (auto* instance = FindPutInstanceLocked(resolved)) s = ClosePutInstanceLocked(instance);
  return s.ok() ? resources_.Recover(resolved) : s;
}
Status ExecutionDispatcher::Recover(GetExecutionConfig c) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (!graph_nodes_.empty()) return Status::Busy("release Get graph leases before recovery");
  GetExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  if (!s.ok()) return s;
  auto* instance = FindGetInstanceLocked(resolved);
  if (instance && instance->output_id) return Status::Busy("release this Get output lease before recovery");
  if (pending_get_ && pending_get_ != instance)
    return Status::Busy("complete the active Get before resource recovery");
  if (pending_get_) {
    s = DrainGetLocked();
    if (!s.ok()) return s;
    instance = FindGetInstanceLocked(resolved);
  }
  if (instance) s = CloseGetInstanceLocked(instance);
  return s.ok() ? resources_.Recover(resolved) : s;
}
Status ExecutionDispatcher::ExecuteBatchPut(const BatchPutExecRequest& request) {
  auto* timing = request.execution.dispatch_timing;
  PutDispatchPhase locking(timing, &PutDispatchTiming::lock_ns);
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(put_mutex_);
  locking.Finish();
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (pending_put_) return Status::InvalidState("Put awaits completion/transaction acknowledgement");
  if (!request.request || !request.put_desc) return Status::InvalidArgument("missing Put request");
  PutExecutionConfig c;
  Status s;
  {
    PutDispatchPhase phase(timing, &PutDispatchTiming::resolve_ns);
    s = ResolveExecution(policy_, request.execution.config, &c);
    if (s.ok() && timing) timing->config = c;
  }
  if (s.ok()) s = ValidateInputLocked(*request.request, request.execution, c);
  if (!s.ok()) return s;
  if (prepared_put_) {
    if (!(prepared_put_->config == c) || prepared_request_ != request.request ||
        prepared_stream_ != request.execution.stream)
      return Status::InvalidArgument("Put execution differs from prepared request; cancel preparation first");
  } else {
    // Direct dispatcher users may omit Prepare. Runtime normally arrives with
    // this instance and its execution lease already prepared before metadata.
    s = PreparePutLocked(c, *request.request, request.execution);
    if (!s.ok()) return s;
  }
  pending_put_ = prepared_put_;
  prepared_put_ = nullptr;
  prepared_request_ = nullptr;
  prepared_stream_ = nullptr;
  pending_put_->launched = true;
  {
    PutDispatchPhase phase(timing, &PutDispatchTiming::provider_ns);
    try {
      // Launch consumes request metadata in this call. Recovery retains the
      // provider state/lease, not copies of request vectors or timing pointers.
      auto invocation = request;
      invocation.execution.config = c;
      invocation.execution.dispatch_timing = nullptr;
      s = pending_put_->view().Launch(invocation);
    } catch (const std::exception& e) {
      s = Status::CompletionUnknown(std::string("provider Put threw after submission: ") + e.what());
    } catch (...) {
      s = Status::CompletionUnknown("provider Put threw after submission");
    }
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
    auto s = pending_put_->view().Resolve(&result);
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
  auto s = pending_put_->launched ? pending_put_->view().Acknowledge() : Status::OK();
  if (s.ok()) {
    pending_put_->ClearExecution();
    pending_put_ = nullptr;
  }
  return s;
}
GetExecutionContext* ExecutionDispatcher::FindGetInstanceLocked(GetExecutionConfig c) const {
  for (const auto& instance : get_instances_)
    if (instance->config == c) return instance.get();
  return nullptr;
}
Status ExecutionDispatcher::CloseGetInstanceLocked(GetExecutionContext* instance) {
  if (!instance) return Status::OK();
  auto s = resources_.Close(instance->resources);
  if (!s.ok()) {
    instance->cleanup_only = true;
    pending_get_ = instance;
    return s;
  }
  if (pending_get_ == instance) pending_get_ = nullptr;
  get_instances_.erase(std::find_if(get_instances_.begin(), get_instances_.end(),
      [instance](const auto& entry) { return entry.get() == instance; }));
  return Status::OK();
}
Status ExecutionDispatcher::PrepareGetInstanceLocked(GetExecutionConfig config,
    const BatchGetRequest* request, const ExecutionWarmup& warmup,
    GetExecutionContext** out, GetPreparationTiming* timing) {
  auto* instance = FindGetInstanceLocked(config);
  if (instance && instance->cleanup_only) return Status::Busy("Get instance requires cleanup");
  const bool prepare = !instance || !instance->warmed ||
      instance->warmed_stream != warmup.stream ||
      instance->warmed_request_capacity < warmup.request_capacity;
  std::uint32_t capacity = warmup.request_capacity;
  Status s;
  if (prepare) {
    const auto started = timing ? TraceNowNs() : 0;
    if (request) {
      if (instance) s = instance->view().Describe(*request, &capacity);
      else {
        GetResourceRequirements requirements;
        s = resources_.Describe(config, *request, &requirements);
        if (s.ok()) capacity = requirements.capacity;
      }
    } else if (instance) s = instance->view().ValidateCapacity(capacity);
    if (timing) timing->describe_ns += TraceNowNs() - started;
    if (!s.ok()) return s;
  }
  if (!instance) {
    auto started = timing ? TraceNowNs() : 0;
    auto created = std::make_unique<GetExecutionContext>();
    created->config = config;
    if (timing) timing->context_ns += TraceNowNs() - started;
    started = timing ? TraceNowNs() : 0;
    s = resources_.PinGet({config, capacity}, &created->resources);
    if (timing) timing->acquire_ns += TraceNowNs() - started;
    if (!s.ok()) return s;
    instance = created.get();
    get_instances_.push_back(std::move(created));
    started = timing ? TraceNowNs() : 0;
    try { s = instance->Bind(); }
    catch (const std::exception& e) { s = Status::Internal(std::string("Get state creation failed: ") + e.what()); }
    catch (...) { s = Status::Internal("Get state creation failed"); }
    if (timing) timing->bind_ns += TraceNowNs() - started;
    if (!s.ok()) { (void)CloseGetInstanceLocked(instance); return s; }
  }
  auto started = timing ? TraceNowNs() : 0;
  // This only claims occupancy on an already initialized record. Resident
  // references never keep the old exclusive leased flag set while idle.
  s = instance->resources.Acquire(&instance->lease);
  if (timing) timing->acquire_ns += TraceNowNs() - started;
  if (!s.ok()) return s;
  if (prepare) {
    capacity = std::max(capacity, instance->warmed_capacity);
    started = timing ? TraceNowNs() : 0;
    try { s = instance->view().Warmup({capacity, warmup.stream}); }
    catch (const std::exception& e) { s = Status::Internal(std::string("Get warmup failed: ") + e.what()); }
    catch (...) { s = Status::Internal("Get warmup failed"); }
    if (timing) timing->warmup_ns += TraceNowNs() - started;
    if (!s.ok()) { instance->lease = {}; return s; }
    instance->warmed = true;
    instance->warmed_capacity = capacity;
    instance->warmed_request_capacity = std::max(instance->warmed_request_capacity, warmup.request_capacity);
    instance->warmed_stream = warmup.stream;
  }
  *out = instance;
  return Status::OK();
}
Status ExecutionDispatcher::DrainGetLocked() {
  if (!graph_nodes_.empty()) return Status::Busy("Get graph leases own this direction");
  if (!pending_get_) return Status::OK();
  auto* instance = pending_get_;
  GetDispatchTrace::Row row{};
  if (get_trace_) {
    row.algorithm = GetAlgorithmLabel(instance->config);
    row.path = instance->output_id ? "output" : "ordinary";
  }
  const auto started = get_trace_ ? TraceNowNs() : 0;
  auto s = instance->cleanup_only ? CloseGetInstanceLocked(instance) : instance->view().Drain();
  const auto drained = get_trace_ ? TraceNowNs() : 0;
  if (s.ok() && pending_get_) {
    instance->ClearRequest();
    pending_get_ = nullptr;
  }
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
  const auto& r = *request.request;
  if (r.indices.bytes % sizeof(std::int32_t) || r.indices.bytes / sizeof(std::int32_t) > UINT32_MAX)
    return Status::InvalidArgument("invalid Get request capacity");
  const auto capacity = static_cast<std::uint32_t>(r.indices.bytes / sizeof(std::int32_t));
  if (!graph_nodes_.empty() || request.execution.graph_lease) {
    if (!request.execution.graph_lease)
      return Status::Busy("Get graph leases own this direction");
    auto it = graph_nodes_.find(request.execution.graph_lease);
    if (it == graph_nodes_.end()) return Status::NotFound("Get graph lease not found");
    auto& node = it->second;
    if (!(c == node.instance->config) || request.execution.output_lease)
      return Status::InvalidArgument("Get graph lease/configuration mismatch");
    if (node.ending || node.instance->graph_recovery_required)
      return Status::InvalidState("Get graph is being retired; release its leases before recovery");
    if (node.captured || request.execution.synchronize || request.execution.profile)
      return Status::InvalidState("one asynchronous Get capture is allowed per lease");
    const auto& bound = node.options;
    if (r.output.data != bound.output.data || r.output.bytes != bound.output.bytes ||
        r.output.device != bound.output.device || request.execution.stream != bound.stream ||
        capacity > bound.request_capacity)
      return Status::InvalidArgument("request differs from Get graph output/stream/capacity");
    auto exec = request;
    exec.execution.config = c;
    try {
      return WithGraphNode(node, [&](GetResourceView view) {
        // Treat a failed launch as consumed: capture might have recorded work.
        node.captured = true;
        return view.LaunchPrepared(exec);
      });
    } catch (const std::exception& e) {
      return Status::CompletionUnknown(std::string("graph capture failed: ") + e.what());
    } catch (...) {
      return Status::CompletionUnknown("graph capture failed");
    }
  }
  auto* instance = FindGetInstanceLocked(c);
  const bool output_bound = instance && instance->output_id;
  if (output_bound || request.execution.output_lease) {
    if (!output_bound || instance->output_id != request.execution.output_lease)
      return Status::Busy("Get output lease/configuration mismatch");
    const auto& bound = instance->output_options;
    if (r.output.data != bound.output.data || r.output.bytes != bound.output.bytes ||
        r.output.device != bound.output.device || capacity > bound.request_capacity)
      return Status::InvalidArgument("request differs from bound Get output/capacity");
  }
  GetDispatchTrace::Row row{};
  if (get_trace_) {
    row.event = "submit";
    row.algorithm = GetAlgorithmLabel(c);
    row.path = output_bound ? "output" : "ordinary";
    row.lock_ns = locked - started;
    row.resolve_ns = resolved - locked;
    row.output_check_ns = TraceNowNs() - resolved;
  }
  auto phase = get_trace_ ? TraceNowNs() : 0;
  s = DrainGetLocked();
  if (!s.ok()) return s;
  if (get_trace_) row.previous_drain_ns = TraceNowNs() - phase;
  GetPreparationTiming preparation{};
  s = PrepareGetInstanceLocked(c, &r, {capacity, request.execution.stream}, &instance,
                              get_trace_ ? &preparation : nullptr);
  if (!s.ok()) return s;
  if (get_trace_) {
    row.context_ns = preparation.context_ns;
    row.describe_ns = preparation.describe_ns;
    row.acquire_ns = preparation.acquire_ns;
    row.bind_ns = preparation.bind_ns;
    row.launch.warmup_ns = preparation.warmup_ns;
  }
  phase = get_trace_ ? TraceNowNs() : 0;
  instance->request = r;
  instance->table = request.host_access_table;
  instance->options = request.execution;
  instance->options.config = c;
  if (get_trace_) row.context_ns += TraceNowNs() - phase;
  pending_get_ = instance;
  instance->last_graph_profile.clear();
  s = LaunchGet(*instance, get_trace_ ? &row.launch : nullptr);
  phase = get_trace_ ? TraceNowNs() : 0;
  if (s.ok() && request.execution.synchronize) {
    instance->ClearRequest();
    pending_get_ = nullptr;
  }
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
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (pending_put_ || prepared_put_) return Status::Busy("Put is prepared or awaits transaction acknowledgement");
  if (input_id_ || failed_input_context_) return Status::Busy("release Put input lease before warmup");
  PutExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  PutExecutionContext* instance = nullptr;
  if (s.ok()) s = PreparePutInstanceLocked(resolved, nullptr, r, &instance);
  if (s.ok()) instance->ClearExecution();
  return s;
}
Status ExecutionDispatcher::Warmup(GetExecutionConfig c, const ExecutionWarmup& r) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  GetExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  if (!s.ok()) return s;
  auto* instance = FindGetInstanceLocked(resolved);
  if (instance && instance->output_id) return Status::Busy("release this Get output lease before warmup");
  s = DrainGetLocked();
  if (s.ok()) s = PrepareGetInstanceLocked(resolved, nullptr, r, &instance);
  if (s.ok()) instance->ClearRequest();
  return s;
}
Status ExecutionDispatcher::ConfigureGetProfiling(std::optional<GetExecutionConfig> config,
                                                   bool enabled, bool detailed) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (!graph_nodes_.empty()) return Status::Busy("configure profiling before acquiring Get graph leases");
  GetExecutionConfig resolved;
  auto s = ResolveExecution(policy_, config, &resolved);
  if (s.ok()) s = DrainGetLocked();
  GetExecutionContext* instance = nullptr;
  if (s.ok()) s = PrepareGetInstanceLocked(resolved, nullptr, {}, &instance);
  if (!s.ok()) return s;
  s = instance->view().operation()->SetBatchGetKernelProfiling(enabled, detailed);
  if (s.ok()) {
    instance->profiling_enabled = enabled;
    instance->last_graph_profile.clear();
  }
  instance->ClearRequest();
  return s;
}
Status ExecutionDispatcher::FindGraphNode(GetExecutionContext* instance, std::uint64_t id,
                                          GetGraphNode** out) {
  *out = nullptr;
  if (id) {
    auto it = graph_nodes_.find(id);
    if (it == graph_nodes_.end()) return Status::NotFound("Get graph lease not found");
    if (instance && instance != it->second.instance)
      return Status::InvalidArgument("Get graph lease/configuration mismatch");
    *out = &it->second;
  } else {
    for (auto& [key, node] : graph_nodes_) {
      if (node.instance != instance) continue;
      if (*out) return Status::InvalidArgument("select a graph_lease for this Get query");
      *out = &node;
    }
  }
  if (*out && ((*out)->ending || !(*out)->captured || (*out)->instance->graph_recovery_required))
    return Status::InvalidState("Get graph has no queryable capture");
  return Status::OK();
}
Status ExecutionDispatcher::CopyGetProfiles(std::optional<GetExecutionConfig> config,
                                             std::vector<BatchGetKernelProfile>* profiles,
                                             std::uint64_t graph_lease) {
  if (!profiles) return Status::InvalidArgument("null Get profile output");
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  GetExecutionConfig resolved;
  auto s = ResolveExecution(policy_, config, &resolved);
  if (!s.ok()) return s;
  auto* instance = FindGetInstanceLocked(resolved);
  GetGraphNode* node = nullptr;
  s = FindGraphNode(graph_lease && !config ? nullptr : instance, graph_lease, &node);
  if (!s.ok()) return s;
  if (node) {
    if (config && !(resolved == node->instance->config))
      return Status::InvalidArgument("Get graph lease/configuration mismatch");
    return WithGraphNode(*node, [&](GetResourceView view) {
      return view.operation()->CopyBatchGetKernelProfiles(profiles);
    });
  }
  if (!instance || instance->cleanup_only)
    return Status::NotInitialized("Get instance has no readable profile");
  if (pending_get_ == instance)
    return Status::Busy("complete Get with DrainGet before reading its profile");
  if (!instance->last_graph_profile.empty()) {
    *profiles = instance->last_graph_profile;
    return Status::OK();
  }
  return instance->view().operation()->CopyBatchGetKernelProfiles(profiles);
}
Status ExecutionDispatcher::CheckGetStatus(std::optional<GetExecutionConfig> config, void* stream,
                                           std::uint64_t graph_lease) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  GetExecutionConfig resolved;
  auto s = ResolveExecution(policy_, config, &resolved);
  if (!s.ok()) return s;
  auto* instance = FindGetInstanceLocked(resolved);
  GetGraphNode* node = nullptr;
  s = FindGraphNode(graph_lease && !config ? nullptr : instance, graph_lease, &node);
  if (!s.ok()) return s;
  if (node) {
    if ((config && !(resolved == node->instance->config)) || stream != node->options.stream)
      return Status::InvalidArgument("check Get graph status on its prepared configuration and stream");
    return WithGraphNode(*node, [&](GetResourceView view) {
      return view.operation()->CheckBatchGetStatus(stream);
    });
  }
  if (!instance || instance->cleanup_only)
    return Status::NotInitialized("Get instance is not available for status checking");
  return instance->view().operation()->CheckBatchGetStatus(stream);
}
Status ExecutionDispatcher::BeginGetGraph(GetExecutionConfig c, const GetGraphOptions& r, std::uint64_t* id) {
  if (!id) return Status::InvalidArgument("null graph lease output");
  *id = 0;
  if (!r.request_capacity || !r.stream) return Status::InvalidArgument("graph requires capacity and stream");
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  if (!ready_) return Status::NotInitialized("dispatcher is not accepting work");
  if (graph_stream_ && graph_stream_ != r.stream)
    return Status::InvalidArgument("all live Get graphs of a Runtime must use the same stream");
  for (const auto& [key, node] : graph_nodes_)
    if (node.instance->graph_recovery_required)
      return Status::Busy("retire failed Get graphs before preparing new nodes");
  GetExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  if (!s.ok()) return s;
  auto* instance = FindGetInstanceLocked(resolved);
  if (instance && instance->output_id) return Status::Busy("release this Get output lease before graph capture");
  if (!next_graph_id_) return Status::InvalidState("graph lease ID exhausted");
  const bool has_siblings = std::any_of(graph_nodes_.begin(), graph_nodes_.end(),
      [&](const auto& e) { return e.second.instance == instance; });
  if (graph_nodes_.empty()) s = DrainGetLocked();
  if (s.ok() && has_siblings) s = instance->view().ValidateCapacity(r.request_capacity);
  else if (s.ok()) s = PrepareGetInstanceLocked(resolved, nullptr, {r.request_capacity, r.stream}, &instance);
  if (!s.ok()) return s;
  auto entry = graph_nodes_.try_emplace(next_graph_id_).first;
  auto& node = entry->second;
  node.instance = instance;
  node.options = r;
  try {
    node.state = instance->view().CreateState();
    if (!node.state) {
      s = Status::NotImplemented("Get provider has no graph node state");
    } else {
      node.state->graph_node = true;
      s = WithGraphNode(node, [&](GetResourceView view) { return view.PrepareGraphPrepared(r); });
    }
  }
  catch (const std::exception& e) { s = Status::Internal(std::string("graph preparation failed: ") + e.what()); }
  catch (...) { s = Status::Internal("graph preparation failed"); }
  instance->lease = {};
  if (!s.ok()) {
    graph_nodes_.erase(entry);
    if (has_siblings) {
      if (s.code() == StatusCode::kCompletionUnknown) instance->graph_recovery_required = true;
    } else {
      (void)CloseGetInstanceLocked(instance);
    }
    return s;
  }
  instance->last_graph_profile.clear();
  graph_stream_ = r.stream;
  *id = next_graph_id_++;
  return Status::OK();
}
Status ExecutionDispatcher::EndGetGraph(std::uint64_t id, void* stream, bool discard) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  auto it = graph_nodes_.find(id);
  if (it == graph_nodes_.end()) return Status::NotFound("Get graph lease not found");
  auto& node = it->second;
  auto* instance = node.instance;
  if (stream != node.options.stream)
    return Status::InvalidArgument("Get graph must end on its capture stream");
  if (!node.domain_closed) {
    if (instance->graph_recovery_required && !discard)
      return Status::InvalidState("failed Get graph must be destroyed and ended with discard_result=True");
    bool fenced_for_recovery = node.ending;
    auto s = node.ending ? Status::OK() : WithGraphNode(node, [&](GetResourceView view) {
      auto finished = view.FinishGraph(stream, discard);
      fenced_for_recovery = view.operation()->GetGraphNeedsRecovery();
      if (fenced_for_recovery) instance->graph_recovery_required = true;
      if (finished.ok() && instance->profiling_enabled && node.captured && !discard) {
        std::vector<BatchGetKernelProfile> snapshot;
        if (view.operation()->CopyBatchGetKernelProfiles(&snapshot).ok())
          instance->last_graph_profile = std::move(snapshot);
      }
      return finished;
    });
    if (instance->graph_recovery_required && discard) {
      if (!s.ok() && !fenced_for_recovery) return s;
      node.ending = true;
      for (const auto& [key, sibling] : graph_nodes_)
        if (sibling.instance == instance && !sibling.ending)
          return Status::Busy("discard remaining Get graph leases, then retry this lease");
      s = resources_.Close(instance->resources);
      if (!s.ok()) return s;
      instance->cleanup_only = true;
      for (auto& [key, sibling] : graph_nodes_)
        if (sibling.instance == instance) sibling.domain_closed = true;
    } else if (!s.ok()) {
      return s;
    }
  }
  graph_nodes_.erase(it);
  if (graph_nodes_.empty()) graph_stream_ = nullptr;
  return Status::OK();
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
  if (c.transport != Transport::Urma ||
      !((c.executor == Executor::Aiv && c.algorithm == GetAlgorithm::Direct) ||
        (c.executor == Executor::Aicpu && c.algorithm == GetAlgorithm::Gather)))
    return Status::NotImplemented("output binding supports URMA AIV Direct and AICPU Gather");
  GetExecutionConfig resolved;
  auto s = ResolveExecution(policy_, c, &resolved);
  if (!s.ok()) return s;
  auto* instance = FindGetInstanceLocked(resolved);
  if (instance && instance->output_id) return Status::Busy("this Get instance already has an output binding");
  if (!next_output_id_) return Status::InvalidState("output lease ID exhausted");
  s = DrainGetLocked();
  if (s.ok()) s = PrepareGetInstanceLocked(resolved, nullptr, {r.request_capacity, r.stream}, &instance);
  if (!s.ok()) return s;
  try { s = instance->view().PrepareOutput(r); }
  catch (const std::exception& e) { s = Status::Internal(std::string("output preparation failed: ") + e.what()); }
  catch (...) { s = Status::Internal("output preparation failed"); }
  if (!s.ok()) { (void)CloseGetInstanceLocked(instance); return s; }
  instance->output_options = r;
  instance->output_id = next_output_id_++;
  instance->ClearRequest();
  *id = instance->output_id;
  return Status::OK();
}
Status ExecutionDispatcher::FinishOutputLocked(GetExecutionContext& instance) {
  if (!graph_nodes_.empty()) return Status::Busy("Get graph leases own this direction");
  if (pending_get_ == &instance) {
    auto s = DrainGetLocked();
    if (!s.ok()) return s;
  }
  auto s = instance.view().FinishOutput();
  if (s.ok()) { instance.output_id = 0; instance.output_options = {}; }
  return s;
}
Status ExecutionDispatcher::EndGetOutput(std::uint64_t id) {
  std::shared_lock<std::shared_mutex> life(lifecycle_);
  std::lock_guard<std::mutex> lock(get_mutex_);
  for (auto& instance : get_instances_)
    if (id && instance->output_id == id) return FinishOutputLocked(*instance);
  return Status::NotFound("Get output lease not found");
}
Status ExecutionDispatcher::Shutdown() {
  std::unique_lock<std::shared_mutex> life(lifecycle_);
  if (!graph_nodes_.empty()) return Status::Busy("release Get graph leases before shutdown");
  ready_ = false;
  Status first;
  for (auto it = put_instances_.begin(); it != put_instances_.end();) {
    auto* instance = it->get();
    auto s = resources_.Close(instance->resources);
    if (s.ok()) {
      if (pending_put_ == instance) pending_put_ = nullptr;
      if (prepared_put_ == instance) {
        prepared_put_ = nullptr;
        prepared_request_ = nullptr;
        prepared_stream_ = nullptr;
      }
      if (input_context_ == instance) {
        input_context_ = nullptr;
        input_id_ = 0;
        input_options_ = {};
      }
      if (failed_input_context_ == instance) failed_input_context_ = nullptr;
      it = put_instances_.erase(it);
    } else {
      instance->cleanup_only = true;
      if (first.ok()) first = s;
      ++it;
    }
  }
  // Close owners with resident execution state and bindings still alive. Only
  // successful closes may erase a pin; failed instances retain every dependency.
  for (auto it = get_instances_.begin(); it != get_instances_.end();) {
    auto s = resources_.Close((*it)->resources);
    if (s.ok()) {
      if (pending_get_ == it->get()) pending_get_ = nullptr;
      it = get_instances_.erase(it);
    } else {
      if (first.ok()) first = s;
      ++it;
    }
  }
  auto s = resources_.Shutdown();
  if (!s.ok() && first.ok()) first = s;
  if (first.ok()) region_ = {};
  if (get_trace_) get_trace_->Flush();
  return first;
}
}  // namespace kvaf
