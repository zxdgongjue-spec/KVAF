#include "cpu_sgl_write_aiv_operator.h"
#include "get/scatter/urma_aicpu/include/cpu_push_transport.h"
#include "aicpu_batch_get_capture.h"

namespace kvaf::cpu_push {
namespace {
struct Owner;
struct Invocation final : ExecutionState {
  explicit Invocation(Owner* value) : owner(value) {}
  Owner* owner;
  std::unique_ptr<ExecutionState> inner;
  GetOutputOptions output;
  GetGraphOptions graph;
  bool output_bound{}, graph_active{}, graph_recorded{}, recovery{};
};
struct Owner final : ExecutionResourceOwner {
  explicit Owner(Config c) : transport(std::move(c)), fallback(this) {
    fallback.inner = transport.CreateExecutionState();
    (void)transport.BindExecutionState(fallback.inner.get());
  }
  ~Owner() override { (void)transport.BindExecutionState(nullptr); }
  CpuPushTransport transport;
  Invocation fallback;
  Invocation* call{&fallback};
  Status Initialize(const RemoteRegionDesc& r) override { return transport.Initialize(r); }
  Status Close() override { return transport.Shutdown(); }
};
class Operator final : public AccessOperator {
 public:
  explicit Operator(Config c) : owned_(std::make_unique<Owner>(std::move(c))), owner_(owned_.get()) {}
  ~Operator() override { if (owned_ && !owned_->Close().ok()) (void)owned_.release(); }
  std::unique_ptr<ExecutionResourceOwner> TakeResources() override { return std::move(owned_); }
  Status Initialize(const RemoteRegionDesc& r) override { return owner_->Initialize(r); }
  Status Shutdown() override { return owner_->Close(); }
  std::unique_ptr<ExecutionState> CreateExecutionState() override {
    auto call = std::make_unique<Invocation>(owner_);
    call->inner = owner_->transport.CreateExecutionState();
    return call;
  }
  Status BindExecutionState(ExecutionState* state) override {
    auto* call = state ? dynamic_cast<Invocation*>(state) : &owner_->fallback;
    if (!call || call->owner != owner_) return Status::InvalidArgument("AIV SGL invocation belongs to another resource");
    call->inner->graph_node = call->graph_node;
    auto s = owner_->transport.BindExecutionState(call->inner.get());
    if (s.ok()) owner_->call = call;
    return s;
  }
  Status ExecuteBatchGet(const BatchGetExecRequest& r) override {
    auto& c = *owner_->call;
    if (!r.request) return Status::InvalidArgument("missing AIV SGL request");
    if (r.execution.profile)
      return Status::InvalidArgument("AIV SGL uses cycle profiles; enable start_batch_get_kernel_profile instead");
    if (c.graph_active) {
      if (c.graph_recorded) return Status::NotImplemented("one Get node per graph lease is supported");
      if (r.execution.stream != c.graph.stream || r.execution.synchronize || r.execution.profile ||
          r.request->output.data != c.graph.output.data || r.request->output.bytes != c.graph.output.bytes ||
          r.request->indices.bytes / sizeof(int32_t) > c.graph.request_capacity)
        return Status::InvalidArgument("captured request differs from prepared AIV SGL graph lease");
    }
    if (c.output_bound && (r.execution.stream != c.output.stream ||
        r.request->output.data != c.output.output.data || r.request->output.bytes != c.output.output.bytes ||
        r.request->indices.bytes / sizeof(int32_t) > c.output.request_capacity))
      return Status::InvalidArgument("request differs from bound AIV SGL output");
    auto s = owner_->transport.Prepare(r, c.graph_active);
    if (s.ok()) s = owner_->transport.ExecutePrepared(r);
    if (s.ok() && c.graph_active) c.graph_recorded = true;
    if (s.ok() && r.execution.synchronize && !c.output_bound && !c.graph_active)
      s = owner_->transport.ReleaseOutput();
    return s;
  }
  Status ExecuteBatchPut(const BatchPutExecRequest&) override {
    return Status::NotImplemented("AIV CPU SGL provider supports Get only");
  }
  Status DrainGet() override {
    auto s = owner_->transport.Drain();
    if (s.ok() && !owner_->call->output_bound && !owner_->call->graph_active)
      s = owner_->transport.ReleaseOutput();
    return s;
  }
  Status PrepareGetOutput(const GetOutputOptions& r) override {
    auto& c = *owner_->call;
    if (c.output_bound || c.graph_active || !r.stream || !r.request_capacity || r.request_capacity > kMaxRequests)
      return Status::InvalidArgument("invalid AIV SGL output binding");
    bool capturing = false;
    auto s = detail::BatchGetCapture(static_cast<aclrtStream>(r.stream), true, false, &capturing);
    if (s.ok()) s = owner_->transport.Drain();
    if (s.ok()) s = owner_->transport.RegisterOutput(r.output);
    if (s.ok()) { c.output = r; c.output_bound = true; }
    return s;
  }
  Status FinishGetOutput() override {
    auto& c = *owner_->call;
    if (c.graph_active) return Status::Busy("AIV SGL output belongs to a graph");
    auto s = owner_->transport.Drain();
    if (s.ok()) s = owner_->transport.ReleaseOutput();
    if (s.ok()) { c.output = {}; c.output_bound = false; }
    return s;
  }
  Status PrepareGetGraph(const GetGraphOptions& r) override {
    auto& c = *owner_->call;
    if (c.output_bound) return Status::Busy("release AIV SGL output binding before graph capture");
    auto s = owner_->transport.PrepareGraph(r);
    if (s.ok()) { c.graph = r; c.graph_active = true; c.graph_recorded = c.recovery = false; }
    return s;
  }
  Status FinishGetGraph(void* stream, bool discard) override {
    auto& c = *owner_->call;
    if (!c.graph_active || stream != c.graph.stream)
      return Status::InvalidArgument("AIV SGL graph stream or lease does not match");
    bool capturing = false;
    auto allowed = detail::BatchGetCapture(static_cast<aclrtStream>(stream), true, false, &capturing);
    if (!allowed.ok()) return allowed;
    if (aclrtSynchronizeStream(static_cast<aclrtStream>(stream)) != ACL_SUCCESS)
      return Status::CompletionUnknown("AIV SGL graph is still pending; lease retained");
    auto s = owner_->transport.FinishGraph(stream, discard);
    // Once fenced, a device failure or failed output unregister can retire via
    // the dispatcher's shared-domain discard/STOP path. Unreplayed graphs keep
    // their ordinary replay-or-discard behavior.
    if (!s.ok() && s.code() != StatusCode::kCompletionUnknown &&
        (owner_->transport.View().health->failed || s.code() == StatusCode::kInternal))
      c.recovery = true;
    if (s.ok()) { c.graph_active = false; c.graph = {}; }
    return s;
  }
  bool GetGraphNeedsRecovery() const override { return owner_->call->recovery; }
  Status CheckBatchGetStatus(void* stream) override { return owner_->transport.Check(stream); }
  Status SetBatchGetKernelProfiling(bool enabled) override { return owner_->transport.SetKernelProfiling(enabled); }
  Status CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>* out) const override {
    return owner_->transport.CopyKernelProfiles(out);
  }
 private:
  std::unique_ptr<Owner> owned_;
  Owner* owner_;
};
}  // namespace
std::unique_ptr<AccessOperator> CreateAivCpuSglWriteOperator(Config c) {
  c.aiv = c.direct_output = true;
  return std::make_unique<Operator>(std::move(c));
}
}  // namespace kvaf::cpu_push
