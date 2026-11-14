#include "execution_dispatcher.h"
#include <array>
#include <cstdlib>
#include <iostream>
using namespace kvaf;
#define CHECK(c) do { if (!(c)) { std::cerr << __LINE__ << ": " #c << '\n'; std::abort(); } } while (false)
struct Counters { int profiles{}, copies{}, checks{}, captures{}; };
class DiagnosticProvider final : public AccessOperator {
 public:
  explicit DiagnosticProvider(Counters& value) : calls(value) {}
  Status Initialize(const RemoteRegionDesc&) override { return Status::OK(); }
  Status Shutdown() override { return Status::OK(); }
  Status SetBatchGetKernelProfiling(bool enabled) override {
    calls.profiles += enabled ? 1 : -1; return Status::OK();
  }
  Status CopyBatchGetKernelProfiles(std::vector<BatchGetKernelProfile>* out) const override {
    ++calls.copies; out->assign(1, {}); return Status::OK();
  }
  Status CheckBatchGetStatus(void*) override { ++calls.checks; return Status::OK(); }
  Status PrepareGetGraph(const GetGraphOptions&) override { return Status::OK(); }
  Status FinishGetGraph(void*, bool) override { return Status::OK(); }
  Status ExecuteBatchGet(const BatchGetExecRequest& request) override {
    if (request.execution.graph_lease) ++calls.captures; return Status::OK();
  }
  Status ExecuteBatchPut(const BatchPutExecRequest&) override { return Status::OK(); }
 private:
  Counters& calls;
};

struct RecoveryCounters {
  int initializes{}, closes{}, gets{}, output_binds{}, output_releases{}, close_failures{};
};
class FailedGraphProvider final : public AccessOperator {
 public:
  explicit FailedGraphProvider(RecoveryCounters& counters) : counters_(counters) {}
  Status Initialize(const RemoteRegionDesc&) override {
    ++counters_.initializes;
    ready_ = true;
    return Status::OK();
  }
  Status Shutdown() override {
    ++counters_.closes;
    if (counters_.close_failures) {
      --counters_.close_failures;
      return Status::CompletionUnknown("remote producer has not stopped");
    }
    ready_ = graph_ = false;
    return Status::OK();
  }
  Status PrepareGetOutput(const GetOutputOptions&) override {
    ++counters_.output_binds;
    return Status::OK();
  }
  Status FinishGetOutput() override {
    ++counters_.output_releases;
    return Status::OK();
  }
  Status PrepareGetGraph(const GetGraphOptions&) override {
    if (!ready_) return Status::NotInitialized("owner was retired");
    graph_ = true;
    return Status::OK();
  }
  Status FinishGetGraph(void*, bool discard) override {
    if (!graph_) return Status::NotFound("graph lease is gone");
    if (!discard) return Status::Internal("graph reported a device protocol error");
    // Model AICPU's confirmed-failure discard: retire the whole session rather
    // than pretend that its queues are reusable. Only Recover initializes it.
    return Shutdown();
  }
  Status ExecuteBatchGet(const BatchGetExecRequest&) override {
    if (!ready_) return Status::NotInitialized("owner was retired");
    ++counters_.gets;
    return Status::OK();
  }
  Status ExecuteBatchPut(const BatchPutExecRequest&) override { return Status::OK(); }
 private:
  RecoveryCounters& counters_;
  bool ready_{false}, graph_{false};
};

void CheckFailedGraphRecovery(GetExecutionConfig target) {
  const GetExecutionConfig direct{Transport::Urma, Executor::Aiv, GetAlgorithm::Direct};
  ExecutionPolicy policy;
  policy.enabled_put_configs = std::vector<PutExecutionConfig>{};
  policy.enabled_get_configs = std::vector<GetExecutionConfig>{direct, target};
  policy.default_get = direct;
  ExecutionDispatcher dispatcher(policy);
  RecoveryCounters bound, failed;
  CHECK(dispatcher.RegisterGet(direct, [&] { return std::make_unique<FailedGraphProvider>(bound); }).ok());
  CHECK(dispatcher.RegisterGet(target, [&] { return std::make_unique<FailedGraphProvider>(failed); }).ok());
  CHECK(dispatcher.Initialize({4096, 4096, 64, DeviceKind::kCpu}).ok());
  int index = 0;
  char output[64]{};
  BatchGetRequest request;
  request.indices = {&index, sizeof(index), DeviceKind::kNpu};
  request.output = {output, sizeof(output), DeviceKind::kNpu};
  void* stream = reinterpret_cast<void*>(1);
  std::uint64_t output_lease = 0, graph_lease = 0;
  CHECK(dispatcher.BeginGetOutput(direct, {request.output, stream, 1}, &output_lease).ok());
  CHECK(dispatcher.BeginGetGraph(target, {request.output, stream, 1}, &graph_lease).ok());
  BatchGetExecRequest call{{}, &request, {}};
  call.execution.config = target;
  call.execution.stream = stream;
  call.execution.graph_lease = graph_lease;
  CHECK(dispatcher.ExecuteBatchGet(call).ok());
  CHECK(dispatcher.Recover(target).code() == StatusCode::kBusy);
  CHECK(!dispatcher.EndGetGraph(graph_lease, stream, false).ok());
  failed.close_failures = 1;
  CHECK(!dispatcher.EndGetGraph(graph_lease, stream, true).ok());
  CHECK(dispatcher.Recover(target).code() == StatusCode::kBusy);
  CHECK(dispatcher.Shutdown().code() == StatusCode::kBusy);
  CHECK(bound.closes == 0 && bound.output_releases == 0);
  CHECK(dispatcher.EndGetGraph(graph_lease, stream, true).ok());
  call.execution.graph_lease = 0;
  call.execution.synchronize = true;
  CHECK(dispatcher.ExecuteBatchGet(call).code() == StatusCode::kNotInitialized);
  CHECK(dispatcher.Recover(target).ok());
  CHECK(failed.initializes == 2);
  CHECK(dispatcher.ExecuteBatchGet(call).ok());
  CHECK(bound.initializes == 1 && bound.output_binds == 1 && bound.closes == 0);
  call.execution.config = direct;
  call.execution.output_lease = output_lease;
  CHECK(dispatcher.ExecuteBatchGet(call).ok());
  CHECK(bound.gets == 1 && bound.output_binds == 1);
  CHECK(dispatcher.EndGetOutput(output_lease).ok());
  CHECK(bound.output_releases == 1);
  CHECK(dispatcher.Shutdown().ok());
}
int main() {
  const std::array<GetExecutionConfig, 4> configs{{
      {Transport::Urma, Executor::Aiv, GetAlgorithm::Direct},
      {Transport::Ubmem, Executor::Aiv, GetAlgorithm::Direct},
      {Transport::Urma, Executor::Aicpu, GetAlgorithm::Gather},
      {Transport::Urma, Executor::Aicpu, GetAlgorithm::Scatter}}};
  ExecutionPolicy policy;
  policy.transport_mode = TransportMode::UrmaAndUbmem;
  policy.enabled_put_configs = std::vector<PutExecutionConfig>{};
  policy.enabled_get_configs = std::vector<GetExecutionConfig>(configs.begin(), configs.end());
  policy.default_get = configs[2];
  ExecutionDispatcher dispatcher(policy);
  std::array<Counters, 4> counters{};
  for (unsigned i = 0; i < configs.size(); ++i)
    CHECK(dispatcher.RegisterGet(configs[i], [&, i] { return std::make_unique<DiagnosticProvider>(counters[i]); }).ok());
  CHECK(dispatcher.Initialize({4096, 4096, 64, DeviceKind::kCpu}).ok());
  std::vector<BatchGetKernelProfile> profiles;
  CHECK(!dispatcher.CopyGetProfiles(configs[0], &profiles).ok());
  CHECK(dispatcher.ConfigureGetProfiling(std::nullopt, true, false).ok());
  CHECK(counters[2].profiles == 1 && counters[0].profiles == 0);
  CHECK(!dispatcher.ConfigureGetProfiling(configs[0], true, true).ok());
  int index = 0;
  char output[64]{};
  BatchGetRequest request;
  request.indices = {&index, sizeof(index), DeviceKind::kNpu};
  request.output = {output, sizeof(output), DeviceKind::kNpu};
  void* stream = reinterpret_cast<void*>(1);
  for (unsigned i = 0; i < configs.size(); ++i) {
    CHECK(dispatcher.ConfigureGetProfiling(configs[i], true, false).ok());
    std::uint64_t lease = 0, duplicate = 0;
    CHECK(dispatcher.BeginGetGraph(configs[i], {request.output, stream, 1}, &lease).ok());
    CHECK(!dispatcher.ConfigureGetProfiling(configs[i], false, false).ok());
    CHECK(!dispatcher.BeginGetGraph(configs[(i + 1) % 4], {request.output, stream, 1}, &duplicate).ok());
    BatchGetExecRequest call{{}, &request, {}};
    call.execution.config = configs[i]; call.execution.stream = stream; call.execution.graph_lease = lease;
    CHECK(dispatcher.ExecuteBatchGet(call).ok());
    CHECK(!dispatcher.ExecuteBatchGet(call).ok()); // One node per graph remains enforced.
    CHECK(!dispatcher.CheckGetStatus(configs[i], reinterpret_cast<void*>(2)).ok());
    CHECK(dispatcher.CheckGetStatus(configs[i], stream).ok());
    CHECK(dispatcher.CopyGetProfiles(configs[i], &profiles).ok() && profiles.size() == 1);
    CHECK(counters[i].checks == 1 && counters[i].copies == 1 && counters[i].captures == 1);
    CHECK(dispatcher.EndGetGraph(lease, stream, false).ok());
    CHECK(dispatcher.ConfigureGetProfiling(configs[i], false, false).ok());
  }
  CHECK(dispatcher.Shutdown().ok());
  CHECK(!dispatcher.CheckGetStatus(configs[2], stream).ok());
  CheckFailedGraphRecovery(configs[2]);
  CheckFailedGraphRecovery(configs[3]);
}
