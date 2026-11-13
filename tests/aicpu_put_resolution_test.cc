#include "aicpu_urma/session.h"
#include "aicpu_urma/kernel_loader.h"
#include "aicpu_batch_put/include/kvaf_aicpu_batch_put.h"
#include "meta_manager.h"
#include "aicpu_execution_resources.h"
#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

#define CHECK(c) do { if (!(c)) { std::cerr << "line " << __LINE__ << ": " #c "\n"; std::abort(); } } while (false)
namespace {
using namespace kvaf::aicpu_urma;
int single_unregisters = 0;
int writes = 0, unregisters = 0, allocations = 0, registrations = 0, destroys = 0, frees = 0;
bool fail_wait = false, fail_copy = false, write_error = false;
bool fail_launch = false, fail_memset = false, fail_wait_on_write = false, fail_wait_on_prepare = false;
std::vector<UmdkTransfer> submitted;
}
extern "C" {
aclError aclrtCreateStream(aclrtStream* p) { *p = reinterpret_cast<aclrtStream>(1); return 0; }
aclError aclrtDestroyStream(aclrtStream) { return 0; }
aclError aclrtMalloc(void** p, size_t n, aclrtMemMallocPolicy) { ++allocations; *p = std::malloc(n); return *p ? 0 : 1; }
aclError aclrtFree(void* p) { ++frees; std::free(p); return 0; }
aclError aclrtMemset(void* p, size_t, int value, size_t n) {
  if (fail_memset) return 1;
  std::memset(p, value, n); return 0;
}
aclError aclrtMemcpy(void* p, size_t, const void* s, size_t n, aclrtMemcpyKind kind) {
  if (fail_copy && kind == ACL_MEMCPY_DEVICE_TO_HOST) return 1;
  std::memcpy(p, s, n); return 0;
}
aclError aclrtSynchronizeStream(aclrtStream) { return fail_wait ? 1 : 0; }
aclError aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus* status, aclmdlRI*) {
  *status = ACL_MODEL_RI_CAPTURE_STATUS_NONE; return 0;
}
}
namespace kvaf::peer {
PeerLink::PeerLink(std::string address) : address_(std::move(address)) {}
Status PeerLink::LookupRegion(uint64_t, PeerRegion* region) const {
  region->segment_key_bytes = 28; return Status::OK();
}
Status PeerLink::CreateChannel(uint64_t, const uint8_t*, uint32_t, const uint8_t*, uint32_t,
    uint64_t, uint32_t, std::array<uint8_t, kQpKeyBytes>*, uint32_t* bytes,
    uint64_t* tp, uint32_t* psn, bool) {
  *bytes = 28; *tp = 1; *psn = 1; return Status::OK();
}
}
namespace kvaf::aicpu_urma {
// This test exercises real Put transport/session code. The unused Gather
// resource is a fake so the test needs neither the Get kernel nor NPU hardware.
struct BatchGetTransport::Impl {
  explicit Impl(UmdkSession& s) : session(s) {}
  UmdkSession& session;
  uint64_t region = 0;
  uint8_t workspace[64]{};
};
BatchGetTransport::BatchGetTransport(UmdkSession& s, std::uint32_t, bool) : impl_(std::make_unique<Impl>(s)) {}
BatchGetTransport::~BatchGetTransport() = default;
Status BatchGetTransport::Initialize(const RemoteRegionDesc& r) { impl_->region = r.base; return Status::OK(); }
Status BatchGetTransport::Warmup() { return impl_->session.Ensure(impl_->region, impl_->workspace, sizeof(impl_->workspace)); }
Status BatchGetTransport::Shutdown() { return Status::OK(); }
Status BatchGetTransport::Drain() { return Status::OK(); }
Status BatchGetTransport::Prepare(const BatchGetExecRequest& r) {
  auto s = Warmup();
  return s.ok() ? impl_->session.Ensure(impl_->region, r.request->output.data, r.request->output.bytes) : s;
}
Status BatchGetTransport::ExecutePrepared(const BatchGetExecRequest&) { return Status::NotImplemented("unused"); }
GatherLaunchView BatchGetTransport::View() const { return {}; }
Status BatchGetTransport::LaunchPrepared(const BatchGetExecRequest&, const GatherLaunchView&, ExecutionState*) { return Status::NotImplemented("unused"); }
std::unique_ptr<ExecutionState> BatchGetTransport::CreateExecutionState() { return {}; }
Status BatchGetTransport::BindExecutionState(ExecutionState*) { return Status::OK(); }
Status PrepareUmdkKernel(const std::string&, bool) { return Status::OK(); }
int32_t LaunchUmdkKernel(const char*, const void* input, size_t, aclrtStream, uint32_t) {
  const auto& args = *static_cast<const UmdkControlArgs*>(input);
  auto& status = *reinterpret_cast<UmdkControlStatus*>(args.status);
  status = {}; status.magic = kUmdkControlMagic; status.session = 1; status.owner_pid = 123;
  status.key_bytes = 28;
  std::memcpy(status.eid, args.local_eid, sizeof(status.eid));
  if (args.op == UmdkControlOp::kUnregisterBuffers) ++unregisters;
  if (args.op == UmdkControlOp::kUnregisterBuffer) ++single_unregisters;
  if (args.op == UmdkControlOp::kPrepare || args.op == UmdkControlOp::kRegisterBuffer) ++registrations;
  if (args.op == UmdkControlOp::kPrepare && fail_wait_on_prepare) fail_wait = true;
  if (args.op == UmdkControlOp::kDestroy) ++destroys;
  return 0;
}
int32_t LaunchBatchPut(const BatchPutKernelArgs* args, aclrtStream stream) {
  ++writes;
  if (fail_wait_on_write) fail_wait = true;
  CHECK(args->op == UmdkControlOp::kWrite);
  const auto* transfers = reinterpret_cast<const UmdkTransfer*>(args->transfer_spans);
  submitted.assign(transfers, transfers + args->transfer_count);
  LaunchUmdkKernel("", args, sizeof(*args), stream, 1);
  reinterpret_cast<UmdkControlStatus*>(args->status)->result = write_error ? -1 : 0;
  return fail_launch ? -1 : 0;
}
}
int main() {
  using namespace kvaf;
  using namespace kvaf::aicpu_urma;
  UmdkSessionConfig config;
  config.peer_link = std::make_shared<PeerLink>("fake:0");
  config.kernel_json = "fake.json"; config.workers = 2; config.local_eid[0] = 1;
  {
    UmdkSession partial(config);
    uint8_t setup[64]{};
    fail_wait_on_prepare = true;
    CHECK(!partial.EnsureSession(0x100000, setup, sizeof(setup)).ok());
    const auto partial_regs = registrations;
    CHECK(!partial.EnsureSession(0x100000, setup, sizeof(setup)).ok());
    CHECK(registrations == partial_regs); // A pending Prepare must not be replayed.
    fail_wait_on_prepare = false; fail_wait = false;
    CHECK(partial.Shutdown().ok()); // Recover ownership, then destroy the partial session.
    CHECK(partial.EnsureSession(0x100000, setup, sizeof(setup)).ok());
    CHECK(registrations == partial_regs + 1);
    CHECK(partial.Shutdown().ok());
  }
  UmdkSession session(config);
  BatchPutTransport transport(session);
  UmdkSession::WriteState invocation;
  session.BindWriteState(&invocation);
  CHECK(session.Load().ok());
  uint8_t input[256]{};
  RemoteRegionDesc region; region.base = 0x100000; region.length = 4096; region.device = DeviceKind::kCpu;
  CHECK(session.Ensure(region.base, input, sizeof(input)).ok());
  BatchPutRequest request;
  request.keys = {1, 2, 3}; request.input_offsets = {0, 80, 160}; request.lengths = {64, 64, 64};
  request.input = {input, sizeof(input), DeviceKind::kNpu};
  PutAccessDesc destinations;
  for (unsigned i = 0; i < 3; ++i) destinations.entries.push_back({i + 1, region.base + 128 * i, 64});
  BatchPutExecRequest call{&request, &destinations, {reinterpret_cast<void*>(2)}};
  fail_wait = true;
  CHECK(!transport.Execute(region, call).ok());
  CHECK(writes == 1 && submitted.size() == 3);
  for (unsigned i = 0; i < 3; ++i) {
    CHECK(submitted[i].local == reinterpret_cast<uint64_t>(input) + request.input_offsets[i]);
    CHECK(submitted[i].remote == destinations.entries[i].address && submitted[i].bytes == 64);
  }
  PutCompletion completion;
  CHECK(!session.ResolveWrite(&completion).ok());
  CHECK(!session.ReleaseBuffers().ok() && unregisters == 0);
  fail_wait = false; fail_copy = true;
  CHECK(!session.ResolveWrite(&completion).ok());
  fail_copy = false;
  CHECK(session.ResolveWrite(&completion).ok() && completion.outcome == PutOutcome::Succeeded);
  CHECK(writes == 1); // Recover the original whole list, never issue another WRITE.
  // Later cleanup failure must preserve the original data result and retry cleanup only.
  fail_wait = true;
  CHECK(!session.ReleaseBuffers().ok() && unregisters == 1);
  CHECK(session.ResolveWrite(&completion).ok() && completion.outcome == PutOutcome::Succeeded);
  fail_wait = false;
  CHECK(session.ReleaseBuffers().ok() && unregisters == 1 && writes == 1);
  invocation = {};
  write_error = true;
  CHECK(!transport.Execute(region, call).ok() && writes == 2);
  CHECK(!session.ResolveWrite(&completion).ok()); // Failed remote WRs are not safe to abort.
  CHECK(invocation.completion.outcome == PutOutcome::Unknown);
  CHECK(!transport.Execute(region, call).ok() && writes == 2); // No replay/reuse while poisoned.
  CHECK(transport.Shutdown().ok());
  CHECK(session.Shutdown().ok());
  invocation = {};
  write_error = false;
  CHECK(session.Ensure(region.base, input, sizeof(input)).ok());
  fail_memset = true;
  CHECK(!transport.Execute(region, call).ok() && writes == 2 && !invocation.attempted);
  CHECK(session.ResolveWrite(&completion).ok() && completion.outcome == PutOutcome::Failed);
  fail_memset = false; fail_launch = true;
  CHECK(!transport.Execute(region, call).ok() && writes == 3);
  CHECK(session.ResolveWrite(&completion).ok() && completion.outcome == PutOutcome::Succeeded);
  CHECK(writes == 3);
  CHECK(transport.Shutdown().ok());
  CHECK(session.Shutdown().ok());
  session.BindWriteState(nullptr);
  fail_launch = false;
  UmdkAccessOperatorConfig provider;
  provider.kernel_json = config.kernel_json; provider.workers = config.workers;
  provider.local_eid = config.local_eid; provider.peer_link = config.peer_link;
  provider.enable_get = false; provider.managed_completion = true;
  auto adapter = std::make_unique<UmdkAccessOperator>(provider);
  auto owner = adapter->TakeResources();
  CHECK(owner && !adapter->TakeResources());
  // Initialize and reserve without entering any operator lifecycle method.
  CHECK(owner->Initialize(region).ok());
  CHECK(owner->Warmup({3, call.execution.stream}).ok());
  auto state = adapter->CreateExecutionState();
  CHECK(adapter->BindExecutionState(state.get()).ok());
  auto* resources = dynamic_cast<AicpuExecutionResources*>(owner.get());
  auto* invocation_state = dynamic_cast<AicpuInvocation*>(state.get());
  CHECK(resources && invocation_state);
  AicpuExecutionResources foreign_resources(provider);
  auto foreign_state = foreign_resources.CreateInvocation();
  CHECK(!resources->Bind(foreign_state.get()).ok());
  CHECK(resources->PreparePutResources(call).ok());
  const auto old_put_view = resources->PutView();
  CHECK(resources->PreparePutResources(call).ok());
  CHECK(!UrmaAicpuPutOperator::Launch(call, old_put_view, *invocation_state).ok());
  const auto before_allocations = allocations, before_registrations = registrations;
  const auto before_writes = writes;
  auto same_owner_unbound = resources->CreateInvocation();
  CHECK(!UrmaAicpuPutOperator::Launch(call, resources->PutView(),
        *static_cast<AicpuInvocation*>(same_owner_unbound.get())).ok());
  auto swapped = request;
  auto swapped_call = call; swapped_call.request = &swapped;
  CHECK(!UrmaAicpuPutOperator::Launch(swapped_call, resources->PutView(), *invocation_state).ok());
  // Concrete views reject foreign invocation identities before any submission.
  CHECK(!UrmaAicpuPutOperator::Launch(call, resources->PutView(),
        *static_cast<AicpuInvocation*>(foreign_state.get())).ok());
  CHECK(writes == before_writes);
  CHECK(UrmaAicpuPutOperator::Launch(call, resources->PutView(), *invocation_state).ok());
  CHECK(writes == before_writes + 1 && allocations == before_allocations && registrations == before_registrations);
  CHECK(adapter->AcknowledgePut().ok());
  CHECK(!UrmaAicpuPutOperator::Launch(call, resources->PutView(), *invocation_state).ok());
  CHECK(writes == before_writes + 1); // A prepared transfer slot is consumed once.
  // The owned setup buffer is only needed to establish the session. Ack removed
  // it with the input registrations; later warmups must keep the session and
  // descriptor storage without repeatedly registering those same 64 bytes.
  const auto warmed_allocations = allocations;
  for (int i = 0; i < 3; ++i) {
    const auto regs = registrations, unregs = unregisters;
    CHECK(owner->Warmup({3, call.execution.stream}).ok());
    CHECK(registrations == regs && allocations == warmed_allocations);
    PutProviderTiming timing;
    call.execution.host_timing = &timing;
    CHECK(adapter->ExecuteBatchPut(call).ok());
    call.execution.host_timing = nullptr;
    CHECK(registrations == regs + 1 && unregisters == unregs + 1);
    CHECK(allocations == warmed_allocations);
    CHECK(std::strcmp(timing.transport, "urma") == 0 && std::strcmp(timing.executor, "aicpu") == 0);
    CHECK(timing.prepare_ns >= timing.registration_ns && timing.registration_ns > 0);
    CHECK(timing.metadata_ns > 0 && timing.submit_ns > 0 && timing.completion_ns > 0);
    CHECK(timing.cleanup_ns > 0);
    CHECK(adapter->AcknowledgePut().ok());
    CHECK(unregisters == unregs + 1); // Managed completion already released inputs.
  }
  fail_wait_on_write = true;
  CHECK(!adapter->ExecuteBatchPut(call).ok());
  fail_wait_on_write = false;
  const auto pending_writes = writes, pending_frees = frees;
  const auto pending_regs = registrations;
  CHECK(!owner->Warmup({3, call.execution.stream}).ok() && registrations == pending_regs);
  CHECK(!owner->Close().ok() && frees == pending_frees);
  fail_wait = false;
  CHECK(adapter->ResolvePut(&completion).ok() && completion.outcome == PutOutcome::Succeeded);
  CHECK(writes == pending_writes); // Owner failure must not replay the original DMA.
  CHECK(adapter->AcknowledgePut().ok());
  CHECK(adapter->BindExecutionState(nullptr).ok());
  state.reset();
  const auto before_destroys = destroys;
  adapter.reset();
  CHECK(destroys == before_destroys); // Destroying the borrowed algorithm does not close communication.
  CHECK(owner->Close().ok() && destroys == before_destroys + 1);
  CHECK(owner->Close().ok() && destroys == before_destroys + 1);
  const auto closed_regs = registrations;
  CHECK(owner->Initialize(region).ok());
  CHECK(owner->Warmup({3, call.execution.stream}).ok() && registrations == closed_regs + 1);
  CHECK(owner->Close().ok());
  // A stale view from a closed owner cannot launch a prepared transfer.
  AicpuInvocation closed_call(resources);
  CHECK(!UrmaAicpuPutOperator::Launch(call, resources->PutView(), closed_call).ok());
  provider.enable_get = true;
  UmdkAccessOperator gather(provider);
  auto gather_owner = gather.TakeResources();
  auto* gather_resources = dynamic_cast<AicpuExecutionResources*>(gather_owner.get());
  CHECK(gather_owner->Initialize(region).ok());
  auto gather_state = gather.CreateExecutionState();
  CHECK(gather.BindExecutionState(gather_state.get()).ok());
  const auto base_regs = registrations, base_unregs = single_unregisters, base_all = unregisters;
  GetOutputOptions binding{{input, sizeof(input), DeviceKind::kNpu}, call.execution.stream, 3};
  CHECK(gather.PrepareGetOutput(binding).ok());
  CHECK(registrations == base_regs + 2); // Workspace and output, once each.
  BatchGetRequest r; r.output = binding.output;
  BatchGetExecRequest g; g.request = &r; g.execution.stream = call.execution.stream;
  for (int i = 0; i < 3; ++i) {
    CHECK(gather_resources->PrepareGetResources(g).ok());
    CHECK(gather_resources->ReleaseGetBuffers().ok());
    CHECK(gather.DrainGet().ok());
  }
  CHECK(registrations == base_regs + 2 && single_unregisters == base_unregs);
  CHECK(gather.BindExecutionState(nullptr).code() == StatusCode::kBusy);
  fail_wait = true;
  CHECK(!gather.FinishGetOutput().ok() && single_unregisters == base_unregs + 1);
  fail_wait = false;
  CHECK(gather.FinishGetOutput().ok() && single_unregisters == base_unregs + 1); // Never resubmit.
  CHECK(unregisters == base_all);
  CHECK(gather.PrepareGetOutput(binding).ok() && registrations == base_regs + 3); // Workspace reused.
  CHECK(gather.FinishGetOutput().ok());
  for (int i = 0; i < 2; ++i) {
    CHECK(gather_resources->PrepareGetResources(g).ok());
    CHECK(gather.DrainGet().ok());
  }
  CHECK(registrations == base_regs + 5 && single_unregisters == base_unregs + 4);
  CHECK(unregisters == base_all); // Ordinary transient Gets also keep workspace.
  CHECK(gather.BindExecutionState(nullptr).ok());
  CHECK(gather_owner->Close().ok());
  std::cout << "AICPU typed resources, prepare/launch separation and WRITE recovery passed\n";
}
