#include "access_operator.h"
#include "aiv_execution_resources.h"
#include "meta_manager.h"
#include "kvaf_aiv_batch_get.h"
#include "kvaf_aiv_batch_put.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>

#define CHECK(c) do { if (!(c)) { std::cerr << "line " << __LINE__ << ": " #c "\n"; std::abort(); } } while (false)
struct KvafBatchPutTransport {};
struct KvafBatchPutLocalBuffer {};
struct KvafBatchGetTransport {};
struct KvafBatchGetLocalBuffer {};
namespace {
int launches = 0, destroys = 0, retirements = 0, acquisitions = 0;
bool fail_wait = false, fail_copy = false, fail_retire = false, complete_payload = true;
bool fail_close = false, fail_create = false, fail_free = false;
bool fail_acquire = false, fail_get_acquire = false, fail_memset = false;
int get_launches = 0, get_retirements = 0;
std::vector<KvafBatchPutLocalBuffer*> registrations;
std::vector<KvafBatchGetLocalBuffer*> get_registrations;
}
extern "C" {
aclError aclrtMalloc(void** p, size_t n, aclrtMemMallocPolicy) { *p = std::malloc(n); return *p ? 0 : 1; }
aclError aclrtMallocHost(void** p, size_t n) { *p = std::malloc(n); return *p ? 0 : 1; }
aclError aclrtFree(void* p) { if (fail_free) return 1; std::free(p); return 0; }
aclError aclrtFreeHost(void* p) { std::free(p); return 0; }
aclError aclrtMemset(void* p, size_t, int value, size_t n) {
  if (fail_memset) return 1;
  std::memset(p, value, n); return 0;
}
aclError aclrtMemsetAsync(void* p, size_t size, int value, size_t n, aclrtStream) { return aclrtMemset(p, size, value, n); }
aclError aclrtMemcpy(void* p, size_t, const void* s, size_t n, aclrtMemcpyKind kind) {
  if (fail_copy && kind == ACL_MEMCPY_DEVICE_TO_HOST) return 1;
  std::memcpy(p, s, n); return 0;
}
aclError aclrtMemcpyAsync(void* p, size_t size, const void* s, size_t n, aclrtMemcpyKind kind, aclrtStream) {
  return aclrtMemcpy(p, size, s, n, kind);
}
aclError aclrtSynchronizeStream(aclrtStream) { return fail_wait ? 1 : 0; }
aclError aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus* status, aclmdlRI*) {
  *status = ACL_MODEL_RI_CAPTURE_STATUS_NONE; return 0;
}
int32_t KvafExecuteBatchPutLoadKernel(const char*) { return 0; }
int32_t KvafBatchPutTransportCreateWithPeerLink(void*, uint64_t, uint32_t, uint32_t, void*, const uint8_t*, KvafBatchPutTransport** out) {
  *out = new KvafBatchPutTransport; return fail_create ? -1 : 0;
}
int32_t KvafBatchPutTransportClose(KvafBatchPutTransport* p) {
  if (fail_close) return -1;
  if (p) {
    for (auto* buffer : registrations) delete buffer;
    registrations.clear();
    ++destroys; delete p;
  }
  return 0;
}
void KvafBatchPutTransportDestroy(KvafBatchPutTransport* p) { (void)KvafBatchPutTransportClose(p); }
int32_t KvafBatchPutLocalBufferAcquire(KvafBatchPutTransport*, void*, uint64_t, KvafBatchPutLocalBuffer** out) {
  ++acquisitions;
  *out = new KvafBatchPutLocalBuffer; registrations.push_back(*out); return fail_acquire ? -1 : 0;
}
void KvafBatchPutLocalBufferRelease(KvafBatchPutLocalBuffer* p) { delete p; }
int32_t KvafBatchPutLocalBufferRetire(KvafBatchPutLocalBuffer* p) {
  if (fail_retire) return -4;
  registrations.erase(std::remove(registrations.begin(), registrations.end(), p), registrations.end());
  ++retirements; delete p; return 0;
}
int32_t KvafExecuteBatchPutSync(KvafBatchPutLocalBuffer*, const KvafExecuteBatchPutKernelArgs* args,
                               aclrtStream, KvafBatchPutCoreStatus*, uint32_t count) {
  ++launches;
  if (complete_payload) {
    auto* reports = reinterpret_cast<KvafBatchPutCoreStatus*>(args->core_status);
    for (unsigned i = 0; i < count; ++i) {
      reports[i] = {};
      reports[i].magic = KVAF_AIV_BATCH_PUT_STATUS_MAGIC;
      reports[i].failed_request_index = UINT64_MAX;
      reports[i].processed_request_count = std::uint64_t(args->row_count) * (i + 1) / count -
                                          std::uint64_t(args->row_count) * i / count;
    }
  }
  // Simulate a Host wait/report failure after a potentially accepted launch.
  return KVAF_BATCH_PUT_RUNTIME_ERROR;
}
int32_t KvafExecuteBatchGetLoadKernel(const char*) { return 0; }
int32_t KvafBatchGetTransportCreateWithPeerLinkMultiEid(void*, uint64_t, uint32_t, uint32_t, void*, const uint8_t*, uint32_t, KvafBatchGetTransport** out) {
  *out = new KvafBatchGetTransport; return 0;
}
void KvafBatchGetTransportDestroy(KvafBatchGetTransport*) {}
int32_t KvafBatchGetTransportClose(KvafBatchGetTransport* p) {
  if (fail_close) return -1;
  for (auto* buffer : get_registrations) delete buffer;
  get_registrations.clear(); delete p; return 0;
}
int32_t KvafBatchGetLocalBufferAcquire(KvafBatchGetTransport*, void*, uint64_t, KvafBatchGetLocalBuffer** out, uint64_t* entity) {
  *out = new KvafBatchGetLocalBuffer; get_registrations.push_back(*out);
  *entity = fail_get_acquire ? 0 : 1; return fail_get_acquire ? -1 : 0;
}
void KvafBatchGetLocalBufferRelease(KvafBatchGetLocalBuffer*) {}
int32_t KvafBatchGetLocalBufferRetire(KvafBatchGetLocalBuffer* p) {
  if (fail_retire) return -1;
  get_registrations.erase(std::remove(get_registrations.begin(), get_registrations.end(), p), get_registrations.end());
  ++get_retirements; delete p; return 0;
}
int32_t KvafExecuteBatchGetLaunch(const KvafExecuteBatchGetKernelArgs*, aclrtStream) { ++get_launches; return -1; }
}
int main() {
  using namespace kvaf;
  AccessOperatorConfig config;
  config.npu_core_count = 2;
  config.enable_get = false;
  config.managed_completion = true;
  config.batch_put_kernel_object_path = "test-put.o";
  config.peer_local_eid[0] = 1;
  // This test's transport factory treats bootstrap identity as opaque.
  config.peer_link = std::shared_ptr<PeerLink>(reinterpret_cast<PeerLink*>(1), [](PeerLink*) {});
  auto op = std::make_unique<DefaultAccessOperator>(config);
  auto resources = op->TakeResources();
  CHECK(resources && !op->TakeResources());
  RemoteRegionDesc region; region.base = 0x100000; region.length = 4096; region.device = DeviceKind::kCpu;
  CHECK(resources->Initialize(region).ok());
  auto state = op->CreateExecutionState();
  CHECK(op->BindExecutionState(state.get()).ok());
  DefaultAccessOperator other(config);
  auto foreign = other.CreateExecutionState();
  CHECK(!op->BindExecutionState(foreign.get()).ok());
  std::uint8_t input[64]{};
  BatchPutRequest request;
  request.keys = {1}; request.input_offsets = {0}; request.lengths = {64};
  request.input = {input, sizeof(input), DeviceKind::kNpu};
  PutAccessDesc destinations;
  destinations.entries.push_back({1, 0x100000, 64});
  BatchPutExecRequest call{&request, &destinations, {reinterpret_cast<void*>(1)}};
  CHECK(!op->ExecuteBatchPut(call).ok() && launches == 1);
  PutCompletion completion;
  fail_wait = true;
  CHECK(!op->ResolvePut(&completion).ok() && retirements == 0);
  fail_wait = false; fail_copy = true;
  CHECK(!op->ResolvePut(&completion).ok() && retirements == 0);
  fail_copy = false; fail_retire = true;
  CHECK(!op->ResolvePut(&completion).ok());
  fail_retire = false;
  CHECK(op->ResolvePut(&completion).ok() && completion.outcome == PutOutcome::Succeeded);
  CHECK(retirements == 1 && launches == 1);
  CHECK(op->ResolvePut(&completion).ok() && completion.outcome == PutOutcome::Succeeded);
  CHECK(op->AcknowledgePut().ok());
  CHECK(op->BindExecutionState(nullptr).ok());
  state = op->CreateExecutionState();
  CHECK(op->BindExecutionState(state.get()).ok());
  // The last successful report must not make a failed next launch look done.
  complete_payload = false;
  CHECK(!op->ExecuteBatchPut(call).ok() && launches == 2);
  CHECK(!op->ResolvePut(&completion).ok() && retirements == 1);
  fail_close = true;
  CHECK(!resources->Close().ok() && destroys == 0 && retirements == 1);
  complete_payload = true;
  CHECK(!op->ResolvePut(&completion).ok()); // Partially closed queues may only retry Close.
  fail_close = false;
  CHECK(resources->Close().ok());
  CHECK(op->BindExecutionState(nullptr).ok());
  state.reset(); op.reset();
  CHECK(destroys == 1);
  resources.reset();
  CHECK(destroys == 1 && retirements == 1 && registrations.empty());
  // A factory can return a partial transport after failed setup/cleanup.
  fail_create = true; fail_close = true;
  DefaultAccessOperator partial(config);
  auto partial_resources = partial.TakeResources();
  CHECK(!partial_resources->Initialize(region).ok());
  CHECK(!partial_resources->Close().ok() && destroys == 1);
  fail_close = false;
  CHECK(partial_resources->Close().ok() && destroys == 2);
  fail_create = false;
  DefaultAccessOperator partial_registration(config);
  CHECK(partial_registration.Initialize(region).ok());
  fail_acquire = true; fail_retire = true;
  const auto old_launches = launches, old_retirements = retirements;
  CHECK(!partial_registration.ExecuteBatchPut(call).ok() && launches == old_launches);
  CHECK(!partial_registration.ResolvePut(&completion).ok() && !registrations.empty());
  fail_retire = false; fail_copy = true;
  CHECK(partial_registration.ResolvePut(&completion).ok() && completion.outcome == PutOutcome::Failed);
  CHECK(retirements == old_retirements + 1 && registrations.empty());
  CHECK(partial_registration.AcknowledgePut().ok());
  fail_copy = false; fail_acquire = false; fail_memset = true; fail_retire = true;
  CHECK(!partial_registration.ExecuteBatchPut(call).ok() && launches == old_launches);
  CHECK(!partial_registration.ResolvePut(&completion).ok() && !registrations.empty());
  fail_memset = false; fail_retire = false;
  CHECK(partial_registration.ResolvePut(&completion).ok() && completion.outcome == PutOutcome::Failed);
  CHECK(partial_registration.AcknowledgePut().ok());
  CHECK(partial_registration.Shutdown().ok());
  config.enable_get = true;
  config.batch_get_kernel_object_path = "test-get.o";
  config.batch_put_kernel_object_path.clear();
  DefaultAccessOperator partial_get(config);
  CHECK(partial_get.Initialize(region).ok());
  int32_t index = 0, count = 1;
  uint64_t address = region.base;
  uint8_t valid = 1;
  BatchGetRequest get;
  get.indices = get.offsets = get.output_indices = {&index, sizeof(index), DeviceKind::kNpu};
  get.miss_count = {&count, sizeof(count), DeviceKind::kNpu};
  get.output = {input, sizeof(input), DeviceKind::kNpu};
  get.output_block_bytes = 64; get.tokens_per_block = 1;
  get.device_access_table.addresses = {&address, sizeof(address), DeviceKind::kNpu};
  get.device_access_table.valid = {&valid, sizeof(valid), DeviceKind::kNpu};
  BatchGetExecRequest get_call;
  get_call.request = &get; get_call.execution.stream = reinterpret_cast<void*>(1);
  fail_get_acquire = true; fail_retire = true;
  CHECK(!partial_get.ExecuteBatchGet(get_call).ok() && get_launches == 0);
  CHECK(!partial_get.DrainGet().ok() && get_registrations.size() == 1);
  fail_retire = false; fail_copy = true;
  CHECK(partial_get.DrainGet().ok() && get_retirements == 1 && get_registrations.empty());
  fail_copy = false;
  CHECK(!partial_get.PrepareGetGraph({get.output, get_call.execution.stream, 1}).ok());
  CHECK(get_registrations.size() == 1);
  fail_close = true;
  CHECK(partial_get.Shutdown().code() == StatusCode::kCompletionUnknown); // Not a graph-lease Busy.
  fail_close = false;
  CHECK(partial_get.Shutdown().ok() && get_registrations.empty());
  // Resource setup/teardown must work after the compatibility operator is gone.
  config.enable_get = false;
  config.batch_put_kernel_object_path = "test-put.o";
  auto detached_adapter = std::make_unique<DefaultAccessOperator>(config);
  auto detached_owner = detached_adapter->TakeResources();
  detached_adapter.reset();
  const auto before_detached = destroys;
  CHECK(detached_owner->Initialize(region).ok());
  CHECK(detached_owner->Warmup({8, call.execution.stream}).ok());
  CHECK(detached_owner->Close().ok() && destroys == before_detached + 1);

  // Algorithm views carry a resource identity; a foreign invocation is rejected
  // before any registration/submission. Reusing a session preserves workspace.
  AivExecutionResources direct(config), unrelated(config);
  CHECK(direct.Initialize(region).ok());
  auto direct_state = direct.CreateState();
  auto* direct_call = dynamic_cast<AivInvocation*>(direct_state.get());
  auto unrelated_state = unrelated.CreateState();
  auto* unrelated_call = dynamic_cast<AivInvocation*>(unrelated_state.get());
  CHECK(direct_call && unrelated_call && direct.Bind(direct_call).ok());
  const auto before_launch = launches;
  auto unbound = direct.CreateState();
  CHECK(!UrmaAivPutOperator::Launch(call, direct.PutView(),
      *dynamic_cast<AivInvocation*>(unbound.get())).ok());
  CHECK(!UrmaAivPutOperator::Launch(call, direct.PutView(), *unrelated_call).ok());
  CHECK(launches == before_launch && registrations.empty());
  // Prepared workspace is required by the algorithm; only the owner may grow it.
  CHECK(!UrmaAivPutOperator::Launch(call, direct.PutView(), *direct_call).ok());
  CHECK(launches == before_launch && registrations.empty());
  CHECK(direct.Warmup({8, call.execution.stream}).ok());
  const auto workspace = direct.PutView().session.workspace.metadata;
  auto prepared = direct.PutView();
  CHECK(direct.PreparePut(call, *direct_call, &prepared).ok());
  const auto registered_before_launch = acquisitions;
  CHECK(direct.Bind(unbound.get()).code() == StatusCode::kBusy);
  auto changed_stream = call;
  changed_stream.execution.stream = reinterpret_cast<void*>(2);
  CHECK(!UrmaAivPutOperator::Launch(changed_stream, prepared, *direct_call).ok());
  CHECK(launches == before_launch && acquisitions == registered_before_launch);
  CHECK(!UrmaAivPutOperator::Launch(call, prepared, *direct_call).ok());
  CHECK(launches == before_launch + 1 && acquisitions == registered_before_launch);
  CHECK(direct.Warmup({16, call.execution.stream}).code() == StatusCode::kBusy);
  CHECK(UrmaAivPutOperator::Launch(call, prepared, *direct_call).code() == StatusCode::kBusy);
  CHECK(launches == before_launch + 1 && acquisitions == registered_before_launch);
  CHECK(UrmaAivPutOperator::Resolve(&completion, direct.PutView(), *direct_call).ok());
  CHECK(completion.outcome == PutOutcome::Succeeded);
  CHECK(UrmaAivPutOperator::Acknowledge(direct.PutView(), *direct_call).ok());
  CHECK(direct.Warmup({4, call.execution.stream}).ok());
  CHECK(workspace == direct.PutView().session.workspace.metadata);
  const auto closed_transports = destroys;
  fail_free = true;
  CHECK(!direct.Close().ok() && destroys == closed_transports + 1);
  CHECK(direct.PutView().session.workspace.metadata == workspace);
  CHECK(!UrmaAivPutOperator::Launch(call, prepared, *direct_call).ok());
  CHECK(launches == before_launch + 1);
  fail_free = false;
  CHECK(direct.Close().ok() && destroys == closed_transports + 1);
  CHECK(!UrmaAivPutOperator::Launch(call, prepared, *direct_call).ok());
  CHECK(direct.Bind(nullptr).ok());
  std::cout << "AIV independent resources, typed views and non-replaying Put resolution passed\n";
}
