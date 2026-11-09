#include "ubmem/access_operator.h"
#include "ubmem/batch_abi.h"
#include "ubmem/device_mapping.h"
#include "peer_link.h"
#include "meta_manager.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>

#define CHECK(c) do { if (!(c)) { std::cerr << "line " << __LINE__ << ": " #c "\n"; std::abort(); } } while (false)
namespace {
int launches = 0, mappings_closed = 0, releases = 0, allocations = 0;
bool fail_free = false;
bool fail_wait = false, fail_copy = false, incomplete = false, fail_launch = false;
}
extern "C" {
aclError aclrtSetDevice(int32_t) { return 0; }
aclError aclrtMalloc(void** p, size_t n, aclrtMemMallocPolicy) { ++allocations; *p = std::malloc(n); return *p ? 0 : 1; }
aclError aclrtMallocHost(void** p, size_t n) { ++allocations; *p = std::malloc(n); return *p ? 0 : 1; }
aclError aclrtFree(void* p) { if (fail_free) return 1; std::free(p); return 0; }
aclError aclrtFreeHost(void* p) { std::free(p); return 0; }
aclError aclrtCreateEvent(aclrtEvent* p) { *p = reinterpret_cast<aclrtEvent>(1); return 0; }
aclError aclrtDestroyEvent(aclrtEvent) { return 0; }
aclError aclrtRecordEvent(aclrtEvent, aclrtStream) { return 0; }
aclError aclrtSynchronizeEvent(aclrtEvent) { return fail_wait ? 1 : 0; }
aclError aclrtMemsetAsync(void* p, size_t, int value, size_t n, aclrtStream) {
  std::memset(p, value, n); return 0;
}
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
aclrtBinary aclrtCreateBinary(const void*, size_t) { return reinterpret_cast<aclrtBinary>(1); }
aclError aclrtDestroyBinary(aclrtBinary) { return 0; }
aclError aclrtBinaryLoad(aclrtBinary, aclrtBinHandle* p) { *p = reinterpret_cast<aclrtBinHandle>(1); return 0; }
aclError aclrtBinaryUnLoad(aclrtBinHandle) { return 0; }
aclError aclrtBinaryGetFunction(aclrtBinHandle, const char*, aclrtFuncHandle* p) {
  *p = reinterpret_cast<aclrtFuncHandle>(1); return 0;
}
aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle, uint32_t blocks, aclrtStream,
    aclrtLaunchKernelCfg*, void* args, size_t, aclrtPlaceHolderInfo*, size_t) {
  ++launches;
  if (fail_launch) return 1;
  const auto& a = *static_cast<kvaf::ubmem::BatchArgs*>(args);
  auto* reports = reinterpret_cast<kvaf::ubmem::CoreStatus*>(a.statuses);
  for (unsigned i = 0; i < blocks; ++i) {
    reports[i] = {};
    reports[i].completed = std::uint64_t(a.capacity) * (i + 1) / blocks -
                           std::uint64_t(a.capacity) * i / blocks;
  }
  if (incomplete) reports[blocks - 1].completed = 0;
  return 0;
}
}
namespace kvaf::peer {
PeerLink::PeerLink(std::string address) : address_(std::move(address)) {}
Status PeerLink::LookupRegion(uint64_t, PeerRegion* region) const { region->fabric_mapping_bytes = 4096; return Status::OK(); }
Status PeerLink::RetainUbmemMapping(uint64_t) { return Status::OK(); }
void PeerLink::ReleaseUbmemMapping(uint64_t) { ++releases; }
}
namespace kvaf::ubmem {
Status Acl(const char* operation, aclError code) { return code ? Status::Internal(operation) : Status::OK(); }
DeviceMapping::~DeviceMapping() { (void)Close(); }
Status DeviceMapping::Import(const FabricHandle&, uint64_t) { address_ = reinterpret_cast<void*>(0x200000); return Status::OK(); }
Status DeviceMapping::Close() { if (address_) ++mappings_closed; address_ = nullptr; return Status::OK(); }
Status DeviceMappingCache::Acquire(uint32_t, const FabricHandle&, uint64_t, std::shared_ptr<DeviceMapping>*) {
  return Status::NotImplemented("unused cache in provider test");
}
Status DeviceMappingCache::Release(std::shared_ptr<DeviceMapping>*) {
  return Status::NotImplemented("unused cache in provider test");
}
}  // namespace kvaf::ubmem
int main() {
  using namespace kvaf;
  ubmem::OperatorOptions options;
  options.batch_put_kernel_object_path = __FILE__; // The fake loader accepts any nonempty bytes.
  options.cores = 2;
  options.managed_completion = true;
  options.peer_link = std::make_shared<PeerLink>("fake:0");
  auto op = std::make_unique<ubmem::AccessOperator>(options);
  auto resources = op->TakeResources();
  CHECK(resources && !op->TakeResources());
  RemoteRegionDesc region; region.base = 0x100000; region.length = 4096; region.device = DeviceKind::kCpu;
  CHECK(resources->Initialize(region).ok());
  auto state = op->CreateExecutionState();
  CHECK(op->BindExecutionState(state.get()).ok());
  ubmem::AccessOperator other(options);
  auto foreign = other.CreateExecutionState();
  CHECK(!op->BindExecutionState(foreign.get()).ok());
  uint8_t input[64]{};
  BatchPutRequest request;
  request.keys = {1}; request.input_offsets = {0}; request.lengths = {64};
  request.input = {input, sizeof(input), DeviceKind::kNpu};
  PutAccessDesc destinations; destinations.entries.push_back({1, region.base, 64});
  BatchPutExecRequest call{&request, &destinations, {reinterpret_cast<void*>(1)}};
  fail_wait = true;
  CHECK(!op->ExecuteBatchPut(call).ok() && launches == 1);
  PutCompletion completion;
  CHECK(!op->ResolvePut(&completion).ok());
  CHECK(!op->AcknowledgePut().ok());
  CHECK(!resources->Close().ok() && mappings_closed == 0 && releases == 0);
  fail_wait = false; fail_copy = true;
  CHECK(!op->ResolvePut(&completion).ok() && mappings_closed == 0);
  fail_copy = false;
  CHECK(op->ResolvePut(&completion).ok() && completion.outcome == PutOutcome::Succeeded);
  CHECK(launches == 1 && mappings_closed == 0);
  CHECK(op->ResolvePut(&completion).ok() && completion.outcome == PutOutcome::Succeeded);
  CHECK(op->AcknowledgePut().ok());
  // A completed stream with a partial local MTE result can safely abort, not commit.
  incomplete = true;
  CHECK(!op->ExecuteBatchPut(call).ok() && launches == 2);
  CHECK(op->ResolvePut(&completion).ok() && completion.outcome == PutOutcome::Failed);
  CHECK(!completion.result.ok());
  CHECK(op->AcknowledgePut().ok());
  // Failed submission after a preceding success cannot reuse its status contents.
  incomplete = false;
  CHECK(op->ExecuteBatchPut(call).ok() && launches == 3);
  CHECK(op->AcknowledgePut().ok());
  fail_launch = true;
  CHECK(!op->ExecuteBatchPut(call).ok() && launches == 4);
  CHECK(op->ResolvePut(&completion).ok() && completion.outcome == PutOutcome::Failed);
  CHECK(op->AcknowledgePut().ok());
  CHECK(op->BindExecutionState(nullptr).ok());
  state.reset(); op.reset();
  CHECK(mappings_closed == 0); // Only the independent owner closes the mapping.
  CHECK(resources->Close().ok() && mappings_closed == 1 && releases == 1);
  resources.reset();
  CHECK(mappings_closed == 1);
  // A manager-owned resource can initialize and warm up after its facade has
  // been destroyed. Stateless algorithms only see already prepared resources.
  options.batch_get_kernel_object_path = __FILE__;
  auto facade = std::make_unique<ubmem::AccessOperator>(options);
  auto independent = facade->TakeResources();
  facade.reset();
  CHECK(independent->Initialize(region).ok());
  CHECK(independent->Warmup({2, reinterpret_cast<void*>(1)}).ok());
  auto* mapped = dynamic_cast<ubmem::UbmemResources*>(independent.get());
  CHECK(mapped);
  auto invocation = mapped->CreateState();
  CHECK(mapped->Bind(invocation.get()).ok());
  auto* typed = dynamic_cast<ubmem::UbmemInvocation*>(invocation.get());
  CHECK(typed);
  ubmem::UbmemPutView put_view;
  CHECK(mapped->PreparePut(reinterpret_cast<aclrtStream>(1), 2, &put_view).ok());
  static_assert(!std::is_convertible_v<ubmem::UbmemPutView, ubmem::UbmemGetView>);
  const auto allocation_count = allocations;
  const auto launch_count = launches;
  auto* wrong = dynamic_cast<ubmem::UbmemInvocation*>(foreign.get());
  CHECK(wrong && !ubmem::UbmemPutOperator::Launch(call, put_view, *wrong).ok());
  CHECK(launches == launch_count && allocations == allocation_count);
  auto unbound = mapped->CreateState();
  auto* unbound_call = dynamic_cast<ubmem::UbmemInvocation*>(unbound.get());
  CHECK(unbound_call && !ubmem::UbmemPutOperator::Launch(call, put_view, *unbound_call).ok());
  auto wrong_stream = call;
  wrong_stream.execution.stream = reinterpret_cast<void*>(2);
  CHECK(!ubmem::UbmemPutOperator::Launch(wrong_stream, put_view, *typed).ok());
  CHECK(launches == launch_count && allocations == allocation_count);
  fail_launch = false;
  CHECK(ubmem::UbmemPutOperator::Launch(call, put_view, *typed).ok());
  CHECK(launches == launch_count + 1 && allocations == allocation_count);
  std::int32_t index = 0, count = 1;
  std::uint64_t address = region.base;
  std::uint8_t valid = 1;
  BatchGetRequest get;
  get.indices = get.offsets = get.output_indices = {&index, sizeof(index), DeviceKind::kNpu};
  get.miss_count = {&count, sizeof(count), DeviceKind::kNpu};
  get.output = {input, sizeof(input), DeviceKind::kNpu};
  get.output_block_bytes = 64; get.tokens_per_block = 1;
  get.device_access_table.addresses = {&address, sizeof(address), DeviceKind::kNpu};
  get.device_access_table.valid = {&valid, sizeof(valid), DeviceKind::kNpu};
  BatchGetExecRequest get_call;
  get_call.request = &get;
  get_call.execution.stream = reinterpret_cast<void*>(1);
  get_call.execution.synchronize = true;
  ubmem::UbmemGetView get_view;
  CHECK(mapped->PrepareGet(reinterpret_cast<aclrtStream>(1), &get_view).ok());
  CHECK(!ubmem::UbmemGetOperator::Launch(get_call, get_view, *wrong).ok());
  CHECK(ubmem::UbmemGetOperator::Launch(get_call, get_view, *typed).ok());
  CHECK(launches == launch_count + 2 && allocations == allocation_count);

  // A borrowed view becomes unusable after Close; incomplete work and graph
  // leases must prevent physical cleanup in the meantime.
  typed->graph_active = true;
  CHECK(independent->Close().code() == StatusCode::kBusy);
  CHECK(mappings_closed == 1);
  typed->graph_active = false;
  fail_wait = true;
  CHECK(!ubmem::UbmemPutOperator::Launch(call, put_view, *typed).ok());
  CHECK(!independent->Close().ok() && mappings_closed == 1);
  fail_wait = false;
  fail_free = true;
  CHECK(!independent->Close().ok() && mappings_closed == 1);
  fail_free = false;
  CHECK(independent->Close().ok() && mappings_closed == 2 && releases == 2);
  const auto after_close_launches = launches;
  CHECK(!ubmem::UbmemPutOperator::Launch(call, put_view, *typed).ok());
  CHECK(launches == after_close_launches);
  CHECK(mapped->Bind(nullptr).ok());
  invocation.reset();
  independent.reset();
  CHECK(mappings_closed == 2 && releases == 2);
  std::cout << "UBMEM independent resources, typed launch views and completion recovery passed\n";
}
