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
bool fail_free = false, capturing = false;
bool fail_wait = false, fail_copy = false, incomplete = false, fail_launch = false;
kvaf::ubmem::BatchArgs captured_args{};
std::uint32_t captured_blocks = 0;
std::uint64_t profile_cycle = 1000;

void ExecuteKernel(const kvaf::ubmem::BatchArgs& args, std::uint32_t blocks) {
  using namespace kvaf::ubmem;
  auto* reports = reinterpret_cast<CoreStatus*>(args.statuses);
  const auto count = args.operation == Operation::kGet
      ? static_cast<std::uint64_t>(*reinterpret_cast<const std::int32_t*>(args.miss_count))
      : args.capacity;
  profile_cycle += 1000;
  for (unsigned i = 0; i < blocks; ++i) {
    reports[i] = {};
    reports[i].completed = count * (i + 1) / blocks - count * i / blocks;
    if (args.profile) {
      reports[i].profile_magic = kProfileMagic;
      reports[i].start_cycle = profile_cycle + i;
      reports[i].finish_cycle = profile_cycle + i + 100;
      reports[i].profile_detail = args.profile == 2;
      if (args.profile == 2) {
        reports[i].setup_cycles = 10;
        reports[i].address_cycles = 20;
        reports[i].submit_cycles = 30;
        reports[i].drain_cycles = 40;
      }
    }
  }
  if (incomplete) reports[blocks - 1].completed = 0;
}
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
  *status = capturing ? ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE : ACL_MODEL_RI_CAPTURE_STATUS_NONE; return 0;
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
  if (capturing) {
    // Capture stores kernel arguments; only replay produces a device report.
    captured_args = a;
    captured_blocks = blocks;
  } else {
    ExecuteKernel(a, blocks);
  }
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
  PutProviderTiming timing;
  call.execution.host_timing = &timing;
  CHECK(op->ExecuteBatchPut(call).ok() && launches == 3);
  call.execution.host_timing = nullptr;
  CHECK(std::strcmp(timing.transport, "ubmem") == 0 && std::strcmp(timing.executor, "aiv") == 0);
  CHECK(timing.prepare_ns > 0 && timing.metadata_ns > 0 && timing.submit_ns > 0);
  CHECK(timing.completion_ns > 0 && timing.registration_ns == 0 && timing.cleanup_ns == 0);
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
  CHECK(mapped->PreparePutResources(reinterpret_cast<aclrtStream>(1), 2, &put_view).ok());
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
  CHECK(mapped->PrepareGetResources(reinterpret_cast<aclrtStream>(1), &get_view).ok());
  CHECK(!ubmem::UbmemGetOperator::Launch(get_call, get_view, *wrong).ok());
  CHECK(ubmem::UbmemGetOperator::Launch(get_call, get_view, *typed).ok());
  CHECK(launches == launch_count + 2 && allocations == allocation_count);

  // A borrowed view becomes unusable after Close; incomplete work and graph
  // leases must prevent physical cleanup in the meantime.
  typed->graph_active = true;
  CHECK(independent->Close().code() == StatusCode::kBusy);
  CHECK(mappings_closed == 1);
  typed->graph_active = false;
  // Preparing Get invalidated the earlier Put workspace view.
  CHECK(!ubmem::UbmemPutOperator::Launch(call, put_view, *typed).ok());
  CHECK(mapped->PreparePutResources(reinterpret_cast<aclrtStream>(1), 2, &put_view).ok());
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
  // Graph preparation owns stream slots once; capture only submits device work.
  ubmem::AccessOperator graph_op(options);
  CHECK(graph_op.Initialize(region).ok());
  auto graph_state = graph_op.CreateExecutionState();
  CHECK(graph_op.BindExecutionState(graph_state.get()).ok());
  CHECK(graph_op.SetBatchGetKernelProfiling(true).ok());
  std::vector<BatchGetKernelProfile> profiles;
  CHECK(!graph_op.CopyBatchGetKernelProfiles(&profiles).ok());
  // The unified benchmark first profiles eager calls, then captures/replays
  // on the same bound invocation. Capture must replace the eager sample.
  CHECK(graph_op.ExecuteBatchGet(get_call).ok());
  CHECK(graph_op.CopyBatchGetKernelProfiles(&profiles).ok());
  CHECK(profiles.size() == options.cores);
  auto previous_cycle = profiles.front().kernel_start_cycle;
  CHECK(graph_op.PrepareGetGraph({get.output, get_call.execution.stream, 1}).ok());
  const auto graph_allocations = allocations;
  get_call.execution.synchronize = false;
  capturing = true;
  CHECK(graph_op.ExecuteBatchGet(get_call).ok());
  CHECK(allocations == graph_allocations);
  const auto capture_profile = graph_op.CopyBatchGetKernelProfiles(&profiles);
  CHECK(capture_profile.code() == StatusCode::kInvalidState);
  CHECK(!graph_op.FinishGetGraph(get_call.execution.stream, true).ok());
  capturing = false;
  // Capturing alone has not executed the kernel or produced valid timestamps.
  CHECK(!graph_op.CopyBatchGetKernelProfiles(&profiles).ok());
  CHECK(captured_blocks == options.cores && captured_args.profile == 1);
  for (int active : {1, 0, 1}) {
    count = active;
    ExecuteKernel(captured_args, captured_blocks);
    CHECK(aclrtSynchronizeStream(get_call.execution.stream) == ACL_SUCCESS);
    CHECK(graph_op.CopyBatchGetKernelProfiles(&profiles).ok());
    CHECK(profiles.size() == options.cores);
    CHECK(profiles.front().kernel_start_cycle > previous_cycle);
    previous_cycle = profiles.front().kernel_start_cycle;
    for (unsigned i = 0; i < options.cores; ++i) {
      CHECK(profiles[i].kernel_finish_cycle > profiles[i].kernel_start_cycle);
      CHECK(profiles[i].active_requests == std::uint64_t(active) * (i + 1) / options.cores -
                                         std::uint64_t(active) * i / options.cores);
      CHECK(!profiles[i].ubmem_profile_detail);
    }
  }
  CHECK(!graph_op.FinishGetGraph(reinterpret_cast<void*>(99), false).ok());
  fail_wait = true;
  CHECK(!graph_op.FinishGetGraph(get_call.execution.stream, true).ok());
  CHECK(graph_op.Shutdown().code() == StatusCode::kBusy);
  fail_wait = false;
  fail_copy = true;
  CHECK(!graph_op.FinishGetGraph(get_call.execution.stream, false).ok());
  fail_copy = false;
  CHECK(graph_op.FinishGetGraph(get_call.execution.stream, false).ok());
  CHECK(graph_op.CopyBatchGetKernelProfiles(&profiles).ok());
  CHECK(profiles.front().kernel_start_cycle == previous_cycle);
  // Detailed graph profiling uses the same replay stream and supports a new
  // graph lease after the preceding graph has been released.
  CHECK(graph_op.SetBatchGetKernelProfiling(true, true).ok());
  CHECK(!graph_op.CopyBatchGetKernelProfiles(&profiles).ok());
  CHECK(graph_op.PrepareGetGraph({get.output, get_call.execution.stream, 1}).ok());
  capturing = true;
  CHECK(graph_op.ExecuteBatchGet(get_call).ok());
  capturing = false;
  CHECK(captured_args.profile == 2);
  ExecuteKernel(captured_args, captured_blocks);
  CHECK(aclrtSynchronizeStream(get_call.execution.stream) == ACL_SUCCESS);
  CHECK(graph_op.CopyBatchGetKernelProfiles(&profiles).ok());
  for (const auto& profile : profiles) {
    CHECK(profile.ubmem_profile_detail);
    CHECK(profile.ubmem_setup_cycles == 10 && profile.ubmem_address_cycles == 20);
    CHECK(profile.ubmem_submit_cycles == 30 && profile.ubmem_drain_cycles == 40);
  }
  CHECK(graph_op.FinishGetGraph(get_call.execution.stream, false).ok());
  CHECK(graph_op.SetBatchGetKernelProfiling(false).ok());
  CHECK(!graph_op.CopyBatchGetKernelProfiles(&profiles).ok());
  CHECK(graph_op.Shutdown().ok());
  CHECK(graph_op.BindExecutionState(nullptr).ok());
  std::cout << "UBMEM independent resources, typed launch views and completion recovery passed\n";
}
