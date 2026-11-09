#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "aicpu_urma/session.h"
#include "cpu_gather.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>

#define CHECK(c) do { if (!(c)) { std::cerr << "line " << __LINE__ << ": " #c "\n"; std::abort(); } } while (false)
namespace {
int allocations = 0, registrations = 0, launches = 0, enables = 0;
std::set<void*> buffers;
}
extern "C" {
aclError aclrtMalloc(void** p, size_t n, aclrtMemMallocPolicy) {
  *p = std::calloc(1, n); if (!*p) return 1;
  ++allocations; buffers.insert(*p); return 0;
}
aclError aclrtMallocHost(void** p, size_t n) { return aclrtMalloc(p, n, ACL_MEM_MALLOC_HUGE_FIRST); }
aclError aclrtFree(void* p) { CHECK(buffers.erase(p) == 1); std::free(p); return 0; }
aclError aclrtFreeHost(void* p) { return aclrtFree(p); }
aclError aclrtMemcpy(void* p, size_t, const void* s, size_t n, aclrtMemcpyKind) {
  std::memcpy(p, s, n); return 0;
}
aclError aclrtMemcpyAsync(void* p, size_t size, const void* s, size_t n, aclrtMemcpyKind kind, aclrtStream) {
  return aclrtMemcpy(p, size, s, n, kind);
}
aclError aclrtSynchronizeStream(aclrtStream) { return 0; }
const char* aclGetRecentErrMsg() { return "fake"; }
}
namespace kvaf::peer {
PeerLink::PeerLink(std::string address) : address_(std::move(address)) {}
Status PeerLink::LookupRegion(uint64_t, PeerRegion* region) const {
  region->length = 128ULL * 1024 * 1024; return Status::OK();
}
Status PeerLink::EnableGatherLanes(uint64_t, uint64_t, uint32_t) const { ++enables; return Status::OK(); }
}
namespace kvaf::aicpu_urma {
// Session registration is a counted resource service. The real Gather host
// algorithm below must not invoke it or allocate memory during prepared Launch.
Status UmdkSession::Ensure(std::uint64_t, void*, std::size_t) { ++registrations; return Status::OK(); }
int32_t LaunchBatchGet(const UmdkBatchGetArgs* args, aclrtStream) {
  ++launches;
  auto* statuses = reinterpret_cast<GatherReadStatus*>(args->gather.core_status);
  for (unsigned w = 0; w < args->gather.worker_count; ++w) {
    statuses[w] = {};
    statuses[w].magic = kGatherStatusMagic;
    statuses[w].sequence = args->gather.sequence;
  }
  return 0;
}
}
int main() {
  using namespace kvaf;
  using namespace kvaf::aicpu_urma;
  UmdkSessionConfig config;
  config.peer_link = std::make_shared<PeerLink>("fake:0"); config.workers = 2;
  UmdkSession session(config);
  BatchGetTransport gather(session, 16);
  CHECK(gather.Initialize({0x100000, 4096, 16, DeviceKind::kCpu}).ok());
  auto state = gather.CreateExecutionState();
  CHECK(gather.BindExecutionState(state.get()).ok());
  uint64_t address = 0x100000;
  int32_t index = 0, count = 1;
  uint8_t valid = 1, output[16]{};
  BatchGetRequest request;
  request.indices = request.offsets = request.output_indices = {&index, sizeof(index), DeviceKind::kNpu};
  request.miss_count = {&count, sizeof(count), DeviceKind::kNpu};
  request.output = {output, sizeof(output), DeviceKind::kNpu};
  request.output_block_bytes = 16; request.tokens_per_block = 1;
  request.device_access_table.addresses = {&address, sizeof(address), DeviceKind::kNpu};
  request.device_access_table.valid = {&valid, sizeof(valid), DeviceKind::kNpu};
  BatchGetProfile profile;
  BatchGetExecRequest call; call.request = &request;
  call.execution.stream = reinterpret_cast<void*>(1);
  call.execution.synchronize = true; call.execution.profile = &profile;
  CHECK(!gather.ExecutePrepared(call).ok() && launches == 0);
  CHECK(gather.Prepare(call).ok() && registrations == 2 && enables == 1);
  const auto allocations_before_launch = allocations, registrations_before_launch = registrations;
  auto other = request;
  auto mismatched = call; mismatched.request = &other;
  CHECK(!gather.ExecutePrepared(mismatched).ok() && launches == 0);
  CHECK(gather.ExecutePrepared(call).ok() && launches == 1);
  CHECK(allocations == allocations_before_launch && registrations == registrations_before_launch);
  CHECK(profile.worker_count == 2);
  CHECK(!gather.ExecutePrepared(call).ok() && launches == 1);
  // Warm workspaces and CPU Gather session are reused across invocations.
  CHECK(gather.Prepare(call).ok() && enables == 1 && allocations == allocations_before_launch);
  CHECK(gather.ExecutePrepared(call).ok() && launches == 2);
  CHECK(gather.BindExecutionState(nullptr).ok());
  state.reset();
  CHECK(gather.Shutdown().ok() && buffers.empty());
  std::cout << "Gather preparation, typed invocation and allocation-free launch passed\n";
}
