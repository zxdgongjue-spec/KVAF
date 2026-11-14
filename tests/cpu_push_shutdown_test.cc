#include "aicpu_batch_get/include/cpu_push_transport.h"
#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include "peer_link.h"
#include <urma_api.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#define CHECK(condition) do { if (!(condition)) { \
  std::cerr << "line " << __LINE__ << ": " #condition "\n"; std::exit(1); \
} } while (false)

namespace {
bool cpu_running = false, fail_launch = false, bad_result = false, capturing = false;
std::vector<std::function<void()>> captured;
unsigned sync_failures = 0, stop_failures = 0;
int handle;
std::set<void*> allocations;
std::deque<std::function<void()>> queued;
std::vector<std::string> events;

std::size_t Event(const std::string& name) {
  auto found = std::find(events.begin(), events.end(), name);
  CHECK(found != events.end());
  return static_cast<std::size_t>(found - events.begin());
}
void Reset() {
  CHECK(allocations.empty() && queued.empty() && !cpu_running);
  events.clear();
  fail_launch = bad_result = capturing = false;
  captured.clear();
  sync_failures = stop_failures = 0;
}
}  // namespace

aclError aclrtMalloc(void** out, std::size_t bytes, aclrtMemMallocPolicy) {
  CHECK(!capturing);
  CHECK(posix_memalign(out, 64, bytes) == 0);
  allocations.insert(*out);
  return ACL_SUCCESS;
}
aclError aclrtMallocHost(void** out, std::size_t bytes) {
  return aclrtMalloc(out, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
}
aclError aclrtFree(void* buffer) {
  CHECK(!cpu_running && queued.empty());
  CHECK(allocations.erase(buffer) == 1);
  events.push_back("free");
  std::free(buffer);
  return ACL_SUCCESS;
}
aclError aclrtFreeHost(void* buffer) { return aclrtFree(buffer); }
aclError aclrtMemset(void* buffer, std::size_t, int value, std::size_t bytes) {
  CHECK(!capturing);
  std::memset(buffer, value, bytes);
  return ACL_SUCCESS;
}
aclError aclrtMemsetAsync(void* buffer, std::size_t, int value, std::size_t bytes, aclrtStream) {
  queued.emplace_back([=] { std::memset(buffer, value, bytes); });
  return ACL_SUCCESS;
}
aclError aclrtMemcpyAsync(void* dst, std::size_t, const void* src, std::size_t bytes,
                          aclrtMemcpyKind, aclrtStream) {
  CHECK(!capturing);
  queued.emplace_back([=] { std::memcpy(dst, src, bytes); });
  return ACL_SUCCESS;
}
aclError aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus* status, aclmdlRI* graph) {
  *status = capturing ? ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE : ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  *graph = nullptr;
  return ACL_SUCCESS;
}
aclError aclrtMemcpy(void* dst, std::size_t, const void* src, std::size_t bytes, aclrtMemcpyKind) {
  CHECK(!capturing);
  std::memcpy(dst, src, bytes);
  return ACL_SUCCESS;
}
aclError aclrtSynchronizeStream(aclrtStream) {
  CHECK(!capturing);
  events.push_back("sync");
  if (sync_failures) { --sync_failures; return 1; }
  while (!queued.empty()) { queued.front()(); queued.pop_front(); }
  return ACL_SUCCESS;
}

namespace kvaf::aicpu_urma {
bool LoadUmdkKernel(const char*) { return true; }
}
namespace kvaf::cpu_push {
int32_t LaunchCpuPushScatter(const KernelArgs* input, aclrtStream) {
  if (fail_launch) return -1;
  const auto args = *input;
  auto launch = [=] {
    // The CPU has not accepted the request until this queued kernel runs.
    CHECK(cpu_running);
    events.push_back("kernel completed");
    auto* statuses = reinterpret_cast<WorkerStatus*>(args.status);
    auto* coordination = reinterpret_cast<Coordination*>(args.coordination);
    const auto sequence = ++coordination->finished.value;
    coordination->claimed.value += args.workers;
    if (bad_result) coordination->result.value = std::uint64_t(Result::kTransportError);
    for (std::uint32_t w = 0; w < args.workers; ++w)
      statuses[w] = {kMagic, sequence,
          std::uint64_t(bad_result ? Result::kTransportError : Result::kOk), w, 0, {}};
    if (args.profile) {
      auto* profiles = reinterpret_cast<BatchGetWorkerProfile*>(args.profile);
      for (std::uint32_t w = 0; w < args.workers; ++w) {
        profiles[w].enter_ns = 100 * sequence;
        profiles[w].exit_ns = 100 * sequence + 50;
        profiles[w].spans = 1;
      }
    }
  };
  if (capturing) captured.push_back(launch);
  else queued.push_back(launch);
  return 0;
}
}
namespace kvaf::peer {
PeerLink::PeerLink(std::string address) : address_(std::move(address)) {}
Status PeerLink::LookupRegion(uint64_t base, PeerRegion* region) const {
  region->base = base;
  region->length = 4096;
  region->token_value = 1;
  return Status::OK();
}
Status PeerLink::OpenCpuPushGet(uint64_t, OpenCpuPushGetRequest,
                               OpenCpuPushGetResponse* response) const {
  *response = {};
  return Status::OK();
}
Status PeerLink::StartCpuPushGet(uint64_t) const { cpu_running = true; return Status::OK(); }
Status PeerLink::StopCpuPushGet(uint64_t) const {
  events.push_back("stop CPU");
  if (stop_failures) { --stop_failures; return Status::Internal("injected STOP failure"); }
  cpu_running = false;
  return Status::OK();
}
}
namespace kvaf::direct_hccp {
bool LoadDirectHccp(DirectHccpApi* api) {
  api->ctx_init = [](CtxInitCfg*, CtxInitAttr*, void** out) { *out = &handle; return 0; };
  api->ctx_deinit = [](void*) { return 0; };
  api->token_alloc = [](void*, TokenId*, void** out) { *out = &handle; return 0; };
  api->token_free = [](void*, void*) { return 0; };
  api->lmem_register = [](void*, MrRegInfo* info, void** out) {
    urma_seg_t segment{};
    segment.ubva.va = info->in.mem.addr;
    segment.len = info->in.mem.size;
    info->out.key.size = sizeof(segment);
    std::memcpy(info->out.key.value, &segment, sizeof(segment));
    *out = &handle;
    return 0;
  };
  api->lmem_unregister = [](void*, void*) {
    CHECK(!cpu_running && queued.empty());
    events.push_back("unregister"); return 0;
  };
  api->cq_create = [](void*, CqInfo*, void** out) { *out = &handle; return 0; };
  api->cq_destroy = [](void*, void*) { return 0; };
  api->qp_create = [](void*, QpCreateAttr*, QpCreateInfo* info, void** out) {
    info->key.size = 28; *out = &handle; return 0;
  };
  api->qp_destroy = [](void*) { CHECK(!cpu_running); return 0; };
  api->qp_unimport = [](void*, void*) { CHECK(!cpu_running); return 0; };
  return true;
}
void CloseDirectHccp(DirectHccpApi*) {}
bool GetCurrentNpuPhysicalDeviceId(std::uint32_t* device) { *device = 0; return true; }
bool AcquireDirectHccpSession(std::uint32_t, std::uint32_t, DirectHccpSession** out) {
  *out = reinterpret_cast<DirectHccpSession*>(&handle); return true;
}
void ReleaseDirectHccpSession(DirectHccpSession*) {}
bool GetPeerTp(const DirectHccpApi&, void*, const std::array<std::uint8_t, kEidBytes>&,
               const std::array<std::uint8_t, kEidBytes>&, const char*, std::uint64_t* tp) {
  *tp = 1; return true;
}
bool ImportPeerJetty(const DirectHccpApi&, void*, const std::uint8_t*, std::uint32_t,
                     std::uint32_t, std::uint64_t, std::uint64_t, std::uint32_t,
                     std::uint32_t, const char*, QpImportInfoT*, void** out) {
  *out = &handle; return true;
}
}

namespace {
using namespace kvaf;
cpu_push::Config Config() {
  cpu_push::Config config;
  config.peer_link = std::make_shared<peer::PeerLink>("mock");
  config.kernel_json = "mock.json";
  config.local_eid[0] = 1;
  config.token_bytes = 16;
  config.workers = 5;
  return config;
}
Status Enqueue(cpu_push::CpuPushTransport& transport, void* stream = &handle) {
  static std::uint64_t address = 4096;
  static std::int32_t index = 0, offset = 0, count = 1, row = 0;
  static std::uint8_t output[16];
  BatchGetRequest request{};
  request.indices = {&index, sizeof(index), DeviceKind::kNpu};
  request.offsets = {&offset, sizeof(offset), DeviceKind::kNpu};
  request.miss_count = {&count, sizeof(count), DeviceKind::kNpu};
  request.output_indices = {&row, sizeof(row), DeviceKind::kNpu};
  request.output = {output, sizeof(output), DeviceKind::kNpu};
  request.output_block_bytes = sizeof(output);
  request.tokens_per_block = 1;
  request.device_access_table.addresses = {&address, sizeof(address), DeviceKind::kNpu};
  return transport.Execute({{}, &request, {stream, false}});
}

void CheckShutdown(unsigned scenario) {
  Reset();
  cpu_push::CpuPushTransport transport(Config());
  CHECK(transport.Initialize({4096, 4096, 16, DeviceKind::kCpu}).ok());
  if (scenario == 1) fail_launch = true;
  CHECK(Enqueue(transport).ok() == !fail_launch);
  if (scenario == 2) stop_failures = 1;
  if (scenario == 3) sync_failures = 2;
  if (scenario == 4) bad_result = true;
  const auto result = transport.Shutdown();
  if (scenario == 2 || scenario == 3) {
    CHECK(!result.ok() && !allocations.empty());
    CHECK(std::find(events.begin(), events.end(), "free") == events.end());
    if (scenario == 3) {
      CHECK(!cpu_running);  // STOP still runs after the first failed drain.
      queued.clear();  // Simulate the device aborting work before the retry.
    }
    CHECK(transport.Shutdown().ok());
  } else {
    CHECK(result.ok() == (scenario != 4));  // Report deferred execution errors.
  }
  if (scenario == 1) CHECK(Event("stop CPU") < Event("sync"));
  else CHECK(Event("sync") < Event("stop CPU"));
  if (scenario == 0) CHECK(Event("kernel completed") < Event("stop CPU"));
  CHECK(Event("stop CPU") < Event("unregister"));
  CHECK(Event("unregister") < Event("free"));
  CHECK(allocations.empty() && queued.empty() && !cpu_running);
  CHECK(transport.Shutdown().ok());
}
void CheckCapturedReplay() {
  Reset();
  cpu_push::CpuPushTransport transport(Config());
  CHECK(transport.Initialize({4096, 4096, 16, DeviceKind::kCpu}).ok());
  CHECK(!transport.Check(&handle).ok());
  CHECK(transport.SetKernelProfiling(true).ok());
  capturing = true;
  CHECK(Enqueue(transport).ok());
  CHECK(captured.size() == 1 && queued.empty());
  CHECK(!transport.Drain().ok());
  CHECK(!transport.Check(&handle).ok());
  std::vector<BatchGetKernelProfile> profiles;
  CHECK(!transport.CopyKernelProfiles(&profiles).ok());
  CHECK(!transport.Shutdown().ok());
  CHECK(cpu_running && std::find(events.begin(), events.end(), "stop CPU") == events.end());
  capturing = false;
  for (std::uint64_t sequence = 1; sequence <= 3; ++sequence) {
    queued.push_back(captured.front());
    CHECK(transport.Check(&handle).ok());
    CHECK(transport.CopyKernelProfiles(&profiles).ok());
    CHECK(profiles.size() == Config().workers);
    CHECK(profiles.front().device_start_ns == 100 * sequence);
  }
  int other_stream;
  CHECK(!Enqueue(transport, &other_stream).ok());
  CHECK(!transport.Check(&other_stream).ok());
  CHECK(transport.Shutdown().ok());
  CHECK(allocations.empty() && queued.empty() && !cpu_running);
}

void CheckManagedGraph(unsigned scenario) {
  Reset();
  cpu_push::CpuPushTransport transport(Config());
  CHECK(transport.Initialize({4096, 4096, 16, DeviceKind::kCpu}).ok());
  // Establish an old report; it cannot be mistaken for this graph's replay.
  CHECK(transport.SetKernelProfiling(true).ok());
  CHECK(Enqueue(transport).ok());
  CHECK(transport.Drain().ok());
  std::uint8_t output[16]{};
  GetGraphOptions options{{output, sizeof(output), DeviceKind::kNpu}, &handle, 1};
  CHECK(transport.PrepareGraph(options).ok());
  CHECK(!transport.PrepareGraph(options).ok());
  int other_stream;
  CHECK(!transport.FinishGraph(&other_stream, true).ok());
  capturing = true;
  CHECK(Enqueue(transport).ok());
  CHECK(!transport.FinishGraph(&handle, true).ok());
  capturing = false;
  CHECK(!transport.Check(&handle).ok());
  CHECK(!transport.FinishGraph(&handle, false).ok());
  if (scenario == 0) {
    // Explicitly discard an unexecuted capture, then reuse the same resource.
    CHECK(transport.FinishGraph(&handle, true).ok());
  } else {
    queued.push_back(captured.front());
    if (scenario == 2) sync_failures = 1;
    if (scenario == 3) bad_result = true;
    if (scenario == 2) {
      CHECK(!transport.FinishGraph(&handle, true).ok());
      CHECK(!queued.empty() && cpu_running && !allocations.empty());
    }
    if (scenario == 3) {
      CHECK(!transport.FinishGraph(&handle, true).ok());
      CHECK(!Enqueue(transport).ok());
      // discard does not swallow a real device error. The owner must STOP the
      // remote producer and close the transport before releasing its lease.
      stop_failures = 1;
      CHECK(!transport.Shutdown().ok() && !allocations.empty());
      CHECK(transport.Shutdown().ok());
      return;
    }
    CHECK(transport.FinishGraph(&handle, false).ok());
  }
  CHECK(!transport.FinishGraph(&handle, true).ok());
  // EndGraph releases its stream association, but the last replay's resident
  // profile must remain readable until profiling is disabled or reused.
  std::vector<BatchGetKernelProfile> profiles;
  CHECK(transport.CopyKernelProfiles(&profiles).ok() == (scenario != 0));
  if (scenario != 0) {
    CHECK(profiles.size() == Config().workers);
    CHECK(profiles.front().device_start_ns == 200);
  }
  CHECK(transport.SetKernelProfiling(false).ok());
  CHECK(!transport.CopyKernelProfiles(&profiles).ok());
  CHECK(Enqueue(transport, &other_stream).ok());
  CHECK(transport.Drain().ok());
  CHECK(transport.PrepareGraph(options).ok());
  CHECK(transport.FinishGraph(&handle, true).ok());
  CHECK(transport.Shutdown().ok());
}

}

int main() {
  // Healthy asynchronous work, known failure, STOP failure, stream failure,
  // and an execution error first observed during shutdown.
  for (unsigned scenario = 0; scenario < 5; ++scenario) CheckShutdown(scenario);
  CheckCapturedReplay();
  for (unsigned scenario = 0; scenario < 4; ++scenario) CheckManagedGraph(scenario);
  std::cout << "Scatter asynchronous shutdown and cleanup retry tests passed\n";
}
