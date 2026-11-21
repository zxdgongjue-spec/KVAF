#include "access_operator.h"
#include "kvaf_aiv_batch_get.h"

#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::cerr << "line " << __LINE__ << ": " << #x << '\n'; std::exit(1); } } while (false)

namespace {
std::vector<std::function<void()>> pending;
int synchronizations = 0, allocations = 0, destroyed = 0, launched = 0;
bool fail_sync = false, fail_launch = false, fail_reset = false, capture = false;
uint64_t device_result = 0;
uint32_t last_chunk_bytes = UINT32_MAX;
}

aclError aclrtMalloc(void** p, std::size_t n, int) { *p = std::malloc(n); ++allocations; return *p ? 0 : 1; }
aclError aclrtFree(void* p) { CHECK(pending.empty()); std::free(p); --allocations; return 0; }
aclError aclrtMemset(void* p, std::size_t, int v, std::size_t n) { std::memset(p, v, n); return 0; }
aclError aclrtMemsetAsync(void* p, std::size_t, int v, std::size_t n, aclrtStream) {
  if (fail_reset) return 1;
  pending.push_back([=] { std::memset(p, v, n); }); return 0;
}
aclError aclrtMemcpy(void* p, std::size_t, const void* q, std::size_t n, int) {
  CHECK(pending.empty()); std::memcpy(p, q, n); return 0;
}
aclError aclrtSynchronizeStream(aclrtStream) {
  ++synchronizations;
  if (fail_sync) return 1;
  for (auto& work : pending) work();
  pending.clear(); return 0;
}
aclError aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus* s, aclmdlRI* g) {
  *s = capture ? ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE : ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  *g = nullptr; return 0;
}

struct KvafBatchGetTransport {};
struct KvafBatchGetLocalBuffer {};
extern "C" {
int32_t KvafBatchGetTransportCreateWithPeerLinkMultiEid(void*, uint64_t, uint32_t, uint32_t,
    void*, const uint8_t*, uint32_t, KvafBatchGetTransport** t) {
  *t = new KvafBatchGetTransport; return 0;
}
void KvafBatchGetTransportDestroy(KvafBatchGetTransport* t) { CHECK(pending.empty()); delete t; ++destroyed; }
int32_t KvafBatchGetLocalBufferAcquire(KvafBatchGetTransport*, void*, uint64_t,
    KvafBatchGetLocalBuffer** b, uint64_t* e) {
  static KvafBatchGetLocalBuffer cached; *b = &cached; *e = 1; return 0;
}
void KvafBatchGetLocalBufferRelease(KvafBatchGetLocalBuffer*) {}
int32_t KvafExecuteBatchGetLoadKernel(const char*) { return 0; }
int32_t KvafExecuteBatchGetLaunch(const KvafExecuteBatchGetKernelArgs*, aclrtStream) { return 0; }
int32_t KvafExecuteBatchGetRangesLaunch(const KvafExecuteBatchGetRangesKernelArgs* a, aclrtStream) {
  ++launched;
  last_chunk_bytes = a->read_chunk_bytes;
  if (fail_launch) return -2;
  const auto args = *a;
  const auto result = device_result;
  pending.push_back([=] {
    auto* s = reinterpret_cast<KvafBatchGetCoreStatus*>(args.core_status);
    for (uint32_t i = 0; i < args.block_count; ++i) {
      s[i].magic = KVAF_AIV_BATCH_GET_STATUS_MAGIC; s[i].result = result;
      s[i].failed_request_index = result ? 2 : UINT64_MAX;
    }
  });
  return 0;
}
}

int main() {
  using namespace kvaf;
  AccessOperatorConfig config;
  config.batch_get_kernel_object_path = "model.o";
  config.npu_core_count = 2; config.jettys_per_core = 1;
  config.peer_link = std::shared_ptr<peer::PeerLink>(
      reinterpret_cast<peer::PeerLink*>(1), [](peer::PeerLink*) {});
  config.peer_local_eid[0] = 1;
  DefaultAccessOperator op(config);
  CHECK(op.Initialize({4096, 8192, 4096, DeviceKind::kCpu}).ok());
  int32_t index = 0, count = 1;
  int64_t zero = 0, length = 656;
  uint64_t address = 4096, object_length = 656;
  uint8_t valid = 1, output[656];
  BatchGetRangesRequest r{{1}, {&index, 4, DeviceKind::kNpu}, {&zero, 8, DeviceKind::kNpu},
      {&length, 8, DeviceKind::kNpu}, {&count, 4, DeviceKind::kNpu}, {&zero, 8, DeviceKind::kNpu},
      {output, sizeof(output), DeviceKind::kNpu}, {{&address, 8, DeviceKind::kNpu},
      {&valid, 1, DeviceKind::kNpu}, {&object_length, 8, DeviceKind::kNpu}}};
  auto stream = reinterpret_cast<void*>(1);
  BatchGetRangesExecRequest e{{}, &r, {stream, false, nullptr}};
  CHECK(op.CheckBatchGetRangesStatus(stream).code() == StatusCode::kInvalidState);
  CHECK(op.ExecuteBatchGetRanges(e).ok() && synchronizations == 0 && pending.size() == 2);
  CHECK(last_chunk_bytes == 0);  // default removes the former 64KiB policy
  CHECK(op.CheckBatchGetRangesStatus(reinterpret_cast<void*>(2)).code() == StatusCode::kInvalidArgument);
  CHECK(op.CheckBatchGetRangesStatus(stream).ok() && synchronizations == 1 && pending.empty());
  CHECK(op.CheckBatchGetRangesStatus(stream).ok() && synchronizations == 1);
  r.read_chunk_bytes = 4 * 1024 * 1024;
  // A failed status reset must not expose an older successful device status.
  fail_reset = true;
  CHECK(op.ExecuteBatchGetRanges(e).code() == StatusCode::kInternal && pending.empty());
  fail_reset = false;
  CHECK(op.CheckBatchGetRangesStatus(stream).code() == StatusCode::kInternal);
  device_result = 8;
  e.execution.synchronize = true;
  CHECK(op.ExecuteBatchGetRanges(e).code() == StatusCode::kInvalidArgument);
  CHECK(last_chunk_bytes == r.read_chunk_bytes);
  CHECK(op.CheckBatchGetRangesStatus(stream).code() == StatusCode::kInvalidArgument);
  device_result = 6;
  CHECK(op.ExecuteBatchGetRanges(e).code() == StatusCode::kInternal);
  device_result = 0;
  e.execution.synchronize = false;
  capture = true;
  const auto before = launched;
  CHECK(op.ExecuteBatchGetRanges(e).code() == StatusCode::kNotImplemented && launched == before);
  capture = false;
  CHECK(op.ExecuteBatchGetRanges(e).ok());
  CHECK(op.ExecuteBatchGetRanges(e).ok());  // drains the previous invocation
  CHECK(pending.size() == 2);
  CHECK(op.SetBatchGetKernelProfiling(false).ok() && pending.empty());
  fail_launch = true;
  CHECK(op.ExecuteBatchGetRanges(e).code() == StatusCode::kInternal && pending.size() == 1);
  fail_sync = true;
  CHECK(op.Shutdown().code() == StatusCode::kInternal && allocations == 2 && destroyed == 0);
  fail_sync = false;
  CHECK(op.Shutdown().ok() && pending.empty() && allocations == 0 && destroyed == 1);
  fail_launch = false;
  CHECK(op.Initialize({4096, 8192, 4096, DeviceKind::kCpu}).ok());
  CHECK(op.CheckBatchGetRangesStatus(stream).code() == StatusCode::kInvalidState);
  CHECK(op.ExecuteBatchGetRanges(e).ok());
  CHECK(op.Shutdown().ok() && allocations == 0 && destroyed == 2);
  std::cout << "RANGE_GET_OPERATOR_PASS: enqueue/wait, status errors, capture rejection, reuse and failed-launch cleanup\n";
}
