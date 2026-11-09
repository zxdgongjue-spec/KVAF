#include "aicpu_urma/session.h"
#include "aicpu_urma/kernel_loader.h"
#include "aicpu_batch_put/include/kvaf_aicpu_batch_put.h"
#include "meta_manager.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

#define CHECK(c) do { if (!(c)) { std::cerr << "line " << __LINE__ << ": " #c "\n"; std::abort(); } } while (false)
namespace {
using namespace kvaf::aicpu_urma;
int writes = 0, unregisters = 0;
bool fail_wait = false, fail_copy = false, write_error = false;
bool fail_launch = false, fail_memset = false;
std::vector<UmdkTransfer> submitted;
}
extern "C" {
aclError aclrtCreateStream(aclrtStream* p) { *p = reinterpret_cast<aclrtStream>(1); return 0; }
aclError aclrtDestroyStream(aclrtStream) { return 0; }
aclError aclrtMalloc(void** p, size_t n, aclrtMemMallocPolicy) { *p = std::malloc(n); return *p ? 0 : 1; }
aclError aclrtFree(void* p) { std::free(p); return 0; }
aclError aclrtMemset(void* p, size_t, int value, size_t n) {
  if (fail_memset) return 1;
  std::memset(p, value, n); return 0;
}
aclError aclrtMemcpy(void* p, size_t, const void* s, size_t n, aclrtMemcpyKind kind) {
  if (fail_copy && kind == ACL_MEMCPY_DEVICE_TO_HOST) return 1;
  std::memcpy(p, s, n); return 0;
}
aclError aclrtSynchronizeStream(aclrtStream) { return fail_wait ? 1 : 0; }
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
Status PrepareUmdkKernel(const std::string&, bool) { return Status::OK(); }
int32_t LaunchUmdkKernel(const char*, const void* input, size_t, aclrtStream, uint32_t) {
  const auto& args = *static_cast<const UmdkControlArgs*>(input);
  auto& status = *reinterpret_cast<UmdkControlStatus*>(args.status);
  status = {}; status.magic = kUmdkControlMagic; status.session = 1; status.owner_pid = 123;
  status.key_bytes = 28;
  std::memcpy(status.eid, args.local_eid, sizeof(status.eid));
  if (args.op == UmdkControlOp::kUnregisterBuffers) ++unregisters;
  return 0;
}
int32_t LaunchBatchPut(const BatchPutKernelArgs* args, aclrtStream stream) {
  ++writes;
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
  std::cout << "AICPU whole-batch WRITE resolution and cleanup retry passed\n";
}
