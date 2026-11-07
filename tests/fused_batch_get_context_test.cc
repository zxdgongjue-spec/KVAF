#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "access_operator.h"
#include "batch_get_abi.h"
#include "kvaf_aiv_batch_get.h"
#include "meta_manager.h"
#include "peer_link.h"
#include "runtime.h"

namespace {
struct FakeDriver {
  int acquires{0};
  int releases{0};
  int destroys{0};
  int launches{0};
  int device_syncs{0};
  bool fail_acquire{false};
  bool fail_stream_sync{false};
} driver;

void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

kvaf::AccessOperatorConfig Config() {
  kvaf::AccessOperatorConfig config;
  config.batch_get_kernel_object_path = "mock-kernel";
  config.npu_core_count = 2;
  config.jettys_per_core = 2;
  config.peer_local_eid[0] = 1;
  config.peer_link = std::make_shared<kvaf::peer::PeerLink>("127.0.0.1:28999");
  return config;
}

class Allocator final : public kvaf::DdrMemoryAllocator {
 public:
  kvaf::Status Allocate(const kvaf::RuntimeInitRequest& request,
                        kvaf::RemoteRegionDesc* region) override {
    data.resize(request.ddr_bytes);
    *region = {reinterpret_cast<std::uint64_t>(data.data()), request.ddr_bytes,
               request.slot_size, kvaf::DeviceKind::kCpu};
    return kvaf::Status::OK();
  }
  kvaf::Status Release(const kvaf::RemoteRegionDesc&) override {
    ++releases;
    data.clear();
    return kvaf::Status::OK();
  }
  int releases{0};
  std::vector<std::uint8_t> data;
};

void TestLeaseAndRuntimeTeardown() {
  driver = {};
  kvaf::DefaultAccessOperator concrete(Config());
  kvaf::AccessOperator& op = concrete;
  auto metadata = kvaf::CreateInMemoryMetaManager();
  Allocator allocator;
  kvaf::RuntimeCore runtime(*metadata, op, allocator);
  Require(runtime.Initialize({4096, 256, kvaf::DeviceKind::kCpu}).ok(), "initialize");
  std::vector<std::uint8_t> output(656 * 4);
  int stream_marker = 0;
  kvaf::FusedBatchGetContext context;
  Require(!op.AcquireFusedBatchGetContext(output.data(), output.size() - 1,
                                         656, 128, &context).ok(), "reject unaligned rows");
  driver.fail_acquire = true;
  Require(!op.AcquireFusedBatchGetContext(output.data(), output.size(),
                                         656, 128, &context).ok(), "registration failure");
  Require(context.lease_id == 0, "failure does not publish a lease");
  driver.fail_acquire = false;
  Require(op.AcquireFusedBatchGetContext(output.data(), output.size(),
                                        656, 128, &context).ok(), "acquire");
  Require(context.worker_count == 2 && context.jettys_per_core == 2 &&
              context.token_bytes == 656 && context.tokens_per_block == 128 &&
              context.output_address == reinterpret_cast<std::uint64_t>(output.data()) &&
              context.output_bytes == output.size() &&
              context.device_channel_entity != 0 && context.device_core_status != 0,
          "export complete transport and geometry");
  Require(driver.device_syncs == 2, "acquire drains previous launches once per attempt");
  auto* queues = reinterpret_cast<KvafBatchGetQueueState*>(context.device_queue_state);
  queues[0].sq_producer_index = 73;
  queues[0].cq_consumer_index = 11;
  kvaf::FusedBatchGetContext duplicate;
  Require(op.AcquireFusedBatchGetContext(output.data(), output.size(), 656, 128,
                                        &duplicate).code() == kvaf::StatusCode::kBusy,
          "reject concurrent lease");
  Require(op.ExecuteBatchGet({}).code() == kvaf::StatusCode::kBusy,
          "ordinary BatchGet cannot use leased queues");
  Require(concrete.SetBatchGetKernelProfiling(true).code() == kvaf::StatusCode::kBusy,
          "profiling cannot reset leased status");
  Require(runtime.Shutdown().code() == kvaf::StatusCode::kBusy,
          "runtime must propagate active lease");
  Require(!runtime.initialized() && runtime.needs_cleanup() &&
              allocator.releases == 0 && driver.destroys == 0,
          "busy shutdown preserves remote DDR and transport");
  auto wrong = context;
  ++wrong.lease_id;
  Require(!op.ReleaseFusedBatchGetContext(wrong, &stream_marker).ok(), "wrong lease");
  Require(!op.ReleaseFusedBatchGetContext(context, nullptr).ok(), "requires stream");
  Require(op.ReleaseFusedBatchGetContext(context, &stream_marker).ok(), "release drained lease");
  Require(driver.releases == 1, "exactly one registration release");
  Require(!op.ReleaseFusedBatchGetContext(context, &stream_marker).ok(), "reject stale release");
  kvaf::FusedBatchGetContext next;
  Require(op.AcquireFusedBatchGetContext(output.data(), output.size(), 656, 128,
                                        &next).ok(), "reacquire persistent queues");
  Require(next.lease_id != context.lease_id &&
              next.device_queue_state == context.device_queue_state &&
              queues[0].sq_producer_index == 73 && queues[0].cq_consumer_index == 11,
          "lease changes, queue state does not reset");
  Require(!op.ReleaseFusedBatchGetContext(context, &stream_marker).ok(), "old lease cannot release new");
  Require(op.ReleaseFusedBatchGetContext(next, &stream_marker).ok(), "release next");
  Require(op.ExecuteBatchGet({}).code() == kvaf::StatusCode::kInvalidArgument,
          "ordinary path is accessible after release");
  Require(runtime.Shutdown().ok() && allocator.releases == 1 && driver.destroys == 1,
          "clean shutdown releases DDR after context release");
}

void TestFailedCompletionRetainsRegistration(bool stream_failure) {
  driver = {};
  // Failure intentionally retains resources until process teardown, matching
  // a device timeout. This test supplies no real device or DMA allocation.
  auto op = std::make_unique<kvaf::DefaultAccessOperator>(Config());
  std::vector<std::uint8_t> ddr(4096), output(656 * 4);
  Require(op->Initialize({reinterpret_cast<std::uint64_t>(ddr.data()), ddr.size(),
                          256, kvaf::DeviceKind::kCpu}).ok(), "failure fixture init");
  kvaf::FusedBatchGetContext context;
  Require(op->AcquireFusedBatchGetContext(output.data(), output.size(), 656, 128,
                                         &context).ok(), "failure fixture acquire");
  std::vector<std::uint8_t> second_output(656 * 8);
  kvaf::FusedBatchGetContext view;
  Require(op->RegisterFusedBatchGetOutput(context, second_output.data(),
                                         second_output.size(), &view).ok(),
          "failure fixture output view");
  auto* queues = reinterpret_cast<KvafBatchGetQueueState*>(context.device_queue_state);
  int stream_marker = 0;
  if (stream_failure) driver.fail_stream_sync = true;
  else queues[3].last_error = 7;
  Require(!op->ReleaseFusedBatchGetContext(context, &stream_marker).ok(),
          "completion failure must not release context");
  Require(!op->RegisterFusedBatchGetOutput(context, second_output.data(),
                                          second_output.size(), &view).ok(),
          "quarantine rejects new views");
  driver.fail_stream_sync = false;
  queues[3].last_error = 0;
  Require(!op->ReleaseFusedBatchGetContext(context, &stream_marker).ok(),
          "clearing status cannot undo uncertain DMA quarantine");
  Require(op->Shutdown().code() == kvaf::StatusCode::kBusy, "failed context blocks shutdown");
  op.reset();
  Require(driver.releases == 0 && driver.destroys == 0,
          "destructor never deregisters potentially live DMA resources");
}

void TestSharedLeaseOutputViews() {
  driver = {};
  kvaf::DefaultAccessOperator op(Config());
  std::vector<std::uint8_t> ddr(4096), first(656 * 4), second(656 * 8);
  Require(op.Initialize({reinterpret_cast<std::uint64_t>(ddr.data()), ddr.size(),
                         256, kvaf::DeviceKind::kCpu}).ok(), "view fixture init");
  kvaf::FusedBatchGetContext owner, view;
  Require(!op.RegisterFusedBatchGetOutput(owner, second.data(), second.size(), &view).ok(),
          "reject view without lease");
  Require(op.AcquireFusedBatchGetContext(first.data(), first.size(), 656, 128, &owner).ok(),
          "view fixture acquire");
  auto* queues = reinterpret_cast<KvafBatchGetQueueState*>(owner.device_queue_state);
  queues[0].sq_producer_index = 73;
  auto wrong = owner;
  ++wrong.lease_id;
  Require(!op.RegisterFusedBatchGetOutput(wrong, second.data(), second.size(), &view).ok(),
          "reject stale owner");
  Require(!op.RegisterFusedBatchGetOutput(owner, second.data(), second.size() - 1, &view).ok(),
          "reject partial token row");
  driver.fail_acquire = true;
  Require(!op.RegisterFusedBatchGetOutput(owner, second.data(), second.size(), &view).ok(),
          "propagate view registration failure");
  Require(view.lease_id == 0, "failed view not published");
  driver.fail_acquire = false;
  Require(op.RegisterFusedBatchGetOutput(owner, second.data(), second.size(), &view).ok(),
          "register second output");
  Require(view.lease_id == owner.lease_id &&
              view.device_queue_state == owner.device_queue_state &&
              view.device_core_status == owner.device_core_status &&
              view.output_address == reinterpret_cast<std::uint64_t>(second.data()) &&
              view.output_bytes == second.size() &&
              view.device_channel_entity != owner.device_channel_entity &&
              queues[0].sq_producer_index == 73 && driver.device_syncs == 1,
          "view shares cursors without synchronization or reset");
  int stream_marker = 0;
  Require(!op.ReleaseFusedBatchGetContext(view, &stream_marker).ok(), "view cannot release owner");
  Require(op.ReleaseFusedBatchGetContext(owner, &stream_marker).ok(), "owner releases all views");
  Require(driver.releases == 2, "both registrations released once");
  Require(!op.RegisterFusedBatchGetOutput(owner, second.data(), second.size(), &view).ok(),
          "released owner cannot register views");
  Require(op.Shutdown().ok(), "view fixture shutdown");
}
}  // namespace

struct KvafBatchGetLocalBuffer { std::uint64_t entity{0xabc}; };
struct KvafBatchGetTransport {
  std::vector<std::unique_ptr<KvafBatchGetLocalBuffer>> registrations;
};

extern "C" {
aclError aclrtMalloc(void** ptr, std::size_t bytes, aclrtMemMallocPolicy) {
  *ptr = std::malloc(bytes);
  return *ptr == nullptr ? 1 : ACL_SUCCESS;
}
aclError aclrtFree(void* ptr) { std::free(ptr); return ACL_SUCCESS; }
aclError aclrtMemset(void* ptr, std::size_t capacity, int32_t value, std::size_t bytes) {
  if (bytes > capacity) return 1;
  std::memset(ptr, value, bytes);
  return ACL_SUCCESS;
}
aclError aclrtMemcpy(void* dst, std::size_t capacity, const void* src,
                     std::size_t bytes, aclrtMemcpyKind) {
  if (dst == nullptr || src == nullptr || bytes > capacity) return 1;
  std::memcpy(dst, src, bytes);
  return ACL_SUCCESS;
}
aclError aclrtSynchronizeDevice() { ++driver.device_syncs; return ACL_SUCCESS; }
aclError aclrtSynchronizeStream(aclrtStream) { return driver.fail_stream_sync ? 1 : ACL_SUCCESS; }
int32_t KvafBatchGetTransportCreateWithPeerLink(void*, uint64_t, uint32_t,
                                                uint32_t, void*, const uint8_t*,
                                                KvafBatchGetTransport** transport) {
  *transport = new KvafBatchGetTransport;
  return KVAF_BATCH_GET_SUCCESS;
}
int32_t KvafBatchGetTransportCreateWithPeerLinkMultiEid(
    void* host, uint64_t bytes, uint32_t cores, uint32_t jettys, void* peer,
    const uint8_t* eids, uint32_t, KvafBatchGetTransport** transport) {
  return KvafBatchGetTransportCreateWithPeerLink(host, bytes, cores, jettys,
                                                peer, eids, transport);
}
void KvafBatchGetTransportDestroy(KvafBatchGetTransport* transport) {
  ++driver.destroys;
  delete transport;
}
int32_t KvafBatchGetLocalBufferAcquire(KvafBatchGetTransport* transport, void*, uint64_t,
                                       KvafBatchGetLocalBuffer** buffer, uint64_t* entity) {
  ++driver.acquires;
  if (driver.fail_acquire) return KVAF_BATCH_GET_RUNTIME_ERROR;
  transport->registrations.push_back(std::make_unique<KvafBatchGetLocalBuffer>());
  *buffer = transport->registrations.back().get();
  *entity = reinterpret_cast<std::uint64_t>(&(*buffer)->entity);
  return KVAF_BATCH_GET_SUCCESS;
}
void KvafBatchGetLocalBufferRelease(KvafBatchGetLocalBuffer*) { ++driver.releases; }
int32_t KvafExecuteBatchGetLoadKernel(const char*) { return KVAF_BATCH_GET_SUCCESS; }
int32_t KvafExecuteBatchGetLaunch(const KvafExecuteBatchGetKernelArgs*, aclrtStream) {
  ++driver.launches;
  return KVAF_BATCH_GET_SUCCESS;
}
}

int main() {
  try {
    TestLeaseAndRuntimeTeardown();
    TestSharedLeaseOutputViews();
    TestFailedCompletionRetainsRegistration(false);
    TestFailedCompletionRetainsRegistration(true);
    std::cout << "fused BatchGet host lifecycle tests passed (mock transport only)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
