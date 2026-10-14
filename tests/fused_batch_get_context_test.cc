#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include "access_operator.h"
#include "batch_get_abi.h"
#include "kvaf_aiv_batch_get.h"
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
#include "kvaf_aiv_batch_put.h"
#endif
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
  int local_creates{0};
  int peer_creates{0};
  std::uint32_t get_workers{0};
  std::uint32_t put_workers{0};
  std::uint32_t put_launch_workers{0};
  std::uint32_t put_status_workers{0};
  int put_creates{0};
  int put_destroys{0};
  int put_launches{0};
  bool fail_put_destroy{false};
  bool fail_destroy{false};
  bool fail_local_create{false};
  bool fail_acquire{false};
  bool fail_stream_sync{false};
} driver;

void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

class StderrCapture {
 public:
  StderrCapture() {
    std::fflush(stderr);
    saved_ = dup(fileno(stderr));
    file_ = std::tmpfile();
    if (saved_ < 0 || file_ == nullptr || dup2(fileno(file_), fileno(stderr)) < 0) {
      if (saved_ >= 0) close(saved_);
      if (file_ != nullptr) std::fclose(file_);
      throw std::runtime_error("cannot capture diagnostic stderr");
    }
  }
  ~StderrCapture() {
    Restore();
    if (file_ != nullptr) std::fclose(file_);
  }
  std::string Read() {
    Restore();
    std::rewind(file_);
    std::string result;
    char buffer[1024];
    while (const auto size = std::fread(buffer, 1, sizeof(buffer), file_)) {
      result.append(buffer, size);
    }
    return result;
  }
 private:
  void Restore() {
    if (saved_ >= 0) {
      std::fflush(stderr);
      dup2(saved_, fileno(stderr));
      close(saved_);
      saved_ = -1;
    }
  }
  int saved_{-1};
  std::FILE* file_{nullptr};
};

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

void TestLocalHostOwnership() {
  driver = {};
  auto config = Config();
  config.local_host_ddr = true;
  config.peer_link.reset();
  config.peer_local_eid.fill(0);
  kvaf::DefaultAccessOperator op(config);
  auto metadata = kvaf::CreateInMemoryMetaManager();
  Allocator allocator;
  kvaf::RuntimeCore runtime(*metadata, op, allocator);
  Require(runtime.Initialize({4096, 256, kvaf::DeviceKind::kCpu}).ok(), "local initialize without Link");
  Require(driver.local_creates == 1 && driver.peer_creates == 0, "local transport selected");
  driver.fail_destroy = true;
  Require(!runtime.Shutdown().ok(), "failed queue retirement reaches Runtime");
  Require(allocator.releases == 0 && !allocator.data.empty(), "failed retirement retains DDR");
  driver.fail_destroy = false;
  Require(runtime.Shutdown().ok(), "retry retired transport");
  Require(allocator.releases == 1 && driver.destroys == 1, "successful retirement releases once");
}

void TestPartialLocalInitialization() {
  driver = {};
  auto config = Config();
  config.local_host_ddr = true;
  config.peer_link.reset();
  kvaf::DefaultAccessOperator op(config);
  auto metadata = kvaf::CreateInMemoryMetaManager();
  Allocator allocator;
  kvaf::RuntimeCore runtime(*metadata, op, allocator);
  driver.fail_local_create = true;
  driver.fail_destroy = true;
  Require(!runtime.Initialize({4096, 256, kvaf::DeviceKind::kCpu}).ok(), "partial initialization fails");
  Require(runtime.initialized() && allocator.releases == 0, "failed rollback retains DDR ownership");
  Require(!op.ExecuteBatchGet({}).ok(), "partial transport cannot execute");
  driver.fail_destroy = false;
  Require(runtime.Shutdown().ok(), "retire partial transport before DDR release");
  Require(allocator.releases == 1 && driver.destroys == 1, "partial resources released once");
}

void TestLeaseAndRuntimeTeardown() {
  driver = {};
  kvaf::DefaultAccessOperator op(Config());
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
  Require(op.SetBatchGetKernelProfiling(true).code() == kvaf::StatusCode::kBusy,
          "profiling cannot reset leased status");
  Require(runtime.Shutdown().code() == kvaf::StatusCode::kBusy,
          "runtime must propagate active lease");
  Require(runtime.initialized() && allocator.releases == 0 && driver.destroys == 0,
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
  auto* queues = reinterpret_cast<KvafBatchGetQueueState*>(context.device_queue_state);
  int stream_marker = 0;
  if (stream_failure) driver.fail_stream_sync = true;
  else queues[3].last_error = 7;
  Require(!op->ReleaseFusedBatchGetContext(context, &stream_marker).ok(),
          "completion failure must not release context");
  driver.fail_stream_sync = false;
  queues[3].last_error = 0;
  Require(!op->ReleaseFusedBatchGetContext(context, &stream_marker).ok(),
          "clearing status cannot undo uncertain DMA quarantine");
  Require(op->Shutdown().code() == kvaf::StatusCode::kBusy, "failed context blocks shutdown");
  op.reset();
  Require(driver.releases == 0 && driver.destroys == 0,
          "destructor never deregisters potentially live DMA resources");
}

void TestQueueHealthDiagnostics(bool release_boundary) {
  driver = {};
  auto op = std::make_unique<kvaf::DefaultAccessOperator>(Config());
  std::vector<std::uint8_t> ddr(4096), output(656 * 4);
  Require(op->Initialize({reinterpret_cast<std::uint64_t>(ddr.data()), ddr.size(),
                         256, kvaf::DeviceKind::kCpu}).ok(), "queue diagnostic fixture init");
  kvaf::FusedBatchGetContext context;
  {
    StderrCapture capture;
    Require(op->AcquireFusedBatchGetContext(output.data(), output.size(), 656, 128,
                                           &context).ok(), "healthy queue acquire");
    Require(capture.Read().empty(), "healthy setup adds no status diagnostics");
  }
  auto* queues = reinterpret_cast<KvafBatchGetQueueState*>(context.device_queue_state);
  auto* statuses = reinterpret_cast<KvafBatchGetCoreStatus*>(context.device_core_status);
  int stream_marker = 0;
  if (!release_boundary) {
    StderrCapture capture;
    Require(op->ReleaseFusedBatchGetContext(context, &stream_marker).ok(), "healthy queue release");
    Require(capture.Read().empty(), "healthy release adds no status diagnostics");
  }
  // Populate host-backed ABI records only; this does not simulate any CQ or
  // device execution. Two poisoned Jettys must both remain visible in stderr.
  queues[1].last_error = 7;
  queues[1].sq_producer_index = 51;
  queues[1].cq_consumer_index = 4;
  queues[1].completed_batch_count = 3;
  queues[3].last_error = 9;
  queues[3].sq_producer_index = 83;
  queues[3].cq_consumer_index = 6;
  queues[3].completed_batch_count = 5;
  statuses[0].magic = statuses[1].magic = KVAF_AIV_BATCH_GET_STATUS_MAGIC;
  statuses[1].result = 7;
  statuses[1].processed_request_count = 12;
  statuses[1].completed_batch_count = 5;
  statuses[1].sq_producer_index = 81;
  statuses[1].cq_consumer_index = 6;
  statuses[1].failed_request_index = 11;
  statuses[1].last_wqe_protocol = 0x1234;
  const int previous_acquires = driver.acquires;
  const int previous_releases = driver.releases;
  std::string diagnostics;
  {
    StderrCapture capture;
    kvaf::Status result;
    if (release_boundary) {
      result = op->ReleaseFusedBatchGetContext(context, &stream_marker);
    } else {
      kvaf::FusedBatchGetContext next;
      result = op->AcquireFusedBatchGetContext(output.data(), output.size(), 656, 128, &next);
      Require(next.lease_id == 0, "poisoned queue does not publish a lease");
    }
    Require(!result.ok(), "poisoned queue fails health check");
    diagnostics = capture.Read();
  }
  Require(diagnostics.find("BATCH_GET_QUEUE_ERROR core=0 jetty=1 last_error=7 sq_pi=51 cq_ci=4 completed_batches=3")
              != std::string::npos, "log first poisoned core and Jetty with persistent indices");
  Require(diagnostics.find("BATCH_GET_QUEUE_ERROR core=1 jetty=1 last_error=9 sq_pi=83 cq_ci=6 completed_batches=5")
              != std::string::npos, "log every poisoned queue rather than just the first");
  Require(diagnostics.find("BATCH_GET_DEVICE_ERROR core=1") != std::string::npos &&
              diagnostics.find("result=7 processed=12 completed_batches=5 sq_pi=81 cq_ci=6 failed_request=11 last_wqe=0x1234")
              != std::string::npos, "log existing device ABI fields alongside queue poison");
  Require(diagnostics.find("BATCH_GET_DEVICE_SNAPSHOT core=0") != std::string::npos &&
              diagnostics.find("BATCH_GET_DEVICE_ERROR core=0") == std::string::npos,
          "failed queue retains a snapshot even when its core status is partial or stale");
  Require(driver.acquires == previous_acquires && driver.releases == previous_releases &&
              driver.launches == 0 && driver.destroys == 0,
          "failure diagnostics post no work and do not deregister or retire transport");
  Require(queues[1].last_error == 7 && queues[3].last_error == 9 &&
              queues[1].sq_producer_index == 51 && queues[3].cq_consumer_index == 6,
          "logging never clears poison or modifies queue state");
  if (release_boundary) {
    Require(op->Shutdown().code() == kvaf::StatusCode::kBusy,
            "failed release still quarantines its registration");
  } else {
    Require(op->Shutdown().ok(), "unleased mock queue retires outside diagnostic check");
  }
  op.reset();
}

void TestQueueHealthCqProfileDiagnostics(bool profile_enabled) {
  driver = {};
  auto op = std::make_unique<kvaf::DefaultAccessOperator>(Config());
  std::vector<std::uint8_t> ddr(4096), output(656 * 4);
  Require(op->Initialize({reinterpret_cast<std::uint64_t>(ddr.data()), ddr.size(),
                         256, kvaf::DeviceKind::kCpu}).ok(), "CQ profile diagnostic fixture init");
  kvaf::FusedBatchGetContext context;
  Require(op->AcquireFusedBatchGetContext(output.data(), output.size(), 656, 128,
                                         &context).ok(), "CQ profile fixture obtains ABI pointers");
  auto* queues = reinterpret_cast<KvafBatchGetQueueState*>(context.device_queue_state);
  auto* statuses = reinterpret_cast<KvafBatchGetCoreStatus*>(context.device_core_status);
  int stream_marker = 0;
  Require(op->ReleaseFusedBatchGetContext(context, &stream_marker).ok(), "CQ profile fixture releases lease");
  Require(op->SetBatchGetKernelProfiling(profile_enabled).ok(), "configure real profile magic");
  // Populate real, host-backed ABI snapshots, without simulating device CQ
  // execution or DMA. Timing on a failed core remains useful even though the
  // public successful-profile API correctly rejects its aggregate error.
  statuses[0].magic = statuses[1].magic = KVAF_AIV_BATCH_GET_STATUS_MAGIC;
  statuses[0].result = 6;
  statuses[0].profile.active_requests = 307;
  statuses[0].profile.total_posts = 10;
  statuses[0].profile.jetty_completion_polls[0] = 20000;
  statuses[0].profile.jetty_completion_polls[1] = 1;
  statuses[0].profile.jetty_completion_wait_cycles[0] = 11000123;
  statuses[0].profile.jetty_completion_wait_cycles[1] = 321;
  statuses[1].profile.active_requests = 307;
  statuses[1].profile.total_posts = 10;
  statuses[1].profile.jetty_completion_polls[0] = 8;
  statuses[1].profile.jetty_completion_polls[1] = 2;
  statuses[1].profile.jetty_completion_wait_cycles[0] = 1000;
  statuses[1].profile.jetty_completion_wait_cycles[1] = 400;
  queues[0].last_error = queues[1].last_error = 6;
  queues[0].sq_producer_index = 160;
  queues[1].sq_producer_index = 147;
  queues[1].cq_consumer_index = 1;
  queues[1].completed_batch_count = 1;
  const int previous_acquires = driver.acquires;
  const int previous_releases = driver.releases;
  std::string diagnostics;
  {
    StderrCapture capture;
    kvaf::FusedBatchGetContext next;
    Require(!op->AcquireFusedBatchGetContext(output.data(), output.size(), 656, 128,
                                            &next).ok(), "poisoned profile fixture fails health check");
    Require(next.lease_id == 0, "CQ profile logging does not publish a lease");
    diagnostics = capture.Read();
  }
  Require(diagnostics.find("BATCH_GET_QUEUE_ERROR core=0 jetty=0 last_error=6") != std::string::npos &&
              diagnostics.find("BATCH_GET_DEVICE_ERROR core=0") != std::string::npos,
          "queue and core fault remain visible alongside timings");
  if (profile_enabled) {
    Require(diagnostics.find("BATCH_GET_CQ_PROFILE core=0 jetty=0 core_result=6 polls=20000 wait_ticks=11000123 "
                             "active_requests=307 total_posts=10 clock_hz=unconfirmed") != std::string::npos,
            "retain failed Jetty timing with an explicit aggregate result and unknown clock");
    Require(diagnostics.find("BATCH_GET_CQ_PROFILE core=0 jetty=1 core_result=6 polls=1 wait_ticks=321")
                != std::string::npos,
            "second Jetty keeps its independent polls and ticks without claiming its own result");
    Require(diagnostics.find("BATCH_GET_CQ_PROFILE core=1 jetty=0 core_result=0 polls=8 wait_ticks=1000")
                != std::string::npos &&
                diagnostics.find("BATCH_GET_CQ_PROFILE core=1 jetty=1 core_result=0 polls=2 wait_ticks=400")
                != std::string::npos,
            "successful cores in a failed owner snapshot retain both Jetty timings");
  } else {
    Require(diagnostics.find("BATCH_GET_CQ_PROFILE") == std::string::npos,
            "without profile magic no unverified timing is emitted");
  }
  Require(driver.acquires == previous_acquires && driver.releases == previous_releases &&
              driver.launches == 0 && driver.destroys == 0,
          "CQ profile logging launches no work and leaves registrations retained");
  Require(queues[0].last_error == 6 && queues[1].cq_consumer_index == 1 &&
              statuses[0].result == 6 && statuses[0].profile.jetty_completion_polls[0] == 20000,
          "CQ profile logging does not clear errors or mutate device snapshots");
  Require(op->Shutdown().ok(), "unleased CPU fixture retires outside the diagnostic check");
}

#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
void TestIndependentPutPool(std::uint32_t requested_put_workers,
                            std::uint32_t expected_put_workers, bool local_host) {
  driver = {};
  auto config = Config();
  config.npu_core_count = 30;
  config.batch_put_kernel_object_path = "mock-put-kernel";
  config.batch_put_npu_core_count = requested_put_workers;
  config.local_host_ddr = local_host;
  if (local_host) {
    config.peer_link.reset();
    config.peer_local_eid.fill(0);
  }
  kvaf::DefaultAccessOperator op(config);
  auto metadata = kvaf::CreateInMemoryMetaManager();
  Allocator allocator;
  kvaf::RuntimeCore runtime(*metadata, op, allocator);
  Require(runtime.Initialize({4096, 256, kvaf::DeviceKind::kCpu}).ok(), "independent Put init");
  Require(driver.get_workers == 30 && driver.put_workers == expected_put_workers &&
              driver.put_creates == 1,
          "Put transport uses separate workers; Get pool remains 30");
  std::vector<std::uint8_t> source(512), output(656 * 4);
  int stream_marker = 0;
  Require(runtime.BatchPut({{1, 2},
                            {source.data(), source.size(), kvaf::DeviceKind::kNpu},
                            {0, 256}, {256, 256}}, {&stream_marker}).ok(),
          "eager Put runs with independent status and transport geometry");
  Require(driver.put_launches == 1 && driver.put_launch_workers == expected_put_workers &&
              driver.put_status_workers == expected_put_workers,
          "Put launch and status storage use the Put pool");
  kvaf::FusedBatchGetContext context;
  Require(op.AcquireFusedBatchGetContext(output.data(), output.size(), 656, 128,
                                        &context).ok(), "lease large Get pool");
  Require(context.worker_count == 30 && context.jettys_per_core == 2,
          "Put workers do not shrink the fused Get lease");
  Require(op.ReleaseFusedBatchGetContext(context, &stream_marker).ok(), "release Get lease");
  driver.fail_put_destroy = true;
  Require(!runtime.Shutdown().ok() && allocator.releases == 0 && !allocator.data.empty(),
          "failed independent Put retirement retains DDR");
  driver.fail_put_destroy = false;
  Require(runtime.Shutdown().ok() && allocator.releases == 1 &&
              driver.put_destroys == 1 && driver.destroys == 1,
          "successful retry retires both pools before DDR release");
}

void TestInvalidPutPoolRejectedBeforeTransportCreation() {
  driver = {};
  auto config = Config();
  config.local_host_ddr = true;
  config.peer_link.reset();
  config.batch_put_kernel_object_path = "mock-put-kernel";
  config.batch_put_npu_core_count = 49;
  kvaf::DefaultAccessOperator op(config);
  std::vector<std::uint8_t> ddr(4096);
  Require(op.Initialize({reinterpret_cast<std::uint64_t>(ddr.data()), ddr.size(),
                         256, kvaf::DeviceKind::kCpu}).code() == kvaf::StatusCode::kInvalidArgument,
          "reject Put workers above ABI limit");
  Require(driver.local_creates == 0 && driver.put_creates == 0,
          "invalid Put count creates no transport");
}
#endif
}  // namespace

struct KvafBatchGetLocalBuffer { std::uint64_t entity{0xabc}; };
struct KvafBatchGetTransport {
  std::vector<std::unique_ptr<KvafBatchGetLocalBuffer>> registrations;
};
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
struct KvafBatchPutLocalBuffer {};
struct KvafBatchPutTransport {
  std::uint32_t workers{0};
  KvafBatchPutLocalBuffer registration;
};
#endif

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
int32_t KvafBatchGetTransportCreateWithPeerLink(void*, uint64_t, uint32_t workers,
                                                uint32_t, void*, const uint8_t*,
                                                KvafBatchGetTransport** transport) {
  ++driver.peer_creates;
  driver.get_workers = workers;
  *transport = new KvafBatchGetTransport;
  return KVAF_BATCH_GET_SUCCESS;
}
int32_t KvafBatchGetTransportCreateLocalHost(void*, uint64_t, uint32_t workers,
                                              uint32_t, KvafBatchGetTransport** transport) {
  ++driver.local_creates;
  driver.get_workers = workers;
  *transport = new KvafBatchGetTransport;
  return driver.fail_local_create ? KVAF_BATCH_GET_RUNTIME_ERROR : KVAF_BATCH_GET_SUCCESS;
}
int32_t KvafBatchGetTransportDestroyChecked(KvafBatchGetTransport* transport) {
  if (driver.fail_destroy) return KVAF_BATCH_GET_RUNTIME_ERROR;
  ++driver.destroys;
  delete transport;
  return KVAF_BATCH_GET_SUCCESS;
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

#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
int32_t KvafBatchPutTransportCreateLocalHost(void*, uint64_t, uint32_t workers,
                                              uint32_t, KvafBatchPutTransport** transport) {
  ++driver.put_creates;
  driver.put_workers = workers;
  *transport = new KvafBatchPutTransport{workers, {}};
  return KVAF_BATCH_PUT_SUCCESS;
}
int32_t KvafBatchPutTransportCreateWithPeerLink(void* ddr, uint64_t bytes, uint32_t workers,
                                                uint32_t jettys, void*, const uint8_t*,
                                                KvafBatchPutTransport** transport) {
  return KvafBatchPutTransportCreateLocalHost(ddr, bytes, workers, jettys, transport);
}
int32_t KvafBatchPutTransportDestroyChecked(KvafBatchPutTransport* transport) {
  if (driver.fail_put_destroy) return KVAF_BATCH_PUT_RUNTIME_ERROR;
  ++driver.put_destroys;
  delete transport;
  return KVAF_BATCH_PUT_SUCCESS;
}
int32_t KvafBatchPutLocalBufferAcquire(KvafBatchPutTransport* transport, void*, uint64_t,
                                       KvafBatchPutLocalBuffer** buffer) {
  *buffer = &transport->registration;
  return KVAF_BATCH_PUT_SUCCESS;
}
void KvafBatchPutLocalBufferRelease(KvafBatchPutLocalBuffer*) {}
int32_t KvafExecuteBatchPutLoadKernel(const char*) { return KVAF_BATCH_PUT_SUCCESS; }
int32_t KvafExecuteBatchPutSync(KvafBatchPutLocalBuffer*,
                                const KvafExecuteBatchPutKernelArgs* arguments,
                                aclrtStream, KvafBatchPutCoreStatus* statuses,
                                uint32_t status_workers) {
  ++driver.put_launches;
  driver.put_launch_workers = arguments->aiv_core_count;
  driver.put_status_workers = status_workers;
  if (arguments->aiv_core_count != driver.put_workers || status_workers != driver.put_workers) {
    return KVAF_BATCH_PUT_INVALID_ARGUMENT;
  }
  auto* device_statuses = reinterpret_cast<KvafBatchPutCoreStatus*>(arguments->core_status);
  for (std::uint32_t core = 0; core < status_workers; ++core) {
    device_statuses[core] = {};
    device_statuses[core].magic = KVAF_AIV_BATCH_PUT_STATUS_MAGIC;
    statuses[core] = device_statuses[core];
  }
  return KVAF_BATCH_PUT_SUCCESS;
}
#endif
}

int main() {
  try {
    TestLocalHostOwnership();
    TestPartialLocalInitialization();
    TestLeaseAndRuntimeTeardown();
    TestFailedCompletionRetainsRegistration(false);
    TestFailedCompletionRetainsRegistration(true);
    TestQueueHealthDiagnostics(false);
    TestQueueHealthDiagnostics(true);
    TestQueueHealthCqProfileDiagnostics(false);
    TestQueueHealthCqProfileDiagnostics(true);
#if defined(KVAF_ENABLE_AIV_BATCH_PUT)
    TestIndependentPutPool(1, 1, true);
    TestIndependentPutPool(0, 30, true);
    TestIndependentPutPool(48, 48, true);
    TestIndependentPutPool(2, 2, false);
    TestInvalidPutPoolRejectedBeforeTransportCreation();
#endif
    std::cout << "fused BatchGet host lifecycle tests passed (mock transport only)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
