// Exercises the production Operator against an asynchronous dependency model.
// This verifies host ownership/order/offsets, not AIV or ACL device execution.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include "access_operator.h"
#include "kvaf_aiv_batch_get.h"

struct Task {
  std::vector<std::shared_ptr<Task>> dependencies;
  std::function<void()> action;
  bool complete{false};
};
struct Stream { std::shared_ptr<Task> tail; };
struct Event { std::shared_ptr<Task> recorded; };
struct Launch {
  KvafExecuteBatchGetKernelArgs args;
  aclrtStream stream;
  std::shared_ptr<Task> task;
};
struct KvafBatchGetLocalBuffer {
  std::vector<KvafBatchGetDirectUrmaEntity> entities;
  int refs{0};
};
struct KvafBatchGetTransport {
  std::size_t workers;
  int id;
  std::map<void*, KvafBatchGetLocalBuffer> buffers;
};

namespace {
std::vector<Launch> launches;
std::map<void*, std::size_t> allocations;
int live_streams = 0, live_events = 0, live_transports = 0;
int transport_creates = 0, synchronizations = 0, launch_attempts = 0;
int fail_launch = 0, fail_event_create = 0, event_attempts = 0;
int transport_attempts = 0, fail_transport = 0, acquire_attempts = 0, fail_acquire = 0;
int allocation_attempts = 0, fail_allocation = 0;
bool fail_synchronize = false;
std::map<std::uint64_t, std::size_t> entity_allocations;

void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
std::shared_ptr<Task> Enqueue(aclrtStream handle, std::function<void()> action = {}) {
  auto* stream = static_cast<Stream*>(handle);
  auto task = std::make_shared<Task>();
  if (stream->tail) task->dependencies.push_back(stream->tail);
  task->action = std::move(action);
  stream->tail = task;
  return task;
}
void Run(const std::shared_ptr<Task>& task) {
  if (!task || task->complete) return;
  for (const auto& dependency : task->dependencies) Run(dependency);
  if (task->action) task->action();
  task->complete = true;
}
bool DependsOn(const std::shared_ptr<Task>& task, const std::shared_ptr<Task>& ancestor) {
  if (task == ancestor) return true;
  for (const auto& dependency : task->dependencies) {
    if (DependsOn(dependency, ancestor)) return true;
  }
  return false;
}
void CheckAllocation(std::uint64_t pointer, std::size_t bytes) {
  for (const auto& allocation : allocations) {
    const auto begin = reinterpret_cast<std::uintptr_t>(allocation.first);
    if (pointer >= begin && pointer - begin <= allocation.second &&
        bytes <= allocation.second - (pointer - begin)) return;
  }
  throw std::runtime_error("kernel state slice exceeds allocated storage");
}
kvaf::AccessOperatorConfig Config(std::uint32_t cores, std::uint32_t streams,
                                  std::uint32_t jettys = 2, std::uint32_t transports = 1) {
  kvaf::AccessOperatorConfig config;
  config.batch_get_kernel_object_path = "fake-kernel";
  config.npu_core_count = cores;
  config.batch_get_stream_count = streams;
  config.batch_get_transport_count = transports;
  config.jettys_per_core = jettys;
  config.peer_local_eid[0] = 1;
  config.peer_link = std::shared_ptr<kvaf::PeerLink>(reinterpret_cast<kvaf::PeerLink*>(1), [](auto*) {});
  return config;
}
kvaf::BatchGetRequest Request(void* output) {
  kvaf::BatchGetRequest request;
  using kvaf::DeviceKind;
  // Device pointers are intentionally unreadable: launch must not download
  // miss_count or inspect request data on the host.
  request.indices = {reinterpret_cast<void*>(0x1000), 400, DeviceKind::kNpu};
  request.offsets = {reinterpret_cast<void*>(0x2000), 400, DeviceKind::kNpu};
  request.output_indices = {reinterpret_cast<void*>(0x3000), 400, DeviceKind::kNpu};
  request.miss_count = {reinterpret_cast<void*>(0x4000), 4, DeviceKind::kNpu};
  request.device_access_table.addresses = {reinterpret_cast<void*>(0x5000), 800, DeviceKind::kNpu};
  request.device_access_table.valid = {reinterpret_cast<void*>(0x6000), 100, DeviceKind::kNpu};
  request.output = {output, 6400, DeviceKind::kNpu};
  request.output_block_bytes = 64;
  request.tokens_per_block = 1;
  return request;
}

void TestOperator(std::uint32_t streams, std::uint32_t cores, std::uint32_t jettys,
                  std::uint32_t transports = 1) {
  launches.clear();
  launch_attempts = 0;
  const auto creations_before = transport_creates;
  kvaf::DefaultAccessOperator op(Config(cores, streams, jettys, transports));
  Check(op.Initialize({0x100000, 6400, 64}).ok(), "operator initialization");
  Check(transport_creates == creations_before + static_cast<int>(transports), "requested transport count");
  Check(op.SetBatchGetKernelProfiling(true).ok(), "profile allocation");
  Stream caller;
  const auto upstream = Enqueue(&caller);
  auto request = Request(reinterpret_cast<void*>(0x7000));
  kvaf::BatchGetExecRequest exec;
  exec.request = &request;
  exec.execution.stream = &caller;
  const auto syncs_before = synchronizations;
  Check(op.ExecuteBatchGet(exec).ok(), "first enqueue");
  Check(synchronizations == syncs_before, "enqueue must not synchronize host");
  Check(launches.size() == streams, "kernel count");
  for (const auto& launch : launches) {
    Check(launch.args.block_count == cores, "core count is per stream");
    Check(launch.args.shard_count == streams, "shard count");
    Check(launch.args.output == 0x7000 && launch.args.indices == 0x1000 &&
          launch.args.miss_count == 0x4000, "global input/output pointers");
    Check(DependsOn(launch.task, upstream), "both kernels depend on producer");
  }
  if (streams == 2) {
    Check(!DependsOn(launches[0].task, launches[1].task) &&
          !DependsOn(launches[1].task, launches[0].task), "kernels must be independent");
    const auto& side = launches[0].args;
    const auto& first = launches[1].args;
    Check(side.shard_id == 1 && first.shard_id == 0, "distinct shards");
    if (transports == 1) {
      Check(side.channel_entity - first.channel_entity == cores * jettys * sizeof(KvafBatchGetDirectUrmaEntity), "shared transport entity offset");
    }
    const auto* side_entity = reinterpret_cast<const KvafBatchGetDirectUrmaEntity*>(side.channel_entity);
    const auto* first_entity = reinterpret_cast<const KvafBatchGetDirectUrmaEntity*>(first.channel_entity);
    Check((side_entity->reserved[0] == first_entity->reserved[0]) == (transports == 1), "transport ownership per stream");
    Check(side_entity->sq_va != first_entity->sq_va && side_entity->cq_va != first_entity->cq_va,
          "physical communication queues cannot overlap");
    Check(side.queue_state - first.queue_state == cores * jettys * sizeof(KvafBatchGetQueueState), "queue offset");
    Check(side.core_status - first.core_status == cores * sizeof(KvafBatchGetCoreStatus), "status offset");
  }
  const auto downstream = Enqueue(&caller);
  for (const auto& launch : launches) Check(DependsOn(downstream, launch.task), "join precedes consumer");
  // Change output and enqueue again without a host wait. Queue indices must
  // survive both calls while each output gets its own entity table.
  request.output.data = reinterpret_cast<void*>(0x8000);
  Check(op.ExecuteBatchGet(exec).ok(), "second enqueue");
  Check(synchronizations == syncs_before, "repeated enqueue must not synchronize");
  for (std::size_t i = 0; i < streams; ++i) {
    Check(launches[i].args.queue_state == launches[i + streams].args.queue_state, "queue state persists across outputs");
    Check(launches[i].args.channel_entity != launches[i + streams].args.channel_entity, "output binding changes");
    Check(DependsOn(launches[i + streams].task, launches[i].task), "queue reuse ordered");
  }
  Check(aclrtSynchronizeStream(&caller) == ACL_SUCCESS, "caller synchronization");
  for (const auto& launch : launches) Check(launch.task->complete, "caller sync completes side kernel");
  std::vector<kvaf::BatchGetKernelProfile> profiles;
  Check(op.CopyBatchGetKernelProfiles(&profiles).ok(), "profile copy covers all streams");
  Check(profiles.size() == cores * streams, "total profile count");
  for (std::size_t i = 0; i < streams; ++i) {
    const auto* queue = reinterpret_cast<KvafBatchGetQueueState*>(launches[i].args.queue_state);
    Check(queue->sq_producer_index == 34, "persistent queue not zeroed per call");
  }
  Check(op.Shutdown().ok(), "shutdown");
}

void TestFailures() {
  for (const std::uint32_t transports : {1U, 2U}) {
  for (const int failed_launch : {1, 2}) {
    launches.clear(); launch_attempts = 0; fail_launch = failed_launch;
    kvaf::DefaultAccessOperator op(Config(1, 2, 2, transports));
    Check(op.Initialize({0x100000, 6400, 64}).ok(), "failure test init");
    Stream caller;
    auto request = Request(reinterpret_cast<void*>(0x7000));
    kvaf::BatchGetExecRequest exec;
    exec.request = &request; exec.execution.stream = &caller;
    Check(!op.ExecuteBatchGet(exec).ok(), "launch failure returned");
    const auto attempts = launch_attempts;
    Check(!op.ExecuteBatchGet(exec).ok() && launch_attempts == attempts, "failed queues cannot be reused");
    aclrtSynchronizeStream(&caller);
    for (const auto& launch : launches) Check(launch.task->complete, "successful sibling joined on error");
    Check(op.Shutdown().ok(), "error teardown");
  }
  fail_launch = 0;
  for (const int failed_event : {1, 2}) {
    fail_event_create = failed_event; event_attempts = 0;
    kvaf::DefaultAccessOperator op(Config(1, 2, 2, transports));
    Check(!op.Initialize({0x100000, 6400, 64}).ok(), "event creation failure returned");
    Check(live_streams == 0 && live_events == 0 && live_transports == 0 && allocations.empty(), "partial initialization cleaned up");
  }
  fail_event_create = 0;
  }
  for (const int failure : {1, 2}) {
    transport_attempts = 0; fail_transport = failure;
    kvaf::DefaultAccessOperator op(Config(1, 2, 1, 2));
    Check(!op.Initialize({0x100000, 6400, 64}).ok(), "transport creation failure");
    Check(live_transports == 0 && allocations.empty(), "all partial transports released");
  }
  fail_transport = 0;
  for (const int failure : {1, 2}) {
    allocation_attempts = 0; fail_allocation = failure;
    kvaf::DefaultAccessOperator op(Config(1, 2, 1, 2));
    Check(!op.Initialize({0x100000, 6400, 64}).ok(), "state allocation failure");
    Check(live_transports == 0 && allocations.empty(), "transports and partial state released");
  }
  fail_allocation = 0;
  for (const int failure : {1, 2}) {
    launches.clear(); acquire_attempts = 0; fail_acquire = failure;
    kvaf::DefaultAccessOperator op(Config(1, 2, 1, 2));
    Check(op.Initialize({0x100000, 6400, 64}).ok(), "acquire failure init");
    Stream caller;
    auto request = Request(reinterpret_cast<void*>(0x7000));
    kvaf::BatchGetExecRequest exec;
    exec.request = &request; exec.execution.stream = &caller;
    Check(!op.ExecuteBatchGet(exec).ok() && launches.empty(), "acquire all bindings before any launch");
    fail_acquire = 0;
    Check(op.ExecuteBatchGet(exec).ok(), "retry after acquisition failure");
    aclrtSynchronizeStream(&caller);
    Check(op.Shutdown().ok(), "acquire retry teardown");
  }
  {
    kvaf::DefaultAccessOperator op(Config(1, 2, 1, 2));
    Check(op.Initialize({0x100000, 6400, 64}).ok(), "drain failure init");
    Stream caller;
    auto request = Request(reinterpret_cast<void*>(0x7000));
    kvaf::BatchGetExecRequest exec;
    exec.request = &request; exec.execution.stream = &caller;
    Check(op.ExecuteBatchGet(exec).ok(), "drain failure enqueue");
    fail_synchronize = true;
    Check(!op.Shutdown().ok() && live_transports == 2 && !allocations.empty(),
          "failed drain retains both transports and queue state");
    fail_synchronize = false;
    aclrtSynchronizeStream(&caller);
    Check(op.Shutdown().ok(), "drain retry teardown");
  }
  for (auto config : {Config(25, 2), Config(1, 0), Config(1, 3),
                      Config(1, 1, 1, 2), Config(1, 2, 1, 0), Config(1, 2, 1, 3)}) {
    kvaf::DefaultAccessOperator op(config);
    Check(!op.Initialize({0x100000, 6400, 64}).ok(), "invalid topology rejected");
  }
  auto config = Config(1, 2, 1, 2);
  config.peer_local_eids = {{{1}}, {{2}}};
  kvaf::DefaultAccessOperator op(config);
  Check(!op.Initialize({0x100000, 6400, 64}).ok(), "EID count limited by workers per transport");
}
}  // namespace

aclError aclrtCreateStream(aclrtStream* stream) { *stream = new Stream; ++live_streams; return 0; }
aclError aclrtDestroyStream(aclrtStream stream) { delete static_cast<Stream*>(stream); --live_streams; return 0; }
aclError aclrtCreateEventExWithFlag(aclrtEvent* event, std::uint32_t flag) {
  Check(flag == ACL_EVENT_SYNC, "sync events required");
  if (++event_attempts == fail_event_create) return 1;
  *event = new Event; ++live_events; return 0;
}
aclError aclrtDestroyEvent(aclrtEvent event) { delete static_cast<Event*>(event); --live_events; return 0; }
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
  static_cast<Event*>(event)->recorded = Enqueue(stream); return 0;
}
aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
  auto snapshot = static_cast<Event*>(event)->recorded;
  Check(snapshot != nullptr, "wait requires record");
  Enqueue(stream)->dependencies.push_back(snapshot); return 0;
}
aclError aclrtSynchronizeStream(aclrtStream stream) {
  ++synchronizations;
  if (fail_synchronize) return 1;
  Run(static_cast<Stream*>(stream)->tail); return 0;
}
aclError aclrtMalloc(void** pointer, std::size_t bytes, int) {
  if (++allocation_attempts == fail_allocation) { *pointer = nullptr; return 1; }
  *pointer = std::malloc(bytes); allocations[*pointer] = bytes; return *pointer ? 0 : 1;
}
aclError aclrtFree(void* pointer) { allocations.erase(pointer); std::free(pointer); return 0; }
aclError aclrtMemset(void* pointer, std::size_t capacity, int value, std::size_t bytes) {
  Check(bytes <= capacity, "memset capacity"); std::memset(pointer, value, bytes); return 0;
}
aclError aclrtMemcpy(void* dst, std::size_t capacity, const void* src, std::size_t bytes, int) {
  Check(bytes <= capacity, "memcpy capacity"); std::memcpy(dst, src, bytes); return 0;
}
extern "C" int32_t KvafExecuteBatchGetLoadKernel(const char*) { return 0; }
extern "C" int32_t KvafBatchGetTransportCreateWithPeerLinkMultiEid(
    void* ddr, uint64_t bytes, uint32_t cores, uint32_t jettys, void* link, const uint8_t* eids, uint32_t eid_count,
    KvafBatchGetTransport** result) {
  Check(ddr == reinterpret_cast<void*>(0x100000) && bytes == 6400 && link == reinterpret_cast<void*>(1),
        "transports share one remote region and peer link");
  Check(eid_count == 1 && eids[0] == 1, "each transport uses the same EID");
  if (++transport_attempts == fail_transport) { *result = nullptr; return -2; }
  *result = new KvafBatchGetTransport{cores * jettys, ++transport_creates, {}};
  ++live_transports; return 0;
}
extern "C" void KvafBatchGetTransportLogResources(const KvafBatchGetTransport* transport,
    uint32_t, uint32_t, uint32_t cores, uint32_t jettys) {
  Check(transport->workers % (cores * jettys) == 0, "whole streams per transport");
}
extern "C" void KvafBatchGetTransportDestroy(KvafBatchGetTransport* transport) {
  for (const auto& binding : transport->buffers) {
    Check(binding.second.refs == 0, "all local buffer references released");
    entity_allocations.erase(reinterpret_cast<uint64_t>(binding.second.entities.data()));
  }
  delete transport; --live_transports;
}
extern "C" int32_t KvafBatchGetLocalBufferAcquire(KvafBatchGetTransport* transport, void* output,
    uint64_t, KvafBatchGetLocalBuffer** buffer, uint64_t* entity) {
  if (++acquire_attempts == fail_acquire) { *buffer = nullptr; *entity = 0; return -2; }
  auto& value = transport->buffers[output]; value.entities.resize(transport->workers);
  ++value.refs;
  for (std::size_t i = 0; i < transport->workers; ++i) {
    value.entities[i].reserved[0] = transport->id;
    value.entities[i].local_hbm_address = reinterpret_cast<uint64_t>(output);
    value.entities[i].sq_va = static_cast<uint64_t>(transport->id) * 0x1000000 + i * 4096;
    value.entities[i].cq_va = value.entities[i].sq_va + 2048;
  }
  entity_allocations[reinterpret_cast<uint64_t>(value.entities.data())] = transport->workers * sizeof(value.entities[0]);
  *buffer = &value; *entity = reinterpret_cast<uint64_t>(value.entities.data()); return 0;
}
extern "C" void KvafBatchGetLocalBufferRelease(KvafBatchGetLocalBuffer* buffer) {
  Check(buffer->refs > 0, "balanced local buffer acquire/release"); --buffer->refs;
}
extern "C" int32_t KvafExecuteBatchGetLaunch(const KvafExecuteBatchGetKernelArgs* args, aclrtStream stream) {
  if (++launch_attempts == fail_launch) return -2;
  const auto copy = *args;
  const auto entity_bytes = copy.block_count * copy.jettys_per_core * sizeof(KvafBatchGetDirectUrmaEntity);
  const auto allocation = entity_allocations.upper_bound(copy.channel_entity);
  Check(allocation != entity_allocations.begin(), "entity belongs to an output binding");
  const auto& range = *std::prev(allocation);
  Check(copy.channel_entity >= range.first && copy.channel_entity - range.first <= range.second &&
        entity_bytes <= range.second - (copy.channel_entity - range.first), "entity slice within its transport");
  const auto* entity = reinterpret_cast<const KvafBatchGetDirectUrmaEntity*>(copy.channel_entity);
  Check(entity->local_hbm_address == copy.output, "per-transport output binding matches kernel");
  CheckAllocation(copy.queue_state, copy.block_count * copy.jettys_per_core * sizeof(KvafBatchGetQueueState));
  CheckAllocation(copy.core_status, copy.block_count * sizeof(KvafBatchGetCoreStatus));
  auto task = Enqueue(stream, [copy] {
    // Model state writes only; data transfer/shard correctness is exercised
    // by the real NPU integration test, not duplicated in this mock.
    auto* queues = reinterpret_cast<KvafBatchGetQueueState*>(copy.queue_state);
    for (unsigned i = 0; i < copy.block_count * copy.jettys_per_core; ++i) queues[i].sq_producer_index += 17;
    auto* statuses = reinterpret_cast<KvafBatchGetCoreStatus*>(copy.core_status);
    for (unsigned i = 0; i < copy.block_count; ++i) {
      statuses[i].magic = KVAF_AIV_BATCH_GET_STATUS_MAGIC;
      statuses[i].profile.kernel_start_cycle = 1;
      statuses[i].profile.setup_done_cycle = 2;
      statuses[i].profile.compact_done_cycle = 2;
      statuses[i].profile.post_done_cycle = 3;
      statuses[i].profile.completion_done_cycle = 4;
      statuses[i].profile.kernel_finish_cycle = 5;
    }
  });
  launches.push_back({copy, stream, task}); return 0;
}

int main() {
  try {
    TestOperator(1, 2, 1);
    TestOperator(2, 1, 1);
    TestOperator(2, 3, 2);
    TestOperator(2, 24, 64);
    TestOperator(2, 1, 1, 2);
    TestOperator(2, 2, 1, 2);
    TestOperator(2, 3, 2, 2);
    TestOperator(2, 24, 64, 2);
    TestFailures();
    Check(live_streams == 0 && live_events == 0 && live_transports == 0 && allocations.empty() && entity_allocations.empty(), "all resources released");
    std::cout << "PASS: BatchGet stream/transport ownership, dependencies, profiles, reuse and failures\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n'; return 1;
  }
}
