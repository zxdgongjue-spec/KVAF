// Real host transport functions with fake RA/ACL handles. No device is used.
#if defined(NDEBUG)
#undef NDEBUG  // Keep failure-injection assertions active in Release builds.
#endif
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(KVAF_TEST_PUT_TEARDOWN)
#include "../access_operator/demo/aiv_batch_put/host/batch_put_host.cpp"
using Transport = KvafBatchPutTransport;
using LocalBuffer = KvafBatchPutLocalBuffer;
#else
#include "../access_operator/demo/aiv_batch_get/host/batch_get_host.cpp"
using Transport = KvafBatchGetTransport;
using LocalBuffer = KvafBatchGetLocalBuffer;
#endif

namespace {
struct Event { std::string name; void* handle; };
std::vector<Event> events;
void* failing = nullptr;
int session_releases = 0;
void* Handle(unsigned value) { return reinterpret_cast<void*>(static_cast<uintptr_t>(value)); }
int Record(const char* name, void* handle) {
  events.push_back({name, handle});
  return handle == failing ? 17 : 0;
}
int Unimport(void*, void* handle) { return Record("unimport", handle); }
int Destroy(void* handle) { return Record("destroy", handle); }
int CqDestroy(void*, void* handle) { return Record("cq", handle); }
int Unregister(void*, void* handle) { return Record("memory", handle); }
int TokenFree(void*, void* handle) { return Record("token", handle); }
int ContextFree(void* handle) { return Record("context", handle); }
int Count(const char* name, void* handle) {
  int count = 0;
  for (const auto& event : events) if (event.name == name && event.handle == handle) ++count;
  return count;
}
int DestroyChecked(Transport* transport) {
#if defined(KVAF_TEST_PUT_TEARDOWN)
  return KvafBatchPutTransportDestroyChecked(transport);
#else
  return KvafBatchGetTransportDestroyChecked(transport);
#endif
}

Transport* Fixture() {
  events.clear(); failing = nullptr; session_releases = 0;
  auto* transport = new Transport;
  transport->api.library = Handle(1);
  transport->api.qp_unimport = Unimport;
  transport->api.qp_destroy = Destroy;
  transport->api.cq_destroy = CqDestroy;
  transport->api.lmem_unregister = Unregister;
  transport->api.rmem_unimport = Unimport;
  transport->api.token_free = TokenFree;
  transport->api.ctx_deinit = ContextFree;
  transport->session = reinterpret_cast<hccp::DirectHccpSession*>(Handle(2));
  transport->device_context = Handle(10);
  transport->device_transport_token_handle = Handle(11);
  transport->device_peer_memory_import = Handle(12);
  transport->worker_count = 1;
  auto& device = transport->workers[0];
  device.device_peer_target_jetty = Handle(20);
  device.device_user_ctl_jetty.handle = Handle(21);
  device.device_user_ctl_cq = Handle(22);
  transport->local_host = std::make_unique<hccp::LocalHostEndpoint>();
  auto& host = *transport->local_host;
  host.context = Handle(30); host.token_handle = Handle(31); host.memory.handle = Handle(32);
  host.workers[0].device_target = Handle(40);
  host.workers[0].jetty.handle = Handle(41); host.workers[0].cq = Handle(42);
  auto local = std::make_unique<LocalBuffer>();
  local->transport = transport;
  local->memory.handle = Handle(50); local->token_handle = Handle(51);
  local->device_entities = std::malloc(64);
  local->device_channel_entity = reinterpret_cast<uint64_t>(local->device_entities);
  transport->local_buffers.push_back(std::move(local));
  return transport;
}

void TestFailurePreservesDependencies(unsigned target) {
  auto* transport = Fixture();
  failing = Handle(target);
  assert(DestroyChecked(transport) != 0);
  assert(transport->retirement_started && session_releases == 0);
  assert(transport->api.library == Handle(1) && transport->session != nullptr);
  if (target == 50 || target == 51) {
    assert(transport->local_buffers.size() == 1);
    auto& local = *transport->local_buffers.front();
    assert(local.device_entities != nullptr && local.token_handle == Handle(51));
    assert(local.memory.handle == (target == 50 ? Handle(50) : nullptr));
    assert(Count("context", Handle(10)) == 0 && Count("memory", Handle(32)) == 0);
  }
  if (target == 10) {
    assert(transport->device_transport_token_handle == nullptr);
    assert(transport->device_context == Handle(10));
    assert(Count("token", Handle(11)) == 1);
  }
  if (target == 11) assert(Count("context", Handle(10)) == 0);
  failing = nullptr;
  assert(DestroyChecked(transport) == 0);
  assert(session_releases == 1);
  assert(Count(target == 10 ? "context" : target == 50 || target == 32 ? "memory" :
               target == 20 || target == 12 ? "unimport" : "token", Handle(target)) == 2);
  if (target == 10) assert(Count("token", Handle(11)) == 1);
}

void TestCleanupOrder() {
  auto* transport = Fixture();
  void* entity = transport->local_buffers.front()->device_entities;
  assert(DestroyChecked(transport) == 0);
  assert(session_releases == 1);
  const auto position = [](const char* name, void* handle) {
    for (std::size_t index = 0; index < events.size(); ++index)
      if (events[index].name == name && events[index].handle == handle) return index;
    std::abort();
  };
  assert(position("unimport", Handle(20)) < position("destroy", Handle(21)));
  assert(position("destroy", Handle(21)) < position("cq", Handle(22)));
  assert(position("cq", Handle(22)) < position("memory", Handle(50)));
  assert(position("memory", Handle(50)) < position("token", Handle(51)));
  assert(position("token", Handle(51)) < position("free", entity));
  assert(position("memory", Handle(32)) < position("context", Handle(30)));
  assert(position("context", Handle(30)) < position("token", Handle(11)));
  assert(position("token", Handle(11)) < position("context", Handle(10)));
}

void TestDeviceViewFailure(unsigned target) {
  auto* owner = Fixture();
  hccp::DeviceAivTransportResources view{
      &owner->api, owner->session, owner->device_context,
      owner->device_transport_token_handle, owner->device_peer_memory_import,
      owner->workers.data(), owner->worker_count};
  failing = Handle(target);
  assert(!hccp::ReleaseDeviceAivTransport(view));
  assert(view.session != nullptr && owner->api.library != nullptr && session_releases == 0);
  if (target == 10) assert(view.device_transport_token_handle == nullptr);
  if (target == 11) assert(Count("context", Handle(10)) == 0);
  failing = nullptr;
  assert(hccp::ReleaseDeviceAivTransport(view));
  assert(view.session == nullptr && view.device_context == nullptr &&
         view.device_transport_token_handle == nullptr && view.device_peer_memory_import == nullptr);
  assert(session_releases == 1 && owner->api.library == nullptr);
  const auto count = events.size();
  assert(hccp::ReleaseDeviceAivTransport(view) && events.size() == count && session_releases == 1);
  // The standalone view test did not own the fixture's other resources.
  std::free(owner->local_buffers.front()->device_entities);
  delete owner;
}

void TestEntityFreeFailure() {
  auto* owner = Fixture();
  void* entity = owner->local_buffers.front()->device_entities;
  failing = entity;
  assert(DestroyChecked(owner) != 0 && session_releases == 0);
  assert(owner->local_buffers.front()->memory.handle == nullptr &&
         owner->local_buffers.front()->token_handle == nullptr &&
         owner->local_buffers.front()->device_entities == entity);
  failing = nullptr;
  assert(DestroyChecked(owner) == 0 && session_releases == 1);
  assert(Count("memory", Handle(50)) == 1 && Count("token", Handle(51)) == 1 && Count("free", entity) == 2);
}

int AllocateToken(void*, hccp::TokenId* id, void** handle) {
  id->token_id = 0x3300; *handle = Handle(51); return 0;
}
int FailRegister(void*, hccp::MrRegInfo*, void** handle) {
  *handle = Handle(50); return 29;
}
void TestPartialAcquireRollbackFailure() {
  auto* owner = Fixture();
  assert(ReleaseLocalBufferResources(owner, owner->local_buffers.front().get()));
  owner->local_buffers.clear(); events.clear();
  owner->api.token_alloc = AllocateToken; owner->api.lmem_register = FailRegister;
  owner->device_transport_token_value = 1;
  std::vector<std::uint8_t> buffer(128);
  failing = Handle(50);
  LocalBuffer* local = nullptr;
#if defined(KVAF_TEST_PUT_TEARDOWN)
  assert(KvafBatchPutLocalBufferAcquire(owner, buffer.data(), buffer.size(), &local) != 0);
#else
  uint64_t entity = 0;
  assert(KvafBatchGetLocalBufferAcquire(owner, buffer.data(), buffer.size(), &local, &entity) != 0);
#endif
  assert(local == nullptr && owner->retirement_started && owner->local_buffers.size() == 1);
  assert(owner->local_buffers.front()->memory.handle == Handle(50) &&
         owner->local_buffers.front()->token_handle == Handle(51));
  const auto count = events.size();
#if defined(KVAF_TEST_PUT_TEARDOWN)
  assert(KvafBatchPutLocalBufferAcquire(owner, buffer.data(), buffer.size(), &local) != 0);
#else
  assert(KvafBatchGetLocalBufferAcquire(owner, buffer.data(), buffer.size(), &local, &entity) != 0);
#endif
  assert(events.size() == count);  // Failed rollback cannot be reused as a cache hit.
  assert(DestroyChecked(owner) != 0 && session_releases == 0);
  failing = nullptr;
  assert(DestroyChecked(owner) == 0 && session_releases == 1);
}
}  // namespace

extern "C" {
aclError aclrtMalloc(void** pointer, std::size_t bytes, aclrtMemMallocPolicy) {
  *pointer = std::malloc(bytes); return *pointer ? ACL_SUCCESS : 1;
}
aclError aclrtFree(void* pointer) {
  const int result = Record("free", pointer);
  if (result == 0) std::free(pointer);
  return result;
}
aclError aclrtMemset(void* pointer, std::size_t capacity, int32_t value, std::size_t bytes) {
  if (bytes > capacity) return 1;
  std::memset(pointer, value, bytes); return ACL_SUCCESS;
}
aclError aclrtMemcpy(void* destination, std::size_t capacity, const void* source,
                     std::size_t bytes, aclrtMemcpyKind) {
  if (!destination || !source || bytes > capacity) return 1;
  std::memcpy(destination, source, bytes); return ACL_SUCCESS;
}
aclError aclrtSynchronizeStream(aclrtStream) { return ACL_SUCCESS; }
aclError aclrtGetDevice(int32_t* device) { *device = 0; return ACL_SUCCESS; }
#if defined(KVAF_TEST_PUT_TEARDOWN)
int32_t KvafExecuteBatchPutLaunchInternal(const KvafExecuteBatchPutLaunchArgs*, aclrtStream) { return 0; }
#endif
}
namespace kvaf::direct_hccp {
bool AcquireDirectHccpSession(uint32_t, uint32_t, DirectHccpSession**) { return false; }
void ReleaseDirectHccpSession(DirectHccpSession*) { ++session_releases; }
bool VerboseLoggingEnabled() { return false; }
void VerboseLog(const char*, ...) {}
}

int main() {
  TestCleanupOrder();
  for (unsigned target : {10, 11, 12, 20, 31, 32, 50, 51}) TestFailurePreservesDependencies(target);
  for (unsigned target : {10, 11, 12, 20, 21, 22}) TestDeviceViewFailure(target);
  TestEntityFreeFailure();
  TestPartialAcquireRollbackFailure();
  std::cout << "real host transport checked teardown passed (CPU RA/ACL failure injection)\n";
}
