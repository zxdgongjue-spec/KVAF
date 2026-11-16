// Compile twice to exercise the real Get and Put teardown implementations.
// Only the external driver/ACL calls are replaced; transports and retries are real.
#ifdef KVAF_TEST_CLOSE_PUT
#include "../access_operator/algorithms/put/urma_aiv/host/batch_put_host.cpp"
#else
#include "../access_operator/algorithms/get/direct/urma_aiv/host/batch_get_host.cpp"
#endif

#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

#define CHECK(c) do { if (!(c)) throw std::runtime_error(std::string("line ") + std::to_string(__LINE__) + ": " #c); } while (0)
namespace {
enum class Kind { Context, ImportedJetty, Jetty, Cq, LocalMr, LocalToken, Entity,
                  QueueState, RemoteMr, TransportToken };
struct Resource { Kind kind; void* context; };
std::map<void*, Resource> live;
std::set<void*> retired;
std::uintptr_t next_handle = 4096;
void* fail_handle{};
int failures{}, released_sessions{}, closed_libraries{};
bool freed_too_soon{}, repeated_release{}, create_failure{}, acquire_failure{};
Kind acquire_failure_kind{};
#ifdef KVAF_TEST_CLOSE_PUT
bool sync_launch_success{}, sync_wait_failure{}, sync_bad_status{};
#endif
void* Add(Kind kind, void* context = nullptr) {
  void* handle;
  if (kind == Kind::Entity || kind == Kind::QueueState) handle = std::malloc(64);
  else handle = reinterpret_cast<void*>(++next_handle);
  CHECK(live.emplace(handle, Resource{kind, context}).second);
  return handle;
}
bool Has(Kind kind) {
  for (const auto& entry : live) if (entry.second.kind == kind) return true;
  return false;
}
int Release(void* handle, Kind expected, void* context = nullptr) {
  const auto found = live.find(handle);
  if (found == live.end()) { repeated_release = true; return -1; }
  CHECK(found->second.kind == expected);
  if (expected != Kind::Context && expected != Kind::Entity && expected != Kind::QueueState)
    CHECK(found->second.context == context || expected == Kind::Jetty);
  if (handle == fail_handle) { ++failures; return -1; }
  if (expected == Kind::Jetty && Has(Kind::ImportedJetty)) freed_too_soon = true;
  if (expected == Kind::Cq && Has(Kind::Jetty)) freed_too_soon = true;
  if ((expected == Kind::LocalMr || expected == Kind::LocalToken || expected == Kind::Entity ||
       expected == Kind::QueueState || expected == Kind::RemoteMr) &&
      (Has(Kind::ImportedJetty) || Has(Kind::Jetty) || Has(Kind::Cq))) freed_too_soon = true;
  if (expected == Kind::Context) {
    for (const auto& entry : live)
      if (entry.second.context == handle) freed_too_soon = true;
  }
  retired.insert(handle);
  live.erase(found);
  if (expected == Kind::Entity || expected == Kind::QueueState) std::free(handle);
  return 0;
}
int UnimportJetty(void* ctx, void* h) { return Release(h, Kind::ImportedJetty, ctx); }
int DestroyJetty(void* h) { return Release(h, Kind::Jetty); }
int DestroyCq(void* ctx, void* h) { return Release(h, Kind::Cq, ctx); }
int Unregister(void* ctx, void* h) { return Release(h, Kind::LocalMr, ctx); }
int FreeToken(void* ctx, void* h) {
  auto found = live.find(h);
  if (found == live.end()) { repeated_release = true; return -1; }
  return Release(h, found->second.kind == Kind::LocalToken ? Kind::LocalToken : Kind::TransportToken, ctx);
}
int UnimportMemory(void* ctx, void* h) { return Release(h, Kind::RemoteMr, ctx); }
int ImportMemory(void* ctx, hccp::MrImportInfo*, void** out) {
  *out = Add(Kind::RemoteMr, ctx);
  return 0;
}
int Deinit(void* ctx) { return Release(ctx, Kind::Context); }
int InitializeContext(hccp::CtxInitCfg*, hccp::CtxInitAttr*, void** out) {
  *out = Add(Kind::Context);
  fail_handle = *out;
  return 0;
}
int FailAllocateToken(void*, hccp::TokenId*, void**) { return -1; }
int AllocateLocalToken(void* ctx, hccp::TokenId* token, void** out) {
  *out = Add(Kind::LocalToken, ctx);
  token->token_id = 256;
  if (acquire_failure_kind == Kind::LocalToken) fail_handle = *out;
  return 0;
}
int RegisterLocal(void* ctx, hccp::MrRegInfo* info, void** out) {
  *out = Add(Kind::LocalMr, ctx);
  info->out.key.size = 1;
  if (acquire_failure_kind == Kind::LocalMr) { fail_handle = *out; return -1; }
  return 0;
}
#ifdef KVAF_TEST_CLOSE_PUT
using Transport = KvafBatchPutTransport;
using Buffer = KvafBatchPutLocalBuffer;
int Close(Transport* t) { return KvafBatchPutTransportClose(t); }
#else
using Transport = KvafBatchGetTransport;
using Buffer = KvafBatchGetLocalBuffer;
int Close(Transport* t) { return KvafBatchGetTransportClose(t); }
#endif
Transport* Make() {
  auto* t = new Transport;
  t->session = reinterpret_cast<hccp::DirectHccpSession*>(1);
  t->worker_count = 2;
  t->api.qp_unimport = UnimportJetty;
  t->api.qp_destroy = DestroyJetty;
  t->api.cq_destroy = DestroyCq;
  t->api.lmem_unregister = Unregister;
  t->api.token_free = FreeToken;
  t->api.rmem_unimport = UnimportMemory;
  t->api.rmem_import = ImportMemory;
  t->api.ctx_deinit = Deinit;
  auto local = std::make_unique<Buffer>();
  local->transport = t;
  local->reference_count = 1; // invocation still holds this opaque buffer
  local->device_entities = Add(Kind::Entity);
#ifdef KVAF_TEST_CLOSE_PUT
  t->device_context = Add(Kind::Context);
  auto* ctx = t->device_context;
  kvaf::UrmaRemoteMemoryDesc remote;
  remote.registered_address = 4096; remote.registered_bytes = 4096;
  remote.token_value = 1; remote.key_bytes = 1;
  CHECK(t->bindings.Acquire(kvaf::UrmaBindingKey::Make(kvaf::MemoryDirection::Put,
      ctx, t->context_generation, remote, hccp::MemImportFlags()),
      [&](auto& binding) { return binding.Import(ctx, ImportMemory, UnimportMemory, remote); },
      &t->peer_binding).ok());
  t->device_transport_token_handle = Add(Kind::TransportToken, ctx);
  local->memory.handle = Add(Kind::LocalMr, ctx);
  local->token_handle = Add(Kind::LocalToken, ctx);
  t->device_queue_state = Add(Kind::QueueState);
#else
  t->paths.resize(2);
  local->path_registrations.resize(2);
  for (unsigned i = 0; i < 2; ++i) {
    auto& p = t->paths[i];
    p.device_context = Add(Kind::Context);
    kvaf::UrmaRemoteMemoryDesc remote;
    remote.registered_address = 4096; remote.registered_bytes = 4096;
    remote.token_value = 1; remote.key_bytes = 1;
    CHECK(t->bindings.Acquire(kvaf::UrmaBindingKey::Make(kvaf::MemoryDirection::Get,
        p.device_context, p.context_generation, remote, hccp::MemImportFlags()),
        [&](auto& binding) { return binding.Import(p.device_context, ImportMemory, UnimportMemory, remote); },
        &p.peer_binding).ok());
    p.device_transport_token_handle = Add(Kind::TransportToken, p.device_context);
    local->path_registrations[i].memory.handle = Add(Kind::LocalMr, p.device_context);
    local->path_registrations[i].token_handle = Add(Kind::LocalToken, p.device_context);
  }
#endif
  t->local_buffers.push_back(std::move(local));
  for (unsigned i = 0; i < 2; ++i) {
#ifdef KVAF_TEST_CLOSE_PUT
    auto* ctx = t->device_context;
#else
    auto* ctx = t->paths[i].device_context;
    t->workers[i].path_index = i;
#endif
    t->workers[i].device_peer_target_jetty = Add(Kind::ImportedJetty, ctx);
    t->workers[i].device_user_ctl_jetty.handle = Add(Kind::Jetty, ctx);
    t->workers[i].device_user_ctl_cq = Add(Kind::Cq, ctx);
  }
  return t;
}
void Scenario(Kind failure) {
  CHECK(live.empty());
  retired.clear(); failures = released_sessions = closed_libraries = 0;
  freed_too_soon = repeated_release = false;
  auto* transport = Make();
  auto* borrowed_buffer = transport->local_buffers.front().get();
  fail_handle = nullptr;
  for (const auto& entry : live) if (entry.second.kind == failure) fail_handle = entry.first;
  CHECK(fail_handle);
  CHECK(Close(transport) != 0 && failures == 1);
  CHECK(transport->closing && !live.empty());
  CHECK(released_sessions == 0 && closed_libraries == 0);
  // Even failure late in path teardown leaves opaque buffers valid for their
  // context holder; only the underlying confirmed-released handles are cleared.
  CHECK(transport->local_buffers.front().get() == borrowed_buffer);
  CHECK(borrowed_buffer->transport == transport);
  Buffer* ignored{};
#ifdef KVAF_TEST_CLOSE_PUT
  CHECK(KvafBatchPutLocalBufferAcquire(transport, reinterpret_cast<void*>(1), 64, &ignored) != 0);
#else
  std::uint64_t entity{};
  CHECK(KvafBatchGetLocalBufferAcquire(transport, reinterpret_cast<void*>(1), 64, &ignored, &entity) != 0);
#endif
  CHECK(Close(transport) != 0 && failures == 2); // repeat identical failure safely
  CHECK(!freed_too_soon && !repeated_release);
  fail_handle = nullptr;
  CHECK(Close(transport) == 0);
  CHECK(live.empty() && released_sessions == 1 && closed_libraries == 1);
  CHECK(!freed_too_soon && !repeated_release);
}
void PartialBufferFailure(Kind failure) {
  CHECK(live.empty());
  acquire_failure = true;
  acquire_failure_kind = failure;
  fail_handle = nullptr;
  failures = released_sessions = closed_libraries = 0;
  auto* transport = new Transport;
  transport->session = reinterpret_cast<hccp::DirectHccpSession*>(1);
  transport->worker_count = 1;
  transport->api.ctx_deinit = Deinit;
  transport->api.token_alloc = AllocateLocalToken;
  transport->api.token_free = FreeToken;
  transport->api.lmem_register = RegisterLocal;
  transport->api.lmem_unregister = Unregister;
#ifdef KVAF_TEST_CLOSE_PUT
  transport->device_context = Add(Kind::Context);
#else
  transport->paths.resize(1);
  transport->paths[0].device_context = Add(Kind::Context);
  transport->peer_memory.token_value = 1;
#endif
  Buffer* output = nullptr;
#ifdef KVAF_TEST_CLOSE_PUT
  auto result = KvafBatchPutLocalBufferAcquire(transport, reinterpret_cast<void*>(4096), 4096, &output);
#else
  std::uint64_t entity = 0;
  auto result = KvafBatchGetLocalBufferAcquire(transport, reinterpret_cast<void*>(4096), 4096, &output, &entity);
  CHECK(entity == 0);
#endif
  CHECK(result != 0 && output && !transport->closing && failures == 1);
  CHECK(transport->local_buffers.size() == 1 && !transport->local_buffers.front()->ready);
  CHECK(output->reference_count == 1);
#ifdef KVAF_TEST_CLOSE_PUT
  CHECK(KvafBatchPutLocalBufferRetire(output) != 0 && failures == 2);
#else
  CHECK(KvafBatchGetLocalBufferRetire(output) != 0 && failures == 2);
#endif
  CHECK(released_sessions == 0 && closed_libraries == 0);
  fail_handle = nullptr;
#ifdef KVAF_TEST_CLOSE_PUT
  CHECK(KvafBatchPutLocalBufferRetire(output) == 0);
#else
  CHECK(KvafBatchGetLocalBufferRetire(output) == 0);
#endif
  CHECK(transport->local_buffers.empty());
  CHECK(Close(transport) == 0);
  CHECK(live.empty() && released_sessions == 1 && closed_libraries == 1);
  CHECK(!freed_too_soon && !repeated_release);
  acquire_failure = false;
}
void PartialCreationFailure() {
  CHECK(live.empty());
  create_failure = true;
  failures = released_sessions = closed_libraries = 0;
  std::uint8_t eid[16]{1};
  Transport* transport = nullptr;
#ifdef KVAF_TEST_CLOSE_PUT
  auto result = KvafBatchPutTransportCreateWithPeerLink(reinterpret_cast<void*>(4096), 4096,
      1, 1, reinterpret_cast<void*>(1), eid, &transport);
#else
  auto result = KvafBatchGetTransportCreateWithPeerLink(reinterpret_cast<void*>(4096), 4096,
      1, 1, reinterpret_cast<void*>(1), eid, &transport);
#endif
  CHECK(result != 0 && transport && transport->closing && failures == 1);
  CHECK(live.size() == 1 && released_sessions == 0 && closed_libraries == 0);
  fail_handle = nullptr;
  CHECK(Close(transport) == 0);
  CHECK(live.empty() && released_sessions == 1 && closed_libraries == 1);
  create_failure = false;
}
}  // namespace

// Setup functions are linked but deliberately unused by teardown-only tests.
namespace kvaf::direct_hccp {
bool GetCurrentNpuPhysicalDeviceId(std::uint32_t* device) { *device = 0; return create_failure; }
bool LoadPeerMemory(PeerLink*, std::uint64_t, UrmaRemoteMemoryDesc*, std::array<std::uint8_t, kEidBytes>*) { return create_failure; }
bool LoadDirectHccp(DirectHccpApi* api) {
  api->ctx_init = InitializeContext;
  api->ctx_deinit = Deinit;
  api->token_alloc = FailAllocateToken;
  return create_failure;
}
bool AcquireDirectHccpSession(std::uint32_t, std::uint32_t, DirectHccpSession** out) {
  *out = reinterpret_cast<DirectHccpSession*>(1);
  return create_failure;
}
bool CreateDeviceAivJetty(const DirectHccpApi&, void*, void*, void*, std::uint32_t,
                          std::uint32_t, const char*, QpCreateInfo*, void**) { return false; }
bool CreatePeerChannel(const PeerChannelRequest&, PeerChannel*) { return false; }
void VerboseLog(const char*, ...) {}
void ReleaseDirectHccpSession(DirectHccpSession*) {
  CHECK(live.empty()); ++released_sessions;
}
void CloseDirectHccp(DirectHccpApi*) { CHECK(live.empty()); ++closed_libraries; }
}
extern "C" {
aclError aclrtFree(void* p) {
  const auto found = live.find(p);
  if (found == live.end()) { repeated_release = true; return 1; }
  return Release(p, found->second.kind);
}
aclError aclrtMalloc(void** p, size_t n, aclrtMemMallocPolicy) {
  *p = std::malloc(n);
  if (*p && acquire_failure) {
    CHECK(live.emplace(*p, Resource{Kind::Entity, nullptr}).second);
    if (acquire_failure_kind == Kind::Entity) fail_handle = *p;
  }
  return *p ? 0 : 1;
}
aclError aclrtMemcpy(void* p, size_t, const void* source, size_t n, aclrtMemcpyKind) {
  if (acquire_failure) return 1; // fail after registration/entity allocation
  std::memcpy(p, source, n); return 0;
}
#ifdef KVAF_TEST_CLOSE_PUT
aclError aclrtMemset(void* p, size_t, int value, size_t n) { std::memset(p, value, n); return 0; }
aclError aclrtMemsetAsync(void* p, size_t size, int value, size_t n, aclrtStream) { return aclrtMemset(p, size, value, n); }
aclError aclrtMemcpyAsync(void* p, size_t size, const void* s, size_t n, aclrtMemcpyKind kind, aclrtStream) {
  return aclrtMemcpy(p, size, s, n, kind);
}
aclError aclrtSynchronizeStream(aclrtStream) { return sync_wait_failure ? 1 : 0; }
int32_t KvafExecuteBatchPutLaunchInternal(const KvafExecuteBatchPutLaunchArgs* args, aclrtStream) {
  if (!sync_launch_success) return -1;
  auto* report = reinterpret_cast<KvafBatchPutCoreStatus*>(args->request.core_status);
  report->magic = KVAF_AIV_BATCH_PUT_STATUS_MAGIC;
  report->result = sync_bad_status ? 1 : KVAF_BATCH_PUT_CORE_SUCCESS;
  report->processed_request_count = args->request.row_count;
  report->failed_request_index = UINT64_MAX;
  return 0;
}
#endif
}
#ifdef KVAF_TEST_CLOSE_PUT
void SyncTimingAndStatus() {
  KvafBatchPutTransport transport;
  transport.aiv_core_count = 1;
  KvafBatchPutLocalBuffer buffer;
  buffer.transport = &transport; buffer.reference_count = 1; buffer.ready = true;
  buffer.input_hbm_address = 0x1000; buffer.input_hbm_bytes = 64;
  KvafBatchPutCoreStatus device{}, host{};
  KvafExecuteBatchPutKernelArgs args{};
  args.input = buffer.input_hbm_address; args.input_bytes = buffer.input_hbm_bytes;
  args.input_offsets = args.lengths = args.destination_addresses = 1;
  args.core_status = reinterpret_cast<uint64_t>(&device);
  args.row_count = args.aiv_core_count = 1;
  uint64_t submit = 0, completion = 0;
  auto run = [&] {
    return KvafExecuteBatchPutSyncTimed(&buffer, &args, reinterpret_cast<void*>(1),
                                        &host, 1, &submit, &completion);
  };
  device.magic = KVAF_AIV_BATCH_PUT_STATUS_MAGIC;
  CHECK(run() == KVAF_BATCH_PUT_RUNTIME_ERROR);
  CHECK(submit > 0 && completion == 0 && device.magic == 0);
  sync_launch_success = true;
  CHECK(run() == KVAF_BATCH_PUT_SUCCESS && completion > 0);
  CHECK(host.magic == KVAF_AIV_BATCH_PUT_STATUS_MAGIC && host.processed_request_count == 1);
  sync_wait_failure = true;
  CHECK(run() == KVAF_BATCH_PUT_RUNTIME_ERROR);
  sync_wait_failure = false; sync_bad_status = true;
  CHECK(run() == KVAF_BATCH_PUT_KERNEL_ERROR);
  sync_bad_status = false;
  CHECK(KvafExecuteBatchPutSync(&buffer, &args, reinterpret_cast<void*>(1), &host, 1) ==
        KVAF_BATCH_PUT_SUCCESS);
}
#endif
int main() {
  try {
    CHECK(Close(nullptr) == 0);
    for (auto kind : {Kind::ImportedJetty, Kind::Jetty, Kind::Cq, Kind::LocalMr,
                      Kind::LocalToken, Kind::Entity, Kind::RemoteMr,
                      Kind::TransportToken, Kind::Context}) Scenario(kind);
#ifdef KVAF_TEST_CLOSE_PUT
    Scenario(Kind::QueueState);
#endif
    PartialCreationFailure();
    for (auto kind : {Kind::LocalMr, Kind::LocalToken, Kind::Entity}) PartialBufferFailure(kind);
#ifdef KVAF_TEST_CLOSE_PUT
    SyncTimingAndStatus();
#endif
    std::cout << "AIV transport close retains failed handles, retries, and respects dependencies\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
