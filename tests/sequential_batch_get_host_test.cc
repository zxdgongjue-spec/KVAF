// Exercise the real Host owner with deferred ACL work and transport doubles.
#include "scatter_batch_get.h"
#include "gather_batch_get.h"
#include "kvaf_aiv_batch_get.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include "peer_link.h"
#include "cpu_push_get_layout.h"
#include "scatter_host_profile.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <vector>
#ifdef _WIN32
#include <malloc.h>
#endif

namespace {
std::vector<std::function<void()>> work;
int allocations{}, refs{}, launches{}, completed{}, stopped{}, quiesced{}, hccp_refs{}, push_stopped{};
bool fail_sync{}, fail_copy{}, fail_quiesce{}, fail_enable{}, capturing{};
bool wrong_phase{}, wrong_sequence{}, wrong_magic{};
bool stale_profile{}, missing_cpu_profile{}, bad_profile_cycle{};
bool skip_last_status{};
void* mailbox_jetty{};
bool fail_mailbox_destroy{}, fail_mailbox_import{};
int async_memsets{}, async_copies{}, synchronizations{};
int capture_queries{};
uint64_t kernel_result{};
std::vector<uint64_t> sequences;
std::vector<uint32_t> launch_options;
}
int aclrtMalloc(void** p, size_t n, int) {
#ifdef _WIN32
  *p = _aligned_malloc(n, 64); const int rc = *p ? 0 : 1;
#else
  const int rc = posix_memalign(p, 64, n);
#endif
  if (!rc) ++allocations; return rc;
}
int aclrtMallocHost(void** p, size_t n) { return aclrtMalloc(p, n, 0); }
int aclrtFree(void* p) {
  if (p) {
    --allocations;
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
  return 0;
}
int aclrtFreeHost(void* p) { return aclrtFree(p); }
int aclrtMemset(void* p, size_t, int32_t v, size_t n) { std::memset(p, v, n); return 0; }
int aclrtMemsetAsync(void* p, size_t, int32_t v, size_t n, aclrtStream) {
  ++async_memsets;
  work.push_back([=] { std::memset(p, v, n); }); return 0;
}
int aclrtMemcpyAsync(void* d, size_t, const void* s, size_t n, aclrtMemcpyKind, aclrtStream) {
  ++async_copies;
  if (fail_copy) return 507035;
  work.push_back([=] { std::memcpy(d, s, n); }); return 0;
}
int aclrtMemcpy(void* d, size_t, const void* s, size_t n, aclrtMemcpyKind) {
  if (fail_copy) return 507035;
  std::memcpy(d, s, n); return 0;
}
int aclrtSynchronizeStream(aclrtStream) {
  ++synchronizations;
  if (fail_sync) return 507034;
  for (auto& f : work) f(); work.clear(); return 0;
}
int aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus* s, aclmdlRI*) {
  ++capture_queries;
  *s = capturing ? ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE : ACL_MODEL_RI_CAPTURE_STATUS_NONE; return 0;
}
uint32_t loaded_mode{};
extern "C" int32_t KvafScatterBatchGetLoadKernel(const char*) { loaded_mode = KVAF_SEQ_SCATTER; return 0; }
extern "C" int32_t KvafGatherBatchGetLoadKernel(const char*) { loaded_mode = KVAF_SEQ_GATHER; return 0; }
int Launch(const KvafScatterBatchGetArgs* a) {
  // The previous request must finish before its workspace is reused.
  assert(completed == launches);
  const auto args = *a;
  ++launches; sequences.push_back(a->sequence);
  assert(kvaf::ScatterOptionsValid(args.reserved));
  launch_options.push_back(args.reserved);
  work.push_back([=] {
    auto* status = reinterpret_cast<KvafSequentialStatus*>(args.request.core_status);
    const auto active = uint64_t(*reinterpret_cast<const int32_t*>(args.request.miss_count));
    for (uint32_t i = 0; i < args.request.block_count; ++i) {
      if (skip_last_status && i + 1 == args.request.block_count) continue;
      const auto requests = active * (i + 1) / args.request.block_count -
                            active * i / args.request.block_count;
      status[i] = {wrong_magic ? 0 : KVAF_SEQ_STATUS_MAGIC,
                   args.sequence + wrong_sequence, kernel_result,
                   kernel_result || wrong_phase ? uint64_t(KVAF_SEQ_WAIT_REMOTE) : uint64_t(KVAF_SEQ_COMPLETE),
                   requests, 0, 0, 0};
      if (kvaf::ScatterProfileEnabled(args.reserved)) {
        auto* profiles = reinterpret_cast<kvaf::ScatterBatchGetCoreProfile*>(
            args.request.core_status + args.request.block_count * sizeof(KvafSequentialStatus));
        auto& p = profiles[i];
        p = {}; p.magic = kvaf::kScatterProfileMagic; p.sequence = args.sequence + stale_profile;
        p.start_cycle = 10; p.inputs_ready_cycle = 20;
        p.notify_cycle = i || !active ? 0 : 30;
        p.done_cycle = bad_profile_cycle ? 15 : 40; p.finish_cycle = 50;
        p.requests = requests; p.bytes = requests * args.request.token_bytes;
        p.pipeline = kvaf::ScatterOptionEnabled(args.reserved, kvaf::kScatterPipeline);
        if (requests) {
          p.scatter_start_cycle = p.pipeline ? 35 : 40; p.scatter_finish_cycle = 45;
          p.scatter_cycles = p.scatter_finish_cycle - p.scatter_start_cycle; p.scatter_chunks = 1;
        }
        if (!i && active && !missing_cpu_profile) {
          p.cpu_magic = kvaf::cpu_push::ProfileToken(args.sequence);
          p.cpu_write_ns = 1000; p.cpu_comm_ns = 800;
        }
      }
    }
    ++completed;
  }); return 0;
}
extern "C" int32_t KvafScatterBatchGetLaunch(const KvafScatterBatchGetArgs* a, void*) {
  assert(loaded_mode == KVAF_SEQ_SCATTER);
  kvaf::cpu_push::Layout layout;
  assert(kvaf::cpu_push::MakeLayout(a->request.token_bytes, a->request.block_count, &layout));
  assert(a->descriptor_offset == layout.descriptor && a->staging_offset == layout.staging);
  assert(a->workspace_bytes == layout.bytes + sizeof(KvafScatterMailboxContext) +
         uint64_t(a->request.block_count + 1) * 64);
  assert(a->request.channel_entity == a->workspace + layout.bytes);
  assert(a->request.queue_state == a->request.channel_entity + offsetof(KvafScatterMailboxContext, queue));
  const auto& mailbox = *reinterpret_cast<const KvafScatterMailboxContext*>(a->request.channel_entity);
  assert(mailbox.entity.local_hbm_address == a->workspace && mailbox.entity.local_token_id == 7);
  assert(mailbox.entity.remote_token_id == 8 && mailbox.max_write_bytes == 65536);
  for (const auto& words : mailbox.write_templates) {
    assert(words[1] == 3U << 8 && words[3] == 8 && words[13] == 7);
    assert(words[2] == (1U << 24 | mailbox.entity.tp_id));
  }
  assert((a->workspace + a->workspace_bytes) % 64 == 0);
  return Launch(a);
}
extern "C" int32_t KvafGatherBatchGetLaunch(const KvafGatherBatchGetArgs* a, void*) {
  assert(loaded_mode == KVAF_SEQ_GATHER);
  KvafScatterBatchGetArgs model{};
  model.request = a->request; model.sequence = a->sequence;
  return Launch(&model);
}
struct KvafBatchGetTransport {};
struct KvafBatchGetLocalBuffer {};
extern "C" int32_t KvafBatchGetTransportCreateWithPeerLink(
    void*, uint64_t, uint32_t, uint32_t, void*, const uint8_t*, KvafBatchGetTransport** t) {
  *t = new KvafBatchGetTransport; return 0;
}
extern "C" int32_t KvafBatchGetLocalBufferAcquire(KvafBatchGetTransport*, void*, uint64_t,
    KvafBatchGetLocalBuffer** b, uint64_t* e) {
  ++refs; *b = new KvafBatchGetLocalBuffer; *e = 1; return 0;
}
extern "C" void KvafBatchGetLocalBufferRelease(KvafBatchGetLocalBuffer* b) { --refs; delete b; }
extern "C" int32_t KvafBatchGetTransportQuiesce(KvafBatchGetTransport*) {
  assert(work.empty()); if (fail_quiesce) return 1; ++quiesced; return 0;
}
extern "C" void KvafBatchGetTransportDestroy(KvafBatchGetTransport* t) { delete t; }

namespace kvaf::peer {
PeerLink::PeerLink(std::string a) : address_(std::move(a)) {}
Status PeerLink::LookupRegion(uint64_t base, PeerRegion* r) const {
  *r = {}; r->base = base; r->length = 16 * 1024 * 1024; r->token_value = 3; return Status::OK();
}
Status PeerLink::EnableGatherLanes(uint64_t, uint64_t, uint32_t) const {
  return fail_enable ? Status::Internal("lost enable response") : Status::OK();
}
Status PeerLink::StopGatherLanes(uint64_t) const { assert(work.empty()); ++stopped; return Status::OK(); }
Status PeerLink::OpenCpuPushGet(uint64_t, OpenCpuPushGetRequest, OpenCpuPushGetResponse* r) const {
  r->mailbox_base = 0x1000000; r->mailbox_bytes = 0x100000;
  r->mailbox_segment_bytes = 48; r->mailbox_token_id = 0x800; r->max_write_bytes = 65536;
  return Status::OK();
}
Status PeerLink::StartCpuPushGet(uint64_t) const { return Status::OK(); }
Status PeerLink::StopCpuPushGet(uint64_t) const {
  assert(work.empty());
  if (kernel_result) assert(!mailbox_jetty); // Failed initiator retired before remote MR release.
  ++push_stopped; return Status::OK();
}
}
namespace kvaf::direct_hccp {
void* Hold() { ++hccp_refs; return new int; }
int Release(void* p) { assert(p && hccp_refs); --hccp_refs; delete static_cast<int*>(p); return 0; }
bool LoadDirectHccp(DirectHccpApi* api) {
  api->ctx_init = [](CtxInitCfg*, CtxInitAttr*, void** out) { *out = Hold(); return 0; };
  api->ctx_deinit = Release;
  api->token_alloc = [](void*, TokenId* id, void** out) { id->token_id = 7; *out = Hold(); return 0; };
  api->token_free = [](void*, void* handle) { return Release(handle); };
  api->lmem_register = [](void*, MrRegInfo* mr, void** out) {
    *out = Hold(); mr->out.key.size = 48;
    std::memcpy(mr->out.key.value + 20, &mr->in.mem.addr, 8);
    std::memcpy(mr->out.key.value + 32, &mr->in.mem.size, 8);
    return 0;
  };
  api->lmem_unregister = [](void*, void* handle) { return Release(handle); };
  api->rmem_import = [](void*, MrImportInfo* info, void** out) {
    assert(info->in.ub.flags == 12); // Remote READ/WRITE, matching exported rights.
    *out = Hold(); return fail_mailbox_import ? 1 : 0;
  };
  api->rmem_unimport = [](void*, void* handle) { return Release(handle); };
  api->cq_create = [](void*, CqInfo* info, void** out) {
    *out = Hold(); info->out.buffer_addr = 0x200000; info->out.swdb_addr = 0x200100;
    info->out.cqe_size = 64; return 0;
  };
  api->cq_destroy = [](void*, void* handle) { return Release(handle); };
  api->qp_create = [](void*, QpCreateAttr*, QpCreateInfo* info, void** out) {
    *out = Hold(); info->key.size = 32; return 0;
  };
  api->qp_destroy = [](void* handle) {
    if (handle == mailbox_jetty) {
      if (fail_mailbox_destroy) return 1;
      mailbox_jetty = nullptr;
    }
    return Release(handle);
  };
  api->qp_unimport = [](void*, void* handle) { return Release(handle); };
  return true;
}
void CloseDirectHccp(DirectHccpApi*) {}
bool GetCurrentNpuPhysicalDeviceId(uint32_t* device) { *device = 0; return true; }
bool AcquireDirectHccpSession(uint32_t, uint32_t, DirectHccpSession** out) {
  *out = reinterpret_cast<DirectHccpSession*>(Hold()); return true;
}
void ReleaseDirectHccpSession(DirectHccpSession* session) { Release(session); }
bool GetPeerTp(const DirectHccpApi&, void*, const std::array<uint8_t, kEidBytes>&,
               const std::array<uint8_t, kEidBytes>&, const char*, uint64_t* tp) {
  assert(reinterpret_cast<uintptr_t>(tp) % alignof(uint64_t) == 0);
  *tp = 1; return true;
}
bool CreateDeviceAivJetty(const DirectHccpApi&, void*, void*, void*, uint32_t,
                          uint32_t depth, const char*, QpCreateInfo* info, void** out) {
  assert(depth == 1024);
  *out = mailbox_jetty = Hold(); info->key.size = 32; info->ub.sq_buffer_va = 0x300000;
  info->ub.db_addr = 0x300100; info->ub.wqebb_size = 64; return true;
}
bool ImportPeerJetty(const DirectHccpApi&, void*, const uint8_t*, uint32_t, uint32_t,
                     uint64_t, uint64_t, uint32_t, uint32_t, const char*, QpImportInfoT* info, void** out) {
  *out = Hold(); info->out.ub.tpn = 3; return true;
}
}

int main() {
  using namespace kvaf;
  unsetenv("KVAF_SCATTER_PIPELINE");
  unsetenv("KVAF_SCATTER_HEAD_START");
  AccessOperatorConfig config;
  config.aiv_batch_get_mode = "gather_read"; config.batch_get_token_bytes = 16;
  config.npu_core_count = 3; config.peer_link = std::make_shared<peer::PeerLink>("test");
  RemoteRegionDesc region{0x100000, 1024 * 1024, 16, DeviceKind::kCpu};
  uint64_t address = region.base; uint8_t valid = 1; int32_t index = 0, count = 1;
  uint8_t output[16]{};
  const auto npu = DeviceKind::kNpu;
  BatchGetRequest request;
  request.device_access_table = {{&address, 8, npu}, {&valid, 1, npu}};
  request.indices = request.offsets = request.output_indices = {&index, 4, npu};
  request.miss_count = {&count, 4, npu}; request.output = {output, 16, npu};
  request.output_block_bytes = 16; request.tokens_per_block = 1;
  BatchGetExecRequest exec{{}, &request, {reinterpret_cast<void*>(1), false}};
  {
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).ok());
    assert(owner->Execute(exec).ok()); assert(completed == 0 && refs == 2);
    assert(owner->Execute(exec).ok()); assert(completed == 1 && refs == 2);
    assert(owner->Drain().ok()); assert(completed == 2 && refs == 1);
    capturing = true; assert(!owner->Execute(exec).ok()); capturing = false;
    assert(launches == 2);
    request.output.bytes = 15; assert(!owner->Execute(exec).ok()); request.output.bytes = 16;
    exec.execution.synchronize = true;
    assert(owner->Execute(exec).ok()); assert(completed == 3 && refs == 1);
    exec.execution.synchronize = false;
    assert(owner->Execute(exec).ok());
    assert(owner->Shutdown().ok()); assert(completed == 4 && !refs && !allocations);
    assert((sequences == std::vector<uint64_t>{1, 2, 3, 4}));
  }
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    kernel_result = KVAF_SEQ_CQ_ERROR;
    assert(owner->Execute(exec).ok()); assert(!owner->Drain().ok());
    assert(!owner->Execute(exec).ok()); assert(refs == 2);
    fail_quiesce = true; assert(!owner->Shutdown().ok());
    assert(refs == 2 && allocations == 4);
    fail_quiesce = false; assert(owner->Shutdown().ok());
    assert(!refs && !allocations); kernel_result = 0;
  }
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    assert(owner->Execute(exec).ok()); fail_sync = true;
    const auto failure = owner->Drain();
    assert(failure.code() == StatusCode::kDeviceError);
    assert(failure.message().find("sequence=1 phase=synchronize acl=507034") != std::string::npos);
    const auto cleanup = owner->Shutdown();
    assert(cleanup.message().find("phase=shutdown_synchronize acl=507034") != std::string::npos);
    assert(refs == 2 && allocations == 4);
    fail_sync = false; assert(owner->Shutdown().ok()); assert(!refs && !allocations);
  }
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    assert(owner->Execute(exec).ok()); fail_copy = true;
    const auto failure = owner->Drain();
    assert(failure.code() == StatusCode::kDeviceError);
    assert(failure.message().find("sequence=1 phase=copy_status acl=507035") != std::string::npos);
    assert(refs == 2 && allocations == 4);
    fail_copy = false; assert(owner->Shutdown().ok()); assert(!refs && !allocations);
  }
  {
    auto owner = CreateSequentialBatchGet(config); fail_enable = true;
    assert(!owner->Initialize(region).ok()); fail_enable = false;
    assert(owner->Shutdown().ok()); assert(!refs && !allocations);
  }
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    kernel_result = KVAF_SEQ_TIMEOUT;
    assert(owner->Execute(exec).ok());
    // Closing an unobserved failed async request must surface its status.
    assert(!owner->Shutdown().ok()); assert(!refs && !allocations);
    assert(owner->Shutdown().ok()); kernel_result = 0;
  }
  assert(stopped == 6 && quiesced == 6);
  config.aiv_batch_get_mode = "cpu_push_scatter";
  config.aiv_gather_sgl_size = 0; // A gather-only option must not affect scatter.
  {
    auto owner = CreateSequentialBatchGet(config);
    fail_mailbox_import = true;
    assert(!owner->Initialize(region).ok());
    fail_mailbox_import = false;
    assert(owner->Shutdown().ok());
    assert(!allocations && !hccp_refs && !mailbox_jetty);
    assert(owner->Initialize(region).ok());
    kernel_result = KVAF_SEQ_CQ_ERROR;
    exec.execution.synchronize = true;
    assert(!owner->Execute(exec).ok());
    const auto stops_before = push_stopped;
    fail_mailbox_destroy = true;
    assert(!owner->Shutdown().ok());
    assert(push_stopped == stops_before && mailbox_jetty && hccp_refs);
    fail_mailbox_destroy = false;
    assert(owner->Shutdown().ok());
    assert(!allocations && !hccp_refs && !mailbox_jetty);
    kernel_result = 0;
  }
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    const auto before = launches;
    const auto memsets_before = async_memsets;
    exec.execution.synchronize = false;
    assert(owner->Execute(exec).ok()); assert(completed == before);
    assert(owner->Execute(exec).ok()); assert(completed == before + 1);
    exec.execution.synchronize = true;
    assert(owner->Execute(exec).ok()); assert(completed == before + 3);
    assert(async_memsets == memsets_before); // No extra device task per call.
    assert(launch_options.back() == (kScatterOptionsV1 | kScatterPipeline));
    capturing = true; assert(!owner->Execute(exec).ok()); capturing = false;
    count = 0; assert(owner->Execute(exec).ok()); count = 1;
    assert(owner->Shutdown().ok()); assert(owner->Shutdown().ok());
    assert(!refs && !allocations && !hccp_refs);
  }
  for (bool previous_success : {false, true}) {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    if (previous_success) assert(owner->Execute(exec).ok());
    // A missing core is rejected both on the first request and when its old
    // successful record is still in HBM. No per-call clear is required.
    skip_last_status = true;
    assert(!owner->Execute(exec).ok());
    skip_last_status = false;
    assert(owner->Shutdown().ok());
    assert(!refs && !allocations && !hccp_refs);
  }
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    scatter_host::Profile profile;
    const auto copies_before = async_copies, syncs_before = synchronizations;
    {
      scatter_host::Session session(&profile);
      assert(owner->Execute(exec).ok());
    }
    assert(scatter_host::current == nullptr);
    assert(profile.ns[scatter_host::kOperator] > 0);
    assert(profile.ns[scatter_host::kOperator] >= profile.ns[scatter_host::kStreamSync]);
    assert(async_copies == copies_before + 1 && synchronizations == syncs_before + 1);
    assert(owner->Shutdown().ok());
  }
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    auto trusted = exec;
    trusted.execution.trusted_npu_request = true;
    trusted.execution.synchronize = true;
    scatter_host::Profile profile;
    const auto captures_before = capture_queries;
    const auto copies_before = async_copies, syncs_before = synchronizations;
    {
      scatter_host::Session session(&profile);
      assert(owner->Execute(trusted).ok());
    }
    assert(ScatterOptionEnabled(launch_options.back(), kScatterTrustedInputs));
    assert(capture_queries == captures_before);
    assert(profile.ns[scatter_host::kCaptureQuery] == 0);
    assert(async_copies == copies_before + 1 && synchronizations == syncs_before + 1);
    // Trusted input bypasses only preflight, not the remote execution result.
    kernel_result = KVAF_SEQ_TIMEOUT;
    assert(!owner->Execute(trusted).ok());
    kernel_result = KVAF_SEQ_OK;
    assert(owner->Shutdown().ok());
    assert(!refs && !allocations && !hccp_refs);
  }
  for (uint32_t mismatch = 0; mismatch < 4; ++mismatch) {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    kernel_result = mismatch == 0 ? KVAF_SEQ_TIMEOUT : KVAF_SEQ_OK;
    wrong_phase = mismatch == 1; wrong_sequence = mismatch == 2; wrong_magic = mismatch == 3;
    const auto result = owner->Execute(exec);
    assert(result.code() == StatusCode::kInternal);
    assert(!owner->Execute(exec).ok());
    assert(owner->Shutdown().ok()); assert(!refs && !allocations && !hccp_refs);
    kernel_result = 0; wrong_phase = wrong_sequence = wrong_magic = false;
  }
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    fail_sync = true;
    const auto result = owner->Execute(exec);
    assert(result.code() == StatusCode::kDeviceError);
    assert(result.message().find("phase=synchronize acl=507034") != std::string::npos);
    assert(!owner->Shutdown().ok() && allocations && hccp_refs);
    fail_sync = false; assert(owner->Shutdown().ok());
    assert(!refs && !allocations && !hccp_refs);
  }
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    fail_copy = true;
    const auto result = owner->Execute(exec);
    assert(result.code() == StatusCode::kDeviceError);
    assert(result.message().find("phase=copy_status acl=507035") != std::string::npos);
    assert(allocations && hccp_refs);
    fail_copy = false; assert(owner->Shutdown().ok());
    assert(!refs && !allocations && !hccp_refs);
  }
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    exec.execution.synchronize = false; kernel_result = KVAF_SEQ_TIMEOUT;
    assert(owner->Execute(exec).ok());
    assert(!owner->Shutdown().ok()); assert(!refs && !allocations && !hccp_refs);
    assert(owner->Shutdown().ok()); kernel_result = 0;
  }
  {
    auto owner = CreateSequentialBatchGet(config);
    assert(!owner->SetScatterProfiling(true).ok());
    assert(owner->Initialize(region).ok());
    assert(owner->SetScatterProfiling(true).ok());
    std::vector<ScatterBatchGetCoreProfile> profiles;
    assert(!owner->CopyScatterProfiles(&profiles).ok());
    // CopyProfile must drain an asynchronous request before reading its records.
    assert(owner->Execute(exec).ok()); assert(owner->CopyScatterProfiles(&profiles).ok());
    assert(profiles.size() == config.npu_core_count && profiles[0].sequence == 1);
    for (uint32_t mismatch = 0; mismatch < 3; ++mismatch) {
      stale_profile = mismatch == 0; missing_cpu_profile = mismatch == 1; bad_profile_cycle = mismatch == 2;
      assert(owner->Execute(exec).ok()); assert(!owner->CopyScatterProfiles(&profiles).ok());
      stale_profile = missing_cpu_profile = bad_profile_cycle = false;
    }
    count = 0;
    assert(owner->Execute(exec).ok()); assert(owner->CopyScatterProfiles(&profiles).ok());
    count = 1;
    assert(owner->SetScatterProfiling(false).ok());
    assert(!owner->CopyScatterProfiles(&profiles).ok());
    assert(owner->Execute(exec).ok()); assert(owner->Shutdown().ok());
    assert(!refs && !allocations && !hccp_refs);
  }
  // Both algorithm options are independent and fixed at initialization.
  for (int head : {0, 1}) for (int pipeline : {0, 1}) {
    setenv("KVAF_SCATTER_PIPELINE", pipeline ? "1" : "0", 1);
    setenv("KVAF_SCATTER_HEAD_START", head ? "1" : "0", 1);
    const auto expected = kScatterOptionsV1 | (pipeline ? kScatterPipeline : 0U) |
                          (head ? kScatterHeadStart : 0U);
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).ok());
    setenv("KVAF_SCATTER_PIPELINE", pipeline ? "0" : "1", 1);
    std::vector<ScatterBatchGetCoreProfile> profiles;
    assert(owner->Execute(exec).ok()); assert(owner->Drain().ok());
    assert(launch_options.back() == expected);
    assert(owner->SetScatterProfiling(true).ok());
    assert(owner->Execute(exec).ok()); assert(owner->CopyScatterProfiles(&profiles).ok());
    assert(launch_options.back() == (expected | kScatterProfiling));
    assert(owner->SetScatterProfiling(false).ok());
    assert(owner->Execute(exec).ok()); assert(owner->Drain().ok());
    assert(launch_options.back() == expected);
    assert(owner->Shutdown().ok());
    assert(owner->Initialize(region).ok());
    assert(owner->Execute(exec).ok()); assert(owner->Drain().ok());
    assert(launch_options.back() == (expected ^ kScatterPipeline));
    assert(owner->Shutdown().ok());
    assert(!refs && !allocations && !hccp_refs);
  }
  unsetenv("KVAF_SCATTER_PIPELINE");
  unsetenv("KVAF_SCATTER_HEAD_START");
  for (const auto* name : {"KVAF_SCATTER_PIPELINE", "KVAF_SCATTER_HEAD_START"})
  for (const auto* value : {"", "ON", "2"}) {
    setenv(name, value, 1);
    auto owner = CreateSequentialBatchGet(config);
    const auto result = owner->Initialize(region);
    assert(result.code() == StatusCode::kInvalidArgument);
    assert(result.message().find(name) != std::string::npos);
    assert(!refs && !allocations && !hccp_refs);
    assert(owner->Shutdown().ok());
    unsetenv(name);
  }
  std::cout << "sequential Host lifecycle: PASS\n";
}
