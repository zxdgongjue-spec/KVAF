// Exercise the real Host owner with deferred ACL work and transport doubles.
#include "scatter_batch_get.h"
#include "gather_batch_get.h"
#include "kvaf_aiv_batch_get.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include "peer_link.h"
#include "cpu_push_get_layout.h"
#include <algorithm>
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
std::function<void()> graph_task;
int allocations{}, refs{}, launches{}, completed{}, stopped{}, quiesced{}, hccp_refs{}, push_stopped{};
bool fail_sync{}, fail_copy{}, fail_quiesce{}, fail_enable{}, capturing{};
void* mailbox_jetty{};
bool fail_mailbox_destroy{}, fail_mailbox_import{};
bool old_probe_kernel{}, wrong_probe_phase{}, legacy_load_kernel{}, zero_probe_requests{};
bool wrong_gather_probe_wrs{};
uint32_t last_coalesce_sges{};
uint32_t last_pipeline_chunk_bytes{}, last_prepare_ahead{}, last_cooperative_chunks{}, last_first_divisor{};
uint32_t capture_queries{};
bool allow_queued_launches{};
int async_memsets{}, status_copies{}, stream_syncs{};
uint64_t kernel_result{};
uint32_t profile_fault{};
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
  ++status_copies;
  if (fail_copy) return 507035;
  work.push_back([=] { std::memcpy(d, s, n); }); return 0;
}
int aclrtMemcpy(void* d, size_t, const void* s, size_t n, aclrtMemcpyKind) {
  if (fail_copy) return 507035;
  std::memcpy(d, s, n); return 0;
}
int aclrtSynchronizeStream(aclrtStream) {
  ++stream_syncs;
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
  assert(allow_queued_launches || completed == launches);
  const auto args = *a;
  ++launches; sequences.push_back(a->sequence);
  assert(kvaf::ScatterOptionsValid(args.reserved));
  launch_options.push_back(args.reserved);
  work.push_back([=] {
    auto* status = reinterpret_cast<KvafSequentialStatus*>(args.request.core_status);
    const auto active = uint64_t(*reinterpret_cast<const int32_t*>(args.request.miss_count));
    for (uint32_t i = 0; i < args.request.block_count; ++i) {
      const auto requests = active * (i + 1) / args.request.block_count -
                            active * i / args.request.block_count;
      status[i] = {KVAF_SEQ_STATUS_MAGIC, args.sequence, kernel_result,
                   kernel_result ? KVAF_SEQ_WAIT_REMOTE : KVAF_SEQ_COMPLETE, requests, 0, 0, 0};
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
  if (!a->sequence) {
    const auto args = *a;
    auto execute = [args] {
      auto* status = reinterpret_cast<KvafSequentialStatus*>(args.request.core_status);
      for (uint32_t i = 0; i < args.request.block_count; ++i) {
        const auto sequence = status[i].sequence + 1;
        status[i] = {KVAF_SEQ_STATUS_MAGIC, sequence, kernel_result,
                     kernel_result ? KVAF_SEQ_WAIT_REMOTE : KVAF_SEQ_COMPLETE, 0, 0, 0, 0};
      }
    };
    if (capturing) graph_task = execute;
    else work.push_back(execute);
    return 0;
  }
  last_coalesce_sges = a->coalesce_sges;
  last_pipeline_chunk_bytes = a->pipeline_chunk_bytes;
  last_prepare_ahead = a->prepare_ahead;
  last_cooperative_chunks = a->cooperative_chunks;
  last_first_divisor = a->first_chunk_divisor;
  if (a->probe) assert(!last_pipeline_chunk_bytes);
  KvafScatterBatchGetArgs model{};
  model.request = a->request; model.sequence = a->sequence;
  const auto rc = Launch(&model);
  if (a->probe && !old_probe_kernel && !kernel_result) {
    const auto args = *a;
    work.push_back([=] {
      auto* statuses = reinterpret_cast<KvafSequentialStatus*>(args.request.core_status);
      for (uint32_t i = 0; i < args.request.block_count; ++i) {
        statuses[i].result = KVAF_SEQ_PROBE_COMPLETE;
        statuses[i].phase = KVAF_GATHER_AFTER_ACK + args.probe - 1 + wrong_probe_phase;
        if (zero_probe_requests) statuses[i].requests = 0;
        statuses[i].data_wrs = args.probe == KVAF_GATHER_PROBE_FIRST_READ ?
            std::min<uint64_t>(KVAF_GATHER_WRS_PER_POST,
                (statuses[i].requests + args.sgl_size - 1) / args.sgl_size) : 0;
        statuses[i].data_wrs += wrong_gather_probe_wrs;
      }
    });
  }
  if (a->profile) {
    const auto args = *a;
    const auto fault = profile_fault;
    work.push_back([=] {
      auto* profiles = reinterpret_cast<KvafGatherBatchGetCoreProfile*>(args.profile);
      const auto* statuses = reinterpret_cast<const KvafSequentialStatus*>(args.request.core_status);
      for (uint32_t i = 0; i < args.request.block_count; ++i) {
        auto& p = profiles[i];
        p = {};
        if (fault == 1) continue; // Old/missing profile producer must fail closed.
        p.magic = KVAF_GATHER_PROFILE_MAGIC;
        p.sequence = args.sequence + (fault == 2);
        p.result = fault == 5 ? KVAF_SEQ_CQ_ERROR : statuses[i].result;
        p.valid_mask = fault == 3 ? 1 : KVAF_GATHER_PROFILE_ALL_EVENTS;
        p.kernel_start_cycle = 100;
        p.input_done_cycle = 101;
        p.descriptor_done_cycle = 102;
        p.notify_done_cycle = 103;
        p.ack_done_cycle = fault == 4 ? 99 : 104;
        p.read_done_cycle = 105;
        p.kernel_finish_cycle = 106;
        p.requests = statuses[i].requests + (fault == 6);
        p.data_wrs = statuses[i].data_wrs + (fault == 7);
        p.prepare_ahead = p.requests ? args.prepare_ahead : 0;
        p.cooperative_chunks = p.requests ? args.cooperative_chunks : 0;
        if (args.pipeline_chunk_bytes && p.requests) {
          p.pipeline_chunk_bytes = args.cooperative_chunks ?
              kvaf::GatherCooperativeChunkBytes(p.requests * args.request.token_bytes,
                  args.cooperative_chunks, args.first_chunk_divisor) : args.pipeline_chunk_bytes;
          p.ack_done_cycle = 106;
          p.first_ready_cycle = 104;
          p.first_read_submit_cycle = 104;
          p.first_read_ready_bytes = p.requests * args.request.token_bytes;
          p.ready_queries = 2;
          p.ready_wait_cycles = 1;
          if (fault == 14) p.pipeline_chunk_bytes = 0;
          if (fault == 15) p.first_ready_cycle = 102;
          if (fault == 16) ++p.first_read_ready_bytes;
          if (fault == 17) p.first_read_submit_cycle = 106;
        }
        if (fault == 8) p.input_done_cycle = 99;
        if (fault == 9) p.descriptor_done_cycle = 100;
        if (fault == 10) p.notify_done_cycle = 101;
        if (fault == 11) p.read_done_cycle = 103;
        if (fault == 12) p.kernel_finish_cycle = 104;
        if (fault == 13) {
          // Device feedback: correct header/counts and late timestamps, but
          // caller-only mask 0x41 and lost early fields. Never accept it.
          p.valid_mask = 0x41;
          p.kernel_start_cycle = p.input_done_cycle = p.descriptor_done_cycle = p.notify_done_cycle = 0;
          p.ack_done_cycle = 97320160017685ULL;
          p.read_done_cycle = 97320162709862ULL;
          p.kernel_finish_cycle = 97320162710121ULL;
        }
      }
    });
  }
  return rc;
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
Status PeerLink::OpenAivCpuPushGet(uint64_t, OpenCpuPushGetRequest, OpenAivCpuPushGetResponse* r) const {
  r->mailbox_base = 0x1000000; r->mailbox_bytes = 0x100000;
  r->mailbox_segment_bytes = 48; r->mailbox_token_id = 0x800; r->max_write_bytes = 65536;
  return Status::OK();
}
Status PeerLink::StartAivCpuPushGet(uint64_t) const { return Status::OK(); }
Status PeerLink::StopAivCpuPushGet(uint64_t) const {
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
  unsetenv("KVAF_AIV_SCATTER_PROBE");
  unsetenv("KVAF_AIV_GATHER_PROBE");
  unsetenv("KVAF_AIV_GATHER_COALESCE_SGES");
  unsetenv("KVAF_AIV_GATHER_GRAPH");
  // Keep strict status-failure coverage, now an explicit async diagnostic mode.
  setenv("KVAF_AIV_GATHER_CHECK_STATUS", "1", 1);
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
    assert(last_coalesce_sges == 0);
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
  {
    setenv("KVAF_AIV_GATHER_PROBE", "typo", 1);
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).code() == StatusCode::kInvalidArgument);
    assert(!allocations && !refs);
  }
  for (const char* probe : {"ack", "wqe", "sq", "first_read"}) {
    setenv("KVAF_AIV_GATHER_PROBE", probe, 1);
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    assert(owner->SetGatherProfiling(true).code() == StatusCode::kInvalidState);
    exec.execution.synchronize = false;
    const auto before = launches;
    assert(owner->Execute(exec).code() == StatusCode::kInvalidArgument && launches == before);
    exec.execution.synchronize = true;
    for (uint32_t i = 0; i < 2; ++i) {
      const auto result = owner->Execute(exec);
      assert(result.code() == StatusCode::kNotImplemented);
      assert(result.message().find(std::string("[KVAF_AIV_GATHER_PROBE_COMPLETE] stage=") + probe + " ") == 0);
      assert(owner->Drain().ok());
    }
    assert(owner->Shutdown().ok() && !refs && !allocations);
  }
  for (uint32_t mismatch = 0; mismatch < 5; ++mismatch) {
    setenv("KVAF_AIV_GATHER_PROBE", "wqe", 1);
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    old_probe_kernel = mismatch == 0; wrong_probe_phase = mismatch == 1;
    zero_probe_requests = mismatch == 2;
    kernel_result = mismatch == 3 ? KVAF_SEQ_TIMEOUT : KVAF_SEQ_OK;
    wrong_gather_probe_wrs = mismatch == 4;
    const auto result = owner->Execute(exec);
    assert(result.code() == StatusCode::kInternal);
    assert(result.message().find("[KVAF_AIV_GATHER_PROBE_COMPLETE]") == std::string::npos);
    assert(owner->Shutdown().ok() && !refs && !allocations);
    old_probe_kernel = wrong_probe_phase = zero_probe_requests = false;
    wrong_gather_probe_wrs = false;
    kernel_result = 0;
  }
  unsetenv("KVAF_AIV_GATHER_PROBE");
  exec.execution.synchronize = false;
  {
    setenv("KVAF_AIV_SCATTER_PROBE", "done", 1);
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).ok());
    assert(owner->Execute(exec).ok());
    assert(owner->Shutdown().ok());
    assert(!allocations && !hccp_refs && !refs);
  }
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
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    exec.execution.synchronize = true;
    kernel_result = KVAF_SEQ_TIMEOUT;
    assert(owner->Execute(exec).code() == StatusCode::kInternal);
    assert(owner->Shutdown().ok());
    assert(!refs && !allocations && !hccp_refs);
    kernel_result = KVAF_SEQ_OK;
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
  assert(!allocations && !refs && !hccp_refs);
  unsetenv("KVAF_AIV_SCATTER_PROBE");
  config.aiv_batch_get_mode = "gather_read";
  for (uint32_t sgl : {0U, 7U}) {
    config.aiv_gather_sgl_size = sgl;
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).code() == StatusCode::kInvalidArgument);
    assert(!allocations && !refs && !hccp_refs);
  }
  config.aiv_gather_sgl_size = 6;
  exec.execution.synchronize = true;
  {
    auto owner = CreateSequentialBatchGet(config);
    GatherBatchGetProfile snapshot;
    assert(owner->SetGatherProfiling(true).code() == StatusCode::kNotInitialized);
    assert(owner->Initialize(region).ok());
    assert(owner->CopyGatherProfile(&snapshot).code() == StatusCode::kInvalidState);
    assert(owner->CopyGatherProfile(nullptr).code() == StatusCode::kInvalidArgument);
    assert(owner->SetGatherProfiling(true).ok() && allocations == 6);
    exec.execution.synchronize = false;
    const auto before = launches;
    assert(owner->Execute(exec).code() == StatusCode::kInvalidArgument && launches == before);
    exec.execution.synchronize = true;
    for (uint64_t seq = 1; seq <= 2; ++seq) {
      assert(owner->Execute(exec).ok());
      assert(owner->CopyGatherProfile(&snapshot).ok());
      assert(snapshot.sequence == seq && snapshot.cores.size() == 3);
      uint64_t rows = 0;
      for (const auto& p : snapshot.cores) {
        assert(p.sequence == seq && p.kernel_finish_cycle - p.kernel_start_cycle == 6);
        rows += p.requests;
      }
      assert(rows == 1);
      assert(snapshot.total_ns >= snapshot.prepare_ns + snapshot.launch_ns +
          snapshot.status_enqueue_ns + snapshot.wait_ns + snapshot.finish_ns);
    }
    assert(owner->SetGatherProfiling(false).ok());
    assert(owner->Execute(exec).ok());
    assert(owner->CopyGatherProfile(&snapshot).code() == StatusCode::kInvalidState);
    assert(snapshot.cores.empty());
    assert(owner->SetGatherProfiling(true).ok());
    count = 0;
    assert(owner->Execute(exec).ok() && owner->CopyGatherProfile(&snapshot).ok());
    for (const auto& p : snapshot.cores) assert(p.requests == 0);
    count = 1;
    assert(owner->Shutdown().ok() && !allocations && !refs);
  }
  const char* profile_reasons[] = {"", "magic", "sequence", "valid_mask", "timestamp_notify_ack",
      "result", "requests", "data_wrs", "timestamp_start_input", "timestamp_input_descriptor",
      "timestamp_descriptor_notify", "timestamp_ack_read", "timestamp_read_finish", "valid_mask"};
  setenv("KVAF_AIV_GATHER_CHUNK_BYTES", "0", 1);
  for (profile_fault = 1; profile_fault <= 13; ++profile_fault) {
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).ok() && owner->SetGatherProfiling(true).ok());
    const auto failure = owner->Execute(exec);
    assert(failure.code() == StatusCode::kInternal);
    assert(failure.message().find(std::string("reason=") + profile_reasons[profile_fault] + " ") != std::string::npos);
    for (const char* field : {"magic=0x", "expected_magic=0x", "observed_sequence=", "result=",
        "valid_mask=0x", "expected_mask=0x7f", "cycles=[start:", ",input:", ",descriptor:",
        ",notify:", ",ack:", ",read:", ",finish:", "requests=", "status_requests=",
        "data_wrs=", "status_data_wrs=", "args_bytes=256", "record_bytes=256"})
      assert(failure.message().find(field) != std::string::npos);
    if (profile_fault == 13) {
      assert(failure.message().find("valid_mask=0x41") != std::string::npos);
      assert(failure.message().find("cycles=[start:0,input:0,descriptor:0,notify:0,ack:97320160017685") != std::string::npos);
    }
    GatherBatchGetProfile snapshot;
    assert(!owner->CopyGatherProfile(&snapshot).ok());
    assert(!owner->Execute(exec).ok());
    assert(owner->Shutdown().ok() && !allocations && !refs);
  }
  unsetenv("KVAF_AIV_GATHER_CHUNK_BYTES");
  profile_fault = 0;
  {
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).ok() && owner->SetGatherProfiling(true).ok());
    fail_sync = true;
    assert(owner->Execute(exec).code() == StatusCode::kDeviceError);
    GatherBatchGetProfile snapshot;
    assert(!owner->CopyGatherProfile(&snapshot).ok());
    assert(owner->SetGatherProfiling(false).ok()); // Cleanup preserves original failure.
    assert(!owner->Shutdown().ok() && allocations == 6);
    fail_sync = false;
    assert(owner->Shutdown().ok() && !allocations && !refs);
  }
  unsetenv("KVAF_AIV_GATHER_CHECK_STATUS");
  allow_queued_launches = true;
  exec.execution.synchronize = false;
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    const auto copies = status_copies, syncs = stream_syncs, clears = async_memsets;
    const auto done = completed;
    fail_copy = true;  // Ordinary async work must never enqueue a status D2H.
    assert(owner->Execute(exec).ok() && refs == 1);
    assert(owner->Execute(exec).ok() && completed == done && refs == 1);
    assert(status_copies == copies && stream_syncs == syncs && async_memsets == clears);
    assert(owner->Drain().ok() && completed == done + 2);
    assert(status_copies == copies && stream_syncs == syncs + 1);
    fail_copy = false;
    exec.execution.synchronize = true;
    assert(owner->Execute(exec).ok());
    assert(status_copies == copies + 1 && async_memsets == clears);
    assert(owner->Shutdown().ok() && !allocations && !refs);
    exec.execution.synchronize = false;
  }
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    const auto copies = status_copies, done = completed;
    assert(owner->Execute(exec).ok());
    exec.execution.stream = reinterpret_cast<void*>(2);
    assert(owner->Execute(exec).ok() && completed == done + 1);
    assert(status_copies == copies);  // Stream switch fences without readback.
    assert(owner->Shutdown().ok() && completed == done + 2 && !allocations && !refs);
    exec.execution.stream = reinterpret_cast<void*>(1);
  }
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    kernel_result = KVAF_SEQ_TIMEOUT;
    assert(owner->Execute(exec).ok());
    // As in direct async mode, stream completion does not inspect device status.
    assert(owner->Shutdown().ok() && !allocations && !refs);
    kernel_result = 0;
  }
  {
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).ok() && owner->SetGatherProfiling(true).ok());
    const auto copies = status_copies, clears = async_memsets;
    exec.execution.synchronize = true;
    assert(owner->Execute(exec).ok());
    GatherBatchGetProfile snapshot;
    assert(owner->CopyGatherProfile(&snapshot).ok());
    assert(status_copies == copies + 2 && async_memsets == clears);
    assert(owner->Shutdown().ok() && !allocations && !refs);
  }
  for (const char* value : {"2", "true"}) {
    setenv("KVAF_AIV_GATHER_COALESCE_SGES", value, 1);
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).code() == StatusCode::kInvalidArgument);
    assert(!allocations && !refs);
  }
  setenv("KVAF_AIV_GATHER_COALESCE_SGES", "1", 1);
  {
    const auto token = config.batch_get_token_bytes;
    config.batch_get_token_bytes = 65535;
    auto owner = CreateSequentialBatchGet(config);
    const auto result = owner->Initialize(region);
    assert(result.code() == StatusCode::kInvalidArgument);
    assert(result.message().find("coalesced gather") != std::string::npos);
    assert(!allocations && !refs);
    config.batch_get_token_bytes = token;
  }
  {
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).ok());
    unsetenv("KVAF_AIV_GATHER_COALESCE_SGES"); // Read once at initialization.
    assert(owner->Execute(exec).ok() && last_coalesce_sges == 1);
    assert(owner->Shutdown().ok() && !allocations && !refs);
  }
  for (const char* chunk : {"0", "64", "131072", "4294967232"}) {
    setenv("KVAF_AIV_GATHER_CHUNK_BYTES", chunk, 1);
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).ok());
    unsetenv("KVAF_AIV_GATHER_CHUNK_BYTES"); // Captured at Runtime initialization.
    assert(owner->Execute(exec).ok());
    assert(last_pipeline_chunk_bytes == std::strtoull(chunk, nullptr, 10));
    assert(owner->Shutdown().ok() && !allocations && !refs);
  }
  for (const char* chunk : {"-64", "+64", "63", "65", "64K", "4294967296", " 64"}) {
    setenv("KVAF_AIV_GATHER_CHUNK_BYTES", chunk, 1);
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).code() == StatusCode::kInvalidArgument);
    assert(!allocations && !refs);
  }
  unsetenv("KVAF_AIV_GATHER_CHUNK_BYTES");
  for (profile_fault = 14; profile_fault <= 17; ++profile_fault) {
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).ok() && owner->SetGatherProfiling(true).ok());
    assert(owner->Execute(exec).code() == StatusCode::kInternal);
    assert(owner->Shutdown().ok() && !allocations && !refs);
  }
  profile_fault = 0;
  // Trusted gather bypasses capture/Input revalidation, while retaining status
  // handling, stream-switch fences and the existing resource lifecycle.
  {
    auto owner = CreateSequentialBatchGet(config); assert(owner->Initialize(region).ok());
    const auto before = capture_queries;
    exec.execution.trusted_npu_request = true;
    exec.execution.synchronize = true;
    assert(owner->Execute(exec).ok() && capture_queries == before);
    exec.execution.trusted_npu_request = false;
    assert(owner->Execute(exec).ok() && capture_queries == before + 1);
    assert(owner->Shutdown().ok() && !refs && !allocations);
  }
  setenv("KVAF_AIV_GATHER_GRAPH", "1", 1);
  setenv("KVAF_AIV_GATHER_CHECK_STATUS", "1", 1);
  {
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).ok());
    assert(!owner->SetGatherProfiling(true).ok());
    exec.execution.trusted_npu_request = true;
    exec.execution.synchronize = false;
    capturing = true;
    assert(!owner->Execute(exec).ok() && !graph_task);
    capturing = false;
    exec.execution.synchronize = true;
    assert(owner->Execute(exec).ok());
    const auto syncs = stream_syncs, copies = status_copies;
    exec.execution.synchronize = false;
    capturing = true;
    exec.execution.stream = reinterpret_cast<void*>(2);
    assert(!owner->Execute(exec).ok() && !graph_task);
    exec.execution.stream = reinterpret_cast<void*>(1);
    uint8_t other_output[sizeof(output)]{};
    request.output = {other_output, sizeof(other_output), npu};
    assert(!owner->Execute(exec).ok() && !graph_task);
    request.output = {output, sizeof(output), npu};
    assert(owner->Execute(exec).ok() && graph_task && work.empty());
    assert(!owner->CheckGatherGraphReplay().ok());
    assert(stream_syncs == syncs && status_copies == copies);
    capturing = false;
    assert(!owner->Execute(exec).ok() && !owner->Drain().ok());
    for (int i = 0; i < 2; ++i) {
      work.push_back(graph_task);
      assert(owner->CheckGatherGraphReplay().ok() && work.empty());
      assert(status_copies == copies + i + 1);
    }
    work.push_back(graph_task);
    kernel_result = KVAF_SEQ_REMOTE_ERROR;
    assert(owner->CheckGatherGraphReplay().code() == StatusCode::kInternal);
    assert(!owner->CheckGatherGraphReplay().ok());
    kernel_result = 0;
    graph_task = {};
    assert(owner->Shutdown().ok() && !refs && !allocations);
    exec.execution.trusted_npu_request = false;
    exec.execution.synchronize = true;
  }
  unsetenv("KVAF_AIV_GATHER_GRAPH");
  unsetenv("KVAF_AIV_GATHER_CHECK_STATUS");
  for (const char* name : {"KVAF_AIV_GATHER_SHARD_PIPELINE", "KVAF_AIV_GATHER_PACKED_POSTS",
                           "KVAF_AIV_GATHER_PROGRESS_MODE"}) {
    setenv(name, "0", 1);
    auto owner = CreateSequentialBatchGet(config);
    const auto result = owner->Initialize(region);
    assert(result.code() == StatusCode::kInvalidArgument);
    assert(result.message().find("has been removed") != std::string::npos);
    assert(!allocations && !refs);
    unsetenv(name);
  }
  setenv("KVAF_AIV_GATHER_CHUNK_BYTES", "0", 1);
  for (const char* setting : {"0", "1"}) {
    setenv("KVAF_AIV_GATHER_PREPARE_AHEAD", setting, 1);
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).ok() && owner->SetGatherProfiling(true).ok());
    unsetenv("KVAF_AIV_GATHER_PREPARE_AHEAD");
    assert(owner->Execute(exec).ok() && last_prepare_ahead == uint32_t(*setting - '0'));
    GatherBatchGetProfile profile;
    assert(owner->CopyGatherProfile(&profile).ok());
    for (const auto& lane : profile.cores)
      assert(lane.prepare_ahead == (lane.requests ? last_prepare_ahead : 0));
    assert(owner->Shutdown().ok() && !allocations && !refs);
  }
  setenv("KVAF_AIV_GATHER_PREPARE_AHEAD", "invalid", 1);
  { auto owner = CreateSequentialBatchGet(config); assert(!owner->Initialize(region).ok()); }
  unsetenv("KVAF_AIV_GATHER_PREPARE_AHEAD");
  unsetenv("KVAF_AIV_GATHER_CHUNK_BYTES");
  for (const char* count : {"2", "4", "8", "16", "32"}) {
    setenv("KVAF_AIV_GATHER_COOPERATIVE_CHUNKS", count, 1);
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).ok() && owner->SetGatherProfiling(true).ok());
    unsetenv("KVAF_AIV_GATHER_COOPERATIVE_CHUNKS");
    assert(owner->Execute(exec).ok());
    assert(last_cooperative_chunks == std::strtoul(count, nullptr, 10));
    assert(last_prepare_ahead == 1);
    GatherBatchGetProfile profile;
    assert(owner->CopyGatherProfile(&profile).ok());
    for (const auto& lane : profile.cores)
      assert(lane.cooperative_chunks == (lane.requests ? last_cooperative_chunks : 0));
    assert(owner->Shutdown().ok() && !allocations && !refs);
  }
  for (const char* first : {"16", "32"})
  for (const char* tails : {"1", "4", "6", "8", "32"}) {
    setenv("KVAF_AIV_GATHER_COOPERATIVE_CHUNKS", tails, 1);
    setenv("KVAF_AIV_GATHER_FIRST_CHUNK_DIVISOR", first, 1);
    auto owner = CreateSequentialBatchGet(config);
    assert(owner->Initialize(region).ok() && owner->SetGatherProfiling(true).ok());
    unsetenv("KVAF_AIV_GATHER_FIRST_CHUNK_DIVISOR");
    assert(owner->Execute(exec).ok() && last_first_divisor == std::strtoul(first, nullptr, 10) &&
           last_cooperative_chunks == std::strtoul(tails, nullptr, 10));
    GatherBatchGetProfile profile; assert(owner->CopyGatherProfile(&profile).ok());
    for (const auto& lane : profile.cores)
      assert(lane.pipeline_chunk_bytes == (lane.requests ? kvaf::GatherCooperativeChunkBytes(
          lane.requests * config.batch_get_token_bytes, last_cooperative_chunks, last_first_divisor) : 0));
    assert(owner->Shutdown().ok() && !allocations && !refs);
  }
  for (const char* value : {"1", "4294967296", "-32", "32x"}) {
    setenv("KVAF_AIV_GATHER_FIRST_CHUNK_DIVISOR", value, 1);
    auto owner = CreateSequentialBatchGet(config); assert(!owner->Initialize(region).ok());
  }
  setenv("KVAF_AIV_GATHER_FIRST_CHUNK_DIVISOR", "32", 1);
  for (const char* value : {"0", "33", "64"}) {
    setenv("KVAF_AIV_GATHER_COOPERATIVE_CHUNKS", value, 1);
    auto owner = CreateSequentialBatchGet(config); assert(!owner->Initialize(region).ok());
  }
  unsetenv("KVAF_AIV_GATHER_FIRST_CHUNK_DIVISOR");
  for (const char* count : {"1", "3", "24", "33", "64", "-2", "4x"}) {
    setenv("KVAF_AIV_GATHER_COOPERATIVE_CHUNKS", count, 1);
    auto owner = CreateSequentialBatchGet(config); assert(!owner->Initialize(region).ok());
  }
  setenv("KVAF_AIV_GATHER_COOPERATIVE_CHUNKS", "4", 1);
  setenv("KVAF_AIV_GATHER_PREPARE_AHEAD", "0", 1);
  { auto owner = CreateSequentialBatchGet(config); assert(!owner->Initialize(region).ok()); }
  unsetenv("KVAF_AIV_GATHER_PREPARE_AHEAD");
  setenv("KVAF_AIV_GATHER_CHUNK_BYTES", "0", 1);
  { auto owner = CreateSequentialBatchGet(config); assert(!owner->Initialize(region).ok()); }
  unsetenv("KVAF_AIV_GATHER_CHUNK_BYTES");
  unsetenv("KVAF_AIV_GATHER_COOPERATIVE_CHUNKS");
  std::cout << "sequential Host lifecycle: PASS\n";
}
