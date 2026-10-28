// Run the production host transport and teardown against deterministic RA
// and control-plane doubles. No kernel launch or hardware ABI is emulated.
#include "direct_hccp/direct_hccp_remote_transport.h"
#define LoadDirectHccp LoadHardwareHccp
#define GetCurrentNpuPhysicalDeviceId GetHardwarePhysicalDeviceId
#include "../access_operator/src/direct_hccp/direct_hccp_remote_transport.cc"
#undef LoadDirectHccp
#undef GetCurrentNpuPhysicalDeviceId
#include "../access_operator/demo/aiv_batch_get/host/batch_get_host.cpp"

#include <iostream>
#include <unordered_map>

#define CHECK(x) do { if (!(x)) { std::cerr << "line " << __LINE__ << ": " #x "\n"; std::abort(); } } while (false)

namespace {
struct Resource { void* context; std::string kind; uint32_t id; };
std::unordered_map<void*, Resource> live;
uint32_t next_id = 1U, primary_candidate = 0U, imports_attempted = 0U;
int fail_import_at = -1, fail_channel_at = -1;
bool fail_jetty_import = false;
std::vector<uint32_t> channel_endpoints;
std::unordered_map<uint32_t, kvaf::peer::PeerRegion> endpoints;
alignas(4096) uint8_t ddr[4096];

void* New(void* context, const char* kind) {
  void* handle = new int;
  live.emplace(handle, Resource{context, kind, next_id++});
  return handle;
}
int Drop(void* context, void* handle) {
  CHECK(live.count(handle) && live.at(handle).context == context);
  live.erase(handle); delete static_cast<int*>(handle); return 0;
}
size_t Count(const std::string& kind) {
  return std::count_if(live.begin(), live.end(), [&](const auto& p) { return p.second.kind == kind; });
}
kvaf::peer::PeerRegion Descriptor(uint32_t candidate) {
  kvaf::peer::PeerRegion desc{};
  desc.session_id = 1U;
  desc.endpoint_id = candidate == primary_candidate ? 0U : candidate + 1U;
  desc.base = reinterpret_cast<uint64_t>(ddr); desc.length = sizeof(ddr);
  desc.token_id = (candidate + 32U) << 8U; desc.token_value = 7U;
  desc.eid[0] = 0xabU; desc.eid[15] = static_cast<uint8_t>(candidate + 1U);
  desc.segment_key_bytes = 16U; desc.segment_key[0] = static_cast<uint8_t>(candidate + 1U);
  return desc;
}
}

namespace kvaf::peer {
PeerLink::PeerLink(std::string address) : address_(std::move(address)) {}
Status PeerLink::GetCandidateInfo(uint32_t candidate, PeerCandidateInfo* info) const {
  info->candidate_id = candidate; info->eid = Descriptor(candidate).eid; return Status::OK();
}
Status PeerLink::BindRegionEndpoint(uint64_t base, uint32_t candidate, PeerRegion* desc) {
  CHECK(base == reinterpret_cast<uint64_t>(ddr));
  *desc = Descriptor(candidate); endpoints[desc->endpoint_id] = *desc; return Status::OK();
}
Status PeerLink::LookupRegion(uint64_t base, PeerRegion* desc, uint32_t endpoint) const {
  CHECK(base == reinterpret_cast<uint64_t>(ddr));
  if (!endpoints.count(endpoint)) return Status::NotFound("endpoint");
  *desc = endpoints.at(endpoint); return Status::OK();
}
Status PeerLink::CreateChannel(uint64_t, const uint8_t*, uint32_t, const uint8_t*, uint32_t token,
    uint64_t local_tp, uint32_t, std::array<uint8_t, kQpKeyBytes>* key,
    uint32_t* key_bytes, uint64_t* peer_tp, uint32_t* psn, uint32_t endpoint) {
  CHECK(endpoints.count(endpoint));
  const auto& desc = endpoints.at(endpoint);
  CHECK(token == desc.token_value && local_tp == 100U + desc.eid[15]);
  if (static_cast<int>(channel_endpoints.size()) == fail_channel_at) return Status::Internal("injected channel failure");
  channel_endpoints.push_back(endpoint);
  (*key)[0] = desc.eid[15]; *key_bytes = 32U; *peer_tp = 200U + desc.eid[15]; *psn = 123U;
  return Status::OK();
}
}

namespace kvaf::direct_hccp {
bool GetCurrentNpuPhysicalDeviceId(uint32_t* id) { *id = 0U; return true; }
bool AcquireDirectHccpSession(uint32_t, uint32_t, DirectHccpSession** session) {
  *session = reinterpret_cast<DirectHccpSession*>(New(nullptr, "session")); return true;
}
void ReleaseDirectHccpSession(DirectHccpSession* session) { CHECK(Count("context") == 0U); Drop(nullptr, session); }
bool LoadDirectHccp(DirectHccpApi* api) {
  api->ctx_init = [](CtxInitCfg*, CtxInitAttr*, void** ctx) { *ctx = New(nullptr, "context"); return 0; };
  api->ctx_deinit = [](void* ctx) {
    for (const auto& p : live) CHECK(p.second.context != ctx);
    return Drop(nullptr, ctx);
  };
  api->token_alloc = [](void* ctx, TokenId* id, void** handle) {
    *handle = New(ctx, "token"); id->token_id = live.at(*handle).id << 8U; return 0;
  };
  api->token_free = Drop;
  api->lmem_register = [](void* ctx, MrRegInfo* info, void** handle) {
    *handle = New(ctx, "local_mr"); info->out.key.size = 16U; return 0;
  };
  api->lmem_unregister = Drop;
  api->rmem_import = [](void* ctx, MrImportInfo* info, void** handle) {
    *handle = New(ctx, "remote_mr");
    info->out.ub.target_seg_handle = live.at(*handle).id;
    return static_cast<int>(imports_attempted++) == fail_import_at ? 1 : 0;
  };
  api->rmem_unimport = Drop;
  api->cq_create = [](void* ctx, CqInfo* info, void** handle) {
    *handle = New(ctx, "cq"); const auto id = live.at(*handle).id;
    info->out.id = id; info->out.buffer_addr = uint64_t{id} * 4096U;
    info->out.swdb_addr = info->out.buffer_addr + 64U; info->out.cqe_size = 64U; return 0;
  };
  api->cq_destroy = Drop;
  api->qp_create = [](void* ctx, QpCreateAttr* attr, QpCreateInfo* info, void** handle) {
    CHECK(live.at(attr->scq_handle).context == ctx && live.at(attr->ub.token_id_handle).context == ctx);
    *handle = New(ctx, "qp"); const auto id = live.at(*handle).id;
    info->key.size = 32U; info->ub.id = id;
    info->ub.sq_buffer_va = uint64_t{id} * 4096U; info->ub.db_addr = info->ub.sq_buffer_va + 64U;
    info->ub.wqebb_size = 64U; return 0;
  };
  api->qp_destroy = [](void* handle) { return Drop(live.at(handle).context, handle); };
  api->qp_import = [](void* ctx, QpImportInfoT* info, void** handle) {
    CHECK(info->in.ub.exp_import_cfg.tp_handle == 100U + info->in.key.value[0]);
    CHECK(info->in.ub.exp_import_cfg.peer_tp_handle == 200U + info->in.key.value[0]);
    *handle = New(ctx, "imported_qp"); info->out.ub.tpn = fail_jetty_import ? 0U : 300U + info->in.key.value[0];
    return 0;
  };
  api->qp_unimport = Drop;
  api->get_tp_info_async = [](void*, GetTpCfg* cfg, TpInfo* info, uint32_t* count, void** request) {
    info->tp_handle = 100U + cfg->peer_eid.raw[15]; *count = 1U;
    *request = New(nullptr, "async"); return 0;
  };
  api->get_async_req_result = [](void* request, int* result) { *result = 0; return Drop(nullptr, request); };
  return true;
}
}

int main() {
  for (uint32_t cores : {2U, 4U, 8U}) {
    for (bool distinct : {false, true}) {
      for (int fault : {0, 1, 2, 3}) {
        primary_candidate = 7U;
        endpoints = {{0U, Descriptor(primary_candidate)}};
        imports_attempted = 0U; channel_endpoints.clear();
        fail_import_at = fault == 1 ? 0 : -1;
        fail_channel_at = fault == 2 ? 1 : -1;
        fail_jetty_import = fault == 3;
        kvaf::peer::PeerLink link("fake");
        std::vector<uint32_t> candidates(cores, primary_candidate);
        if (distinct) for (uint32_t i = 0U; i < cores; ++i) candidates[i] = 7U - i;
        std::array<uint8_t, 16> local_eid{}; local_eid[15] = 99U;
        KvafBatchGetTransport* transport = nullptr;
        const int result = KvafBatchGetTransportCreateWithPeerEndpoints(
            ddr, sizeof(ddr), cores, 2U, &link, local_eid.data(), 1U,
            candidates.data(), cores, &transport);
        if (fault != 0) {
          CHECK(result != 0 && transport == nullptr && live.empty());
          continue;
        }
        CHECK(result == 0 && transport != nullptr);
        CHECK(Count("context") == 1U && Count("remote_mr") == (distinct ? cores : 1U));
        uint8_t output[1024]{};
        KvafBatchGetLocalBuffer* buffer = nullptr;
        uint64_t entity_address = 0U;
        CHECK(KvafBatchGetLocalBufferAcquire(transport, output, sizeof(output), &buffer, &entity_address) == 0);
        CHECK(Count("local_mr") == 1U);
        const auto* entities = reinterpret_cast<const KvafBatchGetDirectUrmaEntity*>(entity_address);
        for (uint32_t i = 0U; i < cores * 2U; ++i) {
          const auto expected = Descriptor(candidates[i / 2U]);
          CHECK(channel_endpoints[i] == expected.endpoint_id);
          CHECK(entities[i].remote_token_id == expected.token_id >> 8U);
          CHECK(entities[i].remote_host_ddr_address == expected.base);
          CHECK(entities[i].tp_id == 300U + expected.eid[15]);
          for (uint32_t b = 0U; b < 16U; ++b) CHECK(entities[i].remote_eid[b] == expected.eid[15U - b]);
          CHECK(entities[i].local_token_id == entities[0].local_token_id);
        }
        KvafBatchGetLocalBufferRelease(buffer);
        KvafBatchGetTransportDestroy(transport);
        CHECK(live.empty() && fake_acl::allocations == 0U);
      }
    }
  }
  // Legacy multi-local-EID entry point still imports the default CPU region.
  fail_import_at = fail_channel_at = -1; imports_attempted = 0U; channel_endpoints.clear();
  fail_jetty_import = false;
  primary_candidate = 0U; endpoints = {{0U, Descriptor(0U)}};
  kvaf::peer::PeerLink link("fake");
  std::array<uint8_t, 32> local_eids{}; local_eids[15] = 98U; local_eids[31] = 99U;
  KvafBatchGetTransport* transport = nullptr;
  CHECK(KvafBatchGetTransportCreateWithPeerLinkMultiEid(
      ddr, sizeof(ddr), 2U, 2U, &link, local_eids.data(), 2U, &transport) == 0);
  CHECK(Count("context") == 2U && Count("remote_mr") == 2U);
  CHECK(std::all_of(channel_endpoints.begin(), channel_endpoints.end(), [](uint32_t id) { return id == 0U; }));
  KvafBatchGetTransportDestroy(transport);
  CHECK(live.empty());
  std::cout << "BatchGet host transport: core endpoints, shared context/HBM, legacy paths and rollback passed\n";
}
