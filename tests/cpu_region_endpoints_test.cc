// Exercise the production CPU service with a deterministic URMA provider.
// Requires the public UMDK headers, but no hardware or liburma at runtime.
#include <set>
#include <atomic>
#include <thread>
#include "peer_link.h"

// Some public header releases also declare expanded APIs with different
// typedefs. The service deliberately loads the deployed private ABI by name.
#define urma_get_tp_list unused_public_get_tp_list
#define urma_import_jetty_ex unused_public_import_jetty_ex
#include <urma_api.h>
#undef urma_get_tp_list
#undef urma_import_jetty_ex
#include "../runtime/src/cpu_runtime.cc"

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); std::abort(); } } while (false)

namespace {
urma_device_t device{};
std::set<urma_context_t*> contexts;
std::set<urma_token_id_t*> tokens;
std::set<urma_target_seg_t*> segments;
std::set<urma_jetty_t*> jettys;
std::set<urma_target_jetty_t*> imports;
std::set<urma_jfc_t*> jfcs;
std::set<urma_jfr_t*> jfrs;
uint32_t next_id = 1U;
int fail_registration = -1;
bool fail_import = false;
bool bad_registration_address = false;
}

extern "C" {
urma_status_t urma_init(urma_init_attr_t*) { return URMA_SUCCESS; }
urma_status_t urma_uninit() { CHECK(contexts.empty()); return URMA_SUCCESS; }
urma_device_t** urma_get_device_list(int* count) {
  std::strcpy(device.name, "fake_ub");
  device.type = URMA_TRANSPORT_UB;
  *count = 1;
  return new urma_device_t*[1]{&device};
}
void urma_free_device_list(urma_device_t** list) { delete[] list; }
urma_device_t* urma_get_device_by_name(char*) { return &device; }
urma_eid_info_t* urma_get_eid_list(urma_device_t*, uint32_t* count) {
  *count = 8U;
  auto* list = new urma_eid_info_t[8]{};
  for (uint32_t i = 0U; i < 8U; ++i) {
    list[i].eid_index = i;
    list[i].eid.raw[15] = static_cast<uint8_t>(i + 1U);
  }
  return list;
}
void urma_free_eid_list(urma_eid_info_t* list) { delete[] list; }
urma_context_t* urma_create_context(urma_device_t*, uint32_t index) {
  auto* ctx = new urma_context_t{};
  ctx->eid_index = index;
  ctx->eid.raw[15] = static_cast<uint8_t>(index + 1U);
  contexts.insert(ctx);
  return ctx;
}
urma_status_t urma_delete_context(urma_context_t* ctx) {
  for (auto* token : tokens) CHECK(token->urma_ctx != ctx);
  CHECK(contexts.erase(ctx) == 1U);
  delete ctx;
  return URMA_SUCCESS;
}
urma_token_id_t* urma_alloc_token_id(urma_context_t* ctx) {
  auto* token = new urma_token_id_t{};
  token->urma_ctx = ctx;
  token->token_id = next_id++ << 8U;
  tokens.insert(token);
  return token;
}
urma_status_t urma_free_token_id(urma_token_id_t* token) {
  for (auto* segment : segments) CHECK(segment->token_id != token);
  CHECK(tokens.erase(token) == 1U);
  delete token;
  return URMA_SUCCESS;
}
urma_target_seg_t* urma_register_seg(urma_context_t* ctx, urma_seg_cfg_t* cfg) {
  CHECK(cfg->token_id->urma_ctx == ctx);
  if (static_cast<int>(ctx->eid_index) == fail_registration) return nullptr;
  auto* seg = new urma_target_seg_t{};
  seg->urma_ctx = ctx;
  seg->token_id = cfg->token_id;
  seg->seg.ubva.eid = ctx->eid;
  seg->seg.ubva.va = cfg->va + (bad_registration_address ? 4096U : 0U);
  seg->seg.len = cfg->len;
  seg->seg.token_id = cfg->token_id->token_id;
  segments.insert(seg);
  return seg;
}
urma_status_t urma_unregister_seg(urma_target_seg_t* seg) {
  for (auto* jetty : jettys) CHECK(jetty->urma_ctx != seg->urma_ctx);
  CHECK(segments.erase(seg) == 1U);
  delete seg;
  return URMA_SUCCESS;
}
urma_jfc_t* urma_create_jfc(urma_context_t* ctx, urma_jfc_cfg_t*) {
  auto* jfc = new urma_jfc_t{}; jfc->urma_ctx = ctx; jfc->jfc_id.id = next_id++;
  jfcs.insert(jfc); return jfc;
}
urma_status_t urma_delete_jfc(urma_jfc_t* jfc) { CHECK(jfcs.erase(jfc)); delete jfc; return URMA_SUCCESS; }
urma_jfr_t* urma_create_jfr(urma_context_t* ctx, urma_jfr_cfg_t*) {
  auto* jfr = new urma_jfr_t{}; jfr->urma_ctx = ctx;
  jfrs.insert(jfr); return jfr;
}
urma_status_t urma_delete_jfr(urma_jfr_t* jfr) { CHECK(jfrs.erase(jfr)); delete jfr; return URMA_SUCCESS; }
urma_jetty_t* urma_create_jetty(urma_context_t* ctx, urma_jetty_cfg_t* cfg) {
  CHECK(cfg->jfs_cfg.jfc->urma_ctx == ctx && cfg->shared.jfr->urma_ctx == ctx);
  auto* jetty = new urma_jetty_t{};
  jetty->urma_ctx = ctx; jetty->jetty_cfg = *cfg;
  jetty->jetty_id.eid = ctx->eid; jetty->jetty_id.id = next_id++;
  jettys.insert(jetty); return jetty;
}
urma_status_t urma_delete_jetty(urma_jetty_t* jetty) { CHECK(jettys.erase(jetty)); delete jetty; return URMA_SUCCESS; }
urma_status_t urma_unimport_jetty(urma_target_jetty_t* jetty) { CHECK(imports.erase(jetty)); delete jetty; return URMA_SUCCESS; }
int urma_get_tp_list(urma_context_t* ctx, kvaf::UrmaExtGetTpCfg* cfg, uint32_t*, kvaf::UrmaExtTpInfo* info) {
  CHECK(std::memcmp(cfg->local_eid, ctx->eid.raw, 16U) == 0);
  info->tp_handle = 100U + ctx->eid_index;
  return 0;
}
urma_target_jetty_t* urma_import_jetty_ex(urma_context_t* ctx, kvaf::UrmaExtRjetty* remote,
                                        urma_token_t*, kvaf::UrmaExtActiveTpCfg* cfg) {
  CHECK(cfg->tp_handle == 100U + ctx->eid_index);
  if (fail_import) return nullptr;
  auto* target = new urma_target_jetty_t{};
  target->id = remote->jetty_id;
  imports.insert(target);
  return target;
}
}

int main() {
  using namespace kvaf;
  using namespace kvaf::peer;
  {
    UrmaResourcePool pool;
    CHECK(pool.Initialize("", false, 0U));
    CpuService service(&pool, CpuMemoryPageMode::kDefault, false);
    for (uint32_t count : {2U, 4U, 8U}) {
      AllocateRegionResponse primary{};
      CHECK(service.Allocate(4096U, 64U, 0U, &primary) == Result::kOk);
      CHECK(primary.endpoint_id == 0U);
      for (uint32_t endpoint = 0U; endpoint < count; ++endpoint) {
        BindRegionEndpointRequest bind{};
        bind.session_id = primary.session_id; bind.candidate_id = endpoint;
        BindRegionEndpointResponse bound{};
        CHECK(service.BindRegionEndpoint(bind, &bound) == Result::kOk);
        CHECK(bound.endpoint_id == endpoint && bound.peer_base == primary.peer_base);
        CHECK(bound.peer_length == primary.peer_length && bound.token_value == primary.token_value);
        CHECK(bound.peer_eid[15] == endpoint + 1U);
        CHECK(segments.size() == endpoint + 1U);
        CHECK(service.BindRegionEndpoint(bind, &bound) == Result::kOk);
        CHECK(segments.size() == endpoint + 1U);  // Repeated core mapping reuses registration.
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
          CreateChannelRequest request{};
          request.session_id = primary.session_id; request.endpoint_id = endpoint;
          request.local_qp_key_bytes = 32U; request.peer_token_value = primary.token_value;
          request.local_tp_handle = 99U; request.local_psn = 42U; request.local_eid[15] = 99U;
          CreateChannelResponse response{};
          CHECK(service.CreateChannel(request, &response) == Result::kOk);
          urma_jetty_id_t id{};
          std::memcpy(&id, response.peer_qp_key, sizeof(id));
          CHECK(id.eid.raw[15] == endpoint + 1U);
          CHECK(response.peer_tp_handle == 100U + endpoint);
          fail_import = true;
          const auto before = jettys.size();
          CHECK(service.CreateChannel(request, &response) == Result::kResourceError);
          CHECK(jettys.size() == before && jfcs.size() == before && jfrs.size() == before);
          fail_import = false;
          request.endpoint_id = 999U;
          CHECK(service.CreateChannel(request, &response) == Result::kNotFound);
        }
      }
      CHECK(service.Release(primary.session_id) == Result::kOk);
      CHECK(segments.empty() && tokens.empty() && jettys.empty() && imports.empty());
      CHECK(jfcs.empty() && jfrs.empty());
    }
    AllocateRegionResponse primary{};
    CHECK(service.Allocate(4096U, 64U, 0U, &primary) == Result::kOk);
    BindRegionEndpointRequest bind{};
    bind.session_id = primary.session_id; bind.candidate_id = 1U;
    BindRegionEndpointResponse bound{};
    fail_registration = 1;
    CHECK(service.BindRegionEndpoint(bind, &bound) == Result::kResourceError);
    CHECK(segments.size() == 1U && tokens.size() == 1U);
    fail_registration = -1;
    bad_registration_address = true;
    CHECK(service.BindRegionEndpoint(bind, &bound) == Result::kResourceError);
    CHECK(segments.size() == 1U && tokens.size() == 1U);
    bad_registration_address = false;
    CHECK(service.Release(primary.session_id) == Result::kOk);
    CHECK(service.BindRegionEndpoint(bind, &bound) == Result::kNotFound);
  }
  CHECK(contexts.empty() && tokens.empty() && segments.empty());
  // Round-trip the real v7 TCP client against the real request dispatcher.
  {
    UrmaResourcePool pool;
    CHECK(pool.Initialize("", false, 0U));
    CpuService service(&pool, CpuMemoryPageMode::kDefault, false);
    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    CHECK(listener >= 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    CHECK(bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
    socklen_t address_bytes = sizeof(address);
    CHECK(getsockname(listener, reinterpret_cast<sockaddr*>(&address), &address_bytes) == 0);
    CHECK(listen(listener, 8) == 0);
    std::atomic<bool> stop{false};
    std::thread server([&] {
      while (!stop) {
        const int fd = accept(listener, nullptr, nullptr);
        if (fd < 0) break;
        if (!stop) HandleClient(fd, &service);
        close(fd);
      }
    });
    PeerLink link("127.0.0.1:" + std::to_string(ntohs(address.sin_port)));
    uint32_t count = 0U;
    CHECK(link.GetCandidateCount(&count).ok() && count == 8U);
    PeerCandidateInfo info{};
    CHECK(link.GetCandidateInfo(7U, &info).ok() && info.eid[15] == 8U);
    PeerRegion region{};
    CHECK(link.AllocateRegion(4096U, 64U, 0U, &region).ok());
    PeerRegion endpoint{};
    CHECK(link.BindRegionEndpoint(region.base, 7U, &endpoint).ok());
    CHECK(endpoint.base == region.base && endpoint.eid[15] == 8U && endpoint.endpoint_id == 1U);
    CHECK(link.BindRegionEndpoint(region.base, 7U, &endpoint).ok() && endpoint.endpoint_id == 1U);
    CHECK(!link.BindRegionEndpoint(region.base, 99U, &endpoint).ok());
    CHECK(link.LookupRegion(region.base, &endpoint, 1U).ok() && endpoint.eid[15] == 8U);
    CHECK(link.LookupRegion(region.base, &endpoint).ok() && endpoint.eid[15] == 1U);
    std::array<uint8_t, kQpKeyBytes> local_key{}, peer_key{};
    std::array<uint8_t, kEidBytes> local_eid{};
    local_eid[15] = 99U;
    uint32_t peer_key_bytes = 0U, peer_psn = 0U;
    uint64_t peer_tp = 0U;
    CHECK(link.CreateChannel(region.base, local_key.data(), 32U, local_eid.data(),
                              region.token_value, 99U, 42U, &peer_key, &peer_key_bytes,
                              &peer_tp, &peer_psn, 1U).ok());
    urma_jetty_id_t peer_id{};
    std::memcpy(&peer_id, peer_key.data(), sizeof(peer_id));
    CHECK(peer_id.eid.raw[15] == 8U && peer_tp == 107U);
    CHECK(!link.CreateChannel(region.base, local_key.data(), 32U, local_eid.data(),
                               region.token_value, 99U, 42U, &peer_key, &peer_key_bytes,
                               &peer_tp, &peer_psn, 99U).ok());
    CHECK(link.ReleaseRegion(region.base).ok());
    CHECK(!link.has_region() && !link.LookupRegion(region.base, &endpoint, 1U).ok());
    stop = true;
    shutdown(listener, SHUT_RDWR);
    close(listener);
    server.join();
  }
  CHECK(contexts.empty() && tokens.empty() && segments.empty() && jettys.empty());
  std::puts("CPU endpoints: shared DDR, 2/4/8 EIDs, Jetty contexts and rollback passed");
}
