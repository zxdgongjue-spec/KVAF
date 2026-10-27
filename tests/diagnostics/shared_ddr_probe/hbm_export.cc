// C ABI for the Prefill Python process. The caller owns the Tensor and keeps
// it alive until K confirms that all pull queues/imports have been destroyed.
#include "wire.h"
#include "access_compat.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
namespace h = kvaf::direct_hccp;
using ddr_probe::Message;
constexpr int kHccpAsyncAgain = 128301;  // Matching KVAF direct HCCP ABI.
thread_local std::string error;
void Check(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
struct Exported {
  h::DirectHccpApi api;
  h::DirectHccpSession* session{};
  void *context{}, *token{}, *mr{}, *cq{}, *jetty{}, *peer{};
  std::uint64_t tp{};
  std::uint32_t token_value{0x51a7d001U};
  void Close() {
    if (peer) { Check(api.qp_unimport(context, peer) == 0, "unimport K Jetty"); peer = nullptr; }
    if (jetty) { Check(api.qp_destroy(jetty) == 0, "destroy source Jetty"); jetty = nullptr; }
    if (cq) { Check(api.cq_destroy(context, cq) == 0, "destroy source CQ"); cq = nullptr; }
    if (mr) { Check(api.lmem_unregister(context, mr) == 0, "unregister HBM"); mr = nullptr; }
    if (token) { Check(api.token_free(context, token) == 0, "free HBM token"); token = nullptr; }
    if (context) { Check(api.ctx_deinit(context) == 0, "destroy source context"); context = nullptr; }
    if (session) { h::ReleaseDirectHccpSession(session); session = nullptr; }
    h::CloseDirectHccp(&api);
  }
};
std::uint64_t QueryTp(Exported& e, const std::uint8_t* local, const std::uint8_t* remote) {
  h::GetTpCfg cfg{}; cfg.flag = h::CtpFlag(); cfg.transport_mode = h::kReliableMessage;
  std::memcpy(cfg.local_eid.raw, local, 16); std::memcpy(cfg.peer_eid.raw, remote, 16);
  h::TpInfo info{}; std::uint32_t count = 1; void* request{};
  Check(e.api.get_tp_info_async(e.context, &cfg, &info, &count, &request) == 0 && request,
    "submit source CTP query");
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  for (;;) {
    int result{};
    const int rc = e.api.get_async_req_result(request, &result);
    if (rc != kHccpAsyncAgain) { Check(rc == 0 && result == 0, "source CTP query failed"); break; }
    Check(std::chrono::steady_clock::now() < deadline, "source CTP query timed out");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  Check(count && info.tp_handle, "empty source CTP query");
  return info.tp_handle;
}
}  // namespace

extern "C" const char* probe_hbm_error() { return error.c_str(); }
extern "C" void* probe_hbm_export(std::uint64_t address, std::uint64_t bytes,
    const std::uint8_t* local_eid, const std::uint8_t* k_eid, Message* descriptor) {
  auto* e = new Exported;
  try {
    Check(address && bytes && local_eid && k_eid && descriptor, "invalid HBM export arguments");
    std::uint32_t physical{};
    Check(h::LoadDirectHccp(&e->api) && h::GetCurrentNpuPhysicalDeviceId(&physical) &&
      h::AcquireDirectHccpSession(physical, 18, &e->session), "acquire source direct HCCP");
    h::CtxInitCfg cfg{}; cfg.mode = h::kOfflineHdc;
    h::CtxInitAttr attr{}; attr.phy_id = physical;
    std::memcpy(attr.ub.eid.raw, local_eid, 16);
    Check(e->api.ctx_init(&cfg, &attr, &e->context) == 0 && e->context, "create source RA context");
    h::TokenId token{};
    Check(e->api.token_alloc(e->context, &token, &e->token) == 0 && e->token, "allocate source token");
    h::MrRegInfo mr{};
    mr.in.mem.addr = address; mr.in.mem.size = bytes;
    mr.in.ub.flags = h::MemRegFlags(true); mr.in.ub.token_value = e->token_value;
    mr.in.ub.token_id_handle = e->token;
    Check(e->api.lmem_register(e->context, &mr, &e->mr) == 0 && e->mr, "register existing source Tensor HBM");
    Check(mr.out.key.size >= sizeof(urma_seg_t) && mr.out.key.size <= 64, "HBM segment ABI mismatch");
    urma_seg_t segment{}; std::memcpy(&segment, mr.out.key.value, sizeof(segment));
    Check(segment.ubva.va && segment.len >= bytes, "invalid exported HBM range");
    h::CqInfo cq{}; cq.in.depth = 16384; cq.in.ub.mode = h::kJfcNormal;
    Check(e->api.cq_create(e->context, &cq, &e->cq) == 0 && e->cq, "create source CQ");
    h::QpCreateAttr qp{};
    qp.scq_handle = e->cq; qp.rcq_handle = e->cq; qp.srq_handle = e->cq;
    qp.sq_depth = 1024; qp.rq_depth = 1024; qp.transport_mode = h::kReliableMessage;
    qp.ub.mode = h::kJettyStandard; qp.ub.flag = h::JettyFlagShareJfr();
    qp.ub.token_id_handle = e->token; qp.ub.token_value = e->token_value;
    qp.ub.jfs_flag = h::JfsFlagErrorSuspend(); qp.ub.priority = 2;
    qp.ub.rnr_retry = 7; qp.ub.err_timeout = 16;
    h::QpCreateInfo info{};
    Check(e->api.qp_create(e->context, &qp, &info, &e->jetty) == 0 && e->jetty,
      "create source target Jetty");
    Check(info.key.size == 32, "expected deployed 32-byte NPU Jetty key");
    e->tp = QueryTp(*e, local_eid, k_eid);
    *descriptor = Message{}; descriptor->op = ddr_probe::Export;
    descriptor->base = segment.ubva.va; descriptor->bytes = bytes;
    descriptor->tp = e->tp; descriptor->psn = 1; descriptor->token = e->token_value;
    descriptor->key_bytes = info.key.size; descriptor->segment_bytes = sizeof(segment);
    std::memcpy(descriptor->eid, local_eid, 16);
    std::memcpy(descriptor->key, info.key.value, info.key.size);
    std::memcpy(descriptor->segment, &segment, sizeof(segment));
    return e;
  } catch (const std::exception& ex) {
    error = ex.what();
    // A failed async HDC call can retain references. Keep partial resources
    // until process teardown instead of guessing whether it has quiesced.
    return nullptr;
  }
}
extern "C" int probe_hbm_bind(void* handle, const Message* response) {
  try {
    Check(handle && response && response->key_bytes == 28, "invalid K descriptor");
    auto& e = *static_cast<Exported*>(handle);
    h::QpImportInfoT info{};
    Check(h::ImportPeerJetty(e.api, e.context, response->key, response->key_bytes,
      e.token_value, e.tp, response->tp, 1, response->psn, "shared_ddr_probe", &info, &e.peer),
      "import K initiator Jetty");
    return 0;
  } catch (const std::exception& ex) { error = ex.what(); return -1; }
}
extern "C" int probe_hbm_close(void* handle) {
  try {
    Check(handle, "invalid exporter handle");
    auto* e = static_cast<Exported*>(handle);
    e->Close(); delete e; return 0;
  } catch (const std::exception& ex) { error = ex.what(); return -1; }
}
