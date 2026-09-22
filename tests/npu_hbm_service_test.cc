#include "npu_hbm_service.h"
#include "npu_peer_eids.h"
#include "peer_protocol.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include <urma_api.h>
#include <acl/acl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace {
namespace h = kvaf::direct_hccp;
namespace p = kvaf::peer;
void Check(bool condition, const char* reason) {
  if (!condition) throw std::runtime_error(reason);
}
int selected_device = -1, allocation_device = -1, live_memory = 0;
int contexts = 0, sessions = 0, tokens = 0, mrs = 0, cqs = 0, jettys = 0, imports = 0;
int next_id = 1;
bool fail_mr = false, fail_import = false;
std::uint32_t imported_token = 0U;
std::vector<std::string> destroyed;
struct Context { std::uint32_t device; h::Eid eid; };
std::unordered_set<void*> mr_handles;
std::unordered_set<void*> token_handles;
std::vector<std::uint32_t> created_jettys;

template<class Request, class Response>
Response Exchange(kvaf::NpuHbmService& service, const Request& request) {
  int fds[2];
  Check(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair failed");
  timeval timeout{2, 0};
  (void)setsockopt(fds[0], SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  Check(send(fds[0], &request, sizeof(request), 0) == sizeof(request), "send failed");
  Check(service.ServeConnection(fds[1]).ok(), "service dispatch failed");
  Response response{};
  const auto bytes = recv(fds[0], &response, sizeof(response), MSG_WAITALL);
  close(fds[0]); close(fds[1]);
  Check(bytes == sizeof(response), "incomplete response");
  Check(p::IsValidResponseHeader(response.header, static_cast<p::Command>(request.header.command),
                                 sizeof(response)), "response header mismatch");
  return response;
}
p::AllocateRegionResponse Allocate(kvaf::NpuHbmService& service,
                                   p::Command command = p::Command::kAllocateHbmRegion) {
  p::AllocateRegionRequest request{};
  request.header = p::MakeHeader(command, sizeof(request));
  request.ddr_bytes = 8192; request.slot_size = 4096; request.candidate_id = 1;
  return Exchange<p::AllocateRegionRequest, p::AllocateRegionResponse>(service, request);
}
}

int aclrtSetDevice(std::int32_t device) { selected_device = device; return 0; }
int aclrtMalloc(void** data, std::size_t bytes, int) {
  *data = std::malloc(bytes); allocation_device = selected_device;
  if (*data) ++live_memory;
  return *data ? 0 : 1;
}
int aclrtFree(void* data) { std::free(data); --live_memory; destroyed.push_back("memory"); return 0; }

namespace kvaf {
Status DiscoverNpuPeerEids(std::uint32_t device, const std::vector<std::string>&,
                           std::vector<std::array<std::uint8_t, 16>>* eids) {
  if (device != 3U) return Status::InvalidArgument("unexpected device");
  eids->resize(2); (*eids)[0][15] = 1; (*eids)[1][15] = 2;
  return Status::OK();
}
namespace direct_hccp {
bool LoadDirectHccp(DirectHccpApi* api) {
  api->library = reinterpret_cast<void*>(1);
  api->ctx_init = [](CtxInitCfg*, CtxInitAttr* attr, void** out) {
    ++contexts; *out = new Context{attr->phy_id, attr->ub.eid}; return 0;
  };
  api->ctx_deinit = [](void* c) { delete static_cast<Context*>(c); --contexts; return 0; };
  api->token_alloc = [](void*, TokenId* token, void** out) {
    token->token_id = 0x1200; *out = new int(next_id++); token_handles.insert(*out); ++tokens; return 0;
  };
  api->token_free = [](void*, void* value) {
    Check(token_handles.erase(value) == 1, "double token free");
    delete static_cast<int*>(value); --tokens; destroyed.push_back("token"); return 0;
  };
  api->lmem_register = [](void* c, MrRegInfo* reg, void** out) {
    Check(token_handles.count(reg->in.ub.token_id_handle) == 1, "MR token is not alive");
    urma_seg_t segment{};
    std::copy_n(static_cast<Context*>(c)->eid.raw, 16, segment.ubva.eid.raw);
    segment.ubva.va = reg->in.mem.addr; segment.len = reg->in.mem.size;
    reg->out.key.size = sizeof(segment);
    std::memcpy(reg->out.key.value, &segment, sizeof(segment));
    *out = new int(next_id++); mr_handles.insert(*out); ++mrs;
    return fail_mr ? 1 : 0; // Failure may still produce a handle to release.
  };
  api->lmem_unregister = [](void*, void* value) {
    Check(mr_handles.erase(value) == 1, "double MR unregister");
    delete static_cast<int*>(value); --mrs; destroyed.push_back("mr"); return 0;
  };
  api->cq_create = [](void*, CqInfo*, void** out) { *out = new int(next_id++); ++cqs; return 0; };
  api->cq_destroy = [](void*, void* value) {
    delete static_cast<int*>(value); --cqs; destroyed.push_back("cq"); return 0;
  };
  api->qp_create = [](void*, QpCreateAttr* attr, QpCreateInfo* info, void** out) {
    Check(token_handles.count(attr->ub.token_id_handle) == 1, "Jetty token is not alive");
    info->key.size = 28; info->ub.id = next_id++;
    std::memcpy(info->key.value, &info->ub.id, sizeof(info->ub.id));
    created_jettys.push_back(info->ub.id);
    *out = new int(next_id++); ++jettys; return 0;
  };
  api->qp_destroy = [](void* value) {
    delete static_cast<int*>(value); --jettys; destroyed.push_back("jetty"); return 0;
  };
  api->qp_unimport = [](void*, void* value) {
    delete static_cast<int*>(value); --imports; destroyed.push_back("import"); return 0;
  };
  return true;
}
void CloseDirectHccp(DirectHccpApi* api) { *api = {}; }
bool GetCurrentNpuPhysicalDeviceId(std::uint32_t* id) { *id = selected_device; return true; }
bool AcquireDirectHccpSession(std::uint32_t device, std::uint32_t, DirectHccpSession** out) {
  Check(device == 3, "context created on wrong device");
  ++sessions; *out = reinterpret_cast<DirectHccpSession*>(1); return true;
}
void ReleaseDirectHccpSession(DirectHccpSession*) { --sessions; }
bool QueryPeerCtp(const DirectHccpApi&, void*, const std::array<std::uint8_t, 16>& local,
                  const std::array<std::uint8_t, 16>& peer, const char*, std::uint64_t* tp) {
  Check(local[15] == 2 && peer[15] == 9, "wrong CTP EID pair"); *tp = 42; return true;
}
bool ImportPeerJetty(const DirectHccpApi&, void*, const std::uint8_t*, std::uint32_t,
                    std::uint32_t token, std::uint64_t local, std::uint64_t peer,
                    std::uint32_t tx_psn, std::uint32_t rx_psn,
                    const char*, QpImportInfoT*, void** out) {
  Check(local == 42 && peer == 11 && tx_psn == 1 && rx_psn == 7, "wrong TP/PSN exchange");
  imported_token = token; *out = new int(next_id++); ++imports;
  return !fail_import;
}
} // namespace direct_hccp
} // namespace kvaf

int main() {
  try {
    kvaf::NpuHbmService service;
    kvaf::NpuHbmServiceOptions options;
    options.device_id = 3; options.max_sessions = 2; options.max_channels_per_session = 2;
    Check(service.Initialize(options).ok(), "initialize failed");
    p::CandidateCountRequest count{p::MakeHeader(p::Command::kGetCandidateCount, sizeof(count))};
    auto candidates = Exchange<p::CandidateCountRequest, p::CandidateCountResponse>(service, count);
    Check(candidates.result == 0 && candidates.candidate_count == 2 && candidates.device_count == 1,
          "candidate metadata incorrect");
    Check(Allocate(service, p::Command::kAllocateRegion).result != 0, "CPU allocator accepted by NPU service");
    fail_mr = true;
    Check(Allocate(service).result != 0 && live_memory == 0 && mrs == 0 && tokens == 0,
          "failed registration leaked resources");
    fail_mr = false;
    auto first = Allocate(service); auto second = Allocate(service);
    Check(first.result == 0 && second.result == 0 && first.session_id != second.session_id &&
          first.peer_base != second.peer_base && first.token_id == 0x1200 &&
          first.segment_key_bytes == sizeof(urma_seg_t) && first.peer_eid[15] == 2,
          "HBM descriptor/session mismatch");
    Check(allocation_device == 3 && Allocate(service).result != 0, "wrong device or session limit ignored");
    Check(service.Shutdown().code() == kvaf::StatusCode::kBusy, "shutdown freed active reader HBM");
    p::CreateChannelRequest channel{};
    channel.header = p::MakeHeader(p::Command::kCreateChannel, sizeof(channel));
    channel.session_id = first.session_id; channel.local_qp_key_bytes = 28;
    channel.local_eid[15] = 9; channel.local_tp_handle = 11; channel.local_psn = 7;
    channel.peer_token_value = 0x1234;
    fail_import = true;
    Check(Exchange<p::CreateChannelRequest, p::CreateChannelResponse>(service, channel).result != 0 &&
          imports == 0 && jettys == 0 && cqs == 0, "failed import leaked partial channel");
    fail_import = false;
    auto a = Exchange<p::CreateChannelRequest, p::CreateChannelResponse>(service, channel);
    auto b = Exchange<p::CreateChannelRequest, p::CreateChannelResponse>(service, channel);
    Check(a.result == 0 && b.result == 0 && imported_token == 0x1234 &&
          std::memcmp(a.peer_qp_key, b.peer_qp_key, 28) != 0 && jettys == 2 && cqs == 2,
          "channels share target queues or use wrong token");
    Check(Exchange<p::CreateChannelRequest, p::CreateChannelResponse>(service, channel).result != 0,
          "channel limit ignored");
    channel.local_qp_key_bytes = 65;
    Check(Exchange<p::CreateChannelRequest, p::CreateChannelResponse>(service, channel).result != 0,
          "oversized Jetty key accepted");
    destroyed.clear();
    p::ReleaseRegionRequest release{};
    release.header = p::MakeHeader(p::Command::kReleaseRegion, sizeof(release));
    release.session_id = first.session_id;
    Check(Exchange<p::ReleaseRegionRequest, p::StatusResponse>(service, release).result == 0,
          "release via protocol failed");
    Check(destroyed.size() == 9 && destroyed[6] == "mr" && destroyed[7] == "token" &&
          destroyed[8] == "memory" && service.active_sessions() == 1 && jettys == 0,
          "HBM released before its channels/MR");
    Check(service.ReleaseLink(second.session_id).ok(), "second release failed");
    Check(service.Shutdown().ok() && live_memory == 0 && mrs == 0 && tokens == 0 &&
          contexts == 0 && sessions == 0 && !service.initialized(), "shutdown leaked resources");
    std::cout << "NPU HBM service protocol and lifecycle: PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
