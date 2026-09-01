#include "hbm_rw_protocol.h"

#include <acl/acl.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <sys/file.h>
#include <string>
#include <vector>

#include <urma_api.h>

#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"

namespace {
using namespace kvaf::direct_hccp;
using namespace kvaf::hbm_test;

struct State {
  DirectHccpApi api{};
  DirectHccpSession* session = nullptr;
  void* context = nullptr;
  void* token_id_handle = nullptr;
  void* lmem_handle = nullptr;
  void* cq = nullptr;
  void* jetty = nullptr;
  void* imported_cpu_jetty = nullptr;
  void* hbm = nullptr;
  std::array<std::uint8_t, kEidBytes> eid{};
  QpCreateInfo jetty_info{};
  std::array<std::uint8_t, kMemKeyBytes> segment_key{};
  std::uint32_t segment_key_bytes = 0;
  std::uint32_t token_value = 0xABCFU;
  std::uint32_t physical = 0;
};

bool ReadAll(int fd, void* data, std::size_t bytes) {
  auto* p = static_cast<std::uint8_t*>(data);
  for (std::size_t n = 0; n < bytes;) {
    const ssize_t r = recv(fd, p + n, bytes - n, 0);
    if (r <= 0) return false;
    n += static_cast<std::size_t>(r);
  }
  return true;
}
bool WriteAll(int fd, const void* data, std::size_t bytes) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t n = 0; n < bytes;) {
    const ssize_t r = send(fd, p + n, bytes - n, MSG_NOSIGNAL);
    if (r <= 0) return false;
    n += static_cast<std::size_t>(r);
  }
  return true;
}
int Listen(std::uint16_t port) {
  const int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(port);
  if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 || listen(fd, 1) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}
bool ParseEid(const char* text, std::array<std::uint8_t, kEidBytes>* eid) {
  if (text == nullptr || eid == nullptr || std::strlen(text) != kEidBytes * 2U) return false;
  auto hex = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
  for (std::size_t i = 0; i < eid->size(); ++i) {
    const int h = hex(text[2 * i]), l = hex(text[2 * i + 1]);
    if (h < 0 || l < 0) return false;
    (*eid)[i] = static_cast<std::uint8_t>((h << 4) | l);
  }
  return true;
}
bool PairInfoEid(std::uint32_t physical, const char* field,
                 std::array<std::uint8_t, kEidBytes>* eid) {
  if (field == nullptr || eid == nullptr) return false;
  if (access("/proc/asdrv_ub/dev_id", W_OK) != 0 || access("/proc/asdrv_ub/pair_info", R_OK) != 0) {
    return false;
  }
  const int lock_fd = open("/tmp/kvaf_hbm_eid.lock", O_CREAT | O_RDWR, 0600);
  if (lock_fd < 0 || flock(lock_fd, LOCK_EX) != 0) return false;
  bool found = false;
  { std::ofstream selector("/proc/asdrv_ub/dev_id");
    if (selector.is_open()) { selector << physical << '\n'; selector.close();
      std::ifstream input("/proc/asdrv_ub/pair_info"); std::string line;
      while (std::getline(input, line)) {
        const std::string prefix = std::string(field) + ":";
        const std::size_t pos = line.find(prefix);
        if (pos == std::string::npos) continue;
        std::string hex; for (unsigned char c : line.substr(pos + prefix.size())) if (std::isxdigit(c)) hex.push_back(static_cast<char>(c));
        found = ParseEid(hex.c_str(), eid); if (found) break;
      }
    }
  }
  (void)flock(lock_fd, LOCK_UN); close(lock_fd); return found;
}

bool ResolveNpuEid(const DirectHccpApi& api, std::uint32_t physical,
                   bool list_index_explicit, std::uint32_t list_index,
                   std::array<std::uint8_t, kEidBytes>* eid,
                   std::uint32_t* selected_index) {
  if (eid == nullptr || selected_index == nullptr || api.get_dev_eid_info_num == nullptr ||
      api.get_dev_eid_info_list == nullptr) {
    return false;
  }
  std::array<std::uint8_t, kEidBytes> route_eid{};
  if (!list_index_explicit && !PairInfoEid(physical, "remote_eid", &route_eid)) return false;
  RaInfo ra_info{kOfflineHdc, physical};
  std::uint32_t count = 0U;
  if (api.get_dev_eid_info_num(ra_info, &count) != 0 || count == 0U) return false;
  std::vector<DevEidInfo> list(count);
  if (api.get_dev_eid_info_list(ra_info, list.data(), &count) != 0 || count == 0U) return false;
  std::uint32_t match = UINT32_MAX;
  if (list_index_explicit) {
    if (list_index >= count) return false;
    match = list_index;
  } else {
    for (std::uint32_t i = 0; i < count; ++i) {
      if (std::memcmp(list[i].eid.raw, route_eid.data(), kEidBytes) == 0) {
        match = i;
        break;
      }
    }
    if (match == UINT32_MAX) return false;
  }
  std::memcpy(eid->data(), list[match].eid.raw, kEidBytes);
  *selected_index = match;
  std::fprintf(stderr, "[HBM_TEST][NPU] EID route=%s list_index=%u eid_index=%u\n",
               list_index_explicit ? "explicit" : "pair_info.remote_eid", match,
               list[match].eid_index);
  return true;
}

bool Init(State* state, std::uint32_t device_id, std::uint32_t eid_index,
          bool auto_eid, bool list_index_explicit, std::uint32_t npu_list_index,
          std::array<std::uint8_t, kEidBytes>* eid, std::uint64_t bytes) {
  if (!LoadDirectHccp(&state->api) || !GetCurrentNpuPhysicalDeviceId(&state->physical) ||
      !AcquireDirectHccpSession(state->physical, kHdcType, &state->session)) return false;
  if (auto_eid) {
    std::uint32_t selected_index = 0U;
    if (!ResolveNpuEid(state->api, state->physical, list_index_explicit, npu_list_index,
                       eid, &selected_index)) {
      std::fprintf(stderr, "[HBM_TEST][NPU] cannot resolve EID from pair_info/RA list\n");
      return false;
    }
  }
  CtxInitCfg cfg{kOfflineHdc};
  CtxInitAttr attr{};
  attr.phy_id = state->physical;
  attr.ub.eid_index = eid_index;
  std::memcpy(attr.ub.eid.raw, eid->data(), eid->size());
  if (state->api.ctx_init(&cfg, &attr, &state->context) != 0 || state->context == nullptr) return false;
  CqInfo cq{};
  cq.in.depth = 1024U;
  cq.in.ub.mode = kJfcNormal;
  if (state->api.cq_create(state->context, &cq, &state->cq) != 0 || state->cq == nullptr) return false;
  TokenId token_id{};
  if (state->api.token_alloc(state->context, &token_id, &state->token_id_handle) != 0 || state->token_id_handle == nullptr) return false;
  QpCreateAttr qp{};
  qp.scq_handle = state->cq;
  qp.rcq_handle = state->cq;
  qp.srq_handle = state->cq;
  qp.sq_depth = 16U;
  qp.rq_depth = 256U;
  qp.transport_mode = kReliableMessage;
  qp.ub.mode = kJettyStandard;
  qp.ub.flag = JettyFlagShareJfr();
  qp.ub.token_id_handle = state->token_id_handle;
  qp.ub.token_value = state->token_value;
  qp.ub.jfs_flag = JfsFlagErrorSuspend();
  if (state->api.qp_create(state->context, &qp, &state->jetty_info, &state->jetty) != 0 ||
      state->jetty == nullptr || state->jetty_info.key.size == 0U || state->jetty_info.key.size > kQpKeyBytes) return false;
  if (aclrtMalloc(&state->hbm, bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) return false;
  MrRegInfo mr{};
  mr.in.mem.addr = reinterpret_cast<std::uint64_t>(state->hbm);
  mr.in.mem.size = bytes;
  mr.in.ub.flags = MemRegFlags(true);
  mr.in.ub.token_value = state->token_value;
  mr.in.ub.token_id_handle = state->token_id_handle;
  if (state->api.lmem_register(state->context, &mr, &state->lmem_handle) != 0 || state->lmem_handle == nullptr) return false;
  state->segment_key_bytes = mr.out.key.size;
  if (state->segment_key_bytes == 0U || state->segment_key_bytes > kMemKeyBytes) return false;
  std::memcpy(state->segment_key.data(), mr.out.key.value, state->segment_key_bytes);
  std::fprintf(stderr, "[HBM_TEST][NPU] READY physical=%u device=%u hbm=0x%llx bytes=%llu eid_index=%u\n",
               state->physical, device_id, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(state->hbm)),
               static_cast<unsigned long long>(bytes), eid_index);
  std::memcpy(state->eid.data(), eid->data(), eid->size());
  return true;
}
void Cleanup(State* s) {
  if (s->imported_cpu_jetty) s->api.qp_unimport(s->context, s->imported_cpu_jetty);
  if (s->jetty) s->api.qp_destroy(s->jetty);
  if (s->cq) s->api.cq_destroy(s->context, s->cq);
  if (s->lmem_handle) s->api.lmem_unregister(s->context, s->lmem_handle);
  if (s->token_id_handle) s->api.token_free(s->context, s->token_id_handle);
  if (s->hbm) aclrtFree(s->hbm);
  if (s->context) s->api.ctx_deinit(s->context);
  if (s->session) ReleaseDirectHccpSession(s->session);
  CloseDirectHccp(&s->api);
}
}  // namespace

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::uint16_t port = 29888U;
  std::uint32_t device_id = 0U, eid_index = 0U, npu_list_index = 0U;
  std::uint64_t bytes = 4096U;
  std::array<std::uint8_t, kEidBytes> eid{};
  bool have_eid = false;
  bool list_index_explicit = false;
  for (int i = 1; i < argc; ++i) {
    if (i + 1 >= argc) return 2;
    const std::string arg(argv[i]);
    if (arg == "--port") port = static_cast<std::uint16_t>(std::strtoul(argv[++i], nullptr, 0));
    else if (arg == "--device-id") device_id = std::strtoul(argv[++i], nullptr, 0);
    else if (arg == "--eid-index") eid_index = std::strtoul(argv[++i], nullptr, 0);
    else if (arg == "--npu-eid-list-index") { npu_list_index = std::strtoul(argv[++i], nullptr, 0); list_index_explicit = true; }
    else if (arg == "--bytes") bytes = std::strtoull(argv[++i], nullptr, 0);
    else if (arg == "--eid") have_eid = ParseEid(argv[++i], &eid);
    else return 2;
  }
  if (bytes == 0U || aclrtSetDevice(static_cast<int32_t>(device_id)) != ACL_SUCCESS) return 1;
  State state;
  if (!Init(&state, device_id, eid_index, !have_eid, list_index_explicit, npu_list_index, &eid, bytes)) { Cleanup(&state); return 1; }
  const int listener = Listen(port);
  if (listener < 0) { Cleanup(&state); return 1; }
  std::fprintf(stdout, "[HBM_TEST][NPU] LISTEN port=%u\n", port);
  const int fd = accept(listener, nullptr, nullptr);
  close(listener);
  if (fd < 0) { Cleanup(&state); return 1; }
  Hello hello{};
  bool ok = ReadAll(fd, &hello, sizeof(hello)) && hello.magic == kMagic && hello.version == kVersion;
  if (!ok) std::fprintf(stderr, "[HBM_TEST][NPU] invalid hello or client disconnected magic=0x%x version=%u\n", hello.magic, hello.version);
  Export out{};
  out.magic = kMagic; out.version = kVersion; out.hbm_bytes = bytes; out.token_value = state.token_value;
  out.segment_key_bytes = state.segment_key_bytes; out.jetty_key_bytes = state.jetty_info.key.size;
  std::memcpy(out.peer_eid, state.eid.data(), state.eid.size());
  std::memcpy(out.segment_key, state.segment_key.data(), state.segment_key_bytes);
  std::memcpy(out.jetty_key, state.jetty_info.key.value, state.jetty_info.key.size);
  ok = ok && WriteAll(fd, &out, sizeof(out));
  if (!ok) std::fprintf(stderr, "[HBM_TEST][NPU] export response failed\n");
  if (ok) {
    ChannelRequest request{};
    ok = ReadAll(fd, &request, sizeof(request)) && request.magic == kMagic && request.version == kVersion;
    if (!ok) std::fprintf(stderr, "[HBM_TEST][NPU] invalid channel request or client disconnected magic=0x%x version=%u\n", request.magic, request.version);
    ChannelResponse response{};
    if (ok) {
      GetTpCfg cfg{}; cfg.flag = CtpFlag(); cfg.transport_mode = kReliableMessage;
      std::memcpy(cfg.local_eid.raw, state.eid.data(), state.eid.size());
      std::memcpy(cfg.peer_eid.raw, request.local_eid, kEidBytes);
      TpInfo tp{}; std::uint32_t count = 1U; void* async_request = nullptr;
      const int submit_ret = state.api.get_tp_info_async(state.context, &cfg, &tp, &count, &async_request);
      ok = submit_ret == 0 && async_request != nullptr;
      if (!ok) std::fprintf(stderr, "[HBM_TEST][NPU] get_tp_info_async failed ret=%d request=%p count=%u\n", submit_ret, async_request, count);
      int result = 0;
      bool completed = false;
      for (int tries = 0; ok && tries < 10000; ++tries) {
        const int rc = state.api.get_async_req_result(async_request, &result);
        if (rc == 128301) continue;
        ok = rc == 0 && result == 0 && count != 0U && tp.tp_handle != 0U;
        completed = true;
        if (!ok) std::fprintf(stderr, "[HBM_TEST][NPU] get_tp_info_async completion failed api_ret=%d result=%d count=%u tp=0x%llx\n", rc, result, count, static_cast<unsigned long long>(tp.tp_handle));
        break;
      }
      if (ok && !completed) { ok = false; std::fprintf(stderr, "[HBM_TEST][NPU] get_tp_info_async completion timeout\n"); }
      if (request.local_key_bytes == 0U || request.local_key_bytes > kQpKeyBytes) ok = false;
      QpImportInfoT import_info{};
      void* target = nullptr;
      if (ok) {
        std::memcpy(import_info.in.key.value, request.local_key, request.local_key_bytes);
        import_info.in.key.size = request.local_key_bytes;
        import_info.in.ub.mode = kJettyImportExp;
        import_info.in.ub.token_value = state.token_value;
        import_info.in.ub.policy = kJettyGroupRoundRobin;
        import_info.in.ub.type = kTargetJetty;
        import_info.in.ub.flag = ImportFlagPlainTextToken();
        import_info.in.ub.exp_import_cfg.tp_handle = tp.tp_handle;
        import_info.in.ub.exp_import_cfg.peer_tp_handle = request.local_tp;
        import_info.in.ub.exp_import_cfg.tx_psn = 1U;
        import_info.in.ub.exp_import_cfg.rx_psn = request.local_psn;
        import_info.in.ub.tp_type = CtpProtocolType();
        const int import_ret = state.api.qp_import(state.context, &import_info, &target);
        ok = import_ret == 0 && target != nullptr;
        if (!ok) std::fprintf(stderr, "[HBM_TEST][NPU] CPU jetty import failed ret=%d target=%p key_bytes=%u local_tp=0x%llx peer_tp=0x%llx\n", import_ret, target, request.local_key_bytes, static_cast<unsigned long long>(tp.tp_handle), static_cast<unsigned long long>(request.local_tp));
        if (ok) state.imported_cpu_jetty = target;
      }
      response.status = ok ? 0 : -1; response.peer_psn = 1U; response.peer_tp = tp.tp_handle;
      response.peer_key_bytes = state.jetty_info.key.size;
      std::memcpy(response.peer_key, state.jetty_info.key.value, response.peer_key_bytes);
      ok = WriteAll(fd, &response, sizeof(response));
    }
  }
  close(fd); Cleanup(&state); return ok ? 0 : 1;
}
