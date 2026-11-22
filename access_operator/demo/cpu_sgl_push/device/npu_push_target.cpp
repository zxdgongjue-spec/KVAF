// NPU side of the CPU-SGL push demo. Registers an HBM hot buffer as a
// remotely-writable segment, exports {eid, mem_key, jetty_key, tp, psn} to a
// keyfile, then polls the HBM flag and verifies landed rows.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "acl/acl.h"
#include "direct_hccp/direct_hccp_abi.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include "../common/demo_protocol.h"

namespace hccp = kvaf::direct_hccp;
using kvaf::demo::NowNs;

// Self-contained copies of the direct-hccp helpers (same logic as
// direct_hccp_remote_transport.cc; that TU pulls in runtime headers whose
// optional gather helpers are absent from some checkouts, so the demo keeps
// its own minimal versions).
namespace {

constexpr int kHccpAsyncAgain = 128301;
constexpr uint32_t kCtpTpCount = 1U;
constexpr uint32_t kCtpQueryPollMs = 1U;
constexpr uint32_t kCtpQueryTimeoutMs = 10000U;
constexpr uint32_t kDeviceAivRqDepth = 256U;
constexpr uint32_t kAivWqeBytes = 64U;
constexpr uint32_t kAivWqebbsPerWqe = 4U;

template <typename Function>
bool LoadSymbol(void *library, const char *name, Function **function) {
  *function = reinterpret_cast<Function *>(dlsym(library, name));
  if (*function != nullptr) return true;
  std::fprintf(stderr, "[KVAF_AIV] direct ABI missing symbol %s: %s\n", name, dlerror());
  return false;
}

bool LoadDirectHccpLocal(hccp::DirectHccpApi *api) {
  if (api == nullptr) return false;
  for (const char *candidate : {"libra.so", "libra.so.0", "libhccp.so", "libhccp.so.0"}) {
    api->library = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
    if (api->library != nullptr) break;
  }
  if (api->library == nullptr) {
    std::fprintf(stderr, "[PUSH_TARGET] dlopen libra.so failed: %s\n", dlerror());
    return false;
  }
  const bool loaded =
      LoadSymbol(api->library, "RaCtxInit", &api->ctx_init) &&
      LoadSymbol(api->library, "RaCtxDeinit", &api->ctx_deinit) &&
      LoadSymbol(api->library, "RaCtxTokenIdAlloc", &api->token_alloc) &&
      LoadSymbol(api->library, "RaCtxTokenIdFree", &api->token_free) &&
      LoadSymbol(api->library, "RaCtxLmemRegister", &api->lmem_register) &&
      LoadSymbol(api->library, "RaCtxLmemUnregister", &api->lmem_unregister) &&
      LoadSymbol(api->library, "RaCtxRmemImport", &api->rmem_import) &&
      LoadSymbol(api->library, "RaCtxRmemUnimport", &api->rmem_unimport) &&
      LoadSymbol(api->library, "RaCtxCqCreate", &api->cq_create) &&
      LoadSymbol(api->library, "RaCtxCqDestroy", &api->cq_destroy) &&
      LoadSymbol(api->library, "RaCtxQpCreate", &api->qp_create) &&
      LoadSymbol(api->library, "RaCtxQpDestroy", &api->qp_destroy) &&
      LoadSymbol(api->library, "RaCtxQpImport", &api->qp_import) &&
      LoadSymbol(api->library, "RaCtxQpUnimport", &api->qp_unimport) &&
      LoadSymbol(api->library, "RaCtxGetTpInfoList", &api->get_tp_info) &&
      LoadSymbol(api->library, "RaGetTpInfoListAsync", &api->get_tp_info_async) &&
      LoadSymbol(api->library, "RaGetAsyncReqResult", &api->get_async_req_result);
  api->get_random = reinterpret_cast<hccp::GetRandomFn>(dlsym(api->library, "RaGetSecRandom"));
  if (!loaded) {
    dlclose(api->library);
    *api = {};
    return false;
  }
  return true;
}

uint32_t MakePsn(const hccp::DirectHccpApi &api, uint32_t phy_id) {
  uint32_t psn = static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count() ^
                                       static_cast<int64_t>(getpid()));
  if (psn == 0U) psn = 1U;
  if (api.get_random != nullptr) {
    hccp::RaInfo info{};
    info.mode = hccp::kOfflineHdc;
    info.phy_id = phy_id;
    if (api.get_random(&info, &psn) != 0 || psn == 0U) {
      psn = static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count() ^
                                  static_cast<int64_t>(getpid()));
      if (psn == 0U) psn = 1U;
    }
  }
  return psn;
}

bool GetPeerCtpLocal(const hccp::DirectHccpApi &api, void *context,
                     const std::array<uint8_t, hccp::kEidBytes> &local_eid,
                     const std::array<uint8_t, hccp::kEidBytes> &peer_eid, uint64_t *tp_handle) {
  hccp::GetTpCfg cfg{};
  cfg.flag = hccp::CtpFlag();
  cfg.transport_mode = hccp::kReliableMessage;
  std::memcpy(cfg.local_eid.raw, local_eid.data(), local_eid.size());
  std::memcpy(cfg.peer_eid.raw, peer_eid.data(), peer_eid.size());
  hccp::TpInfo info{};
  uint32_t count = kCtpTpCount;
  void *async_request = nullptr;
  int ret = api.get_tp_info_async(context, &cfg, &info, &count, &async_request);
  if (ret != 0 || async_request == nullptr) {
    std::fprintf(stderr, "[PUSH_TARGET] ctp async submit failed ret=%d\n", ret);
    return false;
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kCtpQueryTimeoutMs);
  while (true) {
    int request_result = 0;
    ret = api.get_async_req_result(async_request, &request_result);
    if (ret == kHccpAsyncAgain) {
      if (std::chrono::steady_clock::now() >= deadline) {
        std::fprintf(stderr, "[PUSH_TARGET] ctp async timeout\n");
        return false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(kCtpQueryPollMs));
      continue;
    }
    async_request = nullptr;
    if (ret != 0 || request_result != 0) {
      std::fprintf(stderr, "[PUSH_TARGET] ctp async complete failed ret=%d req=%d\n", ret, request_result);
      return false;
    }
    break;
  }
  if (count == 0U || info.tp_handle == 0U) {
    std::fprintf(stderr, "[PUSH_TARGET] ctp empty result\n");
    return false;
  }
  *tp_handle = info.tp_handle;
  return true;
}

bool CreateTargetJetty(const hccp::DirectHccpApi &api, void *context, void *cq_handle, void *token_handle,
                       uint32_t token_value, uint32_t sq_depth, hccp::QpCreateInfo *jetty_info,
                       void **jetty_handle) {
  const uint64_t sq_bytes = static_cast<uint64_t>(sq_depth) * kAivWqebbsPerWqe * kAivWqeBytes;
  hccp::QpCreateAttr attr{};
  attr.scq_handle = cq_handle;
  attr.rcq_handle = cq_handle;
  attr.srq_handle = cq_handle;
  attr.sq_depth = sq_depth;
  attr.rq_depth = kDeviceAivRqDepth;
  attr.transport_mode = hccp::kReliableMessage;
  attr.ub.mode = hccp::kJettyUserCtlNormal;
  attr.ub.jetty_id = 0U;
  attr.ub.flag = hccp::JettyFlagShareJfr();
  attr.ub.jfs_flag = hccp::JfsFlagErrorSuspend();
  attr.ub.token_id_handle = token_handle;
  attr.ub.token_value = token_value;
  attr.ub.priority = 15U;
  attr.ub.rnr_retry = 7U;
  attr.ub.err_timeout = 16U;
  attr.ub.ext_mode.sqebb_num = sq_depth;
  attr.ub.ext_mode.sq.buffer_size = static_cast<uint32_t>(sq_bytes);
  attr.ub.ext_mode.sq.buffer_va = 0U;
  attr.ub.ext_mode.cstm_flag = 0U;
  *jetty_info = {};
  *jetty_handle = nullptr;
  const int ret = api.qp_create(context, &attr, jetty_info, jetty_handle);
  if (ret != 0 || *jetty_handle == nullptr || jetty_info->key.size == 0U ||
      jetty_info->key.size > hccp::kQpKeyBytes) {
    std::fprintf(stderr, "[PUSH_TARGET] qp_create failed ret=%d key_bytes=%u\n", ret, jetty_info->key.size);
    return false;
  }
  return true;
}

}  // namespace

namespace {

constexpr uint32_t kPidHdcType = 18U;
constexpr uint32_t kCqDepth = 1024U;
constexpr uint32_t kSqDepth = 128U;
constexpr uint64_t kDefaultHotBytes = 64ULL << 20;

struct Args {
  uint32_t device_id = 0;
  std::string local_eid_hex;
  std::string peer_eid_hex;
  std::string keyfile;
  uint64_t hot_bytes = kDefaultHotBytes;
  uint32_t poll_timeout_s = 600;
  const char *csv = nullptr;
};

bool ParseHexEid(const std::string &text, std::array<uint8_t, hccp::kEidBytes> *eid) {
  if (text.size() != 32) return false;
  return kvaf::demo::ReadHex(text.c_str(), eid->data(), hccp::kEidBytes);
}

Args ParseArgs(int argc, char **argv) {
  Args args;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key = argv[i];
    const std::string value = argv[i + 1];
    if (key == "--device-id") args.device_id = std::stoul(value);
    else if (key == "--local-eid") args.local_eid_hex = value;
    else if (key == "--peer-eid") args.peer_eid_hex = value;
    else if (key == "--keyfile") args.keyfile = value;
    else if (key == "--hot-bytes") args.hot_bytes = std::stoull(value);
    else if (key == "--poll-timeout-s") args.poll_timeout_s = std::stoul(value);
    else if (key == "--csv") args.csv = value.c_str();
    else { std::fprintf(stderr, "unknown arg %s\n", key.c_str()); std::exit(2); }
  }
  if (args.local_eid_hex.empty() || args.peer_eid_hex.empty() || args.keyfile.empty()) {
    std::fprintf(stderr, "need --local-eid --peer-eid --keyfile\n");
    std::exit(2);
  }
  return args;
}

#define CHECK_ACL(call)                                                     \
  do {                                                                      \
    aclError rc_ = (call);                                                  \
    if (rc_ != ACL_SUCCESS) {                                               \
      std::fprintf(stderr, "[PUSH_TARGET] %s failed rc=%d\n", #call, rc_);  \
      return 1;                                                             \
    }                                                                       \
  } while (0)

}  // namespace

int main(int argc, char **argv) {
  const Args args = ParseArgs(argc, argv);
  std::array<uint8_t, hccp::kEidBytes> local_eid{}, peer_eid{};
  if (!ParseHexEid(args.local_eid_hex, &local_eid) || !ParseHexEid(args.peer_eid_hex, &peer_eid)) {
    std::fprintf(stderr, "[PUSH_TARGET] bad eid hex\n");
    return 2;
  }

  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(static_cast<int32_t>(args.device_id)));

  hccp::DirectHccpApi api{};
  if (!LoadDirectHccpLocal(&api)) {
    std::fprintf(stderr, "[PUSH_TARGET] LoadDirectHccp failed\n");
    return 1;
  }
  hccp::DirectHccpSession *session = nullptr;
  if (!hccp::AcquireDirectHccpSession(args.device_id, kPidHdcType, &session)) {
    std::fprintf(stderr, "[PUSH_TARGET] AcquireDirectHccpSession failed\n");
    return 1;
  }

  // Device-side RA contexts use the HDC mode (peer-online is the CPU-side
  // resource-agent mode), matching KVAF's BatchGet transport.
  hccp::CtxInitCfg cfg{};
  cfg.mode = hccp::kOfflineHdc;
  hccp::CtxInitAttr attr{};
  attr.phy_id = args.device_id;
  attr.ub.eid_index = 0U;
  std::memcpy(attr.ub.eid.raw, local_eid.data(), local_eid.size());
  void *ctx = nullptr;
  if (api.ctx_init(&cfg, &attr, &ctx) != 0 || ctx == nullptr) {
    std::fprintf(stderr, "[PUSH_TARGET] ctx_init failed\n");
    return 1;
  }
  void *token_handle = nullptr;
  hccp::TokenId raw_token{};
  if (api.token_alloc(ctx, &raw_token, &token_handle) != 0 || token_handle == nullptr) {
    std::fprintf(stderr, "[PUSH_TARGET] token_alloc failed\n");
    return 1;
  }
  const uint32_t token_value = MakePsn(api, args.device_id);

  // HBM buffer: [flag 4K][data area].
  void *hot = nullptr;
  CHECK_ACL(aclrtMalloc(&hot, args.hot_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemset(hot, args.hot_bytes, 0, args.hot_bytes));

  hccp::MrRegInfo reg{};
  reg.in.mem.addr = reinterpret_cast<uint64_t>(hot);
  reg.in.mem.size = args.hot_bytes;
  reg.in.ub.flags = hccp::MemRegFlags(true);
  reg.in.ub.token_value = token_value;
  reg.in.ub.token_id_handle = token_handle;
  void *mr_handle = nullptr;
  if (api.lmem_register(ctx, &reg, &mr_handle) != 0 || mr_handle == nullptr || reg.out.key.size == 0U) {
    std::fprintf(stderr, "[PUSH_TARGET] lmem_register failed\n");
    return 1;
  }
  std::fprintf(stderr, "[PUSH_TARGET] HBM registered addr=%p bytes=%llu key_bytes=%u\n", hot,
               static_cast<unsigned long long>(args.hot_bytes), reg.out.key.size);

  // Target jetty: the CPU imports it to aim its WRITE WRs at this endpoint.
  hccp::CqInfo cq{};
  cq.in.channel_handle = nullptr;
  cq.in.depth = kCqDepth;
  cq.in.ub.mode = hccp::kJfcNormal;
  void *cq_handle = nullptr;
  if (api.cq_create(ctx, &cq, &cq_handle) != 0 || cq_handle == nullptr) {
    std::fprintf(stderr, "[PUSH_TARGET] cq_create failed\n");
    return 1;
  }
  hccp::QpCreateInfo jetty_info{};
  void *jetty_handle = nullptr;
  if (!CreateTargetJetty(api, ctx, cq_handle, token_handle, token_value, kSqDepth, &jetty_info, &jetty_handle)) {
    std::fprintf(stderr, "[PUSH_TARGET] CreateDeviceAivJetty failed\n");
    return 1;
  }

  uint64_t tp_handle = 0;
  if (!GetPeerCtpLocal(api, ctx, local_eid, peer_eid, &tp_handle)) {
    std::fprintf(stderr, "[PUSH_TARGET] GetPeerCtp failed\n");
    return 1;
  }
  const uint32_t psn = MakePsn(api, args.device_id);

  {
    FILE *kf = std::fopen(args.keyfile.c_str(), "w");
    if (kf == nullptr) {
      std::fprintf(stderr, "[PUSH_TARGET] cannot write keyfile %s\n", args.keyfile.c_str());
      return 1;
    }
    kvaf::demo::WriteHex(kf, "npu_eid", local_eid.data(), hccp::kEidBytes);
    kvaf::demo::WriteHex(kf, "mem_key", reg.out.key.value, reg.out.key.size);
    kvaf::demo::WriteHex(kf, "jetty_key", jetty_info.key.value, jetty_info.key.size);
    std::fprintf(kf, "jetty_key_bytes=%u\n", jetty_info.key.size);
    std::fprintf(kf, "mem_token_value=%u\n", token_value);
    std::fprintf(kf, "jetty_token_value=%u\n", token_value);
    std::fprintf(kf, "npu_tp_handle=%llu\n", static_cast<unsigned long long>(tp_handle));
    std::fprintf(kf, "npu_psn=%u\n", psn);
    std::fprintf(kf, "hbm_addr=%llu\n", static_cast<unsigned long long>(reinterpret_cast<uint64_t>(hot)));
    std::fprintf(kf, "hbm_bytes=%llu\n", static_cast<unsigned long long>(args.hot_bytes));
    std::fprintf(kf, "flag_offset=0\n");
    std::fprintf(kf, "data_offset=%llu\n", static_cast<unsigned long long>(kvaf::demo::kFlagBytes));
    std::fclose(kf);
  }
  std::fprintf(stderr, "[PUSH_TARGET] KEYFILE_READY %s\n", args.keyfile.c_str());

  // Poll the flag. Every new seq gets a CLOCK_MONOTONIC stamp (same host clock
  // as the CPU pusher).
  void *flag_host = nullptr;
  CHECK_ACL(aclrtMallocHost(&flag_host, 8));
  uint64_t last_seq = 0;
  std::vector<std::pair<uint64_t, uint64_t>> stamps;
  const uint64_t deadline = NowNs() + uint64_t(args.poll_timeout_s) * 1000000000ULL;
  while (NowNs() < deadline) {
    const aclError rc = aclrtMemcpy(flag_host, 8, hot, 8, ACL_MEMCPY_DEVICE_TO_HOST);
    if (rc != ACL_SUCCESS) {
      std::fprintf(stderr, "[PUSH_TARGET] flag memcpy rc=%d\n", rc);
      break;
    }
    uint64_t seq = 0;
    std::memcpy(&seq, flag_host, 8);
    if (seq != 0 && seq != last_seq) {
      stamps.emplace_back(seq, NowNs());
      last_seq = seq;
      if (seq == kvaf::demo::kStopSeq) break;
    }
  }
  std::fprintf(stderr, "[PUSH_TARGET] poll done, %zu seqs\n", stamps.size());

  // Verify the data area row-by-row until the first unwritten slot.
  const uint32_t token_bytes = [] {
    const char *env = std::getenv("SGL_DEMO_TOKEN_BYTES");
    return env ? uint32_t(std::stoul(env)) : 656U;
  }();
  const uint64_t data_bytes = args.hot_bytes - kvaf::demo::kFlagBytes;
  const uint64_t rows = data_bytes / token_bytes;
  std::vector<uint8_t> chunk(size_t(4) << 20);
  void *verify_host = nullptr;
  CHECK_ACL(aclrtMallocHost(&verify_host, chunk.size()));
  uint64_t verified = 0, bad = 0;
  for (uint64_t base = 0; base < rows * token_bytes; base += chunk.size()) {
    const uint64_t bytes = std::min<uint64_t>(chunk.size(), rows * token_bytes - base);
    CHECK_ACL(aclrtMemcpy(verify_host, bytes, static_cast<uint8_t *>(hot) + kvaf::demo::kFlagBytes + base, bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    for (uint64_t off = 0; off + token_bytes <= bytes; off += token_bytes) {
      const uint64_t row_index = (base + off) / token_bytes;
      const uint8_t *row = static_cast<const uint8_t *>(verify_host) + off;
      uint64_t first_word = 0;
      std::memcpy(&first_word, row, 8);
      if (first_word == 0) { base = rows * token_bytes; break; }  // unwritten tail
      if (kvaf::demo::VerifyRow(row, token_bytes) == ~0ULL) {
        if (bad < 8) std::fprintf(stderr, "[PUSH_TARGET] VERIFY_BAD row=%llu\n",
                                  static_cast<unsigned long long>(row_index));
        ++bad;
      } else {
        ++verified;
      }
    }
  }
  std::fprintf(stderr, "[PUSH_TARGET] VERIFY_OK rows=%llu bad=%llu\n",
               static_cast<unsigned long long>(verified), static_cast<unsigned long long>(bad));

  if (args.csv != nullptr) {
    FILE *out = std::fopen(args.csv, "w");
    if (out != nullptr) {
      std::fprintf(out, "seq,t_flag_seen_ns\n");
      for (const auto &[seq, t] : stamps) std::fprintf(out, "%llu,%llu\n",
          static_cast<unsigned long long>(seq), static_cast<unsigned long long>(t));
      std::fclose(out);
    }
  }

  aclrtFreeHost(flag_host);
  aclrtFreeHost(verify_host);
  aclrtFree(hot);
  if (api.qp_destroy != nullptr && jetty_handle != nullptr) (void)api.qp_destroy(jetty_handle);
  if (api.cq_destroy != nullptr && cq_handle != nullptr) (void)api.cq_destroy(ctx, cq_handle);
  if (api.lmem_unregister != nullptr && mr_handle != nullptr) (void)api.lmem_unregister(ctx, mr_handle);
  if (api.token_free != nullptr && token_handle != nullptr) (void)api.token_free(ctx, token_handle);
  if (api.ctx_deinit != nullptr && ctx != nullptr) (void)api.ctx_deinit(ctx);
  hccp::ReleaseDirectHccpSession(session);
  aclrtResetDevice(static_cast<int32_t>(args.device_id));
  aclFinalize();
  std::fprintf(stderr, "[PUSH_TARGET] EXIT bad=%llu\n", static_cast<unsigned long long>(bad));
  return bad == 0 ? 0 : 3;
}
