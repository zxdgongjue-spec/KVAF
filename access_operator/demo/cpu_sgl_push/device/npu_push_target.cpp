// NPU side of the CPU-SGL push demo. Registers an HBM hot buffer as a
// remotely-writable segment, exports {eid, mem_key, jetty_key, tp, psn} to a
// keyfile, then polls the HBM flag and verifies landed rows.
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "direct_hccp/direct_hccp_abi.h"
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include "../common/demo_protocol.h"

namespace hccp = kvaf::direct_hccp;
using kvaf::demo::NowNs;

// Defined in direct_hccp_remote_transport.cc (non-static, no header decl).
namespace kvaf::direct_hccp {
std::uint32_t MakePeerCtpPsn(const DirectHccpApi& api, std::uint32_t physical_device_id, const char* role);
bool GetPeerCtp(const DirectHccpApi& api, void* context, const std::array<std::uint8_t, kEidBytes>& local_eid,
                const std::array<std::uint8_t, kEidBytes>& peer_eid, const char* role, std::uint64_t* tp_handle);
}  // namespace kvaf::direct_hccp

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
  if (!hccp::LoadDirectHccp(&api)) {
    std::fprintf(stderr, "[PUSH_TARGET] LoadDirectHccp failed\n");
    return 1;
  }
  hccp::DirectHccpSession *session = nullptr;
  if (!hccp::AcquireDirectHccpSession(args.device_id, kPidHdcType, &session)) {
    std::fprintf(stderr, "[PUSH_TARGET] AcquireDirectHccpSession failed\n");
    return 1;
  }

  // RA context bound to the local EID.
  hccp::CtxInitCfg cfg{};
  cfg.mode = hccp::kPeerOnline;
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
  const uint32_t token_value = hccp::MakePeerCtpPsn(api, args.device_id, "push_target");

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
  if (!hccp::CreateDeviceAivJetty(api, ctx, cq_handle, token_handle, token_value, kSqDepth, "push_target",
                                  &jetty_info, &jetty_handle)) {
    std::fprintf(stderr, "[PUSH_TARGET] CreateDeviceAivJetty failed\n");
    return 1;
  }

  uint64_t tp_handle = 0;
  if (!hccp::GetPeerCtp(api, ctx, local_eid, peer_eid, "push_target", &tp_handle)) {
    std::fprintf(stderr, "[PUSH_TARGET] GetPeerCtp failed\n");
    return 1;
  }
  const uint32_t psn = hccp::MakePeerCtpPsn(api, args.device_id, "push_target_psn");

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
  if (api.cq_destroy != nullptr && cq_handle != nullptr) (void)api.cq_destroy(cq_handle);
  if (api.lmem_unregister != nullptr && mr_handle != nullptr) (void)api.lmem_unregister(ctx, mr_handle);
  if (api.token_free != nullptr && token_handle != nullptr) (void)api.token_free(ctx, token_handle);
  if (api.ctx_deinit != nullptr && ctx != nullptr) (void)api.ctx_deinit(ctx);
  hccp::ReleaseDirectHccpSession(session);
  aclrtResetDevice(static_cast<int32_t>(args.device_id));
  aclFinalize();
  std::fprintf(stderr, "[PUSH_TARGET] EXIT bad=%llu\n", static_cast<unsigned long long>(bad));
  return bad == 0 ? 0 : 3;
}
