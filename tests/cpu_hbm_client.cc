#include "hbm_rw_protocol.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <urma_api.h>

namespace {
using namespace kvaf::direct_hccp;
using namespace kvaf::hbm_test;
using namespace kvaf::peer;

struct ImportFlag { std::uint64_t value, stag, dtag; };
struct RemoteJettyEx { urma_jetty_id_t jetty_id; std::uint32_t trans_mode, policy, type, reserved; ImportFlag flag; std::uint32_t tp_type, reserved2; };
struct TpQuery { std::uint32_t flag, trans_mode; std::uint8_t local_eid[16], peer_eid[16]; };
struct LocalTpInfo { std::uint64_t tp_handle; };
struct ActiveTp { std::uint64_t tp_handle, peer_tp_handle, tag; std::uint32_t tx_psn, rx_psn; std::uint64_t reserved; };
using GetTpList = int (*)(urma_context_t*, TpQuery*, std::uint32_t*, LocalTpInfo*);
using ImportJettyEx = urma_target_jetty_t* (*)(urma_context_t*, RemoteJettyEx*, urma_token_t*, ActiveTp*);

struct Block { std::uint8_t* data = nullptr; urma_target_seg_t* seg = nullptr; };
struct State {
  urma_context_t* ctx = nullptr; urma_jfc_t* jfc = nullptr; urma_jfr_t* jfr = nullptr;
  urma_token_id_t* token = nullptr; urma_target_seg_t* remote_seg = nullptr;
  urma_jetty_t* jetty = nullptr; urma_target_jetty_t* remote_jetty = nullptr;
  std::uint64_t remote_va = 0; std::vector<Block> blocks;
};

void Cleanup(State* s) {
  if (s->remote_jetty) urma_unimport_jetty(s->remote_jetty);
  if (s->jetty) urma_delete_jetty(s->jetty);
  if (s->jfr) urma_delete_jfr(s->jfr);
  if (s->remote_seg) urma_unimport_seg(s->remote_seg);
  for (auto& block : s->blocks) {
    if (block.seg) urma_unregister_seg(block.seg);
    std::free(block.data);
  }
  if (s->token) urma_free_token_id(s->token);
  if (s->jfc) urma_delete_jfc(s->jfc);
  if (s->ctx) urma_delete_context(s->ctx);
  s->blocks.clear();
}

bool ReadAll(int fd, void* data, std::size_t bytes) {
  auto* p = static_cast<std::uint8_t*>(data);
  for (std::size_t n = 0; n < bytes;) { const ssize_t r = recv(fd, p + n, bytes - n, 0); if (r <= 0) return false; n += static_cast<std::size_t>(r); }
  return true;
}
bool WriteAll(int fd, const void* data, std::size_t bytes) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t n = 0; n < bytes;) { const ssize_t r = send(fd, p + n, bytes - n, MSG_NOSIGNAL); if (r <= 0) return false; n += static_cast<std::size_t>(r); }
  return true;
}
int Connect(const std::string& host, std::uint16_t port) {
  addrinfo hints{}; hints.ai_socktype = SOCK_STREAM; hints.ai_family = AF_UNSPEC; addrinfo* list = nullptr;
  const int gai = getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &list);
  if (gai != 0) {
    std::fprintf(stderr, "[HBM_TEST][CPU] getaddrinfo(%s:%u) failed: %s\n", host.c_str(), port, gai_strerror(gai));
    return -1;
  }
  int fd = -1; int last_errno = 0;
  for (auto* a = list; a; a = a->ai_next) { fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol); if (fd >= 0 && connect(fd, a->ai_addr, a->ai_addrlen) == 0) break; last_errno = errno; if (fd >= 0) close(fd); fd = -1; }
  freeaddrinfo(list); if (fd < 0) std::fprintf(stderr, "[HBM_TEST][CPU] connect failed: %s\n", std::strerror(last_errno));
  if (fd >= 0) { timeval timeout{30, 0}; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)); setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)); }
  return fd;
}
int EidIndex(urma_device_t* device, const std::array<std::uint8_t, 16>& wanted) {
  std::uint32_t count = 0; auto* list = urma_get_eid_list(device, &count); if (!list) return -1; int result = -1;
  for (std::uint32_t i = 0; i < count; ++i) if (std::memcmp(list[i].eid.raw, wanted.data(), 16) == 0) { result = static_cast<int>(list[i].eid_index); break; }
  urma_free_eid_list(list); return result;
}
bool PairInfoLocalEid(std::uint32_t physical, std::array<std::uint8_t, 16>* eid) {
  if (!eid || access("/proc/asdrv_ub/dev_id", W_OK) != 0 || access("/proc/asdrv_ub/pair_info", R_OK) != 0) return false;
  const int lock_fd = open("/tmp/kvaf_hbm_eid.lock", O_CREAT | O_RDWR, 0600); if (lock_fd < 0 || flock(lock_fd, LOCK_EX) != 0) return false;
  bool found = false; std::ofstream selector("/proc/asdrv_ub/dev_id");
  if (selector.is_open()) { selector << physical << '\n'; selector.close(); std::ifstream input("/proc/asdrv_ub/pair_info"); std::string line;
    while (std::getline(input, line)) { const auto p = line.find("local_eid:"); if (p == std::string::npos) continue; std::string hex; for (unsigned char c : line.substr(p + 10U)) if (std::isxdigit(c)) hex.push_back(static_cast<char>(c)); if (hex.size() != 32) continue; auto h = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }; for (int i = 0; i < 16; ++i) (*eid)[i] = static_cast<std::uint8_t>((h(hex[2 * i]) << 4) | h(hex[2 * i + 1])); found = true; break; }
  }
  (void)flock(lock_fd, LOCK_UN); close(lock_fd); return found;
}

bool Transfer(State* s, bool write, std::size_t index, std::uint64_t offset, std::size_t bytes, std::uint64_t user_ctx) {
  urma_sge_t local{}; local.addr = reinterpret_cast<std::uint64_t>(s->blocks[index].data); local.len = static_cast<std::uint32_t>(bytes); local.tseg = s->blocks[index].seg;
  urma_sge_t remote{}; remote.addr = s->remote_va + offset; remote.len = static_cast<std::uint32_t>(bytes); remote.tseg = s->remote_seg;
  urma_jfs_wr_t wr{}; wr.opcode = write ? URMA_OPC_WRITE : URMA_OPC_READ; wr.flag.bs.complete_enable = 1; wr.tjetty = s->remote_jetty; wr.user_ctx = user_ctx;
  if (write) { wr.rw.src = {&local, 1}; wr.rw.dst = {&remote, 1}; } else { wr.rw.src = {&remote, 1}; wr.rw.dst = {&local, 1}; }
  urma_jfs_wr_t* bad = nullptr; const auto post_ret = urma_post_jetty_send_wr(s->jetty, &wr, &bad); if (post_ret != URMA_SUCCESS) { std::fprintf(stderr, "[HBM_TEST][CPU] transfer post failed op=%s index=%zu ret=%d bad=%p\n", write ? "WRITE" : "READ", index, static_cast<int>(post_ret), static_cast<void*>(bad)); return false; }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) { urma_cr_t cr{}; const int n = urma_poll_jfc(s->jfc, 1, &cr); if (n < 0) { std::fprintf(stderr, "[HBM_TEST][CPU] transfer CQ poll failed op=%s index=%zu\n", write ? "WRITE" : "READ", index); return false; } if (n == 1) { if (cr.status != URMA_CR_SUCCESS || cr.user_ctx != user_ctx) std::fprintf(stderr, "[HBM_TEST][CPU] transfer completion failed op=%s index=%zu status=%d ctx=0x%llx\n", write ? "WRITE" : "READ", index, static_cast<int>(cr.status), static_cast<unsigned long long>(cr.user_ctx)); return cr.status == URMA_CR_SUCCESS && cr.user_ctx == user_ctx; } std::this_thread::yield(); }
  std::fprintf(stderr, "[HBM_TEST][CPU] transfer CQ timeout op=%s index=%zu\n", write ? "WRITE" : "READ", index);
  return false;
}

struct Stats { std::uint64_t ops = 0, bytes = 0, sum_ns = 0, min_ns = UINT64_MAX, max_ns = 0; double wall_ns = 0; };
void LogFail(const char* stage) { std::fprintf(stderr, "[HBM_TEST][CPU] FAILED stage=%s errno=%d (%s)\n", stage, errno, std::strerror(errno)); }
bool RunPass(State* s, bool write, const std::vector<std::size_t>& order, std::size_t block_bytes, std::size_t iterations, Stats* stats) {
  const auto wall_start = std::chrono::steady_clock::now();
  for (std::size_t it = 0; it < iterations; ++it) for (const std::size_t index : order) {
    const auto start = std::chrono::steady_clock::now();
    if (!Transfer(s, write, index, index * block_bytes, block_bytes, index + 1U)) return false;
    const auto ns = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
    ++stats->ops; stats->bytes += block_bytes; stats->sum_ns += ns; stats->min_ns = std::min(stats->min_ns, ns); stats->max_ns = std::max(stats->max_ns, ns);
  }
  stats->wall_ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - wall_start).count()); return true;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  std::string host = argv[1]; std::uint16_t port = 29888; std::size_t blocks = 1, block_bytes = 656, iterations = 1, warmup = 1; std::uint64_t seed = 0x6b766166ULL; std::array<std::uint8_t, 16> wanted{}; bool have = false; bool local_physical_explicit = false; std::uint32_t eid_index = 0, physical = 0, remote_device_id = 0;
  int arg_start = 2; if (arg_start < argc && argv[arg_start][0] != '-') port = static_cast<std::uint16_t>(std::strtoul(argv[arg_start++], nullptr, 0));
  else {
    const std::size_t colon = host.rfind(':');
    if (colon != std::string::npos && colon != 0U && colon + 1U < host.size()) {
      const std::string port_text = host.substr(colon + 1U);
      char* end = nullptr; const unsigned long parsed = std::strtoul(port_text.c_str(), &end, 10);
      if (end != port_text.c_str() && *end == '\0' && parsed <= UINT16_MAX) {
        port = static_cast<std::uint16_t>(parsed); host.resize(colon);
      }
    }
  }
  for (int i = arg_start; i < argc; ++i) { const std::string a = argv[i]; if (i + 1 >= argc) return 2;
    if (a == "--blocks") blocks = std::strtoull(argv[++i], nullptr, 0); else if (a == "--block-bytes") block_bytes = std::strtoull(argv[++i], nullptr, 0); else if (a == "--iterations") iterations = std::strtoull(argv[++i], nullptr, 0); else if (a == "--warmup") warmup = std::strtoull(argv[++i], nullptr, 0); else if (a == "--seed") seed = std::strtoull(argv[++i], nullptr, 0); else if (a == "--bytes") { block_bytes = std::strtoull(argv[++i], nullptr, 0); blocks = 1; } else if (a == "--device-id") remote_device_id = std::strtoul(argv[++i], nullptr, 0); else if (a == "--local-physical") { physical = std::strtoul(argv[++i], nullptr, 0); local_physical_explicit = true; } else if (a == "--eid-index") eid_index = std::strtoul(argv[++i], nullptr, 0);
    else if (a == "--host-eid") { const char* t = argv[++i]; if (std::strlen(t) != 32) return 2; auto h = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }; for (int j = 0; j < 16; ++j) { const int x = h(t[2 * j]), y = h(t[2 * j + 1]); if (x < 0 || y < 0) return 2; wanted[j] = static_cast<std::uint8_t>((x << 4) | y); } have = true; } else return 2;
  }
  if (blocks == 0 || block_bytes == 0 || iterations == 0 || blocks > UINT64_MAX / block_bytes || block_bytes > UINT32_MAX) return 2;
  if (!local_physical_explicit) physical = remote_device_id;
  const std::uint64_t total_bytes = static_cast<std::uint64_t>(blocks * block_bytes);
  std::fprintf(stderr, "[HBM_TEST][CPU] START host=%s port=%u blocks=%zu block_bytes=%zu total_bytes=%llu iterations=%zu warmup=%zu remote_device=%u local_physical=%u\n", host.c_str(), port, blocks, block_bytes, static_cast<unsigned long long>(total_bytes), iterations, warmup, remote_device_id, physical);
  urma_init_attr_t init{}; if (urma_init(&init) != URMA_SUCCESS) { LogFail("urma_init"); return 1; } if (!have && !PairInfoLocalEid(physical, &wanted)) { LogFail("local_eid_discovery"); return 1; }
  const int fd = Connect(host, port); if (fd < 0) { LogFail("connect"); return 1; }
  AllocateHbmRegionRequest alloc{}; alloc.header = MakeHeader(Command::kAllocateHbmRegion, sizeof(alloc)); alloc.hbm_bytes = total_bytes; alloc.slot_size = block_bytes; alloc.device_id = remote_device_id; alloc.eid_index = eid_index;
  if (!WriteAll(fd, &alloc, sizeof(alloc))) { LogFail("allocate_request_send"); return 1; }
  AllocateHbmRegionResponse ex{}; if (!ReadAll(fd, &ex, sizeof(ex))) { LogFail("allocate_response_read"); return 1; } if (!IsValidResponseHeader(ex.header, Command::kAllocateHbmRegion, sizeof(ex)) || ex.result != static_cast<std::int32_t>(Result::kOk) || ex.hbm_bytes != total_bytes) { std::fprintf(stderr, "[HBM_TEST][CPU] FAILED stage=allocate_response result=%d hbm_bytes=%llu expected=%llu\n", ex.result, static_cast<unsigned long long>(ex.hbm_bytes), static_cast<unsigned long long>(total_bytes)); return 1; }
  std::fprintf(stderr, "[HBM_TEST][CPU] ALLOCATED remote_device=%u bytes=%llu\n", remote_device_id, static_cast<unsigned long long>(ex.hbm_bytes));
  urma_eid_t host_eid{}; std::memcpy(host_eid.raw, wanted.data(), wanted.size()); urma_device_t* dev = urma_get_device_by_eid(host_eid, URMA_TRANSPORT_UB); if (!dev) { LogFail("local_urma_device"); return 1; } int idx = EidIndex(dev, wanted); if (idx < 0) idx = static_cast<int>(eid_index);
  State s; s.ctx = urma_create_context(dev, idx); if (!s.ctx) { LogFail("local_context"); return 1; } urma_jfc_cfg_t jc{}; jc.depth = 1024; s.jfc = urma_create_jfc(s.ctx, &jc); if (!s.jfc) { LogFail("local_jfc"); Cleanup(&s); return 1; } s.token = urma_alloc_token_id(s.ctx); if (!s.token) { LogFail("local_token"); Cleanup(&s); return 1; }
  s.blocks.resize(blocks); for (auto& block : s.blocks) { if (posix_memalign(reinterpret_cast<void**>(&block.data), 4096, block_bytes) != 0) { LogFail("host_alloc"); Cleanup(&s); return 1; } urma_seg_cfg_t lc{}; lc.va = reinterpret_cast<std::uint64_t>(block.data); lc.len = block_bytes; lc.token_id = s.token; lc.token_value.token = 0xCAFE; lc.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT; lc.flag.bs.access = URMA_ACCESS_REMOTE_READ | URMA_ACCESS_REMOTE_WRITE | URMA_ACCESS_REMOTE_ATOMIC; lc.flag.bs.non_pin = 1; lc.flag.bs.token_id_valid = 1; block.seg = urma_register_seg(s.ctx, &lc); if (!block.seg) { LogFail("host_segment_register"); Cleanup(&s); return 1; } }
  urma_seg_t remote_desc{}; if (ex.segment_key_bytes < sizeof(remote_desc) || ex.segment_key_bytes > sizeof(ex.segment_key)) { LogFail("remote_descriptor"); Cleanup(&s); return 1; } std::memcpy(&remote_desc, ex.segment_key, sizeof(remote_desc)); s.remote_va = remote_desc.ubva.va; urma_token_t rt{}; rt.token = ex.token_value; urma_import_seg_flag_t rif{}; rif.bs.access = URMA_ACCESS_REMOTE_READ | URMA_ACCESS_REMOTE_WRITE; rif.bs.mapping = URMA_SEG_NOMAP; s.remote_seg = urma_import_seg(s.ctx, &remote_desc, &rt, 0, rif); if (!s.remote_seg) { LogFail("remote_segment_import"); Cleanup(&s); return 1; } std::fprintf(stderr, "[HBM_TEST][CPU] REMOTE_SEGMENT_READY va=0x%llx\n", static_cast<unsigned long long>(s.remote_va));
  urma_jfs_cfg_t jfs{}; jfs.depth = 1024; jfs.trans_mode = URMA_TM_RM; jfs.max_sge = 1; jfs.jfc = s.jfc; urma_jfr_cfg_t jfrc{}; jfrc.depth = 1024; jfrc.trans_mode = URMA_TM_RM; jfrc.max_sge = 1; jfrc.jfc = s.jfc; jfrc.token_value.token = ex.token_value; s.jfr = urma_create_jfr(s.ctx, &jfrc); if (!s.jfr) { LogFail("local_jfr"); Cleanup(&s); return 1; } urma_jetty_cfg_t cfg{}; cfg.flag.bs.share_jfr = 1; cfg.jfs_cfg = jfs; cfg.shared.jfr = s.jfr; cfg.shared.jfc = s.jfc; s.jetty = urma_create_jetty(s.ctx, &cfg); if (!s.jetty) { LogFail("local_jetty"); Cleanup(&s); return 1; }
  auto get_tp = reinterpret_cast<GetTpList>(dlsym(RTLD_DEFAULT, "urma_get_tp_list")); auto import_ex = reinterpret_cast<ImportJettyEx>(dlsym(RTLD_DEFAULT, "urma_import_jetty_ex")); if (!get_tp || !import_ex) { LogFail("urma_symbols"); Cleanup(&s); return 1; } TpQuery q{}; q.flag = 1; q.trans_mode = URMA_TM_RM; std::memcpy(q.local_eid, s.ctx->eid.raw, 16); std::memcpy(q.peer_eid, ex.peer_eid, 16); LocalTpInfo tp{}; std::uint32_t count = 1; if (get_tp(s.ctx, &q, &count, &tp) != 0 || !tp.tp_handle) { LogFail("ctp_lookup"); Cleanup(&s); return 1; }
  ChannelRequest cr{}; cr.magic = kMagic; cr.version = kVersion; cr.local_tp = tp.tp_handle; cr.local_psn = 1; cr.local_key_bytes = sizeof(urma_jetty_id_t) + sizeof(urma_transport_mode_t); std::memcpy(cr.local_eid, s.ctx->eid.raw, 16); std::memcpy(cr.local_key, &s.jetty->jetty_id, sizeof(s.jetty->jetty_id)); auto mode = s.jetty->jetty_cfg.jfs_cfg.trans_mode; std::memcpy(cr.local_key + sizeof(s.jetty->jetty_id), &mode, sizeof(mode)); if (!WriteAll(fd, &cr, sizeof(cr))) { LogFail("channel_request_send"); Cleanup(&s); return 1; } ChannelResponse resp{}; if (!ReadAll(fd, &resp, sizeof(resp)) || resp.status != 0) { LogFail("channel_response"); Cleanup(&s); return 1; }
  RemoteJettyEx rj{}; std::memcpy(&rj.jetty_id, resp.peer_key, sizeof(rj.jetty_id)); std::memcpy(&rj.trans_mode, resp.peer_key + sizeof(rj.jetty_id), sizeof(rj.trans_mode)); rj.type = URMA_JETTY; rj.flag.value = 1; rj.tp_type = 1; ActiveTp active{tp.tp_handle, resp.peer_tp, 0, 1, resp.peer_psn, 0}; urma_token_t pt{}; pt.token = ex.token_value; s.remote_jetty = import_ex(s.ctx, &rj, &pt, &active); if (!s.remote_jetty) { LogFail("remote_jetty_import"); Cleanup(&s); return 1; } std::fprintf(stderr, "[HBM_TEST][CPU] CHANNEL_READY\n");
  std::mt19937_64 rng(seed); std::vector<std::vector<std::uint8_t>> expected(blocks, std::vector<std::uint8_t>(block_bytes)); for (auto& block : expected) for (auto& byte : block) byte = static_cast<std::uint8_t>(rng()); for (std::size_t i = 0; i < blocks; ++i) std::memcpy(s.blocks[i].data, expected[i].data(), block_bytes); std::vector<std::size_t> order(blocks); for (std::size_t i = 0; i < blocks; ++i) order[i] = i; std::shuffle(order.begin(), order.end(), rng);
  Stats warmup_stats; if (warmup && !RunPass(&s, true, order, block_bytes, warmup, &warmup_stats)) { LogFail("warmup_write"); Cleanup(&s); return 1; } Stats write_stats, read_stats; if (!RunPass(&s, true, order, block_bytes, iterations, &write_stats)) { LogFail("write_pass"); Cleanup(&s); return 1; } for (auto& block : s.blocks) std::memset(block.data, 0, block_bytes); if (!RunPass(&s, false, order, block_bytes, iterations, &read_stats)) { LogFail("read_pass"); Cleanup(&s); return 1; }
  bool ok = true; for (std::size_t i = 0; i < blocks; ++i) ok = ok && std::memcmp(s.blocks[i].data, expected[i].data(), block_bytes) == 0;
  const auto print_stats = [](const char* name, const Stats& st) { const double bw = st.bytes * 1e9 / st.wall_ns / (1024.0 * 1024.0); std::printf("[HBM_TEST][CPU] %-5s bytes=%llu ops=%llu bandwidth=%.2f MiB/s avg_latency=%.3f us min=%.3f us max=%.3f us\n", name, static_cast<unsigned long long>(st.bytes), static_cast<unsigned long long>(st.ops), bw, st.sum_ns / static_cast<double>(st.ops) / 1000.0, st.min_ns / 1000.0, st.max_ns / 1000.0); };
  print_stats("WRITE", write_stats); print_stats("READ", read_stats); std::printf("[HBM_TEST][CPU] %s blocks=%zu block_bytes=%zu seed=%llu\n", ok ? "PASS" : "FAIL", blocks, block_bytes, static_cast<unsigned long long>(seed));
  Done done{kMagic, kVersion, static_cast<std::uint16_t>(ok ? 0 : 1)};
  (void)WriteAll(fd, &done, sizeof(done));
  close(fd); Cleanup(&s); urma_uninit(); return ok ? 0 : 1;
}
