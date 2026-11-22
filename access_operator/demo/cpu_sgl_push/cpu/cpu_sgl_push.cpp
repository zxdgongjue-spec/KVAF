// CPU side of the CPU-SGL push demo. Pushes scattered DDR token rows into
// contiguous HBM runs via one chained WRITE WR per run (local SGL -> remote 1
// sge), then an 8B flag WR with strong place_order. Same host clock as the NPU
// target; join on seq.
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <time.h>

#include "urma_api.h"
#include "../common/demo_protocol.h"

namespace {

constexpr uint32_t kEidBytes = 16U;
constexpr uint32_t kSegBytes = sizeof(urma_seg_t);
constexpr uint32_t kMaxGroup = 32U;
constexpr uint32_t kJettyKeyMax = 64U;

// Private expanded-CTP ABI (same layouts as KVAF runtime/src/cpu_runtime.cc).
struct UrmaExtImportJettyFlag {
  uint64_t value;
  uint64_t stag;
  uint64_t dtag;
};
struct UrmaExtRjetty {
  urma_jetty_id_t jetty_id;
  uint32_t trans_mode;
  uint32_t policy;
  uint32_t type;
  uint32_t reserved_before_flag;
  UrmaExtImportJettyFlag flag;
  uint32_t tp_type;
  uint32_t reserved_after_tp_type;
};
struct UrmaExtGetTpCfg {
  uint32_t flag;
  uint32_t trans_mode;
  uint8_t local_eid[kEidBytes];
  uint8_t peer_eid[kEidBytes];
};
struct UrmaExtTpInfo {
  uint64_t tp_handle;
};
struct UrmaExtActiveTpCfg {
  uint64_t tp_handle;
  uint64_t peer_tp_handle;
  uint64_t tag;
  uint32_t tx_psn;
  uint32_t rx_psn;
  uint64_t reserved;
};
static_assert(sizeof(UrmaExtRjetty) == 72U, "rjetty ABI");
static_assert(sizeof(UrmaExtActiveTpCfg) == 40U, "active tp ABI");
static_assert(kSegBytes == 48U, "seg ABI");

using UrmaGetTpListFn = int (*)(urma_context_t *, UrmaExtGetTpCfg *, uint32_t *, UrmaExtTpInfo *);
using UrmaImportJettyExFn = urma_target_jetty_t *(*)(urma_context_t *, UrmaExtRjetty *, urma_token_t *,
                                                     UrmaExtActiveTpCfg *);

struct Args {
  std::string keyfile;
  std::string ub_device;
  std::string npu_eid_hex;
  uint32_t eid_index = 0;
  uint64_t ddr_mb = 256;
  uint32_t token_bytes = 656;
  uint32_t tokens_per_block = 128;
  uint32_t miss = 102;
  uint32_t group = 16;
  uint32_t iters = 200;
  uint32_t warmup = 20;
  const char *csv = nullptr;
};

Args ParseArgs(int argc, char **argv) {
  Args args;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key = argv[i];
    const std::string value = argv[i + 1];
    if (key == "--keyfile") args.keyfile = value;
    else if (key == "--ub-device") args.ub_device = value;
    else if (key == "--npu-eid") args.npu_eid_hex = value;
    else if (key == "--eid-index") args.eid_index = std::stoul(value);
    else if (key == "--ddr-mb") args.ddr_mb = std::stoull(value);
    else if (key == "--token-bytes") args.token_bytes = std::stoul(value);
    else if (key == "--tokens-per-block") args.tokens_per_block = std::stoul(value);
    else if (key == "--miss") args.miss = std::stoul(value);
    else if (key == "--group") args.group = std::stoul(value);
    else if (key == "--iters") args.iters = std::stoul(value);
    else if (key == "--warmup") args.warmup = std::stoul(value);
    else if (key == "--csv") args.csv = value.c_str();
    else { std::fprintf(stderr, "unknown arg %s\n", key.c_str()); std::exit(2); }
  }
  if (args.keyfile.empty() || args.npu_eid_hex.empty()) {
    std::fprintf(stderr, "need --keyfile --npu-eid\n");
    std::exit(2);
  }
  if (args.group == 0 || args.group > kMaxGroup || args.miss == 0) {
    std::fprintf(stderr, "bad group/miss\n");
    std::exit(2);
  }
  return args;
}

struct Keyfile {
  uint8_t mem_key[128]{};
  uint32_t mem_key_bytes = 0;
  uint8_t jetty_key[kJettyKeyMax]{};
  uint32_t jetty_key_bytes = 0;
  uint32_t mem_token_value = 0;
  uint32_t jetty_token_value = 0;
  uint64_t npu_tp_handle = 0;
  uint32_t npu_psn = 0;
  uint64_t hbm_addr = 0;
  uint64_t hbm_bytes = 0;
  uint64_t data_offset = 4096;
};

bool LoadKeyfile(const std::string &path, Keyfile *out) {
  FILE *f = std::fopen(path.c_str(), "r");
  if (f == nullptr) return false;
  char line[512];
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    char *eq = std::strchr(line, '=');
    if (eq == nullptr) continue;
    *eq = 0;
    const std::string key = line;
    const std::string value = eq + 1;
    if (key == "mem_key") {
      out->mem_key_bytes = uint32_t(value.size() - 1) / 2;
      if (out->mem_key_bytes > sizeof(out->mem_key) ||
          !kvaf::demo::ReadHex(value.c_str(), out->mem_key, out->mem_key_bytes)) return false;
    } else if (key == "jetty_key") {
      out->jetty_key_bytes = uint32_t(value.size() - 1) / 2;
      if (out->jetty_key_bytes > sizeof(out->jetty_key) ||
          !kvaf::demo::ReadHex(value.c_str(), out->jetty_key, out->jetty_key_bytes)) return false;
    } else if (key == "mem_token_value") out->mem_token_value = std::stoul(value);
    else if (key == "jetty_token_value") out->jetty_token_value = std::stoul(value);
    else if (key == "npu_tp_handle") out->npu_tp_handle = std::stoull(value);
    else if (key == "npu_psn") out->npu_psn = std::stoul(value);
    else if (key == "hbm_addr") out->hbm_addr = std::stoull(value);
    else if (key == "hbm_bytes") out->hbm_bytes = std::stoull(value);
    else if (key == "data_offset") out->data_offset = std::stoull(value);
  }
  std::fclose(f);
  return out->mem_key_bytes >= kSegBytes && out->jetty_key_bytes >= 28 && out->hbm_addr != 0;
}

}  // namespace

int PrintEid() {
  urma_init_attr_t init_attr{};
  if (urma_init(&init_attr) != URMA_SUCCESS) return 1;
  int count = 0;
  urma_device_t **list = urma_get_device_list(&count);
  for (int i = 0; i < count; ++i) {
    if (list[i]->type != URMA_TRANSPORT_UB) continue;
    urma_context_t *ctx = urma_create_context(list[i], 0);
    if (ctx == nullptr) continue;
    for (uint8_t byte : ctx->eid.raw) std::fprintf(stderr, "%02x", byte);
    std::fprintf(stderr, "\n");
    (void)urma_delete_context(ctx);
    return 0;
  }
  return 1;
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--print-eid") return PrintEid();
  const Args args = ParseArgs(argc, argv);
  Keyfile key{};
  if (!LoadKeyfile(args.keyfile, &key)) {
    std::fprintf(stderr, "[SGL_PUSH] keyfile load failed: %s\n", args.keyfile.c_str());
    return 2;
  }
  uint8_t npu_eid[kEidBytes]{};
  if (!kvaf::demo::ReadHex(args.npu_eid_hex.c_str(), npu_eid, kEidBytes)) {
    std::fprintf(stderr, "[SGL_PUSH] bad --npu-eid\n");
    return 2;
  }

  urma_init_attr_t init_attr{};
  if (urma_init(&init_attr) != URMA_SUCCESS) {
    std::fprintf(stderr, "[SGL_PUSH] urma_init failed\n");
    return 1;
  }
  urma_device_t *device = nullptr;
  if (!args.ub_device.empty()) {
    device = urma_get_device_by_name(const_cast<char *>(args.ub_device.c_str()));
  } else {
    int count = 0;
    urma_device_t **list = urma_get_device_list(&count);
    for (int i = 0; i < count && device == nullptr; ++i) {
      if (list[i]->type == URMA_TRANSPORT_UB) device = list[i];
    }
  }
  if (device == nullptr) {
    std::fprintf(stderr, "[SGL_PUSH] no UB device\n");
    return 1;
  }
  urma_context_t *ctx = urma_create_context(device, args.eid_index);
  if (ctx == nullptr) {
    std::fprintf(stderr, "[SGL_PUSH] urma_create_context failed errno=%d\n", errno);
    return 1;
  }
  auto get_tp_list = reinterpret_cast<UrmaGetTpListFn>(dlsym(RTLD_DEFAULT, "urma_get_tp_list"));
  auto import_jetty_ex = reinterpret_cast<UrmaImportJettyExFn>(dlsym(RTLD_DEFAULT, "urma_import_jetty_ex"));
  if (get_tp_list == nullptr || import_jetty_ex == nullptr) {
    std::fprintf(stderr, "[SGL_PUSH] liburma lacks expanded CTP APIs\n");
    return 1;
  }

  // Local DDR region: token rows with self-describing pattern, laid out in
  // tokens_per_block-row pages like the KVAF DDR archive.
  const uint64_t ddr_bytes = args.ddr_mb << 20;
  void *ddr = nullptr;
  if (posix_memalign(&ddr, 4096, ddr_bytes) != 0 || ddr == nullptr) {
    std::fprintf(stderr, "[SGL_PUSH] ddr alloc failed\n");
    return 1;
  }
  const uint64_t total_tokens = ddr_bytes / args.token_bytes;
  for (uint64_t t = 0; t < total_tokens; ++t) {
    kvaf::demo::FillRow(static_cast<uint8_t *>(ddr) + t * args.token_bytes, args.token_bytes, t);
  }
  urma_token_id_t *token_id = urma_alloc_token_id(ctx);
  if (token_id == nullptr) {
    std::fprintf(stderr, "[SGL_PUSH] alloc_token_id failed\n");
    return 1;
  }
  const uint32_t local_token_value = uint32_t(kvaf::demo::NowNs()) | 1U;
  urma_reg_seg_flag_t seg_flag{};
  seg_flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
  seg_flag.bs.cacheable = URMA_NON_CACHEABLE;
// ABI-stable access bit values; UMDK renames the enum members across
// releases (REMOTE_* vs plain), so use the documented bit positions directly.
constexpr uint32_t kAccessRead = 0x1U << 1;
constexpr uint32_t kAccessWrite = 0x1U << 2;
constexpr uint32_t kAccessAtomic = 0x1U << 3;
  seg_flag.bs.access = kAccessRead | kAccessWrite | kAccessAtomic;
  seg_flag.bs.non_pin = 1U;
  seg_flag.bs.token_id_valid = 1U;
  urma_seg_cfg_t seg_cfg{};
  seg_cfg.va = reinterpret_cast<uint64_t>(ddr);
  seg_cfg.len = ddr_bytes;
  seg_cfg.token_id = token_id;
  seg_cfg.token_value.token = local_token_value;
  seg_cfg.flag = seg_flag;
  urma_target_seg_t *local_seg = urma_register_seg(ctx, &seg_cfg);
  if (local_seg == nullptr) {
    std::fprintf(stderr, "[SGL_PUSH] register_seg failed errno=%d\n", errno);
    return 1;
  }
  std::fprintf(stderr, "[SGL_PUSH] DDR registered va=%p ubva=0x%llx tokens=%llu\n", ddr,
               static_cast<unsigned long long>(local_seg->seg.ubva.va),
               static_cast<unsigned long long>(total_tokens));

  // Local jetty with enough SGE slots for one group per WR.
  urma_jfc_cfg_t jfc_cfg{};
  jfc_cfg.depth = 4096;
  urma_jfc_t *jfc = urma_create_jfc(ctx, &jfc_cfg);
  if (jfc == nullptr) {
    std::fprintf(stderr, "[SGL_PUSH] create_jfc failed errno=%d\n", errno);
    return 1;
  }
  urma_jfs_cfg_t jfs{};
  jfs.depth = 1024;
  jfs.flag.bs.error_suspend = 1U;
  jfs.trans_mode = URMA_TM_RM;
  jfs.priority = 15U;
  jfs.max_sge = kMaxGroup;
  jfs.rnr_retry = 7U;
  jfs.err_timeout = 16U;
  jfs.jfc = jfc;
  urma_jfr_cfg_t jfr_cfg{};
  jfr_cfg.depth = 256;
  jfr_cfg.trans_mode = URMA_TM_RM;
  jfr_cfg.max_sge = 1U;
  jfr_cfg.jfc = jfc;
  jfr_cfg.token_value.token = local_token_value;
  jfr_cfg.flag.bs.tag_matching = URMA_NO_TAG_MATCHING;
  urma_jfr_t *jfr = urma_create_jfr(ctx, &jfr_cfg);
  if (jfr == nullptr) {
    std::fprintf(stderr, "[SGL_PUSH] create_jfr failed errno=%d\n", errno);
    return 1;
  }
  urma_jetty_cfg_t jetty_cfg{};
  jetty_cfg.flag.bs.share_jfr = URMA_SHARE_JFR;
  jetty_cfg.jfs_cfg = jfs;
  jetty_cfg.shared.jfr = jfr;
  jetty_cfg.shared.jfc = jfc;
  urma_jetty_t *jetty = urma_create_jetty(ctx, &jetty_cfg);
  if (jetty == nullptr) {
    std::fprintf(stderr, "[SGL_PUSH] create_jetty failed errno=%d\n", errno);
    return 1;
  }

  // CTP + import of the NPU target jetty.
  UrmaExtGetTpCfg tp_cfg{};
  tp_cfg.flag = 1U;
  tp_cfg.trans_mode = static_cast<uint32_t>(URMA_TM_RM);
  std::memcpy(tp_cfg.local_eid, ctx->eid.raw, kEidBytes);
  std::memcpy(tp_cfg.peer_eid, npu_eid, kEidBytes);
  UrmaExtTpInfo tp_info{};
  uint32_t tp_count = 1U;
  if (get_tp_list(ctx, &tp_cfg, &tp_count, &tp_info) != 0 || tp_count == 0 || tp_info.tp_handle == 0) {
    std::fprintf(stderr, "[SGL_PUSH] get_tp_list failed errno=%d\n", errno);
    return 1;
  }
  const uint32_t local_psn = uint32_t(kvaf::demo::NowNs() >> 8) | 1U;
  UrmaExtRjetty remote{};
  std::memcpy(&remote.jetty_id, key.jetty_key, sizeof(remote.jetty_id));
  std::memcpy(&remote.trans_mode, key.jetty_key + sizeof(remote.jetty_id), sizeof(remote.trans_mode));
  remote.policy = static_cast<uint32_t>(URMA_JETTY_GRP_POLICY_RR);
  remote.type = static_cast<uint32_t>(URMA_JETTY);
  urma_import_jetty_flag_t import_flag{};
  import_flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
  remote.flag.value = import_flag.value;
  remote.tp_type = 1U;  // URMA_CTP
  UrmaExtActiveTpCfg active{};
  active.tp_handle = tp_info.tp_handle;
  active.peer_tp_handle = key.npu_tp_handle;
  active.tx_psn = local_psn;
  active.rx_psn = key.npu_psn;
  urma_token_t remote_token{};
  remote_token.token = key.jetty_token_value;
  urma_target_jetty_t *target = import_jetty_ex(ctx, &remote, &remote_token, &active);
  if (target == nullptr) {
    std::fprintf(stderr, "[SGL_PUSH] import_jetty_ex failed errno=%d\n", errno);
    return 1;
  }

  // Import the NPU HBM segment. The HCCP mem key's first bytes are the
  // urma_seg_t export (same 48-byte layout as the CPU-side registration).
  urma_seg_t remote_seg{};
  std::memcpy(&remote_seg, key.mem_key, kSegBytes);
  urma_token_t seg_token{};
  seg_token.token = key.mem_token_value;
  urma_import_seg_flag_t import_seg_flag{};
  import_seg_flag.bs.cacheable = URMA_NON_CACHEABLE;
  import_seg_flag.bs.access = (0x1U << 1) | (0x1U << 2);  // READ | WRITE
  urma_target_seg_t *remote_tseg = urma_import_seg(ctx, &remote_seg, &seg_token, 0, import_seg_flag);
  if (remote_tseg == nullptr) {
    std::fprintf(stderr, "[SGL_PUSH] import_seg failed errno=%d ubva=0x%llx len=%llu\n", errno,
                 static_cast<unsigned long long>(remote_seg.ubva.va),
                 static_cast<unsigned long long>(remote_seg.len));
    return 1;
  }
  std::fprintf(stderr, "[SGL_PUSH] NPU seg imported ubva=0x%llx len=%llu hbm_addr=0x%llx\n",
               static_cast<unsigned long long>(remote_seg.ubva.va),
               static_cast<unsigned long long>(remote_seg.len),
               static_cast<unsigned long long>(key.hbm_addr));

  // Deterministic pseudo-random miss list over the DDR tokens.
  std::vector<uint64_t> misses(args.miss);
  uint64_t rng = 0x9e3779b97f4a7c15ULL;
  for (uint32_t i = 0; i < args.miss; ++i) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    misses[i] = rng % total_tokens;
  }
  const uint32_t run_count = (args.miss + args.group - 1) / args.group;
  const uint64_t run_bytes = uint64_t(args.group) * args.token_bytes;
  const uint64_t data_area = key.hbm_bytes - key.data_offset;
  const uint32_t runs_per_iter_cap = uint32_t(data_area / run_bytes);
  if (run_count > runs_per_iter_cap) {
    std::fprintf(stderr, "[SGL_PUSH] hot area too small: need %u runs, cap %u\n", run_count, runs_per_iter_cap);
    return 2;
  }

  std::vector<urma_sge_t> sges(kMaxGroup);
  std::vector<urma_sge_t> dst_sges(run_count + 1);
  std::vector<urma_jfs_wr_t> wrs(run_count + 1);
  std::vector<urma_sg_t> src_sgs(run_count), dst_sgs(run_count + 1);
  // Use a dedicated registered source for the flag: the first 8B of DDR region
  // are pattern data; keep flag in a separately registered small segment.
  urma_seg_cfg_t flag_cfg{};
  void *flag_buf = nullptr;
  if (posix_memalign(&flag_buf, 4096, 4096) != 0) return 1;
  flag_cfg.va = reinterpret_cast<uint64_t>(flag_buf);
  flag_cfg.len = 4096;
  flag_cfg.token_id = token_id;
  flag_cfg.token_value.token = local_token_value;
  flag_cfg.flag = seg_flag;
  urma_target_seg_t *flag_seg = urma_register_seg(ctx, &flag_cfg);
  if (flag_seg == nullptr) {
    std::fprintf(stderr, "[SGL_PUSH] flag seg register failed\n");
    return 1;
  }

  FILE *csv = args.csv != nullptr ? std::fopen(args.csv, "w") : nullptr;
  if (csv != nullptr) std::fprintf(csv, "seq,t0_ns,t_posted_ns,t_jfc_ns,runs,miss,group\n");

  const uint64_t remote_data_base = key.hbm_addr + key.data_offset;
  uint64_t seq = 0;
  for (uint32_t iter = 0; iter < args.iters + args.warmup; ++iter) {
    ++seq;
    // Reset the flag *before* this iteration's data (previous seq is stale but
    // nonzero, so the NPU only reacts to the new value).
    const uint64_t t0 = kvaf::demo::NowNs();
    uint32_t cursor = 0;
    for (uint32_t run = 0; run < run_count; ++run) {
      const uint32_t count = std::min(args.group, args.miss - cursor);
      for (uint32_t j = 0; j < count; ++j) {
        const uint64_t token = misses[cursor + j];
        sges[j].addr = reinterpret_cast<uint64_t>(ddr) + token * args.token_bytes;
        sges[j].len = args.token_bytes;
        sges[j].tseg = local_seg;
        sges[j].user_tseg = nullptr;
      }
      // Chained WRs must keep their sge arrays alive until completion; the
      // last run reuses the tail of the vector, so rebuild per iteration but
      // store per-run copies for all but the final run.
      static thread_local std::vector<std::vector<urma_sge_t>> keep;
      keep.resize(run_count);
      keep[run].assign(sges.begin(), sges.begin() + count);
      src_sgs[run] = {keep[run].data(), count};
      dst_sges[run].addr = remote_data_base + run * run_bytes;
      dst_sges[run].len = count * args.token_bytes;
      dst_sges[run].tseg = remote_tseg;
      dst_sges[run].user_tseg = nullptr;
      dst_sgs[run] = {&dst_sges[run], 1U};
      urma_jfs_wr_t &wr = wrs[run];
      wr = {};
      wr.opcode = URMA_OPC_WRITE;
      wr.tjetty = target;
      wr.user_ctx = run;
      wr.rw.src = src_sgs[run];
      wr.rw.dst = dst_sgs[run];
      wr.next = &wrs[run + 1];
      cursor += count;
    }
    // Final WR: 8B flag with strong ordering + completion.
    std::memcpy(flag_buf, &seq, 8);
    urma_sge_t flag_src{};
    flag_src.addr = reinterpret_cast<uint64_t>(flag_buf);
    flag_src.len = 8;
    flag_src.tseg = flag_seg;
    urma_sge_t flag_dst{};
    flag_dst.addr = key.hbm_addr;  // flag at offset 0
    flag_dst.len = 8;
    flag_dst.tseg = remote_tseg;
    urma_jfs_wr_t &fwr = wrs[run_count];
    fwr = {};
    fwr.opcode = URMA_OPC_WRITE;
    fwr.flag.bs.complete_enable = 1;
    fwr.flag.bs.place_order = 2;  // strong order: lands after all data WRs
    fwr.tjetty = target;
    fwr.user_ctx = seq;
    fwr.rw.src = {&flag_src, 1U};
    fwr.rw.dst = {&flag_dst, 1U};
    fwr.next = nullptr;
    dst_sgs[run_count] = {&flag_dst, 1U};

    urma_jfs_wr_t *bad = nullptr;
    if (urma_post_jetty_send_wr(jetty, &wrs[0], &bad) != 0) {
      std::fprintf(stderr, "[SGL_PUSH] post failed iter=%u errno=%d\n", iter, errno);
      return 1;
    }
    const uint64_t t_posted = kvaf::demo::NowNs();
    urma_cr_t cr{};
    int got = 0;
    while ((got = urma_poll_jfc(jfc, 1, &cr)) == 0) {
    }
    if (cr.status != URMA_CR_SUCCESS) {
      std::fprintf(stderr, "[SGL_PUSH] completion error iter=%u status=%d\n", iter, cr.status);
      return 1;
    }
    const uint64_t t_jfc = kvaf::demo::NowNs();
    if (csv != nullptr && iter >= args.warmup) {
      std::fprintf(csv, "%llu,%llu,%llu,%llu,%u,%u,%u\n", static_cast<unsigned long long>(seq),
                   static_cast<unsigned long long>(t0), static_cast<unsigned long long>(t_posted),
                   static_cast<unsigned long long>(t_jfc), run_count, args.miss, args.group);
    }
  }

  // Stop sentinel lets the NPU poller exit into verification.
  const uint64_t stop = kvaf::demo::kStopSeq;
  std::memcpy(flag_buf, &stop, 8);
  urma_sge_t flag_src{};
  flag_src.addr = reinterpret_cast<uint64_t>(flag_buf);
  flag_src.len = 8;
  flag_src.tseg = flag_seg;
  urma_sge_t flag_dst{};
  flag_dst.addr = key.hbm_addr;
  flag_dst.len = 8;
  flag_dst.tseg = remote_tseg;
  urma_jfs_wr_t swr{};
  swr.opcode = URMA_OPC_WRITE;
  swr.flag.bs.complete_enable = 1;
  swr.flag.bs.place_order = 2;
  swr.tjetty = target;
  swr.rw.src = {&flag_src, 1U};
  swr.rw.dst = {&flag_dst, 1U};
  urma_jfs_wr_t *bad = nullptr;
  if (urma_post_jetty_send_wr(jetty, &swr, &bad) != 0) {
    std::fprintf(stderr, "[SGL_PUSH] stop post failed\n");
    return 1;
  }
  urma_cr_t cr{};
  while (urma_poll_jfc(jfc, 1, &cr) == 0) {
  }
  if (csv != nullptr) std::fclose(csv);
  std::fprintf(stderr, "[SGL_PUSH] DONE iters=%u miss=%u group=%u runs_per_iter=%u\n", args.iters, args.miss,
               args.group, run_count);
  return 0;
}
