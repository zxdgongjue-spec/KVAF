#include "cpu_pull_urma.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <thread>

#if !defined(URMA_ACCESS_LOCAL_WRITE) || defined(URMA_ACCESS_LOCAL_ONLY)
#error "CPU Pull requires the UMDK 24.12 LOCAL_WRITE access model"
#endif

namespace kvaf::offload {
namespace {
struct ImportFlag { std::uint64_t value, stag, dtag; };
struct RemoteJetty {
  urma_jetty_id_t id;
  std::uint32_t mode, policy, type, padding;
  ImportFlag flag;
  std::uint32_t tp_type, reserved;
};
struct TpQuery { std::uint32_t flag, mode; std::uint8_t local[16], remote[16]; };
struct TpInfo { std::uint64_t handle; };
struct ActiveTp { std::uint64_t local, remote, tag; std::uint32_t tx, rx; std::uint64_t reserved; };
using QueryTp = int (*)(urma_context_t*, TpQuery*, std::uint32_t*, TpInfo*);
using ImportJetty = urma_target_jetty_t* (*)(urma_context_t*, RemoteJetty*, urma_token_t*, ActiveTp*);
#ifdef KVAF_CPU_PULL_EXTENDED_SGE
struct Sge { std::uint64_t addr; std::uint32_t len; urma_target_seg_t* tseg; void* user_tseg; };
static_assert(sizeof(Sge) == 32);
#else
using Sge = urma_sge_t;
static_assert(sizeof(Sge) == 24, "select the matching deployed SGE ABI explicitly");
#endif
static_assert(sizeof(urma_seg_t) == 48 && sizeof(urma_jetty_id_t) == 24);
static_assert(sizeof(RemoteJetty) == 72 && sizeof(TpQuery) == 40 && sizeof(ActiveTp) == 40);
static_assert(offsetof(urma_device_attr_t, dev_cap) + offsetof(urma_device_cap_t, max_msg_size) == 72);
Status Error(const char* operation) { return Status::Internal(std::string("CPU Pull: ") + operation); }

class UrmaBackend final : public Backend {
 public:
  explicit UrmaBackend(urma_context_t* context) : context_(context) {}
  ~UrmaBackend() override {
    if (!Close().ok()) std::fprintf(stderr, "CPU_PULL: retaining non-quiescent resources\n");
  }
  Status Allocate(std::uint64_t bytes, Endpoint* local) override {
    if (!context_ || memory_ || !bytes || bytes > SIZE_MAX) return Error("invalid allocation");
    if (posix_memalign(&memory_, 4096, bytes)) return Error("allocate DDR");
    bytes_ = bytes;
    std::memset(memory_, 0, bytes);
    token_ = urma_alloc_token_id(context_);
    if (!token_) return Error("allocate DDR token");
    urma_seg_cfg_t cfg{};
    cfg.va = reinterpret_cast<std::uint64_t>(memory_); cfg.len = bytes;
    cfg.token_id = token_; cfg.token_value.token = token_->token_id;
    cfg.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    cfg.flag.bs.token_id_valid = 1;
    cfg.flag.bs.access = URMA_ACCESS_LOCAL_WRITE;
    cfg.flag.bs.non_pin = 0;
    destination_ = urma_register_seg(context_, &cfg);
    if (!destination_) return Error("register pinned LOCAL_WRITE DDR");
    local->base = reinterpret_cast<std::uint64_t>(memory_); local->bytes = bytes;
    std::memcpy(local->eid, context_->eid.raw, 16);
    std::fprintf(stderr, "CPU_PULL_DDR base=%p bytes=%llu sge_bytes=%zu access=LOCAL_WRITE\n",
                 memory_, static_cast<unsigned long long>(bytes), sizeof(Sge));
    return Status::OK();
  }
  Status Open(const Endpoint& source, Endpoint* local) override {
    if (source_ || jetty_ || poisoned_ || source.segment_bytes != sizeof(urma_seg_t) ||
        source.key_bytes != 32 || !source.tp || !source.psn || !source.token)
      return Error("invalid source endpoint");
    urma_seg_t segment{}; std::memcpy(&segment, source.segment, sizeof(segment));
    // The CTP route EID and the exported segment's EID need not be equal.
    // Import the provider's segment verbatim, just as aicpu-urma does, while
    // retaining the route EID for the TP query below.
    if (!source.base || source.base > UINT64_MAX-source.bytes || source.base < segment.ubva.va ||
        !Contains(segment.len, source.base-segment.ubva.va, source.bytes))
      return Error("source segment range");
    auto query = reinterpret_cast<QueryTp>(dlsym(RTLD_DEFAULT, "urma_get_tp_list"));
    auto import = reinterpret_cast<ImportJetty>(dlsym(RTLD_DEFAULT, "urma_import_jetty_ex"));
    if (!query || !import) return Error("missing expanded CTP symbols");
    // Both public and deployed attributes share the prefix through max_msg_size.
    // Reserve enough output for the deployed extension; never pass an undersized
    // public struct to a provider compiled with the expanded device attributes.
    alignas(64) std::array<std::uint8_t, 4096> storage{};
    auto* attr = reinterpret_cast<urma_device_attr_t*>(storage.data());
    if (urma_query_device(context_->dev, attr) || !attr->dev_cap.max_jfs_sge ||
        !attr->dev_cap.max_jfs_rsge || !attr->dev_cap.max_msg_size) return Error("query READ limits");
    max_bytes_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(65536, attr->dev_cap.max_msg_size));
    urma_import_seg_flag_t flags{}; flags.bs.access = URMA_ACCESS_REMOTE_READ;
    flags.bs.mapping = URMA_SEG_NOMAP;
    urma_token_t token{}; token.token = source.token;
    source_ = urma_import_seg(context_, &segment, &token, 0, flags);
    if (!source_) return Error("import HBM");
    source_base_ = source.base;
    urma_jfc_cfg_t cq_cfg{}; cq_cfg.depth = 256;
    cq_ = urma_create_jfc(context_, &cq_cfg);
    if (!cq_) return Error("create CQ");
    urma_jfr_cfg_t rq_cfg{}; rq_cfg.depth = 256; rq_cfg.max_sge = 1;
    rq_cfg.trans_mode = URMA_TM_RM; rq_cfg.jfc = cq_; rq_cfg.token_value.token = source.token;
    rq_ = urma_create_jfr(context_, &rq_cfg);
    if (!rq_) return Error("create RQ");
    urma_jetty_cfg_t cfg{}; cfg.flag.bs.share_jfr = URMA_SHARE_JFR;
    cfg.jfs_cfg.depth = 256; cfg.jfs_cfg.max_sge = 1; cfg.jfs_cfg.max_rsge = 1;
    cfg.jfs_cfg.trans_mode = URMA_TM_RM; cfg.jfs_cfg.jfc = cq_;
    cfg.jfs_cfg.flag.bs.error_suspend = 1;
    cfg.jfs_cfg.priority = 15; cfg.jfs_cfg.rnr_retry = 7; cfg.jfs_cfg.err_timeout = 16;
    cfg.shared.jfr = rq_; cfg.shared.jfc = cq_;
    jetty_ = urma_create_jetty(context_, &cfg);
    if (!jetty_) return Error("create Jetty");
    TpQuery query_cfg{}; query_cfg.flag = 1; query_cfg.mode = URMA_TM_RM;
    std::memcpy(query_cfg.local, context_->eid.raw, 16); std::memcpy(query_cfg.remote, source.eid, 16);
    TpInfo tp{}; std::uint32_t count = 1;
    if (query(context_, &query_cfg, &count, &tp) || !count || !tp.handle) return Error("query CTP");
    RemoteJetty remote{};
    std::memcpy(&remote.id, source.key, 24); std::memcpy(&remote.mode, source.key+24, 4);
    if (remote.mode != URMA_TM_RM) return Error("source Jetty transport mode");
    remote.policy = URMA_JETTY_GRP_POLICY_RR; remote.type = URMA_JETTY; remote.tp_type = 1;
    urma_import_jetty_flag_t import_flag{}; import_flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    remote.flag.value = import_flag.value;
    const auto psn = static_cast<std::uint32_t>((++sequence_ % 0xffffff) + 1);
    ActiveTp active{}; active.local = tp.handle; active.remote = source.tp; active.tx = psn; active.rx = source.psn;
    peer_ = import(context_, &remote, &token, &active);
    if (!peer_) return Error("import source Jetty");
    local->tp = tp.handle; local->psn = psn; local->token = source.token;
    local->key_bytes = 28;
    std::memcpy(local->key, &jetty_->jetty_id, 24);
    std::memcpy(local->key+24, &jetty_->jetty_cfg.jfs_cfg.trans_mode, 4);
    return Status::OK();
  }
  Status Read(const std::vector<Span>& spans) override {
    for (const auto& span : spans) {
      for (std::uint64_t offset = 0; offset < span.bytes;) {
        const auto length = static_cast<std::uint32_t>(std::min<std::uint64_t>(max_bytes_, span.bytes-offset));
        source_sge_ = {}; source_sge_.addr = source_base_ + span.source + offset;
        source_sge_.len = length; source_sge_.tseg = source_;
        destination_sge_ = {}; destination_sge_.addr = reinterpret_cast<std::uint64_t>(memory_) + span.destination + offset;
        destination_sge_.len = length; destination_sge_.tseg = destination_;
        wr_ = {}; wr_.opcode = URMA_OPC_READ; wr_.tjetty = peer_; wr_.user_ctx = ++sequence_;
        wr_.flag.bs.complete_enable = 1; wr_.flag.bs.comp_order = 1; wr_.flag.bs.place_order = URMA_STRONG_ORDER;
        wr_.rw.src = {reinterpret_cast<urma_sge_t*>(&source_sge_), 1};
        wr_.rw.dst = {reinterpret_cast<urma_sge_t*>(&destination_sge_), 1};
        urma_jfs_wr_t* bad = nullptr;
        if (urma_post_jetty_send_wr(jetty_, &wr_, &bad)) return Poison("post READ");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
        for (;;) {
          urma_cr_t cr{}; const int n = urma_poll_jfc(cq_, 1, &cr);
          if (n < 0) return Poison("poll CQ");
          if (n) {
            if (cr.status != URMA_CR_SUCCESS || cr.user_ctx != sequence_) return Poison("CQ status/context");
            break;
          }
          if (std::chrono::steady_clock::now() >= deadline) return Poison("CQ timeout");
          std::this_thread::yield();
        }
        offset += length;
      }
    }
    // This backend requires the deployed coherent host-DMA contract. Hardware
    // acceptance must verify CPU reads after CQ; no CPU fence substitutes for it.
    return Status::OK();
  }
  Status ReleaseSource() override {
    if (poisoned_) return Status::InDoubt("READ has not quiesced");
#define RELEASE(handle, operation) \
    if (handle) { if (operation(handle)) return Poison(#operation); handle = nullptr; }
    RELEASE(peer_, urma_unimport_jetty)
    RELEASE(jetty_, urma_delete_jetty)
    RELEASE(rq_, urma_delete_jfr)
    RELEASE(cq_, urma_delete_jfc)
    RELEASE(source_, urma_unimport_seg)
    return Status::OK();
  }
  Status Close() override {
    auto status = ReleaseSource(); if (!status.ok()) return status;
    RELEASE(destination_, urma_unregister_seg)
    RELEASE(token_, urma_free_token_id)
#undef RELEASE
    if (memory_) { free(memory_); memory_ = nullptr; }
    return Status::OK();
  }
  std::uint64_t Checksum(const std::vector<Span>& spans) const override {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto& span : spans)
      hash = Hash(static_cast<const std::uint8_t*>(memory_) + span.destination, span.bytes, hash);
    return hash;
  }
 private:
  Status Poison(const char* operation) { poisoned_ = true; return Status::InDoubt(std::string("CPU Pull: ") + operation); }
  urma_context_t* context_;
  void* memory_{};
  std::uint64_t bytes_{}, source_base_{}, sequence_{};
  std::uint32_t max_bytes_{};
  bool poisoned_{};
  urma_token_id_t* token_{};
  urma_target_seg_t *destination_{}, *source_{};
  urma_jfc_t* cq_{};
  urma_jfr_t* rq_{};
  urma_jetty_t* jetty_{};
  urma_target_jetty_t* peer_{};
  Sge source_sge_{}, destination_sge_{};
  urma_jfs_wr_t wr_{};
};
}
std::unique_ptr<Backend> CreateUrmaBackend(urma_context_t* context) {
  return context ? std::make_unique<UrmaBackend>(context) : nullptr;
}
}
