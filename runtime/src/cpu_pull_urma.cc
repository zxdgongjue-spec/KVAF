#include "cpu_pull_urma.h"
#include "urma_device_abi.h"
#include "urma_transport.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <sstream>

namespace kvaf::offload {
namespace {
// Data-plane baseline: aicpu-urma@8bc5a2a, cpu_push_get_transport.cc.
// Use its deployed ABI, queue helpers, registration flags and single-SGE READ
// path. The Offload protocol supplies tensor ranges instead of HBM controls.
constexpr std::uint32_t kDepth = 8192;
constexpr auto kReadTimeout = std::chrono::seconds(30);
std::atomic<std::uint32_t> next_token{1}, next_psn{1};
std::uint32_t Next(std::atomic<std::uint32_t>& sequence) {
  auto value = ++sequence;
  while (!value) value = ++sequence;
  return value;
}
Status Error(const std::string& detail) { return Status::Internal("CPU Pull: " + detail); }
std::string Eid(const std::uint8_t* bytes) {
  constexpr char hex[] = "0123456789abcdef";
  std::string value;
  for (std::size_t i = 0; i < 16; ++i) { value += hex[bytes[i] >> 4]; value += hex[bytes[i] & 15]; }
  return value;
}
void Describe(std::ostream& out, const char* name, const urma_seg_t& segment) {
  out << ' ' << name << "_base=0x" << std::hex << segment.ubva.va
      << ' ' << name << "_attr=0x" << segment.attr.value << std::dec
      << ' ' << name << "_bytes=" << segment.len
      << ' ' << name << "_access=" << segment.attr.bs.access
      << ' ' << name << "_token_id=" << segment.token_id;
}

class UrmaBackend final : public Backend {
 public:
  explicit UrmaBackend(urma_context_t* context) : context_(context) {}
  ~UrmaBackend() override {
    if (!Close().ok()) std::fprintf(stderr, "CPU_PULL: retaining non-quiescent resources\n");
  }
  Status Allocate(std::uint64_t bytes, Endpoint* local) override {
    if (!context_ || !local || memory_ || !bytes || bytes > SIZE_MAX) return Error("invalid allocation");
    if (posix_memalign(&memory_, 4096, bytes)) return Error("allocate DDR");
    std::memset(memory_, 0, bytes);
    bytes_ = bytes;
    region_token_ = Next(next_token);
    local->base = reinterpret_cast<std::uint64_t>(memory_);
    local->bytes = bytes;
    local->token = region_token_;
    std::memcpy(local->eid, context_->eid.raw, 16);
    return Status::OK();
  }

  Status Open(const Endpoint& source, Endpoint* local) override {
    if (!local || !memory_ || source_ || poisoned_ || source.segment_bytes != sizeof(urma_seg_t) ||
        (source.key_bytes != 28 && source.key_bytes != 32) || !source.tp || !source.psn || !source.token)
      return Error("invalid source endpoint");
    if (source.token != region_token_) return Error("source token differs from region token");
    urma_seg_t segment{};
    std::memcpy(&segment, source.segment, sizeof(segment));
    if (!source.base || source.base > UINT64_MAX-source.bytes || source.base < segment.ubva.va ||
        !Contains(segment.len, source.base-segment.ubva.va, source.bytes)) return Error("source segment range");
    for (const auto& key : source.key) {
      urma_transport_mode_t mode{};
      std::memcpy(&mode, key+sizeof(urma_jetty_id_t), sizeof(mode));
      if (mode != URMA_TM_RM) return Error("source Jetty transport mode");
    }
    auto query = reinterpret_cast<urma::GetTpList>(dlsym(RTLD_DEFAULT, "urma_get_tp_list"));
    auto import = reinterpret_cast<urma::ImportJettyEx>(dlsym(RTLD_DEFAULT, "urma_import_jetty_ex"));
    if (!query || !import) return Error("missing expanded CTP symbols");
    urma::UbDeviceAttributes attr{};
    if (urma_query_device(context_->dev, reinterpret_cast<urma_device_attr_t*>(&attr)))
      return Error("query device");
    max_sge_ = std::min(16U, attr.dev_cap.max_jfs_sge);
    max_bytes_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(65536, attr.dev_cap.max_msg_size));
    if (!max_sge_ || max_bytes_ < 64) return Error("unsupported SGE/message limits");

    // Reference order: import HBM, register the receiving host buffer, query
    // CTP, then create/import four queues. Preserve exported memory/Jetty EIDs.
    urma_import_seg_flag_t flags{};
    flags.bs.cacheable = URMA_NON_CACHEABLE;
    flags.bs.access = (1U << 1U) | (1U << 2U);
    flags.bs.mapping = URMA_SEG_NOMAP;
    urma_token_t token{}; token.token = region_token_;
    source_ = urma_import_seg(context_, &segment, &token, 0, flags);
    if (!source_) return Error("import HBM");
    source_base_ = source.base;
    source_bytes_ = source.bytes;
    auto status = RegisterDestination();
    if (!status.ok()) return status;
    std::uint64_t tp{};
    if (!urma::QueryTp(query, context_, source.eid, &tp)) return Error("query CTP");
    local->tp = tp;
    local->psn = Next(next_psn);
    local->token = region_token_;
    local->key_bytes = sizeof(urma_jetty_id_t) + sizeof(urma_transport_mode_t);
    for (std::size_t index = 0; index < queues_.size(); ++index) {
      auto& q = queues_[index];
      const auto* error = urma::CreateQueues(urma::NativeApi{}, context_, region_token_,
          {16384, kDepth, max_sge_, false, true}, q.cq, q.rq, q.jetty);
      if (error) return Error(error);
      q.local.reserve(1); q.remote.reserve(1); q.wrs.reserve(1);
      q.peer = urma::ImportJetty(import, context_, source.key[index], region_token_,
          tp, source.tp, local->psn, source.psn);
      if (!q.peer) return Error("import NPU Jetty " + std::to_string(index));
      std::memcpy(local->key[index], &q.jetty->jetty_id, sizeof(urma_jetty_id_t));
      std::memcpy(local->key[index]+sizeof(urma_jetty_id_t), &q.jetty->jetty_cfg.jfs_cfg.trans_mode,
                  sizeof(urma_transport_mode_t));
    }
    ready_ = true;
    std::fprintf(stderr, "[KVAF_CPU] CPU_PULL_REFERENCE commit=8bc5a2a protocol=%u sge_bytes=%zu "
        "jettys=%u max_sge=%u max_bytes=%u local_eid=%s peer_eid=%s psn=%u peer_psn=%u\n",
        kVersion, sizeof(urma::Sge), kChannelCount, max_sge_, max_bytes_,
        Eid(context_->eid.raw).c_str(), Eid(source.eid).c_str(), local->psn, source.psn);
    return Status::OK();
  }

  Status Read(const std::vector<Span>& spans) override {
    if (poisoned_) return Status::InDoubt(failure_);
    if (!ready_ || !source_ || !destination_) return Error("READ transport is not ready");
    if (!ValidateSpans(spans, source_bytes_, bytes_)) return Error("invalid READ spans");
    // The reference uses queue 0 for all control READs. Offload uses exactly
    // that path for payload READs as well; no WRITE/scatter kernel is involved.
    auto& q = queues_[0];
    for (const auto& span : spans) {
      for (std::uint64_t offset = 0; offset < span.bytes;) {
        const auto length = static_cast<std::uint32_t>(std::min<std::uint64_t>(max_bytes_, span.bytes-offset));
        q.local.clear(); q.remote.clear(); q.wrs.clear();
        q.local.push_back({reinterpret_cast<std::uint64_t>(memory_)+span.destination+offset, length, destination_, nullptr});
        q.remote.push_back({source_base_+span.source+offset, length, source_, nullptr});
        urma_jfs_wr_t wr{};
        wr.opcode = URMA_OPC_READ;
        wr.flag.bs.comp_order = 1;
        wr.flag.bs.place_order = URMA_STRONG_ORDER;
        wr.flag.bs.complete_enable = 1;
        wr.tjetty = q.peer;
        wr.user_ctx = ++q.sequence;
        wr.rw.src = {reinterpret_cast<urma_sge_t*>(q.remote.data()), 1};
        wr.rw.dst = {reinterpret_cast<urma_sge_t*>(q.local.data()), 1};
        q.wrs.push_back(wr);
        q.pending = wr.user_ctx;
        urma_jfs_wr_t* bad{};
        errno = 0;
        const int ret = urma_post_jetty_send_wr(q.jetty, q.wrs.data(), &bad);
        const int post_errno = errno;
        if (ret) return ReadFailure("post_read", ret, post_errno, bad);
        const auto deadline = std::chrono::steady_clock::now() + kReadTimeout;
        std::uint32_t polls{};
        while (q.pending) {
          std::array<urma_cr_t, 8> records{};
          errno = 0;
          const int count = urma_poll_jfc(q.cq, records.size(), records.data());
          const int poll_errno = errno;
          if (count < 0 || count > static_cast<int>(records.size())) return ReadFailure("poll_cq", count, poll_errno);
          for (int i = 0; i < count; ++i) {
            const auto& cr = records[i];
            if (cr.status != URMA_CR_SUCCESS || cr.user_ctx != q.pending)
              return ReadFailure("cq_completion", count, poll_errno, nullptr, &cr);
            q.pending = 0;
          }
          if ((++polls & 31) == 0 && q.pending && std::chrono::steady_clock::now() >= deadline)
            return ReadFailure("cq_timeout", count, poll_errno);
        }
        offset += length;
      }
    }
    return Status::OK();
  }

  Status ReleaseSource() override {
    if (poisoned_) return Status::InDoubt(failure_);
    ready_ = false;
#define RELEASE(handle, operation) \
    if (handle) { if (operation(handle)) return Poison(#operation); handle = nullptr; }
    // Same queue-first teardown order as cpu_push::Transport::Shutdown.
    for (auto& q : queues_) {
      RELEASE(q.jetty, urma_delete_jetty)
      RELEASE(q.peer, urma_unimport_jetty)
      RELEASE(q.rq, urma_delete_jfr)
      RELEASE(q.cq, urma_delete_jfc)
      q = {};
    }
    RELEASE(source_, urma_unimport_seg)
    source_base_ = source_bytes_ = 0;
    return Status::OK();
  }
  Status Close() override {
    auto status = ReleaseSource(); if (!status.ok()) return status;
    RELEASE(destination_, urma_unregister_seg)
    RELEASE(token_, urma_free_token_id)
#undef RELEASE
    std::free(memory_); memory_ = nullptr; bytes_ = 0; destination_valid_ = false;
    return Status::OK();
  }
  std::uint64_t Checksum(const std::vector<Span>& spans) const override {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto& span : spans)
      hash = Hash(static_cast<const std::uint8_t*>(memory_) + span.destination, span.bytes, hash);
    return hash;
  }
 private:
  Status RegisterDestination() {
    if (destination_valid_) return Status::OK();
    if (!token_) token_ = urma_alloc_token_id(context_);
    if (!token_) return Error("allocate DDR token");
    urma_seg_cfg_t cfg{};
    cfg.va = reinterpret_cast<std::uint64_t>(memory_); cfg.len = bytes_;
    cfg.token_id = token_; cfg.token_value.token = region_token_;
    cfg.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    cfg.flag.bs.access = 1U;
    cfg.flag.bs.non_pin = 0;
    cfg.flag.bs.token_id_valid = 1;
    if (!destination_) destination_ = urma_register_seg(context_, &cfg);
    if (!destination_) return Error("register pinned DDR");
    const auto& segment = destination_->seg;
    const bool range_ok = segment.ubva.va && cfg.va <= UINT64_MAX-bytes_ && cfg.va >= segment.ubva.va &&
                          Contains(segment.len, cfg.va-segment.ubva.va, bytes_);
    const bool access_ok = (segment.attr.bs.access & 1U) && !segment.attr.bs.non_pin;
    std::ostringstream log;
    log << "CPU_PULL_DDR sge_bytes=" << sizeof(urma::Sge);
    Describe(log, "registered", segment);
    log << " range_ok=" << range_ok << " access_ok=" << access_ok;
    std::fprintf(stderr, "%s\n", log.str().c_str());
    if (!range_ok || !access_ok) return Error("invalid registered DDR range/access");
    destination_valid_ = true;
    return Status::OK();
  }
  Status ReadFailure(const char* stage, int ret, int error_number,
                     const urma_jfs_wr_t* bad = nullptr, const urma_cr_t* cr = nullptr) {
    const auto& q = queues_[0];
    std::ostringstream message;
    message << "stage=" << stage << " ret=" << ret << " errno=" << error_number
            << " sge_bytes=" << sizeof(urma::Sge) << " wr_id=" << q.pending
            << " src=0x" << std::hex << q.remote.front().addr << " dst=0x" << q.local.front().addr << std::dec
            << " bytes=" << q.remote.front().len << " bad_wr_nonnull=" << (bad != nullptr)
            << " bad_wr_is_current=" << (bad == q.wrs.data());
    if (cr) message << " cq_status=" << static_cast<unsigned>(cr->status)
                    << " cq_user_ctx=" << cr->user_ctx << " cq_flags=" << unsigned(cr->flag.value)
                    << " local_id=" << cr->local_id << " tpn=" << cr->tpn;
    if (cr && cr->status == URMA_CR_LOC_ACCESS_ERR) message << " cq_status_name=URMA_CR_LOC_ACCESS_ERR";
    if (destination_) Describe(message, "dst_mr", destination_->seg);
    if (source_) Describe(message, "src_mr", source_->seg);
    return Poison(message.str());
  }
  Status Poison(const std::string& reason) {
    if (!poisoned_) { poisoned_ = true; failure_ = "CPU Pull: " + reason; }
    return Status::InDoubt(failure_);
  }
  struct Queue {
    urma_jfc_t* cq{};
    urma_jfr_t* rq{};
    urma_jetty_t* jetty{};
    urma_target_jetty_t* peer{};
    std::vector<urma::Sge> local, remote;
    std::vector<urma_jfs_wr_t> wrs;
    std::uint64_t sequence{}, pending{};
  };
  urma_context_t* context_{};
  void* memory_{};
  std::uint64_t bytes_{}, source_base_{}, source_bytes_{};
  std::uint32_t region_token_{}, max_sge_{}, max_bytes_{};
  urma_token_id_t* token_{};
  urma_target_seg_t *destination_{}, *source_{};
  std::array<Queue, kChannelCount> queues_;
  bool ready_{}, poisoned_{}, destination_valid_{};
  std::string failure_;
};
}
std::unique_ptr<Backend> CreateUrmaBackend(urma_context_t* context) {
  return context ? std::make_unique<UrmaBackend>(context) : nullptr;
}
}
