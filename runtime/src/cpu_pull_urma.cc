#include "cpu_pull_urma.h"
#include "cpu_pull_profile.h"
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
#include <unordered_map>

namespace kvaf::offload {
namespace {
// Data-plane baseline: aicpu-urma@8bc5a2a, cpu_push_get_transport.cc.
// Keep its deployed ABI/registration/link setup and batched doorbell scheme.
// Unlike WRITE gather, each READ describes ONE contiguous remote source.
constexpr std::uint32_t kDepth = 8192;
constexpr std::size_t kWaveDepth = 1024, kDoorbellWrs = 32;
constexpr auto kReadTimeout = std::chrono::seconds(30);
using Clock = std::chrono::steady_clock;
bool Enabled(const char* name) {
  const auto* value = std::getenv(name);
  return value && std::strcmp(value, "1") == 0;
}
double Micros(Clock::duration duration) {
  return std::chrono::duration<double, std::micro>(duration).count();
}
std::atomic<std::uint32_t> next_token{1}, next_psn{1};
std::atomic<std::uint64_t> next_profile_id{0};
std::uint64_t ProfileWarmup() {
  const char* value = std::getenv("KVAF_CPU_PULL_PROFILE_WARMUP");
  if (!value) return 0;
  char* end{};
  errno = 0;
  const auto warmup = std::strtoull(value, &end, 10);
  if (!*value || *value < '0' || *value > '9' || *end || errno) {
    std::fprintf(stderr, "[KVAF_CPU] CPU_PULL_PROFILE_CONFIG invalid_warmup using=0\n");
    return 0;
  }
  return warmup;
}
std::uint32_t Next(std::atomic<std::uint32_t>& sequence) {
  auto value = ++sequence;
  while (!value) value = ++sequence;
  return value;
}
Status Error(const std::string& detail) { return Status::Internal("CPU Pull: " + detail); }
const char* CompletionStatusName(urma_cr_status_t status) {
#define CR_NAME(value) case value: return #value
  switch (status) {
    CR_NAME(URMA_CR_SUCCESS);
    CR_NAME(URMA_CR_UNSUPPORTED_OPCODE_ERR);
    CR_NAME(URMA_CR_LOC_LEN_ERR);
    CR_NAME(URMA_CR_LOC_OPERATION_ERR);
    CR_NAME(URMA_CR_LOC_ACCESS_ERR);
    CR_NAME(URMA_CR_REM_RESP_LEN_ERR);
    CR_NAME(URMA_CR_REM_UNSUPPORTED_REQ_ERR);
    CR_NAME(URMA_CR_REM_OPERATION_ERR);
    CR_NAME(URMA_CR_REM_ACCESS_ABORT_ERR);
    CR_NAME(URMA_CR_ACK_TIMEOUT_ERR);
    CR_NAME(URMA_CR_RNR_RETRY_CNT_EXC_ERR);
    CR_NAME(URMA_CR_WR_FLUSH_ERR);
    CR_NAME(URMA_CR_WR_SUSPEND_DONE);
    CR_NAME(URMA_CR_WR_FLUSH_ERR_DONE);
    CR_NAME(URMA_CR_WR_UNHANDLED);
    default: return "URMA_CR_UNKNOWN";
  }
#undef CR_NAME
}
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

} // namespace

class UrmaDestination {
 public:
  UrmaDestination(urma_context_t* context, void* memory, std::uint64_t bytes, std::uint32_t token)
      : context_(context), memory_(memory), bytes_(bytes), region_token_(token) {}
  ~UrmaDestination() { (void)Close(); }
  urma_target_seg_t* segment() const { return destination_; }
  bool Matches(urma_context_t* context, void* memory, std::uint64_t bytes, std::uint32_t token) const {
    return context_ == context && memory_ == memory && bytes_ == bytes && region_token_ == token;
  }
  Status Close() {
    if (destination_) {
      if (urma_unregister_seg(destination_)) return Status::InDoubt("unregister shared DDR destination");
      destination_ = nullptr;
    }
    if (token_) {
      if (urma_free_token_id(token_)) return Status::InDoubt("free shared DDR destination token");
      token_ = nullptr;
    }
    destination_valid_ = false;
    return Status::OK();
  }
  Status Ensure() {
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
 private:
  urma_context_t* context_{};
  void* memory_{};
  std::uint64_t bytes_{};
  std::uint32_t region_token_{};
  urma_token_id_t* token_{};
  urma_target_seg_t* destination_{};
  bool destination_valid_{};
};
std::shared_ptr<UrmaDestination> CreateUrmaDestination(urma_context_t* context, void* memory,
                                                    std::uint64_t bytes, std::uint32_t token) {
  return std::make_shared<UrmaDestination>(context, memory, bytes, token);
}
Status CloseUrmaDestination(const std::shared_ptr<UrmaDestination>& destination) {
  if (!destination) return Status::OK();
  if (destination.use_count() != 1) return Status::Busy("CPU Pull still retains DDR destination MR");
  return destination->Close();
}

namespace {
class UrmaBackend final : public Backend {
 public:
  explicit UrmaBackend(urma_context_t* context) : context_(context) {}
  UrmaBackend(urma_context_t* context, void* memory, std::uint64_t bytes,
              std::uint32_t token, std::shared_ptr<void> lease, std::shared_ptr<UrmaDestination> destination)
      : context_(context), memory_(memory), bytes_(bytes), region_token_(token),
        owns_memory_(false), lease_(std::move(lease)), borrowed_destination_(bool(destination)),
        shared_destination_(std::move(destination)) {}
  ~UrmaBackend() override {
    if (!Close().ok()) {
      std::fprintf(stderr, "CPU_PULL: retaining non-quiescent resources\n");
      if (lease_) (void)new std::shared_ptr<void>(std::move(lease_));
      if (shared_destination_) (void)new std::shared_ptr<UrmaDestination>(std::move(shared_destination_));
      // Keep descriptors/SGEs as well as MRs and the pool alive on unknown DMA.
      for (auto& q : queues_) (void)new Queue(std::move(q));
    }
  }
  Status Allocate(std::uint64_t bytes, MemoryDesc* local) override {
    if (!context_ || !local || memory_ || !bytes || bytes > SIZE_MAX) return Error("invalid allocation");
    if (posix_memalign(&memory_, 4096, bytes)) return Error("allocate DDR");
    std::memset(memory_, 0, bytes);
    bytes_ = bytes;
    region_token_ = Next(next_token);
    local->base = reinterpret_cast<std::uint64_t>(memory_);
    local->bytes = bytes;
    local->token = region_token_;
    return Status::OK();
  }

  Status Open(const ConnectionDesc& peer, const MemoryDesc& source, ConnectionDesc* local) override {
    if (!local || !memory_ || source_ || poisoned_ || source.segment_bytes != sizeof(urma_seg_t) ||
        (peer.key_bytes != 28 && peer.key_bytes != 32) || !peer.tp || !peer.psn || !source.token)
      return Error("invalid source endpoint");
    if (source.token != region_token_ || peer.token != region_token_) return Error("source token differs from region token");
    urma_seg_t segment{};
    std::memcpy(&segment, source.segment, sizeof(segment));
    if (!source.base || source.base > UINT64_MAX-source.bytes || source.base < segment.ubva.va ||
        !Contains(segment.len, source.base-segment.ubva.va, source.bytes)) return Error("source segment range");
    for (const auto& key : peer.key) {
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
    if (attr.dev_cap.max_read_size) max_bytes_ = std::min(max_bytes_, attr.dev_cap.max_read_size);
    if (!max_sge_ || max_bytes_ < 64) return Error("unsupported SGE/message limits");

    // Reference order: import HBM, register the receiving host buffer, query
    // CTP, then create/import four queues. Preserve exported memory/Jetty EIDs.
    auto status = AddSource(1, source);
    if (!status.ok()) return status;
    status = SelectSource(1);
    if (!status.ok()) return status;
    status = RegisterDestination();
    if (!status.ok()) return status;
    std::uint64_t tp{};
    if (!urma::QueryTp(query, context_, peer.eid, &tp)) return Error("query CTP");
    std::memcpy(local->eid, context_->eid.raw, 16);
    local->tp = tp;
    local->psn = Next(next_psn);
    local->token = region_token_;
    local->key_bytes = sizeof(urma_jetty_id_t) + sizeof(urma_transport_mode_t);
    for (std::size_t index = 0; index < queues_.size(); ++index) {
      auto& q = queues_[index];
      const auto* error = urma::CreateQueues(urma::NativeApi{}, context_, region_token_,
          {16384, kDepth, max_sge_, false, true}, q.cq, q.rq, q.jetty);
      if (error) return Error(error);
      q.peer = urma::ImportJetty(import, context_, peer.key[index], region_token_,
          tp, peer.tp, local->psn, peer.psn);
      if (!q.peer) return Error("import NPU Jetty " + std::to_string(index));
      std::memcpy(local->key[index], &q.jetty->jetty_id, sizeof(urma_jetty_id_t));
      std::memcpy(local->key[index]+sizeof(urma_jetty_id_t), &q.jetty->jetty_cfg.jfs_cfg.trans_mode,
                  sizeof(urma_transport_mode_t));
    }
    profile_ = Enabled("KVAF_CPU_PULL_PROFILE");
    if (profile_) {
      profile_id_ = ++next_profile_id;
      profile_batch_ = 0;
      const char* each = std::getenv("KVAF_CPU_PULL_PROFILE_EACH");
      profile_each_ = !each || std::strcmp(each, "0") != 0;
      profile_summary_.Reset(ProfileWarmup());
    }
    ready_ = true;
    std::fprintf(stderr, "[KVAF_CPU] CPU_PULL_REFERENCE commit=8bc5a2a endpoint_layout=%u sge_bytes=%zu "
        "jettys=%u max_sge=%u max_bytes=%u local_eid=%s peer_eid=%s psn=%u peer_psn=%u\n",
        kEndpointVersion, sizeof(urma::Sge), kChannelCount, max_sge_, max_bytes_,
        Eid(context_->eid.raw).c_str(), Eid(peer.eid).c_str(), local->psn, peer.psn);
    return Status::OK();
  }

  Status AddSource(std::uint64_t id, const MemoryDesc& source) override {
    if (poisoned_) return Status::InDoubt(failure_);
    if (!id || source.segment_bytes != sizeof(urma_seg_t) || source.token != region_token_)
      return Error("invalid source MR identity");
    for (const auto& q : queues_) if (q.pending) return Status::InDoubt("READ is pending");
    auto found = sources_.find(id);
    if (found != sources_.end())
      return std::memcmp(&found->second.endpoint, &source, sizeof(source)) == 0
          ? Status::OK() : Error("source MR identity changed");
    urma_seg_t segment{};
    std::memcpy(&segment, source.segment, sizeof(segment));
    if (!source.base || source.base > UINT64_MAX-source.bytes || source.base < segment.ubva.va ||
        !Contains(segment.len, source.base-segment.ubva.va, source.bytes)) return Error("source segment range");
    // Allocate bookkeeping before importing so an allocation failure cannot
    // orphan an imported segment.
    auto& memory = sources_[id]; memory.endpoint = source;
    urma_import_seg_flag_t flags{};
    flags.bs.cacheable = URMA_NON_CACHEABLE;
    flags.bs.access = (1U << 1U) | (1U << 2U);
    flags.bs.mapping = URMA_SEG_NOMAP;
    urma_token_t token{}; token.token = source.token;
    memory.segment = urma_import_seg(context_, &segment, &token, 0, flags);
    if (!memory.segment) { sources_.erase(id); return Error("import HBM MR"); }
    return Status::OK();
  }
  Status SelectSource(std::uint64_t id) override {
    if (poisoned_) return Status::InDoubt(failure_);
    for (const auto& q : queues_) if (q.pending) return Status::InDoubt("cannot change MR during READ");
    const auto found = sources_.find(id);
    if (found == sources_.end() || !found->second.segment) return Error("unknown source MR");
    if (selected_mr_ != id) cached_spans_.clear(); // Same offsets in another MR are a different plan.
    selected_mr_ = id;
    source_ = found->second.segment;
    source_base_ = found->second.endpoint.base;
    source_bytes_ = found->second.endpoint.bytes;
    return Status::OK();
  }

  Status Read(const std::vector<Span>& spans) override {
    Profile stats;
    const auto begin = profile_ ? Clock::now() : Clock::time_point{};
    const auto status = Transfer(spans, false, profile_ ? &stats : nullptr);
    if (profile_) {
      const auto total_us = Micros(Clock::now()-begin);
      ++profile_batch_;
      profile_summary_.Record(status.ok(), stats.build_us, stats.post_us, stats.cq_us, total_us);
      if (profile_each_) std::fprintf(stderr, "[KVAF_CPU] CPU_PULL_PROFILE mode=batch status=%u spans=%zu bytes=%llu "
          "build_us=%.1f post_us=%.1f cq_wait_us=%.1f total_us=%.1f read_wrs=%zu "
          "post_calls=%zu tail_cqes=%zu waves=%zu active_jettys=%zu plan_reused=%u cpu_threads=1 "
          "checksum_included=0 profile_id=%llu batch=%llu mr=%llu\n", unsigned(status.code()), spans.size(),
          static_cast<unsigned long long>(stats.bytes), stats.build_us, stats.post_us, stats.cq_us,
          total_us, stats.wrs, stats.posts, stats.cqes, stats.waves,
          stats.jettys, unsigned(stats.reused), (unsigned long long)profile_id_, (unsigned long long)profile_batch_,
          (unsigned long long)selected_mr_);
      if (!status.ok()) ReportProfile("read_error");
    }
    return status;
  }

  Status Probe(const std::vector<Span>& spans) override {
    return ProbeQueues(spans, nullptr);
  }

  Status ReleaseSource() override {
    if (poisoned_) return Status::InDoubt(failure_);
    for (const auto& q : queues_)
      if (q.pending) return Poison("release while READ completion is unresolved");
    ready_ = false;
    cached_spans_.clear();
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
    for (auto& item : sources_) { RELEASE(item.second.segment, urma_unimport_seg) }
    sources_.clear(); source_ = nullptr; selected_mr_ = 0;
    ReportProfile("source_release");
    profile_summary_.ReleaseSamples();
    source_base_ = source_bytes_ = 0;
    return Status::OK();
  }
  Status Close() override {
    auto status = ReleaseSource(); if (!status.ok()) return status;
    status = ReleaseScratch(); if (!status.ok()) return status;
    if (shared_destination_) {
      if (!borrowed_destination_) {
        status = shared_destination_->Close(); if (!status.ok()) return status;
      }
      destination_ = nullptr; shared_destination_.reset();
    }
#undef RELEASE
    if (owns_memory_) std::free(memory_);
    memory_ = nullptr; bytes_ = 0; lease_.reset();
    return Status::OK();
  }
  Status ProbeToScratch(std::uint64_t offset, std::uint64_t bytes, std::uint64_t* checksum) override {
    if (poisoned_) return Status::InDoubt(failure_);
    if (!ready_ || !checksum || bytes > 64 || !Contains(source_bytes_, offset, bytes))
      return Error("invalid scratch probe");
    if (posix_memalign(&scratch_, 4096, 4096)) return Error("allocate scratch probe");
    std::memset(scratch_, 0, 4096);
    scratch_token_ = urma_alloc_token_id(context_);
    if (!scratch_token_) return Error("allocate scratch token");
    urma_seg_cfg_t cfg{}; cfg.va = reinterpret_cast<std::uint64_t>(scratch_); cfg.len = 4096;
    cfg.token_id = scratch_token_; cfg.token_value.token = region_token_;
    cfg.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    cfg.flag.bs.access = 1; cfg.flag.bs.token_id_valid = 1; cfg.flag.bs.non_pin = 0;
    scratch_mr_ = urma_register_seg(context_, &cfg);
    if (!scratch_mr_) return Error("register scratch probe");
    if (!scratch_mr_->seg.ubva.va || cfg.va < scratch_mr_->seg.ubva.va ||
        !Contains(scratch_mr_->seg.len, cfg.va-scratch_mr_->seg.ubva.va, 4096) ||
        !(scratch_mr_->seg.attr.bs.access & 1U) || scratch_mr_->seg.attr.bs.non_pin)
      return Error("invalid scratch registration");
    auto* saved_memory = memory_; const auto saved_bytes = bytes_; auto* saved_mr = destination_;
    memory_ = scratch_; bytes_ = 4096; destination_ = scratch_mr_;
    Status status;
    try { status = ProbeQueues({{offset, 0, bytes}}, checksum); }
    catch (...) { status = Poison("exception during scratch READ"); }
    memory_ = saved_memory; bytes_ = saved_bytes; destination_ = saved_mr;
    // Unknown DMA retains scratch and pool MR as well as the source lease.
    if (status.code() == StatusCode::kInDoubt) return status;
    const auto released = ReleaseScratch();
    return released.ok() ? status : released;
  }
  std::uint64_t Checksum(const std::vector<Span>& spans) const override {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto& span : spans)
      hash = Hash(static_cast<const std::uint8_t*>(memory_) + span.destination, span.bytes, hash);
    return hash;
  }
 private:
  Status ProbeQueues(const std::vector<Span>& spans, std::uint64_t* checksum) {
    if (poisoned_) return Status::InDoubt(failure_);
    if (spans.size() != 1 || !spans[0].bytes || spans[0].bytes > 64)
      return Error("probe requires one span of at most 64 bytes");
    std::uint64_t first_hash{};
    for (std::size_t index = 0; index < queues_.size(); ++index) {
      completed_error_ = false;
      // Probe every Jetty, but allow only one outstanding WR. A terminal
      // error can retire this entire candidate without other DMA in flight.
      auto status = Transfer(spans, true, nullptr, index);
      if (!status.ok()) {
        if (!completed_error_) return status;
        const auto failure = failure_;
        queues_[index].pending = 0;
        poisoned_ = false;
        status = ReleaseSource();
        if (!status.ok()) {
          failure_ = failure + "; probe teardown failed: " + status.message();
          return Status::InDoubt(failure_);
        }
        failure_.clear(); completed_error_ = false;
        return Status::Internal(failure + "; probe queues and source retired");
      }
      if (checksum) {
        const auto hash = Hash(static_cast<const std::uint8_t*>(memory_)+spans[0].destination, spans[0].bytes);
        if (index && hash != first_hash)
          return Error("scratch checksum differs across Jettys: queue=" + std::to_string(index));
        first_hash = hash;
      }
    }
    if (checksum) *checksum = first_hash; // Producer verifies the same bytes too.
    return Status::OK();
  }
  void ReportProfile(const char* reason) {
    if (profile_) profile_summary_.Report(stderr, profile_id_, source_base_,
        reinterpret_cast<std::uint64_t>(memory_), reason);
  }
  struct Profile {
    std::uint64_t bytes{};
    std::size_t wrs{}, posts{}, cqes{}, waves{}, jettys{};
    double build_us{}, post_us{}, cq_us{};
    bool reused{};
  };
  bool SamePlan(const std::vector<Span>& spans) const {
    return !cached_spans_.empty() && cached_spans_.size() == spans.size() &&
        std::equal(spans.begin(), spans.end(), cached_spans_.begin(), [](const Span& a, const Span& b) {
          return a.source == b.source && a.destination == b.destination && a.bytes == b.bytes;
        });
  }
  void BuildWave(const std::vector<Span>& spans, std::size_t& cursor,
                 std::uint64_t& offset, std::size_t channels, std::size_t depth,
                 std::size_t first_queue = 0) {
    for (auto& q : queues_) {
      q.local.clear(); q.remote.clear(); q.wrs.clear();
    }
    for (std::size_t n = 0; n < channels*depth && cursor < spans.size(); ++n) {
      const auto& span = spans[cursor];
      const auto length = static_cast<std::uint32_t>(std::min<std::uint64_t>(max_bytes_, span.bytes-offset));
      auto& q = queues_[(first_queue + n % channels) % queues_.size()];
      q.local.push_back({reinterpret_cast<std::uint64_t>(memory_)+span.destination+offset,
                         length, destination_, nullptr});
      q.remote.push_back({source_base_+span.source+offset, length, source_, nullptr});
      offset += length;
      if (offset == span.bytes) { ++cursor; offset = 0; }
    }
    for (auto& q : queues_) {
      q.wrs.resize(q.local.size());
      // Bind pointers only after vectors stop growing; no reallocation until
      // ALL queue tails have completed. Remote READ SGLs always have one entry.
      for (std::size_t i = 0; i < q.wrs.size(); ++i) {
        auto& wr = q.wrs[i];
        wr = {};
        wr.opcode = URMA_OPC_READ;
        wr.flag.bs.comp_order = 1;
        const bool tail = i+1 == q.wrs.size();
        wr.flag.bs.place_order = tail ? URMA_STRONG_ORDER : URMA_RELAX_ORDER;
        wr.flag.bs.complete_enable = tail;
        // WRITE's tail signalling alone is not the whole READ contract:
        // explicitly fence the tail against all preceding READs on this SQ.
        wr.flag.bs.fence = tail && i != 0;
        wr.tjetty = q.peer;
        wr.rw.src = {reinterpret_cast<urma_sge_t*>(&q.remote[i]), 1};
        wr.rw.dst = {reinterpret_cast<urma_sge_t*>(&q.local[i]), 1};
        wr.next = !tail && (i+1)%kDoorbellWrs ? &q.wrs[i+1] : nullptr;
      }
    }
  }
  Status PostWave(Profile* stats) {
    for (std::size_t index = 0; index < queues_.size(); ++index) {
      auto& q = queues_[index];
      for (std::size_t first = 0; first < q.wrs.size(); first += kDoorbellWrs) {
        urma_jfs_wr_t* bad{};
        errno = 0;
        const int ret = urma_post_jetty_send_wr(q.jetty, &q.wrs[first], &bad);
        const int error_number = errno;
        if (stats) ++stats->posts;
        // A failing call may have posted a prefix, including other queues.
        // Do not reset/rebuild any descriptors or authorize source reuse.
        if (ret) return ReadFailure(index, "post_read", ret, error_number, bad);
      }
    }
    return Status::OK();
  }
  Status PollWave(bool probe, Clock::time_point deadline, Profile* stats) {
    std::uint32_t sweeps{};
    for (;;) {
      std::size_t pending = 0;
      for (std::size_t n = 0; n < queues_.size(); ++n) {
        const auto index = (next_poll_+n)%queues_.size();
        auto& q = queues_[index];
        if (!q.pending) continue;
        std::array<urma_cr_t, 8> records{};
        errno = 0;
        const int count = urma_poll_jfc(q.cq, records.size(), records.data());
        const int error_number = errno;
        if (count < 0 || count > static_cast<int>(records.size()))
          return ReadFailure(index, "poll_cq", count, error_number);
        if (count > 1) return ReadFailure(index, "unexpected_cq_count", count, error_number, nullptr, &records[0]);
        if (count) {
          const auto& cr = records[0];
          if (cr.status != URMA_CR_SUCCESS || cr.user_ctx != q.pending || cr.flag.bs.s_r) {
            // Only a single probe's exact terminal error CQE can allow EID
            // fallback. Any error in a multi-WR payload quarantines ALL queues.
            completed_error_ = probe && cr.user_ctx == q.pending &&
                               !cr.flag.bs.s_r && cr.status != URMA_CR_SUCCESS;
            return ReadFailure(index, "cq_completion", count, error_number, nullptr, &cr);
          }
          q.pending = 0;
          if (stats) ++stats->cqes;
        }
        pending += q.pending != 0;
      }
      next_poll_ = (next_poll_+1)%queues_.size();
      if (!pending) return Status::OK();
      if ((++sweeps & 31) == 0 && Clock::now() >= deadline) {
        for (std::size_t i = 0; i < queues_.size(); ++i)
          if (queues_[i].pending) return ReadFailure(i, "cq_timeout", 0, 0);
      }
    }
  }
  Status Transfer(const std::vector<Span>& spans, bool probe, Profile* stats, std::size_t probe_queue = 0) {
    if (poisoned_) return Status::InDoubt(failure_);
    if (!ready_ || !source_ || !destination_) return Error("READ transport is not ready");
    const bool reuse = !probe && SamePlan(spans);
    // A cached plan already passed range/overlap validation against these MRs.
    // Exact equality avoids sorting thousands of unchanged spans every batch.
    if (!reuse && !ValidateSpans(spans, source_bytes_, bytes_)) return Error("invalid READ spans");
    // Each startup probe has exactly one READ on its selected queue.
    std::size_t depth = probe ? 1 : kWaveDepth;
    for (const auto& q : queues_) {
      if (q.pending) return Poison("READ while another wave is pending");
      depth = std::min<std::size_t>(depth, q.jetty->jetty_cfg.jfs_cfg.depth);
    }
    if (!depth) return Error("zero READ SQ depth");
    if (stats) for (const auto& span : spans) stats->bytes += span.bytes;
    if (!reuse) cached_spans_.clear();
    std::size_t cursor = 0;
    std::uint64_t offset = 0;
    while (cursor < spans.size()) {
      const auto build_begin = stats ? Clock::now() : Clock::time_point{};
      const bool first_wave = cursor == 0 && offset == 0;
      if (reuse) cursor = spans.size();
      else BuildWave(spans, cursor, offset, probe ? 1 : queues_.size(), depth, probe ? probe_queue : 0);
      // Cache only a single bounded wave, keyed by ALL offsets and lengths.
      // Larger requests stream through the same storage without growing it.
      if (!probe && !reuse && first_wave && cursor == spans.size()) cached_spans_ = spans;
      for (const auto& q : queues_)
        if (q.sequence > UINT64_MAX-q.wrs.size()) return Error("READ completion sequence exhausted");
      std::size_t active = 0;
      for (auto& q : queues_) {
        if (q.wrs.empty()) continue;
        for (auto& wr : q.wrs) wr.user_ctx = ++q.sequence;
        q.pending = q.wrs.back().user_ctx;
        ++active;
        if (stats) stats->wrs += q.wrs.size();
      }
      const auto post_begin = Clock::now();
      if (stats) {
        stats->build_us += Micros(post_begin-build_begin);
        ++stats->waves; stats->jettys = std::max(stats->jettys, active); stats->reused = reuse;
      }
      auto status = PostWave(stats);  // All four queues are posted before polling any CQ.
      const auto posted = stats ? Clock::now() : Clock::time_point{};
      if (stats) stats->post_us += Micros(posted-post_begin);
      if (!status.ok()) return status;
      status = PollWave(probe, post_begin+kReadTimeout, stats);
      if (stats) stats->cq_us += Micros(Clock::now()-posted);
      if (!status.ok()) return status;
    }
    return Status::OK();
  }
  Status ReleaseScratch() {
    if (scratch_mr_) {
      if (urma_unregister_seg(scratch_mr_)) return Poison("unregister scratch probe");
      scratch_mr_ = nullptr;
    }
    if (scratch_token_) {
      if (urma_free_token_id(scratch_token_)) return Poison("free scratch token");
      scratch_token_ = nullptr;
    }
    std::free(scratch_); scratch_ = nullptr; return Status::OK();
  }
  Status RegisterDestination() {
    if (!shared_destination_)
      shared_destination_ = CreateUrmaDestination(context_, memory_, bytes_, region_token_);
    if (!shared_destination_->Matches(context_, memory_, bytes_, region_token_))
      return Error("shared DDR destination belongs to a different pool/context");
    auto status = shared_destination_->Ensure();
    if (status.ok()) destination_ = shared_destination_->segment();
    return status;
  }
  Status ReadFailure(std::size_t index, const char* stage, int ret, int error_number,
                     const urma_jfs_wr_t* bad = nullptr, const urma_cr_t* cr = nullptr) {
    const auto& q = queues_[index];
    std::size_t failed = q.wrs.size()-1;
    for (std::size_t i = 0; i < q.wrs.size(); ++i)
      if (bad == &q.wrs[i] || (cr && cr->user_ctx == q.wrs[i].user_ctx)) { failed = i; break; }
    std::ostringstream message;
    message << "stage=" << stage << " ret=" << ret << " errno=" << error_number
            << " mr=" << selected_mr_
            << " sge_bytes=" << sizeof(urma::Sge) << " queue=" << index
            << " wr_id=" << q.wrs[failed].user_ctx << " pending_tail=" << q.pending
            << " wave_wrs=" << q.wrs.size() << " wr_index=" << failed
            << " src=0x" << std::hex << q.remote[failed].addr << " dst=0x" << q.local[failed].addr << std::dec
            << " bytes=" << q.remote[failed].len << " bad_wr_nonnull=" << (bad != nullptr)
            << " bad_wr_is_current=" << (bad == &q.wrs[failed]);
    if (cr) message << " cq_status=" << static_cast<unsigned>(cr->status)
                    << " cq_user_ctx=" << cr->user_ctx << " cq_flags=" << unsigned(cr->flag.value)
                    << " local_id=" << cr->local_id << " tpn=" << cr->tpn;
    if (cr) message << " cq_status_name=" << CompletionStatusName(cr->status);
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
  urma_target_seg_t *destination_{}, *source_{};
  struct SourceMemory { MemoryDesc endpoint{}; urma_target_seg_t* segment{}; };
  std::unordered_map<std::uint64_t, SourceMemory> sources_;
  std::uint64_t selected_mr_{};
  std::array<Queue, kChannelCount> queues_;
  std::vector<Span> cached_spans_;
  std::size_t next_poll_{};
  bool profile_{};
  bool profile_each_{true};
  std::uint64_t profile_id_{}, profile_batch_{};
  CpuPullProfileSummary profile_summary_;
  bool ready_{}, poisoned_{}, completed_error_{};
  std::string failure_;
  bool owns_memory_{true};
  std::shared_ptr<void> lease_;
  bool borrowed_destination_{};
  std::shared_ptr<UrmaDestination> shared_destination_;
  void* scratch_{};
  urma_token_id_t* scratch_token_{};
  urma_target_seg_t* scratch_mr_{};
};
}
std::unique_ptr<Backend> CreateUrmaBackend(urma_context_t* context) {
  return context ? std::make_unique<UrmaBackend>(context) : nullptr;
}
std::unique_ptr<Backend> CreateBorrowedUrmaBackend(urma_context_t* context, void* memory,
    std::uint64_t bytes, std::uint32_t token, std::shared_ptr<void> lease,
    std::shared_ptr<UrmaDestination> destination) {
  return context && memory && bytes && token && lease
      ? std::make_unique<UrmaBackend>(context, memory, bytes, token, std::move(lease), std::move(destination)) : nullptr;
}
}
