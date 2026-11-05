#include "cpu_push_get_transport.h"
#include "urma_device_abi.h"
#include "batch_get_profile.h"

#include <chrono>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <sstream>

namespace kvaf::cpu_push {
namespace {
Status Error(const char* operation) { return Status::Internal(std::string("CPU push: ") + operation); }
constexpr std::size_t kJettyDepth = 8192;
#ifdef __OPTIMIZE__
constexpr unsigned kOptimized = 1;
#else
constexpr unsigned kOptimized = 0;
#endif
constexpr std::uint32_t kMaxChunks = (kMaxRequests + kChunkRows - 1) / kChunkRows;
const char* CqStatusName(urma_cr_status_t status) {
  switch (status) {
    case URMA_CR_SUCCESS: return "SUCCESS";
    case URMA_CR_ACK_TIMEOUT_ERR: return "ACK_TIMEOUT";
    case URMA_CR_LOC_ACCESS_ERR: return "LOCAL_ACCESS_ERROR";
    case URMA_CR_REM_ACCESS_ABORT_ERR: return "REMOTE_ACCESS_ABORT";
    case URMA_CR_RNR_RETRY_CNT_EXC_ERR: return "RNR_RETRY_EXCEEDED";
    default: return "OTHER";
  }
}
std::string CqDetails(std::size_t queue, const urma_cr_t& cr,
                      const cpu_runtime_detail::UrmaChannel& channel) {
  std::ostringstream out;
  out << " queue=" << queue << " status=" << static_cast<unsigned>(cr.status)
      << " status_name=" << CqStatusName(cr.status)
      << " actual_ctx=0x" << std::hex << cr.user_ctx << std::dec
      << " flags=" << static_cast<unsigned>(cr.flag.value)
      << " completion_bytes=" << cr.completion_len << " local_id=" << cr.local_id
      << " jetty_id=" << channel.local_jetty->jetty_id.id
      << " cq_id=" << channel.local_jfc->jfc_id.id << " tpn=" << cr.tpn
      << " local_tp=0x" << std::hex << channel.tp.local
      << " peer_tp=0x" << channel.tp.peer
      << " initial_tx_psn=0x" << channel.tp.local_psn
      << " initial_rx_psn=0x" << channel.tp.peer_psn << std::dec;
  return out.str();
}
}  // namespace

Transport::~Transport() {
  if (!Shutdown().ok()) {
    // Outstanding DMA must retain its WR/SGL storage and registered buffers.
    // Owners call Shutdown before destruction; a failed transport is quarantined.
    std::fprintf(stderr, "CPU_PUSH: transport destroyed before successful Shutdown\n");
    std::terminate();
  }
}

Status Transport::Initialize(urma_context_t* context, urma_target_seg_t* source,
    const peer::OpenCpuPushGetRequest& setup,
    urma::GetTpList query, urma::ImportJettyEx import, std::uint32_t psn,
    peer::OpenCpuPushGetResponse* response) {
  if (context_ || !context || !source || !query || !import || !response || !psn ||
      !setup.npu_tp || !setup.npu_psn || !setup.token_value ||
      setup.segment_bytes != sizeof(urma_seg_t) || !setup.source_bytes ||
      !MakeLayout(setup.token_bytes, setup.workers, &layout_) ||
      setup.hbm_base % 64 || setup.hbm_bytes < layout_.bytes ||
      setup.hbm_base > UINT64_MAX - setup.hbm_bytes)
    return Status::InvalidArgument("invalid CPU push setup");
  context_ = context;
  if (const auto* value = std::getenv("KVAF_CPU_PUSH_POST_BATCH_WRS")) {
    char* end = nullptr;
    errno = 0;
    const auto limit = std::strtoul(value, &end, 10);
    if (errno || end == value || *end || !limit || limit > kJettyDepth)
      return Status::InvalidArgument("KVAF_CPU_PUSH_POST_BATCH_WRS must be in 1..8192");
    post_batch_wrs_ = static_cast<std::uint32_t>(limit);
  }
  session_ = setup.session_id;
  source_ = source;
  hbm_base_ = setup.hbm_base;
  workers_ = setup.workers;
  token_bytes_ = setup.token_bytes;
  urma::UbDeviceAttributes attr{};
  if (urma_query_device(context->dev, reinterpret_cast<urma_device_attr_t*>(&attr)))
    return Error("query device");
  max_sge_ = std::min(16U, attr.dev_cap.max_jfs_sge);
  max_bytes_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(65536, attr.dev_cap.max_msg_size));
  if (!max_sge_ || max_bytes_ < sizeof(Request)) return Error("unsupported SGE/message limits");
  std::fprintf(stderr, "[KVAF_CPU] CPU_PUSH_CAP max_sge=%u message_bytes=%u workers=%u token_bytes=%u "
               "post_batch_wrs=%u optimized=%u source_refresh=%s\n",
               max_sge_, max_bytes_, workers_, token_bytes_, post_batch_wrs_, kOptimized,
               std::uint64_t{max_sge_} * token_bytes_ <= max_bytes_ ? "contiguous_runs_v1" : "fragment_map");
  urma_seg_t segment{};
  std::memcpy(&segment, setup.segment, sizeof(segment));
  if (!SourceRange(segment.ubva.va, segment.len, hbm_base_, layout_.bytes))
    return Error("HBM workspace outside exported segment");
  urma_import_seg_flag_t flag{};
  // Match the probe's READ/WRITE-only HBM import, without remote atomics.
  flag.bs.cacheable = URMA_NON_CACHEABLE;
  flag.bs.access = (1U << 1U) | (1U << 2U);
  flag.bs.mapping = URMA_SEG_NOMAP;
  urma_token_t token{};
  token.token = setup.token_value;
  remote_ = urma_import_seg(context_, &segment, &token, 0, flag);
  if (!remote_) return Error("import HBM segment");
  void* buffer = nullptr;
  // Each in-flight ready WRITE has its own immutable registered source word.
  const auto marker_bytes = workers_ * kMaxChunks * kJettyCount * sizeof(std::uint64_t);
  const auto bytes = (layout_.staging + marker_bytes + 4095) & ~std::uint64_t{4095};
  if (posix_memalign(&buffer, 4096, bytes)) return Error("allocate control buffer");
  memory_ = static_cast<std::uint8_t*>(buffer);
  std::memset(memory_, 0, bytes);
  control_token_ = urma_alloc_token_id(context_);
  if (!control_token_) return Error("allocate control token");
  urma_seg_cfg_t cfg{};
  cfg.va = reinterpret_cast<std::uint64_t>(memory_);
  cfg.len = bytes;
  cfg.token_id = control_token_;
  cfg.token_value.token = setup.token_value;
  // AIV pushes header/list here before publishing notify. The same pinned
  // registration supplies immutable local Ready/done WRITE source words.
  cfg.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
  // Public UMDK bit 0 is LOCAL_ONLY (exclusive), not a combinable local
  // write flag. Remote WRITE registration also requires remote READ.
  cfg.flag.bs.access = (1U << 1U) | (1U << 2U);
  cfg.flag.bs.non_pin = 0;
  cfg.flag.bs.token_id_valid = 1;
  control_ = urma_register_seg(context_, &cfg);
  if (!control_) return Error("register control buffer");
  response->mailbox_base = control_->seg.ubva.va;
  response->mailbox_bytes = control_->seg.len;
  response->mailbox_token_id = control_token_->token_id;
  response->mailbox_segment_bytes = sizeof(urma_seg_t);
  response->max_write_bytes = max_bytes_;
  std::memcpy(response->mailbox_segment, &control_->seg, sizeof(urma_seg_t));
  std::uint64_t cpu_tp = 0;
  if (!urma::QueryTp(query, context_, setup.eid, &cpu_tp)) return Error("query CTP");
  response->cpu_tp = cpu_tp;
  response->cpu_psn = psn;
  for (std::uint32_t q = 0; q < kJettyCount; ++q) {
    if (setup.channels[q].bytes != 28 && setup.channels[q].bytes != 32)
      return Error("invalid NPU Jetty key");
    auto& queue = queues_[q];
    auto& c = queue.channel;
    const auto* error = urma::CreateQueues(urma::NativeApi{}, context_, setup.token_value,
        {16384, kJettyDepth, max_sge_, false, true}, c.local_jfc, c.local_jfr, c.local_jetty);
    if (error) return Error(error);
    c.remote_jetty = urma::ImportJetty(import, context_, setup.channels[q].key,
        setup.token_value, cpu_tp, setup.npu_tp, psn, setup.npu_psn);
    if (!c.remote_jetty) return Error("import NPU Jetty");
    // Keep the negotiated initial values for CQ failure diagnostics. These
    // are not the live hardware packet sequence counters.
    std::copy_n(setup.eid, c.tp.peer_eid.size(), c.tp.peer_eid.begin());
    c.tp.local = cpu_tp; c.tp.peer = setup.npu_tp;
    c.tp.local_psn = psn; c.tp.peer_psn = setup.npu_psn;
    auto& key = response->channels[q];
    key.bytes = sizeof(urma_jetty_id_t) + sizeof(urma_transport_mode_t);
    std::memcpy(key.key, &c.local_jetty->jetty_id, sizeof(urma_jetty_id_t));
    std::memcpy(key.key + sizeof(urma_jetty_id_t), &c.local_jetty->jetty_cfg.jfs_cfg.trans_mode,
                sizeof(urma_transport_mode_t));
  }
  PrepareCompletions();
  return Status::OK();
}

void Transport::PrepareCompletions() {
  auto& done = done_.queues[0];
  done.local.push_back({reinterpret_cast<std::uint64_t>(memory_ + layout_.done),
                        sizeof(std::uint64_t), control_, nullptr});
  // Publish only the aligned token. A profiled completion writes the report
  // first and strongly orders the token behind it in the same submission.
  AppendPrepared(done, 0, hbm_base_ + layout_.done, sizeof(std::uint64_t), 1, true);
  SealPrepared(done_);
  auto& profiled = profiled_done_.queues[0];
  profiled.local.push_back({reinterpret_cast<std::uint64_t>(memory_ + layout_.done + 8),
                           sizeof(Completion::reserved), control_, nullptr});
  AppendPrepared(profiled, 0, hbm_base_ + layout_.done + 8, sizeof(Completion::reserved), 1);
  profiled.local.push_back(done.local.front());
  AppendPrepared(profiled, 0, hbm_base_ + layout_.done, sizeof(std::uint64_t), 1, true);
  SealPrepared(profiled_done_);
}

Status Transport::ReadRequest(std::uint64_t* sequence) {
  // Strongly ordered remote WRITE publishes notify only after the request and
  // descriptor list. Acquire also prevents compiler/CPU reordering of local
  // mailbox loads; no URMA READ or CQ round trip is needed while idle.
  const auto notify = __atomic_load_n(
      reinterpret_cast<const std::uint64_t*>(memory_ + layout_.notify), __ATOMIC_ACQUIRE);
  *sequence = notify & ~kProfileNotify;
  profile_.enabled = (notify & kProfileNotify) != 0;
  return Status::OK();
}
Status Transport::ReadDescriptor(std::uint64_t sequence) {
  const bool enabled = profile_.enabled;
  profile_ = {};
  profile_.enabled = enabled;
  const auto& r = request();
  if (r.magic != kMagic || r.sequence != sequence || !sequence || sequence > UINT32_MAX ||
      r.count > kMaxRequests || r.token_bytes != token_bytes_ || r.workers != workers_)
    return Status::InvalidArgument("CPU push request header mismatch");
  if ((r.reserved & kHeadStart) &&
      (r.reserved_words[0] != kChunkRows || r.count <= kChunkRows))
    return Status::InvalidArgument("CPU push descriptor prefix mismatch");
  return Status::OK();
}

// Each prepared chain signals only its queue's strongly ordered tail.
void Transport::AppendPrepared(PreparedQueue& q, std::uint32_t queue,
    std::uint64_t remote, std::uint32_t bytes, std::uint32_t sges, bool ordered) {
  q.remote.push_back({remote, bytes, remote_, nullptr});
  urma_jfs_wr_t wr{};
  wr.opcode = URMA_OPC_WRITE;
  wr.flag.bs.comp_order = 1;
  wr.flag.bs.place_order = ordered ? URMA_STRONG_ORDER : URMA_RELAX_ORDER;
  wr.tjetty = queues_[queue].channel.remote_jetty;
  wr.rw.src.num_sge = sges;
  wr.rw.dst.num_sge = 1;
  q.wrs.push_back(wr);
}
void Transport::SealPrepared(PreparedWrite& plan) {
  for (std::uint32_t index = 0; index < kJettyCount; ++index) {
    auto& q = plan.queues[index];
    if (q.wrs.empty()) continue;
    std::uint32_t local = 0, payloads = 0;
    for (std::size_t i = 0; i < q.wrs.size(); ++i) {
      auto& wr = q.wrs[i];
      const bool payload = q.remote[i].addr >= hbm_base_ + layout_.staging;
      // Budget payload WRs. Their strongly ordered Ready WR belongs to the
      // same submission; it must not force an extra doorbell at the boundary.
      if (!i || (payload && payloads == post_batch_wrs_)) {
        if (i) { q.wrs[i - 1].next = nullptr; q.posts.back().end_sge = local; }
        q.posts.push_back({static_cast<std::uint32_t>(i), local, 0});
        payloads = 0;
      }
      payloads += payload;
      wr.rw.src.sge = reinterpret_cast<urma_sge_t*>(q.local.data() + local);
      wr.rw.dst.sge = reinterpret_cast<urma_sge_t*>(q.remote.data() + i);
      local += wr.rw.src.num_sge;
      wr.next = i + 1 < q.wrs.size() ? &q.wrs[i + 1] : nullptr;
    }
    q.posts.back().end_sge = local;
    q.wrs.back().flag.bs.complete_enable = 1;
    q.wrs.back().flag.bs.place_order = URMA_STRONG_ORDER;
    plan.fences[plan.fence_count++] = {index, q.wrs.size(), 0};
  }
}
Status Transport::PrepareWrites(std::uint32_t count) {
  if (!tracker_.Empty()) return Error("prepare while WRITE is in flight");
  const bool batch_done_only = (request().reserved & kBatchDoneOnly) != 0;
  const auto head_rows = (request().reserved & kHeadStart) ? kChunkRows : 0U;
  if (prepared_count_ == count && prepared_batch_done_only_ == batch_done_only &&
      prepared_head_rows_ == head_rows) {
    profile_.plan_reused = true;
    return Status::OK();
  }
  prepared_.clear();
  const auto build_stage = [&](Range stage, bool suffix) {
    std::array<std::vector<PreparedQueue>, kJettyCount> shards;
    const bool unsplit = std::uint64_t{max_sge_} * token_bytes_ <= max_bytes_;
    for (std::uint32_t queue = 0; queue < kJettyCount; ++queue) {
      auto& waves = shards[queue];
      waves.emplace_back();
      const auto append = [&](std::uint32_t worker, std::uint32_t chunk, Range range) {
        const auto shard = JettyRange(range, queue);
        const auto begin = batch_done_only ? shard.begin : std::max(shard.begin, stage.begin);
        const auto end = batch_done_only ? shard.end : std::min(shard.end, stage.end);
        if (head_rows && begin >= end) return;
        auto row = begin;
        std::uint32_t offset = 0;
        while (row < end) {
          // The probe budgets source SGEs as well as WRs against SQ depth.
          // Continue large requests in another drained wave without narrowing
          // BatchGet's existing maximum request/token sizes.
          if (waves.back().local.size() + max_sge_ > kJettyDepth) waves.emplace_back();
          auto& q = waves.back();
          if (batch_done_only && q.local.empty() && unsplit)
            q.first_row = row;
          const auto first_local = static_cast<std::uint32_t>(q.local.size());
          const auto first_row = row;
          const auto destination = hbm_base_ + layout_.staging + std::uint64_t{row} * token_bytes_ + offset;
          std::uint32_t bytes = 0, sges = 0;
          while (row < end && sges < max_sge_ && bytes < max_bytes_) {
            const auto length = std::min(token_bytes_ - offset, max_bytes_ - bytes);
            if (!unsplit)
              q.sources.push_back({static_cast<std::uint32_t>(q.local.size()), row, offset});
            q.local.push_back({0, length, source_, nullptr});
            ++sges; bytes += length; offset += length;
            if (offset == token_bytes_) { ++row; offset = 0; }
          }
          if (unsplit && !batch_done_only) {
            if (!q.source_runs.empty() && q.source_runs.back().local + q.source_runs.back().count == first_local &&
                q.source_runs.back().row + q.source_runs.back().count == first_row) {
              q.source_runs.back().count += sges;
            } else q.source_runs.push_back({first_local, first_row, sges});
          }
          AppendPrepared(q, queue, destination, bytes, sges);
        }
        if (batch_done_only || (head_rows && end < shard.end)) return;
        if (waves.back().local.size() == kJettyDepth) waves.emplace_back();
        auto& q = waves.back();
        auto* marker = reinterpret_cast<std::uint64_t*>(memory_ + layout_.staging) +
            (worker * kMaxChunks + chunk) * kJettyCount + queue;
        q.markers.push_back({marker, chunk + 1, static_cast<std::uint32_t>(q.local.size())});
        q.local.push_back({reinterpret_cast<std::uint64_t>(marker), 8, control_, nullptr});
        AppendPrepared(q, queue, hbm_base_ + layout_.ready + worker * sizeof(Ready) + queue * 8,
                       8, 1, true);
      };
      // Whole-batch completion has no CPU worker/chunk boundaries. Partition
      // this stage across Jettys; all ordered CQ tails still finish before
      // Complete publishes done on queue 0.
      if (batch_done_only) append(0, 0, stage);
      else ForEachPipelineChunk(count, token_bytes_, workers_, append);
    }
    std::size_t waves = 0;
    for (const auto& shard : shards) waves = std::max(waves, shard.size());
    for (std::size_t w = 0; w < waves; ++w) {
      auto plan = std::make_unique<PreparedWrite>();
      plan->suffix = suffix;
      for (std::uint32_t q = 0; q < kJettyCount; ++q)
        if (w < shards[q].size()) plan->queues[q] = std::move(shards[q][w]);
      SealPrepared(*plan);
      if (plan->fence_count) prepared_.push_back(std::move(plan));
    }
  };
  if (head_rows) build_stage({0, head_rows}, false);
  build_stage({head_rows, count}, head_rows != 0);
  prepared_count_ = count;
  prepared_batch_done_only_ = batch_done_only;
  prepared_head_rows_ = head_rows;
  return Status::OK();
}
void Transport::RefreshPrepared(PreparedQueue& q, const PreparedQueue::Post& post) {
  const auto* addresses = reinterpret_cast<const std::uint64_t*>(memory_ + layout_.descriptor);
  if (q.first_row != UINT32_MAX) {
    // No per-SGE mapping walk or dependent vector indexing for ordinary small
    // tokens. Refresh on every call: plans are reusable, source addresses are not.
    const auto* input = addresses + q.first_row;
    auto* output = q.local.data();
    for (auto i = post.first_sge; i < post.end_sge; ++i) output[i].addr = input[i];
  } else if (!q.source_runs.empty()) {
    auto it = std::lower_bound(q.source_runs.begin(), q.source_runs.end(), post.first_sge,
        [](const PreparedQueue::SourceRun& r, std::uint32_t i) { return r.local + r.count <= i; });
    for (; it != q.source_runs.end() && it->local < post.end_sge; ++it) {
      const auto first = std::max(it->local, post.first_sge);
      const auto end = std::min(it->local + it->count, post.end_sge);
      for (auto i = first; i < end; ++i) q.local[i].addr = addresses[it->row + i - it->local];
    }
  } else {
    auto it = std::lower_bound(q.sources.begin(), q.sources.end(), post.first_sge,
        [](const PreparedQueue::Source& s, std::uint32_t i) { return s.local < i; });
    for (; it != q.sources.end() && it->local < post.end_sge; ++it)
      q.local[it->local].addr = addresses[it->row] + it->offset;
  }
  auto marker = std::lower_bound(q.markers.begin(), q.markers.end(), post.first_sge,
      [](const PreparedQueue::Marker& m, std::uint32_t i) { return m.local < i; });
  for (; marker != q.markers.end() && marker->local < post.end_sge; ++marker)
    *marker->value = Progress(request().sequence, marker->progress);
}
Status Transport::BeginPrepared(PreparedWrite& plan, std::uint64_t id, Profile* drain_profile,
                                  bool drain_for_credits) {
  for (const auto& q : queues_)
    if (q.poisoned) return Error("queue unavailable");
  bool submitted = false;
  auto status = tracker_.TryBegin(id, plan.handle, plan.fences.data(), plan.fence_count, &submitted);
  if (!status.ok()) return status;
  if (!submitted && drain_for_credits) {
    status = DrainPrepared(drain_profile);
    if (!status.ok()) return status;
    status = tracker_.TryBegin(id, plan.handle, plan.fences.data(), plan.fence_count, &submitted);
    if (!status.ok()) return status;
  }
  if (!submitted) return Error("drained queue has insufficient credits");
  for (std::size_t i = 0; i < plan.fence_count; ++i) {
    const auto& fence = plan.fences[i];
    plan.queues[fence.queue].wrs.back().user_ctx = fence.context;
  }
  return Status::OK();
}
Status Transport::PostPrepared(PreparedWrite& plan, bool refresh_sources,
                               std::uint64_t* first_post_ns) {
  std::size_t largest = 0;
  for (const auto& q : plan.queues) largest = std::max(largest, q.posts.size());
  // Start all four Jettys before posting the next slice of any queue. A long
  // queue must not delay the first doorbell to the remaining three Jettys.
  for (std::size_t batch = 0; batch < largest; ++batch) {
    for (std::uint32_t q = 0; q < kJettyCount; ++q) {
      auto& queue = plan.queues[q];
      if (batch >= queue.posts.size()) continue;
      const auto& post = queue.posts[batch];
      // Refresh only this submission. Posted SGEs/marker words stay immutable
      // until CQ completion, while other queues/batches can be prepared.
      if (refresh_sources) RefreshPrepared(queue, post);
      urma_jfs_wr_t* bad = nullptr;
      if (first_post_ns && !*first_post_ns) *first_post_ns = BatchGetProfileNowNs();
      if (urma_post_jetty_send_wr(queues_[q].channel.local_jetty,
                                &queue.wrs[post.wr], &bad)) {
        queues_[q].poisoned = true;
        return Error("post prepared WRITE chain");
      }
    }
  }
  return Status::OK();
}
Status Transport::PollPrepared(std::uint64_t id, Profile* profile) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);
  std::uint32_t polls = 0;
  for (;;) {
    std::array<urma_cr_t, 8> crs;
    std::size_t count = 0;
    auto status = cpu_urma_detail::SweepAsyncCqs(kJettyCount, next_poll_queue_, 8, 1, &count,
        [&](std::size_t q) { return tracker_.Pending(q); },
        [&](std::size_t q, std::size_t limit, std::size_t* polled) -> Status {
          const auto n = urma_poll_jfc(queues_[q].channel.local_jfc, static_cast<int>(limit), crs.data());
          if (n < 0) { queues_[q].poisoned = true; return Error("poll prepared CQ"); }
          *polled = static_cast<std::size_t>(n);
          if (profile) profile->tail_cqes += n;
          return Status::OK();
        },
        [&](std::size_t q, std::size_t index, bool* finished) -> Status {
          const auto& cr = crs[index];
          if (cr.status != URMA_CR_SUCCESS) {
            queues_[q].poisoned = true;
            return Status::Internal("CPU push: prepared CQ failed" +
                                    CqDetails(q, cr, queues_[q].channel));
          }
          std::uint64_t completed = 0;
          auto status = tracker_.Complete(q, cr.user_ctx, &completed);
          if (!status.ok()) {
            queues_[q].poisoned = true;
            return Status::Error(status.code(), status.message() +
                                 CqDetails(q, cr, queues_[q].channel));
          }
          if (completed && id && completed != id) return Error("unexpected prepared completion");
          *finished = completed != 0;
          return Status::OK();
        });
    next_poll_queue_ = (next_poll_queue_ + 1) % kJettyCount;
    if (!status.ok()) return status;
    if (count) return Status::OK();
    if ((++polls & 31) == 0 && std::chrono::steady_clock::now() >= deadline) {
      for (auto& q : queues_) q.poisoned = true;
      return Error("prepared CQ timeout; resources retained");
    }
  }
}
Status Transport::WaitSuffix(const std::atomic<bool>* stop) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);
  std::uint32_t polls = 0;
  for (;;) {
    const auto sequence = __atomic_load_n(
        reinterpret_cast<const std::uint64_t*>(memory_ + kSuffixNotifyOffset), __ATOMIC_ACQUIRE);
    if (sequence == request().sequence) return Status::OK();
    if (sequence > request().sequence) return Error("future descriptor suffix sequence");
    // Stop must also cancel a partial request after a device-side failure.
    if (stop && stop->load(std::memory_order_acquire)) return Error("descriptor suffix wait cancelled");
    if ((++polls & 31) == 0 && std::chrono::steady_clock::now() >= deadline)
      return Error("descriptor suffix timeout; resources retained");
  }
}
Status Transport::DrainPrepared(Profile* profile) {
  BatchGetProfileScope timer(profile ? &profile->cq_wait_ns : nullptr);
  while (!tracker_.Empty()) {
    auto status = PollPrepared(0, profile); // A prefix and suffix can both be in flight.
    if (!status.ok()) return status;
  }
  if (profile) profile->last_cqe_ns = BatchGetProfileNowNs();
  return Status::OK();
}
Status Transport::WriteBatch(const std::atomic<bool>* stop) {
  BatchGetProfileScope timer(profile_.enabled ? &profile_.write_ns : nullptr);
  auto status = PrepareWrites(request().count);
  if (!status.ok()) return status;
  auto* profile = profile_.enabled ? &profile_ : nullptr;
  bool suffix_ready = false;
  for (std::size_t w = 0; w < prepared_.size(); ++w) {
    auto& plan = *prepared_[w];
    if (plan.suffix && !suffix_ready) {
      status = WaitSuffix(stop);
      if (!status.ok()) return status;
      suffix_ready = true;
    }
    // Only a physical SQ-credit shortage drains the prefix before the suffix.
    status = BeginPrepared(plan, w + 1, profile, true);
    if (!status.ok()) return status;
    if (profile) {
      profile->cpu_threads = 1;
      ++profile->waves;
      for (const auto& q : plan.queues) {
        profile->ready_wrs += q.markers.size();
        profile->data_wrs += q.wrs.size() - q.markers.size();
        profile->post_calls += q.posts.size();
      }
    }
    std::uint64_t first_post = 0;
    status = PostPrepared(plan, true, profile ? &first_post : nullptr);
    if (!status.ok()) return status;
    if (profile) {
      if (!profile->first_post_ns) profile->first_post_ns = first_post;
      profile->post_ns += BatchGetProfileNowNs() - first_post;
    }
    const bool keep_prefix = !plan.suffix && w + 1 < prepared_.size() && prepared_[w + 1]->suffix;
    if (!keep_prefix) {
      status = DrainPrepared(profile);
      if (!status.ok()) return status;
    }
  }
  return tracker_.Empty() ? Status::OK() : DrainPrepared(profile);
}
Status Transport::Complete(std::uint64_t sequence, Result result) {
  BatchGetProfileScope timer(profile_.enabled ? &profile_.done_ns : nullptr);
  auto* done = reinterpret_cast<Completion*>(memory_ + layout_.done);
  *done = {Progress(sequence, static_cast<std::uint32_t>(result)), {}};
  const bool report = profile_.enabled && (request().reserved & kInlineProfile) &&
      sequence && result == Result::kOk;
  if (report) {
    done->reserved[0] = ProfileToken(sequence);
    done->reserved[1] = profile_.header_read_ns;
    done->reserved[2] = profile_.address_read_ns;
    done->reserved[3] = profile_.write_ns;
    done->reserved[4] = profile_.first_post_ns ? profile_.last_cqe_ns - profile_.first_post_ns : 0;
    done->reserved[5] = profile_.post_ns;
    done->reserved[6] = profile_.cq_wait_ns;
  }
  // One post and one tail CQE for both paths. The 8-byte strongly ordered
  // token follows the report's placement; do not combine them into an
  // unordered 64-byte WRITE (the token could otherwise become visible first).
  auto& plan = report ? profiled_done_ : done_;
  auto status = BeginPrepared(plan, UINT64_MAX);
  if (status.ok()) status = PostPrepared(plan);
  if (status.ok()) status = PollPrepared(UINT64_MAX);
  return status;
}

void Transport::PrintProfile() const {
  if (!profile_.enabled || (request().reserved & kInlineProfile)) return;
  const auto& p = profile_;
  const auto& r = request();
  const auto build_ns = p.write_ns - p.post_ns - p.cq_wait_ns;
  const auto submit_ns = p.first_post_ns ? p.last_cqe_ns - p.first_post_ns : 0;
  std::fprintf(stderr,
      "CPU_PUSH_PROFILE sequence=%" PRIu64 " workers=%u rows=%u bytes=%" PRIu64
      " header_read_us=%.1f address_read_us=%.1f build_us=%.1f post_us=%.1f cq_wait_us=%.1f"
      " write_us=%.1f submit_to_cqe_us=%.1f done_us=%.1f"
      " data_wrs=%u ready_wrs=%u post_calls=%u waves=%u tail_cqes=%u cpu_threads=%u plan_reused=%u optimized=%u\n",
      r.sequence, r.workers, r.count, std::uint64_t{r.count} * token_bytes_,
      p.header_read_ns / 1000.0, p.address_read_ns / 1000.0, build_ns / 1000.0, p.post_ns / 1000.0,
      p.cq_wait_ns / 1000.0, p.write_ns / 1000.0, submit_ns / 1000.0, p.done_ns / 1000.0,
      p.data_wrs, p.ready_wrs, p.post_calls, p.waves, p.tail_cqes, p.cpu_threads,
      unsigned(p.plan_reused), kOptimized);
}

Status Transport::Shutdown() {
  // Deleting the initiator queues stops all DMA, including partial-post failures.
  // Do not let UrmaChannel's best-effort destructor discard a failed deletion.
  for (auto& q : queues_) {
    auto& c = q.channel;
    if (c.local_jetty) {
      if (urma_delete_jetty(c.local_jetty)) return Error("delete Jetty; DMA resources retained");
      c.local_jetty = nullptr;
    }
    if (c.remote_jetty) {
      if (urma_unimport_jetty(c.remote_jetty)) return Error("unimport Jetty");
      c.remote_jetty = nullptr;
    }
    if (c.local_jfr) {
      if (urma_delete_jfr(c.local_jfr)) return Error("delete RQ");
      c.local_jfr = nullptr;
    }
    if (c.local_jfc) {
      if (urma_delete_jfc(c.local_jfc)) return Error("delete CQ");
      c.local_jfc = nullptr;
    }
  }
  if (remote_) { if (urma_unimport_seg(remote_)) return Error("unimport HBM"); remote_ = nullptr; }
  if (control_) { if (urma_unregister_seg(control_)) return Error("unregister controls"); control_ = nullptr; }
  if (control_token_) { if (urma_free_token_id(control_token_)) return Error("free token"); control_token_ = nullptr; }
  tracker_ = Tracker(kJettyCount, kJettyDepth);
  prepared_.clear();
  done_ = {};
  profiled_done_ = {};
  prepared_count_ = UINT32_MAX;
  prepared_batch_done_only_ = false;
  prepared_head_rows_ = 0;
  post_batch_wrs_ = 128;
  next_poll_queue_ = 0;
  for (auto& q : queues_) q.poisoned = false;
  std::free(memory_); memory_ = nullptr;
  context_ = nullptr;
  return Status::OK();
}
}  // namespace kvaf::cpu_push
