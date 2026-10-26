#include "cpu_push_get_transport.h"
#include "urma_device_abi.h"
#include "batch_get_profile.h"
#include "cpu_push/a12bb16/pipeline.h"

#include <chrono>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <thread>

namespace kvaf::cpu_push {
namespace {
Status Error(const char* operation) { return Status::Internal(std::string("CPU push: ") + operation); }
constexpr std::size_t kJettyDepth = 8192, kDoorbellWrs = 32;
constexpr std::uint32_t kMaxChunks = (kMaxRequests + kChunkRows - 1) / kChunkRows;
void Stage(const char* stage, std::uint32_t queue = UINT32_MAX) {
  if (queue == UINT32_MAX)
    std::fprintf(stderr, "[KVAF_CPU] CPU_PUSH_INIT stage=%s\n", stage);
  else
    std::fprintf(stderr, "[KVAF_CPU] CPU_PUSH_INIT stage=%s queue=%u\n", stage, queue);
  errno = 0;
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
  source_ = source;
  hbm_base_ = setup.hbm_base;
  workers_ = setup.workers;
  token_bytes_ = setup.token_bytes;
  Stage("query_device");
  urma::UbDeviceAttributes attr{};
  if (urma_query_device(context->dev, reinterpret_cast<urma_device_attr_t*>(&attr)))
    return Error("query device");
  max_sge_ = std::min(16U, attr.dev_cap.max_jfs_sge);
  max_bytes_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(65536, attr.dev_cap.max_msg_size));
  if (!max_sge_ || max_bytes_ < sizeof(Request)) return Error("unsupported SGE/message limits");
  std::fprintf(stderr, "[KVAF_CPU] CPU_PUSH_CAP max_sge=%u message_bytes=%u workers=%u token_bytes=%u\n",
               max_sge_, max_bytes_, workers_, token_bytes_);
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
  Stage("import_hbm");
  remote_ = urma_import_seg(context_, &segment, &token, 0, flag);
  if (!remote_) return Error("import HBM segment");
  void* buffer = nullptr;
  // Each in-flight ready WRITE has its own immutable registered source word.
  const auto marker_bytes = workers_ * kMaxChunks * kJettyCount * sizeof(std::uint64_t);
  const auto bytes = (layout_.staging + marker_bytes + 4095) & ~std::uint64_t{4095};
  Stage("allocate_control");
  if (posix_memalign(&buffer, 4096, bytes)) return Error("allocate control buffer");
  memory_ = static_cast<std::uint8_t*>(buffer);
  std::memset(memory_, 0, bytes);
  Stage("allocate_control_token");
  control_token_ = urma_alloc_token_id(context_);
  if (!control_token_) return Error("allocate control token");
  urma_seg_cfg_t cfg{};
  cfg.va = reinterpret_cast<std::uint64_t>(memory_);
  cfg.len = bytes;
  cfg.token_id = control_token_;
  cfg.token_value.token = setup.token_value;
  // This pinned buffer receives local READ results and supplies local WRITE
  // payloads. Use the probe's direct-host registration, not remote DDR flags.
  cfg.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
  cfg.flag.bs.access = 1U;  // LOCAL_WRITE in the deployed public UMDK ABI.
  cfg.flag.bs.non_pin = 0;
  cfg.flag.bs.token_id_valid = 1;
  Stage("register_control");
  control_ = urma_register_seg(context_, &cfg);
  if (!control_) return Error("register control buffer");
  std::uint64_t cpu_tp = 0;
  Stage("query_ctp");
  if (!urma::QueryTp(query, context_, setup.eid, &cpu_tp)) return Error("query CTP");
  response->cpu_tp = cpu_tp;
  response->cpu_psn = psn;
  for (std::uint32_t q = 0; q < kJettyCount; ++q) {
    if (setup.channels[q].bytes != 28 && setup.channels[q].bytes != 32)
      return Error("invalid NPU Jetty key");
    auto& queue = queues_[q];
    auto& c = queue.channel;
    Stage("create_queues", q);
    const auto* error = urma::CreateQueues(urma::NativeApi{}, context_, setup.token_value,
        {16384, kJettyDepth, max_sge_, false, true}, c.local_jfc, c.local_jfr, c.local_jetty);
    if (error) return Error(error);
    // Control READs use one SGE. Data and done use reusable prepared chains.
    queue.local.reserve(1);
    queue.remote.reserve(1);
    queue.wrs.reserve(1);
    Stage("import_jetty", q);
    c.remote_jetty = urma::ImportJetty(import, context_, setup.channels[q].key,
        setup.token_value, cpu_tp, setup.npu_tp, psn, setup.npu_psn);
    if (!c.remote_jetty) return Error("import NPU Jetty");
    auto& key = response->channels[q];
    key.bytes = sizeof(urma_jetty_id_t) + sizeof(urma_transport_mode_t);
    std::memcpy(key.key, &c.local_jetty->jetty_id, sizeof(urma_jetty_id_t));
    std::memcpy(key.key + sizeof(urma_jetty_id_t), &c.local_jetty->jetty_cfg.jfs_cfg.trans_mode,
                sizeof(urma_transport_mode_t));
  }
  auto& done = done_.queues[0];
  done.local.push_back({reinterpret_cast<std::uint64_t>(memory_ + layout_.done),
                        sizeof(Completion), control_, nullptr});
  AppendPrepared(done, 0, hbm_base_ + layout_.done, sizeof(Completion), 1, true);
  SealPrepared(done_);
  Stage("ready");
  return Status::OK();
}

void Transport::ResetRequests(Queue& q) { q.local.clear(); q.remote.clear(); q.wrs.clear(); }
void Transport::Add(Queue& q, urma_opcode_t opcode, std::uint64_t remote,
                    std::uint32_t bytes, std::uint32_t first, std::uint32_t sges) {
  q.remote.push_back({remote, bytes, remote_, nullptr});
  urma_jfs_wr_t wr{};
  wr.opcode = opcode;
  wr.flag.bs.comp_order = 1;
  wr.flag.bs.place_order = URMA_RELAX_ORDER;
  wr.tjetty = q.channel.remote_jetty;
  wr.user_ctx = ++q.sequence;
  if (opcode == URMA_OPC_WRITE) {
    wr.rw.src = {reinterpret_cast<urma_sge_t*>(q.local.data() + first), sges};
    wr.rw.dst = {reinterpret_cast<urma_sge_t*>(&q.remote.back()), 1};
  } else {
    wr.rw.src = {reinterpret_cast<urma_sge_t*>(&q.remote.back()), 1};
    wr.rw.dst = {reinterpret_cast<urma_sge_t*>(q.local.data() + first), sges};
  }
  q.wrs.push_back(wr);
}
Status Transport::Post(Queue& q) {
  if (q.poisoned || q.pending || q.wrs.empty()) return Error("queue unavailable");
  auto& tail = q.wrs.back();
  tail.flag.bs.complete_enable = 1;
  tail.flag.bs.place_order = URMA_STRONG_ORDER;
  q.pending = tail.user_ctx;
  for (std::size_t first = 0; first < q.wrs.size(); first += 32) {
    const auto end = std::min(first + 32, q.wrs.size());
    for (auto i = first; i < end; ++i) q.wrs[i].next = i + 1 < end ? &q.wrs[i + 1] : nullptr;
    urma_jfs_wr_t* bad = nullptr;
    if (urma_post_jetty_send_wr(q.channel.local_jetty, &q.wrs[first], &bad)) {
      q.poisoned = true;
      return Error("post WRITE/READ chain");
    }
  }
  return Status::OK();
}
Status Transport::Wait(Queue& q) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);
  std::uint32_t polls = 0;
  while (q.pending) {
    urma_cr_t crs[8]{};
    const auto count = urma_poll_jfc(q.channel.local_jfc, 8, crs);
    if (count < 0) { q.poisoned = true; return Error("poll CQ"); }
    for (int i = 0; i < count; ++i) {
      if (crs[i].status != URMA_CR_SUCCESS || crs[i].user_ctx != q.pending) {
        q.poisoned = true; return Error("CQ status/context mismatch");
      }
      q.pending = 0;
    }
    if ((++polls & 31) == 0 && std::chrono::steady_clock::now() >= deadline) {
      q.poisoned = true; return Error("CQ timeout; resources retained");
    }
  }
  return Status::OK();
}

Status Transport::Transfer(bool write, std::uint64_t offset, std::uint64_t bytes) {
  auto& q = queues_[0];
  if (q.poisoned || q.pending || offset > layout_.staging || bytes > layout_.staging - offset)
    return Error("invalid control transfer");
  while (bytes) {
    ResetRequests(q);
    const auto length = static_cast<std::uint32_t>(std::min<std::uint64_t>(bytes, max_bytes_));
    q.local.push_back({reinterpret_cast<std::uint64_t>(memory_ + offset), length, control_, nullptr});
    Add(q, write ? URMA_OPC_WRITE : URMA_OPC_READ, hbm_base_ + offset, length, 0, 1);
    auto result = Post(q);
    if (result.ok()) result = Wait(q);
    if (!result.ok()) return result;
    offset += length;
    bytes -= length;
  }
  return Status::OK();
}
Status Transport::ReadRequest(std::uint64_t* sequence) {
  auto result = Transfer(false, layout_.notify, sizeof(Word));
  if (result.ok()) {
    const auto notify = reinterpret_cast<const Word*>(memory_ + layout_.notify)->value;
    *sequence = notify & ~kProfileNotify;
    profile_.enabled = (notify & kProfileNotify) != 0;
  }
  return result;
}
Status Transport::ReadDescriptor(std::uint64_t sequence) {
  if (profile_.enabled) {
    profile_ = {};
    profile_.enabled = true;
  }
  Status result;
  {
    BatchGetProfileScope timer(profile_.enabled ? &profile_.header_read_ns : nullptr);
    result = Transfer(false, layout_.request, sizeof(Request));
  }
  if (!result.ok()) return result;
  const auto& r = request();
  if (r.magic != kMagic || r.sequence != sequence || !sequence || sequence > UINT32_MAX ||
      r.count > kMaxRequests || r.token_bytes != token_bytes_ || r.workers != workers_)
    return Status::InvalidArgument("CPU push request header mismatch");
  BatchGetProfileScope timer(profile_.enabled ? &profile_.address_read_ns : nullptr);
  return Transfer(false, layout_.descriptor, r.count * sizeof(std::uint64_t));
}

// Ported from hbm-sgl-probe PrepareWrite: partition each chunk into contiguous
// Jetty shards, append ordered ready WRs, and signal only each queue's tail.
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
    std::size_t local = 0;
    for (std::size_t i = 0; i < q.wrs.size(); ++i) {
      auto& wr = q.wrs[i];
      wr.rw.src.sge = reinterpret_cast<urma_sge_t*>(q.local.data() + local);
      wr.rw.dst.sge = reinterpret_cast<urma_sge_t*>(q.remote.data() + i);
      local += wr.rw.src.num_sge;
      wr.next = i + 1 < q.wrs.size() && (i + 1) % kDoorbellWrs ? &q.wrs[i + 1] : nullptr;
    }
    q.wrs.back().flag.bs.complete_enable = 1;
    q.wrs.back().flag.bs.place_order = URMA_STRONG_ORDER;
    plan.fences[plan.fence_count++] = {index, q.wrs.size(), 0};
  }
}
Status Transport::PrepareWrites(std::uint32_t count) {
  if (!tracker_.Empty()) return Error("prepare while WRITE is in flight");
  if (prepared_count_ == count) {
    profile_.plan_reused = true;
    return Status::OK();
  }
  std::array<std::vector<PreparedQueue>, kJettyCount> shards;
  const auto config = ReferenceConfig(count, token_bytes_, workers_);
  for (std::uint32_t queue = 0; queue < kJettyCount; ++queue) {
    auto& waves = shards[queue];
    waves.emplace_back();
    aicpu_test::ForEachPipelineChunk(config,
      [&](std::uint32_t worker, std::uint64_t chunk_index, aicpu_test::RowRange original) {
        const auto chunk = static_cast<std::uint32_t>(chunk_index);
        const Range range{static_cast<std::uint32_t>(original.begin),
                          static_cast<std::uint32_t>(original.end)};
        const auto rows = range.end - range.begin;
        const auto base = rows / kJettyCount, remainder = rows % kJettyCount;
        auto row = range.begin + queue * base + std::min(queue, remainder);
        const auto end = row + base + (queue < remainder ? 1U : 0U);
        std::uint32_t offset = 0;
        while (row < end) {
          // The probe budgets source SGEs as well as WRs against SQ depth.
          // Continue large requests in another drained wave without narrowing
          // BatchGet's existing maximum request/token sizes.
          if (waves.back().local.size() + max_sge_ > kJettyDepth) waves.emplace_back();
          auto& q = waves.back();
          const auto destination = hbm_base_ + layout_.staging + std::uint64_t{row} * token_bytes_ + offset;
          std::uint32_t bytes = 0, sges = 0;
          while (row < end && sges < max_sge_ && bytes < max_bytes_) {
            const auto length = std::min(token_bytes_ - offset, max_bytes_ - bytes);
            q.sources.push_back({static_cast<std::uint32_t>(q.local.size()), row, offset});
            q.local.push_back({0, length, source_, nullptr});
            ++sges; bytes += length; offset += length;
            if (offset == token_bytes_) { ++row; offset = 0; }
          }
          AppendPrepared(q, queue, destination, bytes, sges);
        }
        if (waves.back().local.size() == kJettyDepth) waves.emplace_back();
        auto& q = waves.back();
        auto* marker = reinterpret_cast<std::uint64_t*>(memory_ + layout_.staging) +
            (worker * kMaxChunks + chunk) * kJettyCount + queue;
        q.markers.push_back({marker, chunk + 1});
        q.local.push_back({reinterpret_cast<std::uint64_t>(marker), 8, control_, nullptr});
        AppendPrepared(q, queue, hbm_base_ + layout_.ready + worker * sizeof(Ready) + queue * 8,
                       8, 1, true);
      });
  }
  std::size_t waves = 0;
  for (const auto& shard : shards) waves = std::max(waves, shard.size());
  prepared_.clear();
  for (std::size_t w = 0; w < waves; ++w) {
    auto plan = std::make_unique<PreparedWrite>();
    for (std::uint32_t q = 0; q < kJettyCount; ++q)
      if (w < shards[q].size()) plan->queues[q] = std::move(shards[q][w]);
    SealPrepared(*plan);
    if (plan->fence_count) prepared_.push_back(std::move(plan));
  }
  prepared_count_ = count;
  return Status::OK();
}
void Transport::RefreshPrepared(PreparedQueue& q) {
  const auto* addresses = reinterpret_cast<const std::uint64_t*>(memory_ + layout_.descriptor);
  for (const auto& source : q.sources) q.local[source.local].addr = addresses[source.row] + source.offset;
  for (const auto& marker : q.markers) *marker.value = Progress(request().sequence, marker.progress);
}
Status Transport::BeginPrepared(PreparedWrite& plan, std::uint64_t id) {
  for (const auto& q : queues_)
    if (q.poisoned || q.pending) return Error("queue unavailable");
  bool submitted = false;
  auto status = tracker_.TryBegin(id, plan.handle, plan.fences.data(), plan.fence_count, &submitted);
  if (!status.ok()) return status;
  if (!submitted) return Error("drained queue has insufficient credits");
  for (std::size_t i = 0; i < plan.fence_count; ++i) {
    const auto& fence = plan.fences[i];
    plan.queues[fence.queue].wrs.back().user_ctx = fence.context;
  }
  return Status::OK();
}
Status Transport::PostPrepared(PreparedQueue& q, std::uint32_t queue) {
  for (std::size_t first = 0; first < q.wrs.size(); first += kDoorbellWrs) {
    urma_jfs_wr_t* bad = nullptr;
    if (urma_post_jetty_send_wr(queues_[queue].channel.local_jetty, &q.wrs[first], &bad)) {
      queues_[queue].poisoned = true;
      return Error("post prepared WRITE chain");
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
            queues_[q].poisoned = true; return Error("prepared CQ status");
          }
          std::uint64_t completed = 0;
          auto status = tracker_.Complete(q, cr.user_ctx, &completed);
          if (!status.ok()) { queues_[q].poisoned = true; return status; }
          if (completed && completed != id) return Error("unexpected prepared completion");
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
Status Transport::WriteBatch() {
  BatchGetProfileScope timer(profile_.enabled ? &profile_.write_ns : nullptr);
  auto status = PrepareWrites(request().count);
  if (!status.ok()) return status;
  // Match the ordered-WRITE probe: short submissions to all four queues from
  // their owner, then poll their tails together. Avoid waking an OpenMP team
  // and crossing several team barriers for each short request.
  for (std::size_t w = 0; w < prepared_.size(); ++w) {
    auto& plan = *prepared_[w];
    for (auto& queue : plan.queues) RefreshPrepared(queue);
    status = BeginPrepared(plan, w + 1);
    if (!status.ok()) return status;
    if (profile_.enabled) {
      profile_.cpu_threads = 1;
      ++profile_.waves;
      for (const auto& q : plan.queues) {
        profile_.ready_wrs += q.markers.size();
        profile_.data_wrs += q.wrs.size() - q.markers.size();
        profile_.post_calls += (q.wrs.size() + kDoorbellWrs - 1) / kDoorbellWrs;
      }
    }
    const auto phase_begin = profile_.enabled ? BatchGetProfileNowNs() : 0;
    if (profile_.enabled && !profile_.first_post_ns) profile_.first_post_ns = phase_begin;
    for (std::uint32_t q = 0; q < kJettyCount; ++q) {
      status = PostPrepared(plan.queues[q], q);
      if (!status.ok()) return status;
    }
    const auto posted = profile_.enabled ? BatchGetProfileNowNs() : 0;
    if (profile_.enabled) profile_.post_ns += posted - phase_begin;
    status = PollPrepared(w + 1, profile_.enabled ? &profile_ : nullptr);
    if (profile_.enabled) {
      profile_.last_cqe_ns = BatchGetProfileNowNs();
      profile_.cq_wait_ns += profile_.last_cqe_ns - posted;
    }
    if (!status.ok()) return status;
  }
  return status;
}
Status Transport::Complete(std::uint64_t sequence, Result result) {
  BatchGetProfileScope timer(profile_.enabled ? &profile_.done_ns : nullptr);
  auto* done = reinterpret_cast<Completion*>(memory_ + layout_.done);
  *done = {Progress(sequence, static_cast<std::uint32_t>(result)), {}};
  auto status = BeginPrepared(done_, UINT64_MAX);
  if (status.ok()) status = PostPrepared(done_.queues[0], 0);
  if (status.ok()) status = PollPrepared(UINT64_MAX);
  return status;
}

void Transport::PrintProfile() const {
  if (!profile_.enabled) return;
#ifdef __OPTIMIZE__
  constexpr unsigned optimized = 1;
#else
  constexpr unsigned optimized = 0;
#endif
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
      unsigned(p.plan_reused), optimized);
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
    q.pending = 0;
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
  prepared_count_ = UINT32_MAX;
  next_poll_queue_ = 0;
  for (auto& q : queues_) { ResetRequests(q); q.poisoned = false; q.sequence = 0; }
  std::free(memory_); memory_ = nullptr;
  context_ = nullptr;
  return Status::OK();
}
}  // namespace kvaf::cpu_push
