#include "cpu_service/push/cpu_push_get_transport.h"
#include "urma_device_abi.h"
#include "batch_get_profile.h"

#include <chrono>
#include <cstddef>
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
constexpr std::size_t kJettyDepth = 8192;
constexpr std::uint32_t kPostPayloadWrs = 128;
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
                             const peer::OpenCpuPushGetRequest& setup, urma::GetTpList query,
                             urma::ImportJettyEx import, std::uint32_t psn,
                             peer::OpenCpuPushGetResponse* response, std::uint64_t canonical_base,
                             const peer::OpenCpuSglPushRequest* push, peer::OpenCpuSglPushResponse* mailbox) {
  if (context_ || !context || !source || !query || !import || !response || !psn ||
      !setup.npu_tp || !setup.npu_psn || !setup.token_value ||
      setup.segment_bytes != sizeof(urma_seg_t) || !setup.source_bytes ||
      setup.output_mode > kChunkedOutput ||
      setup.sge_per_wr > 16 || (setup.sge_per_wr && setup.output_mode == kScatterOutput) ||
      !MakeLayout(setup.token_bytes, setup.workers, &layout_, setup.output_mode != kScatterOutput) ||
      setup.hbm_base % 64 || setup.hbm_bytes < layout_.bytes ||
      setup.hbm_base > UINT64_MAX - setup.hbm_bytes)
    return Status::InvalidArgument("invalid CPU push setup");
  if (push && (!mailbox || setup.output_mode != kPipelinedOutput || !push->control_tp ||
      !push->control_psn || (push->control.bytes != 28 && push->control.bytes != 32)))
    return Status::InvalidArgument("invalid device-push mailbox setup");
  device_push_ = push != nullptr;
  context_ = context;
  source_ = source;
  source_base_ = canonical_base ? canonical_base : source->seg.ubva.va;
  source_bytes_ = setup.source_bytes;
  hbm_base_ = setup.hbm_base;
  workers_ = setup.workers;
  token_bytes_ = setup.token_bytes;
  direct_output_ = setup.output_mode != kScatterOutput;
  payload_only_ = setup.output_mode == kPayloadOnlyOutput;
  chunked_ = setup.output_mode == kChunkedOutput;
  pipeline_ = chunked_ || setup.output_mode == kPipelinedOutput;
  Stage("query_device");
  urma::UbDeviceAttributes attr{};
  if (urma_query_device(context->dev, reinterpret_cast<urma_device_attr_t*>(&attr)))
    return Error("query device");
  const auto device_max_sge = std::min(16U, attr.dev_cap.max_jfs_sge);
  if (setup.sge_per_wr > device_max_sge)
    return Status::InvalidArgument("requested sge_per_wr=" + std::to_string(setup.sge_per_wr) +
        " exceeds device limit=" + std::to_string(device_max_sge));
  max_sge_ = setup.sge_per_wr ? setup.sge_per_wr : device_max_sge;
  max_bytes_ = static_cast<std::uint32_t>(std::min<std::uint64_t>(65536, attr.dev_cap.max_msg_size));
  if (!max_sge_ || max_bytes_ < sizeof(Request)) return Error("unsupported SGE/message limits");
  std::fprintf(stderr, "[KVAF_CPU] CPU_PUSH_CAP max_sge=%u device_max_sge=%u requested_sge=%u "
               "message_bytes=%u workers=%u token_bytes=%u"
               " post_payload_wrs=%u\n", max_sge_, device_max_sge, setup.sge_per_wr,
               max_bytes_, workers_, token_bytes_, kPostPayloadWrs);
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
  // READ mode uses a local-only buffer. Device-push mode exposes the mailbox
  // to remote WRITEs: LOCAL_ONLY is exclusive, and WRITE also requires READ.
  cfg.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
  cfg.flag.bs.access = device_push_ ? ((1U << 1U) | (1U << 2U)) : 1U;
  cfg.flag.bs.non_pin = 0;
  cfg.flag.bs.token_id_valid = 1;
  Stage("register_control");
  control_ = urma_register_seg(context_, &cfg);
  if (!control_) {
    const auto code = errno;
    return Status::Internal("CPU push: register control buffer (access=" + std::to_string(cfg.flag.bs.access) +
        ", flags=" + std::to_string(cfg.flag.value) + ", bytes=" + std::to_string(cfg.len) +
        ", errno=" + std::to_string(code) + ": " + std::strerror(code) + ")");
  }
  std::uint64_t cpu_tp = 0;
  Stage("query_ctp");
  if (!urma::QueryTp(query, context_, setup.eid, &cpu_tp)) return Error("query CTP");
  response->cpu_tp = cpu_tp;
  response->cpu_psn = psn;
  for (std::uint32_t q = 0; q < kJettyCount + (pipeline_ ? 1U : 0U); ++q) {
    const auto peer_queue = q % kJettyCount;
    if (setup.channels[peer_queue].bytes != 28 && setup.channels[peer_queue].bytes != 32)
      return Error("invalid NPU Jetty key");
    auto& queue = q < kJettyCount ? queues_[q] : control_queue_;
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
    const bool control_push = device_push_ && q == kJettyCount;
    c.remote_jetty = urma::ImportJetty(import, context_, control_push ? push->control.key : setup.channels[peer_queue].key,
        setup.token_value, cpu_tp, control_push ? push->control_tp : setup.npu_tp, psn,
        control_push ? push->control_psn : setup.npu_psn);
    if (!c.remote_jetty) return Error("import NPU Jetty");
    if (q == kJettyCount) {
      if (device_push_) {
        mailbox->control.bytes = sizeof(urma_jetty_id_t) + sizeof(urma_transport_mode_t);
        std::memcpy(mailbox->control.key, &c.local_jetty->jetty_id, sizeof(urma_jetty_id_t));
        std::memcpy(mailbox->control.key + sizeof(urma_jetty_id_t), &c.local_jetty->jetty_cfg.jfs_cfg.trans_mode,
                    sizeof(urma_transport_mode_t));
      }
      continue;
    }  // Private control queue, same remote target as queue 0.
    auto& key = response->channels[q];
    key.bytes = sizeof(urma_jetty_id_t) + sizeof(urma_transport_mode_t);
    std::memcpy(key.key, &c.local_jetty->jetty_id, sizeof(urma_jetty_id_t));
    std::memcpy(key.key + sizeof(urma_jetty_id_t), &c.local_jetty->jetty_cfg.jfs_cfg.trans_mode,
                sizeof(urma_transport_mode_t));
  }
  if (device_push_) {
    mailbox->mailbox_base = reinterpret_cast<std::uint64_t>(memory_);
    mailbox->mailbox_bytes = layout_.staging;
    mailbox->segment_bytes = sizeof(urma_seg_t);
    mailbox->max_write_bytes = max_bytes_;
    std::memcpy(mailbox->segment, &control_->seg, sizeof(urma_seg_t));
  }
  auto& done = done_.queues[0];
  // Publish the report before the ordered done token. The NPU may observe
  // the token before a CPU-side CQ poll or a subsequent control WRITE.
  auto done_bytes = static_cast<std::uint32_t>(sizeof(Completion));
  if (payload_only_) {
    const auto offset = layout_.done + offsetof(Completion, cpu_sgl_write);
    done.local.push_back({reinterpret_cast<std::uint64_t>(memory_ + offset),
                          sizeof(CpuSglWriteSample), control_, nullptr});
    AppendPrepared(done, 0, hbm_base_ + offset, sizeof(CpuSglWriteSample), 1);
    done_bytes = sizeof(Completion::token);
  }
  done.local.push_back({reinterpret_cast<std::uint64_t>(memory_ + layout_.done),
                        done_bytes, control_, nullptr});
  AppendPrepared(done, 0, hbm_base_ + layout_.done, done_bytes, 1, true);
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
  auto& q = pipeline_ ? control_queue_ : queues_[0];
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
  // The profile bit is known only after the READ. Measure just this probe,
  // never the idle interval spent repeatedly reading an unchanged sequence.
  if (device_push_) {
    const auto notify = __atomic_load_n(reinterpret_cast<const std::uint64_t*>(memory_ + layout_.notify), __ATOMIC_ACQUIRE);
    *sequence = notify & ~kProfileNotify;
    profile_.enabled = (notify & kProfileNotify) != 0;
    profile_.notify_read_ns = 0;
    profile_.cpu_start_ns = profile_.enabled ? BatchGetProfileNowNs() : 0;
    return Status::OK();
  }
  const auto start = (payload_only_ || pipeline_) ? BatchGetProfileNowNs() : 0;
  auto result = Transfer(false, layout_.notify, sizeof(Word));
  if (result.ok()) {
    const auto notify = reinterpret_cast<const Word*>(memory_ + layout_.notify)->value;
    *sequence = notify & ~kProfileNotify;
    profile_.enabled = (notify & kProfileNotify) != 0;
    if ((payload_only_ || pipeline_) && profile_.enabled) {
      profile_.notify_read_ns = BatchGetProfileNowNs() - start;
      profile_.cpu_start_ns = start;
    }
  }
  return result;
}
Status Transport::ReadDescriptor(std::uint64_t sequence) {
  if (profile_.enabled) {
    const auto notify_read_ns = profile_.notify_read_ns, cpu_start_ns = profile_.cpu_start_ns;
    profile_ = {};
    profile_.enabled = true;
    profile_.notify_read_ns = notify_read_ns;
    profile_.cpu_start_ns = cpu_start_ns;
  }
  Status result;
  {
    BatchGetProfileScope timer(profile_.enabled ? &profile_.header_read_ns : nullptr);
    result = device_push_ ? Status::OK() : Transfer(false, layout_.request, sizeof(Request));
  }
  if (!result.ok()) return result;
  const auto& r = request();
  if (r.magic != kMagic || r.sequence != sequence || !sequence || sequence > UINT32_MAX ||
      r.count > kMaxRequests || r.token_bytes != token_bytes_ || r.workers != workers_ ||
      r.mode != (chunked_ ? kChunkedOutput : direct_output_ ? kDirectOutput : kScatterOutput))
    return Status::InvalidArgument("CPU push request header mismatch");
  BatchGetProfileScope timer(profile_.enabled ? &profile_.address_read_ns : nullptr);
  const bool detail = (payload_only_ || pipeline_) && profile_.enabled;
  batch_result_ = Result::kOk;
  head_count_ = pipeline_ ? static_cast<std::uint32_t>(r.reserved_words[0]) : 0;
  if (chunked_ && (r.reserved_words[0] != kChunkRows || !r.count))
    return Status::InvalidArgument("invalid CPU SGL chunk size");
  if (pipeline_ && !chunked_ && (r.reserved_words[0] != std::min(r.count, kPipelineHeadRows) || !head_count_))
    return Status::InvalidArgument("invalid CPU SGL pipeline head");
  Status read;
  if (!pipeline_) {
    BatchGetProfileScope stage(detail ? &profile_.address_list_read_ns : nullptr);
    read = Transfer(false, layout_.descriptor, r.count * sizeof(std::uint64_t));
    if (!read.ok()) return read;
  }
  if (!pipeline_) {
    auto checked = ValidateSources(0, r.count);
    if (!checked.ok()) return checked;
  }
  if (direct_output_) {
    if (output_remote_) return Error("previous output is still imported");
    {
      BatchGetProfileScope stage(detail ? &profile_.output_desc_read_ns : nullptr);
      read = device_push_ ? Status::OK() : Transfer(false, layout_.output_target, sizeof(OutputTarget));
    }
    if (!read.ok()) return read;
    output_ = *reinterpret_cast<const OutputTarget*>(memory_ + layout_.output_target);
    if (!output_.base || !output_.token_value || output_.segment_bytes != sizeof(urma_seg_t) ||
        std::uint64_t{r.count} * token_bytes_ > output_.bytes ||
        output_.base > UINT64_MAX - output_.bytes)
      return Status::InvalidArgument("invalid CPU SGL output descriptor");
    urma_seg_t segment{};
    std::memcpy(&segment, output_.segment, sizeof(segment));
    if (!SourceRange(segment.ubva.va, segment.len, output_.base, output_.bytes))
      return Status::InvalidArgument("CPU SGL output outside exported segment");
    urma_import_seg_flag_t flag{};
    flag.bs.cacheable = URMA_NON_CACHEABLE;
    flag.bs.access = 1U << 2U;
    flag.bs.mapping = URMA_SEG_NOMAP;
    urma_token_t token{};
    token.token = output_.token_value;
    {
      BatchGetProfileScope stage(detail ? &profile_.output_import_ns : nullptr);
      output_remote_ = urma_import_seg(context_, &segment, &token, 0, flag);
    }
    if (!output_remote_) return Error("import output segment");
  }
  if (pipeline_ && !chunked_) {
    {
      BatchGetProfileScope stage(detail ? &profile_.address_list_read_ns : nullptr);
      read = device_push_ ? Status::OK() : Transfer(false, layout_.descriptor, std::uint64_t{head_count_} * 8);
    }
    if (!read.ok()) return read;
    return ValidateSources(0, head_count_);
  }
  return Status::OK();
}

Status Transport::ValidateSources(std::uint32_t begin, std::uint32_t end) {
  // Device-push requests trust the generated source list. Skip the CPU scan
  // before starting its timer; legacy READ-based paths keep range validation.
  if (device_push_) return Status::OK();
  BatchGetProfileScope timer((payload_only_ || pipeline_) && profile_.enabled ? &profile_.address_validate_ns : nullptr);
  const auto* addresses = reinterpret_cast<const std::uint64_t*>(memory_ + layout_.descriptor);
  for (auto i = begin; i < end; ++i)
    if (!SourceRange(source_base_, source_bytes_, addresses[i], token_bytes_))
      return Status::InvalidArgument("CPU push source outside cache region");
  return Status::OK();
}

// A separate CQ keeps descriptor READs independent from in-flight data CQEs.
Status Transport::BeginControlRead(std::uint64_t offset, std::uint64_t bytes) {
  auto& q = control_queue_;
  if (q.poisoned || q.pending || !bytes || offset > layout_.staging || bytes > layout_.staging - offset)
    return Error("invalid asynchronous descriptor READ");
  ResetRequests(q);
  const auto parts = (bytes + max_bytes_ - 1) / max_bytes_;
  if (parts > kJettyDepth) return Error("descriptor READ exceeds control queue depth");
  q.local.reserve(parts); q.remote.reserve(parts); q.wrs.reserve(parts);
  while (bytes) {
    const auto length = static_cast<std::uint32_t>(std::min<std::uint64_t>(bytes, max_bytes_));
    const auto index = static_cast<std::uint32_t>(q.local.size());
    q.local.push_back({reinterpret_cast<std::uint64_t>(memory_ + offset), length, control_, nullptr});
    Add(q, URMA_OPC_READ, hbm_base_ + offset, length, index, 1);
    offset += length; bytes -= length;
  }
  return Post(q);
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
    std::uint32_t local = 0, payloads = 0;
    std::size_t destination = 0;
    for (std::uint32_t i = 0; i < q.wrs.size(); ++i) {
      auto& wr = q.wrs[i];
      const bool payload = destination < q.destinations.size() && q.destinations[destination].wr == i;
      destination += payload;
      // Keep a trailing ordered Ready in the same post when payload hits the budget.
      if (!i || (payload && payloads == kPostPayloadWrs)) {
        if (i) { q.wrs[i - 1].next = nullptr; q.posts.back().end_sge = local; }
        q.posts.push_back({i, local, 0});
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
    // Preserve the probe's conservative SGE budget across concurrent chunks,
    // rather than applying that limit only within an individual plan.
    plan.fences[plan.fence_count++] = {index, (chunked_ || device_push_) ? std::max(q.wrs.size(), q.local.size()) : q.wrs.size(), 0};
  }
}
Status Transport::PrepareWrites(std::uint32_t count) {
  if (!tracker_.Empty()) return Error("prepare while WRITE is in flight");
  if (prepared_count_ == count) {
    profile_.plan_reused = true;
    RetargetWrites();
    return Status::OK();
  }
  prepared_.clear();
  head_waves_ = 0;
  chunk_plans_.clear();
  const auto phases = chunked_ ? (count + kChunkRows - 1) / kChunkRows : pipeline_ && head_count_ < count ? 2U : 1U;
  for (std::uint32_t phase = 0; phase < phases; ++phase) {
    const auto first_plan = static_cast<std::uint32_t>(prepared_.size());
    std::array<std::vector<PreparedQueue>, kJettyCount> shards;
    const bool unsplit = std::uint64_t{max_sge_} * token_bytes_ <= max_bytes_;
    for (std::uint32_t queue = 0; queue < kJettyCount; ++queue) {
      auto& waves = shards[queue];
      waves.emplace_back();
      const auto append = [&](std::uint32_t worker, std::uint32_t chunk, Range range) {
          auto shard = JettyRange(range, queue);
          if (chunked_) {
            shard = JettyRange({phase * kChunkRows, std::min(count, (phase + 1) * kChunkRows)}, queue);
          } else if (pipeline_) {
            shard.begin = std::max(shard.begin, phase == 0 ? 0U : head_count_);
            shard.end = std::min(shard.end, phase == 0 ? head_count_ : count);
            if (shard.begin >= shard.end) return;
          }
          auto row = shard.begin;
          const auto end = shard.end;
          std::uint32_t offset = 0;
          while (row < end) {
            // The probe budgets source SGEs as well as WRs against SQ depth.
            // Continue large requests in another drained wave without narrowing
            // BatchGet's existing maximum request/token sizes.
            if (waves.back().local.size() + max_sge_ > kJettyDepth) waves.emplace_back();
            auto& q = waves.back();
            const auto first_local = static_cast<std::uint32_t>(q.local.size());
            const auto first_row = row;
            const auto destination_offset = std::uint64_t{row} * token_bytes_ + offset;
            const auto destination = (direct_output_ ? output_.base : hbm_base_ + layout_.staging) + destination_offset;
            std::uint32_t bytes = 0, sges = 0;
            while (row < end && sges < max_sge_ && bytes < max_bytes_) {
              const auto length = std::min(token_bytes_ - offset, max_bytes_ - bytes);
              if (!unsplit)
                q.sources.push_back({static_cast<std::uint32_t>(q.local.size()), row, offset});
              q.local.push_back({0, length, source_, nullptr});
              ++sges; bytes += length; offset += length;
              if (offset == token_bytes_) { ++row; offset = 0; }
            }
            if (unsplit) {
              // Walk the contiguous descriptor entries directly for whole-token SGEs.
              if (!q.source_runs.empty() &&
                  q.source_runs.back().local + q.source_runs.back().count == first_local &&
                  q.source_runs.back().row + q.source_runs.back().count == first_row)
                q.source_runs.back().count += sges;
              else
                q.source_runs.push_back({first_local, first_row, sges});
            }
            q.destinations.push_back({static_cast<std::uint32_t>(q.wrs.size()), destination_offset});
            AppendPrepared(q, queue, destination, bytes, sges);
          }
          if (payload_only_ || pipeline_) return;  // Direct output consumes only the final done token.
          if (waves.back().local.size() == kJettyDepth) waves.emplace_back();
          auto& q = waves.back();
          auto* marker = reinterpret_cast<std::uint64_t*>(memory_ + layout_.staging) +
              (worker * kMaxChunks + chunk) * kJettyCount + queue;
          q.markers.push_back({marker, chunk + 1, static_cast<std::uint32_t>(q.local.size())});
          q.local.push_back({reinterpret_cast<std::uint64_t>(marker), 8, control_, nullptr});
          AppendPrepared(q, queue, hbm_base_ + layout_.ready + worker * sizeof(Ready) + queue * 8,
                         8, 1, true);
        };
      if (chunked_) append(phase % workers_, phase / workers_, {});
      else ForEachPipelineChunk(count, token_bytes_, workers_, append);
    }
    std::size_t waves = 0;
    for (const auto& shard : shards) waves = std::max(waves, shard.size());
    for (std::size_t w = 0; w < waves; ++w) {
      auto plan = std::make_unique<PreparedWrite>();
      for (std::uint32_t q = 0; q < kJettyCount; ++q)
        if (w < shards[q].size()) plan->queues[q] = std::move(shards[q][w]);
      SealPrepared(*plan);
      if (plan->fence_count) prepared_.push_back(std::move(plan));
    }
    if (chunked_) chunk_plans_.push_back({first_plan, static_cast<std::uint32_t>(prepared_.size())});
    if (!phase) head_waves_ = static_cast<std::uint32_t>(prepared_.size());
  }
  prepared_count_ = count;
  RetargetWrites();
  return Status::OK();
}
void Transport::RetargetWrites() {
  if (!direct_output_) return;
  BatchGetProfileScope timer((payload_only_ || pipeline_) && profile_.enabled ? &profile_.target_update_ns : nullptr);
  for (auto& plan : prepared_)
    for (auto& q : plan->queues)
      for (const auto& destination : q.destinations) {
        auto& remote = q.remote[destination.wr];
        remote.addr = output_.base + destination.offset;
        remote.tseg = output_remote_;
      }
}
void Transport::RefreshPrepared(PreparedQueue& q, const PreparedQueue::Post& post) {
  const auto* addresses = reinterpret_cast<const std::uint64_t*>(memory_ + layout_.descriptor);
  if (!q.source_runs.empty()) {
    auto run = std::lower_bound(q.source_runs.begin(), q.source_runs.end(), post.first_sge,
        [](const PreparedQueue::SourceRun& r, std::uint32_t i) { return r.local + r.count <= i; });
    for (; run != q.source_runs.end() && run->local < post.end_sge; ++run) {
      const auto first = std::max(run->local, post.first_sge);
      const auto end = std::min(run->local + run->count, post.end_sge);
      for (auto i = first; i < end; ++i) q.local[i].addr = source_->seg.ubva.va +
          (addresses[run->row + i - run->local] - source_base_);
    }
  } else {
    auto source = std::lower_bound(q.sources.begin(), q.sources.end(), post.first_sge,
        [](const PreparedQueue::Source& s, std::uint32_t i) { return s.local < i; });
    for (; source != q.sources.end() && source->local < post.end_sge; ++source)
      q.local[source->local].addr = source_->seg.ubva.va +
          (addresses[source->row] - source_base_) + source->offset;
  }
  auto marker = std::lower_bound(q.markers.begin(), q.markers.end(), post.first_sge,
      [](const PreparedQueue::Marker& m, std::uint32_t i) { return m.local < i; });
  for (; marker != q.markers.end() && marker->local < post.end_sge; ++marker)
    *marker->value = Progress(request().sequence, marker->progress);

}
Status Transport::BeginPrepared(PreparedWrite& plan, std::uint64_t id, bool* accepted) {
  for (const auto& q : queues_)
    if (q.poisoned || q.pending) return Error("queue unavailable");
  bool submitted = false;
  auto status = tracker_.TryBegin(id, plan.handle, plan.fences.data(), plan.fence_count, &submitted);
  if (!status.ok()) return status;
  if (accepted) *accepted = submitted;
  if (!submitted) return accepted ? Status::OK() : Error("drained queue has insufficient credits");
  for (std::size_t i = 0; i < plan.fence_count; ++i) {
    const auto& fence = plan.fences[i];
    plan.queues[fence.queue].wrs.back().user_ctx = fence.context;
  }
  return Status::OK();
}
Status Transport::PostPrepared(PreparedWrite& plan, bool refresh_sources, Profile* profile,
                               std::uint64_t* first_post_ns) {
  if (pipeline_ && profile) {
    profile->cpu_threads = 1;
    ++profile->waves;
    for (const auto& q : plan.queues) profile->data_wrs += static_cast<std::uint32_t>(q.wrs.size());
  }
  std::size_t posts = 0;
  for (const auto& q : plan.queues) posts = std::max(posts, q.posts.size());
  // Give every Jetty a doorbell before preparing the next slice of any Jetty.
  for (std::size_t batch = 0; batch < posts; ++batch) {
    for (std::uint32_t queue = 0; queue < kJettyCount; ++queue) {
      auto& q = plan.queues[queue];
      if (batch >= q.posts.size()) continue;
      const auto& post = q.posts[batch];
      if (refresh_sources) {
        BatchGetProfileScope timer(pipeline_ && profile_.enabled ? &profile_.source_sge_update_ns : nullptr);
        RefreshPrepared(q, post);
      }
      const auto start = (profile || first_post_ns) ? BatchGetProfileNowNs() : 0;
      urma_jfs_wr_t* bad = nullptr;
      if (urma_post_jetty_send_wr(queues_[queue].channel.local_jetty, &q.wrs[post.wr], &bad)) {
        queues_[queue].poisoned = true;
        return Error("post prepared WRITE chain");
      }
      if (profile) {
        if (!profile->first_post_ns) profile->first_post_ns = start;
        profile->post_ns += BatchGetProfileNowNs() - start;
        ++profile->post_calls;
      }
      if (first_post_ns) {
        *first_post_ns = start;
        first_post_ns = nullptr;
      }
    }
  }
  return Status::OK();
}
Status Transport::PollAnyPrepared(std::uint64_t* completed, Profile* profile, bool wait) {
  *completed = 0;
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
          auto status = tracker_.Complete(q, cr.user_ctx, completed);
          if (!status.ok()) { queues_[q].poisoned = true; return status; }
          *finished = *completed != 0;
          return Status::OK();
        });
    next_poll_queue_ = (next_poll_queue_ + 1) % kJettyCount;
    if (!status.ok()) return status;
    if (count || !wait) return Status::OK();
    if ((++polls & 31) == 0 && std::chrono::steady_clock::now() >= deadline) {
      for (auto& q : queues_) q.poisoned = true;
      return Error("prepared CQ timeout; resources retained");
    }
  }
}
Status Transport::PollPrepared(std::uint64_t id, Profile* profile) {
  std::uint64_t completed = 0;
  auto status = PollAnyPrepared(&completed, profile, true);
  if (status.ok() && completed != id) return Error("unexpected prepared completion");
  return status;
}
Status Transport::WritePayloadOnly() {
  payload_sample_ = {};
  payload_sample_.waves = static_cast<std::uint16_t>(prepared_.size());
  const bool measure = profile_.enabled && prepared_.size() == 1;
  for (std::size_t w = 0; w < prepared_.size(); ++w) {
    auto& plan = *prepared_[w];
    {
      BatchGetProfileScope timer(profile_.enabled ? &profile_.source_sge_update_ns : nullptr);
      for (auto& q : plan.queues) {
        for (const auto& post : q.posts) RefreshPrepared(q, post);
        payload_sample_.data_wrs += static_cast<std::uint32_t>(q.wrs.size());
      }
    }
    Status status;
    {
      BatchGetProfileScope timer(profile_.enabled ? &profile_.data_begin_ns : nullptr);
      status = BeginPrepared(plan, w + 1);
    }
    if (!status.ok()) return status;
    std::uint64_t first_post_ns = 0;
    status = PostPrepared(plan, false, nullptr, measure ? &first_post_ns : nullptr);
    const auto posted = measure ? BatchGetProfileNowNs() : 0;
    if (!status.ok()) return status;
    status = PollPrepared(w + 1);
    if (!status.ok()) return status;
    if (measure) {
      const auto completed = BatchGetProfileNowNs();
      payload_sample_.elapsed_ns = completed - first_post_ns;
      profile_.payload_post_ns = posted - first_post_ns;
      profile_.payload_cq_wait_ns = completed - posted;
    }
    // PollPrepared succeeds only after every participating queue's tail CQE.
    payload_sample_.tail_cqes += static_cast<std::uint16_t>(plan.fence_count);
  }
  return Status::OK();
}
Status Transport::WritePipeline() {
  auto* p = profile_.enabled ? &profile_ : nullptr;
  // Retain the last head wave in flight while consuming suffix publication.
  for (std::uint32_t w = 0; w < head_waves_; ++w) {
    auto& plan = *prepared_[w];
    auto status = BeginPrepared(plan, w + 1);
    if (!status.ok()) return status;
    {
      BatchGetProfileScope timer(p ? &p->head_submit_ns : nullptr);
      status = PostPrepared(plan, true, p);
    }
    if (!status.ok()) return status;
    if (w + 1 < head_waves_) {
      BatchGetProfileScope timer(p ? &p->head_cq_wait_ns : nullptr);
      status = PollPrepared(w + 1, p);
      if (!status.ok()) return status;
    }
  }
  Status status;
  {
    BatchGetProfileScope timer(p ? &p->suffix_wait_ns : nullptr);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);
    for (;;) {
      status = Transfer(false, kSuffixNotifyOffset, 8);
      if (!status.ok()) return status;
      const auto token = *reinterpret_cast<const std::uint64_t*>(memory_ + kSuffixNotifyOffset);
      const auto sequence = token >> 32;
      if (sequence > request().sequence) { batch_result_ = Result::kSequenceError; break; }
      if (sequence == request().sequence) {
        const auto result = static_cast<std::uint32_t>(token);
        batch_result_ = result >= std::uint32_t(Result::kOk) && result <= std::uint32_t(Result::kCopyError)
            ? static_cast<Result>(result) : Result::kInvalidRequest;
        break;
      }
      if (std::chrono::steady_clock::now() >= deadline) { batch_result_ = Result::kTimeout; break; }
    }
  }
  const bool read_suffix = batch_result_ == Result::kOk && head_count_ < request().count;
  const auto reading = p && read_suffix ? BatchGetProfileNowNs() : 0;
  if (read_suffix) {
    status = BeginControlRead(layout_.descriptor + std::uint64_t{head_count_} * 8,
                              std::uint64_t{request().count - head_count_} * 8);
    if (!status.ok()) return status;
  }
  if (read_suffix) {
    status = Wait(control_queue_);
    if (p) p->suffix_read_ns = BatchGetProfileNowNs() - reading;
    if (!status.ok()) return status;
    // Invalid suffix addresses never become SGE addresses or data WRs.
    if (!ValidateSources(head_count_, request().count).ok()) batch_result_ = Result::kInvalidRequest;
    if (batch_result_ == Result::kOk) {
      // Prepare the suffix while the head DMA is still in flight. Every wave
      // has independent WR/SGE storage, so no submitted entry is overwritten.
      BatchGetProfileScope timer(p ? &p->source_sge_update_ns : nullptr);
      for (std::size_t w = head_waves_; w < prepared_.size(); ++w)
        for (auto& q : prepared_[w]->queues)
          for (const auto& post : q.posts) RefreshPrepared(q, post);
    }
  }
  {
    BatchGetProfileScope timer(p ? &p->head_cq_wait_ns : nullptr);
    status = PollPrepared(head_waves_, p);
  }
  if (!status.ok()) return status;
  if (batch_result_ != Result::kOk) return Status::OK();  // Worker publishes the failure after the head drained.
  for (std::size_t w = head_waves_; w < prepared_.size(); ++w) {
    auto& plan = *prepared_[w];
    status = BeginPrepared(plan, w + 1);
    if (!status.ok()) return status;
    {
      BatchGetProfileScope timer(p ? &p->suffix_submit_ns : nullptr);
      status = PostPrepared(plan, false, p);
    }
    if (!status.ok()) return status;
    {
      BatchGetProfileScope timer(p ? &p->suffix_cq_wait_ns : nullptr);
      status = PollPrepared(w + 1, p);
    }
    if (!status.ok()) return status;
  }
  return Status::OK();
}

// aiv-scatter's mailbox consumption and credit-driven prefix/suffix posting,
// adapted to caller-owned direct output. No device descriptor READ is issued.
Status Transport::WritePushed(const std::atomic<bool>* stop) {
  auto* p = profile_.enabled ? &profile_ : nullptr;
  std::uint32_t inflight = 0;
  auto poll = [&](bool wait) -> Status {
    std::uint64_t completed = 0;
    auto status = PollAnyPrepared(&completed, p, wait);
    if (status.ok() && completed) {
      if (!inflight || completed > prepared_.size()) return Error("unexpected mailbox data CQE");
      --inflight;
    }
    return status;
  };
  auto post = [&](std::size_t begin, std::size_t end) -> Status {
    for (auto w = begin; w < end; ++w) {
      bool accepted = false;
      while (!accepted) {
        if (inflight < 4) {
          auto status = BeginPrepared(*prepared_[w], w + 1, &accepted);
          if (!status.ok()) return status;
        }
        if (!accepted) {
          if (!inflight) return Error("mailbox plan exceeds drained SQ credits");
          auto status = poll(true); if (!status.ok()) return status;
        }
      }
      auto status = PostPrepared(*prepared_[w], true, p);
      if (!status.ok()) return status;
      ++inflight;
      if (p) p->max_inflight = std::max(p->max_inflight, inflight);
    }
    return Status::OK();
  };
  Status status;
  {
    BatchGetProfileScope timer(p ? &p->head_submit_ns : nullptr);
    status = post(0, head_waves_);
  }
  if (!status.ok()) return status;
  if (head_count_ < request().count) {
    const auto suffix_started = p ? BatchGetProfileNowNs() : 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);
    for (;;) {
      const auto token = __atomic_load_n(reinterpret_cast<const std::uint64_t*>(memory_ + kSuffixNotifyOffset), __ATOMIC_ACQUIRE);
      if ((token >> 32) > request().sequence) { batch_result_ = Result::kSequenceError; break; }
      if ((token >> 32) == request().sequence) {
        const auto r = static_cast<std::uint32_t>(token);
        batch_result_ = r >= std::uint32_t(Result::kOk) && r <= std::uint32_t(Result::kCopyError)
            ? static_cast<Result>(r) : Result::kInvalidRequest;
        break;
      }
      if ((stop && stop->load(std::memory_order_acquire)) || std::chrono::steady_clock::now() >= deadline) {
        batch_result_ = Result::kTimeout; break;
      }
      if (inflight) { status = poll(false); if (!status.ok()) return status; }
    }
    if (p) p->suffix_wait_ns = BatchGetProfileNowNs() - suffix_started;
    if (batch_result_ == Result::kOk && !ValidateSources(head_count_, request().count).ok())
      batch_result_ = Result::kInvalidRequest;
    if (batch_result_ == Result::kOk) {
      BatchGetProfileScope timer(p ? &p->suffix_submit_ns : nullptr);
      status = post(head_waves_, prepared_.size());
      if (!status.ok()) return status;
    }
  }
  while (inflight) { status = poll(true); if (!status.ok()) return status; }
  return Status::OK();
}

Status Transport::WriteChunks() {
  auto* p = profile_.enabled ? &profile_ : nullptr;
  const auto count = request().count;
  const auto chunks = (count + kChunkRows - 1) / kChunkRows;
  constexpr std::uint32_t kWindow = 4;  // Bounded even for split-token, multi-wave chunks.
  std::array<std::uint32_t, kMaxWorkers> published{};
  std::vector<bool> posted(chunks);
  std::uint32_t submitted_chunks = 0, inflight = 0, prefetched = chunks;
  std::uint64_t prefetch_start = 0;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);
  auto poll = [&](bool wait) -> Status {
    std::uint64_t completed = 0;
    BatchGetProfileScope timer(p ? &p->chunk_cq_wait_ns : nullptr);
    auto status = PollAnyPrepared(&completed, p, wait);
    if (status.ok() && completed) {
      if (completed > prepared_.size() || !inflight) return Error("unexpected chunk completion");
      --inflight;
    }
    return status;
  };
  while (submitted_chunks < chunks && batch_result_ == Result::kOk) {
    std::uint32_t next = prefetched;
    if (next == chunks) for (std::uint32_t c = 0; c < chunks; ++c)
      if (!posted[c] && published[c % workers_] > c / workers_) { next = c; break; }
    if (next == chunks) {
      {
        BatchGetProfileScope timer(p ? &p->chunk_ready_ns : nullptr);
        auto status = Transfer(false, layout_.ready, workers_ * sizeof(Ready));
        if (!status.ok()) return status;
        if (p) ++p->progress_reads;
      }
      for (std::uint32_t w = 0; w < workers_; ++w) {
        const auto token = reinterpret_cast<const Word*>(memory_ + layout_.ready)[w].value;
        const auto sequence = token >> 32;
        if (sequence < request().sequence) continue;
        if (sequence > request().sequence) { batch_result_ = Result::kSequenceError; break; }
        const auto progress = static_cast<std::uint32_t>(token);
        if (progress & kChunkFailed) {
          const auto error = progress & ~kChunkFailed;
          batch_result_ = error >= std::uint32_t(Result::kInvalidRequest) && error <= std::uint32_t(Result::kCopyError)
              ? static_cast<Result>(error) : Result::kInvalidRequest;
          break;
        }
        const auto owned = chunks / workers_ + (w < chunks % workers_ ? 1U : 0U);
        if (progress > owned || progress < published[w]) { batch_result_ = Result::kInvalidRequest; break; }
        published[w] = progress;
      }
      if (std::chrono::steady_clock::now() >= deadline) batch_result_ = Result::kTimeout;
      // Reap without waiting for an unpublished chunk; another producer may be ready.
      bool ready = false;
      for (std::uint32_t c = 0; c < chunks; ++c)
        ready |= !posted[c] && published[c % workers_] > c / workers_;
      if (inflight && !ready) { auto status = poll(false); if (!status.ok()) return status; }
      continue;
    }
    const auto begin = next * kChunkRows, end = std::min(count, begin + kChunkRows);
    {
      const auto reading = prefetched != chunks ? prefetch_start : p ? BatchGetProfileNowNs() : 0;
      auto status = prefetched != chunks ? Status::OK() :
          BeginControlRead(layout_.descriptor + std::uint64_t{begin} * 8, std::uint64_t{end - begin} * 8);
      if (status.ok()) status = Wait(control_queue_);
      if (p) p->chunk_read_ns += BatchGetProfileNowNs() - reading;
      prefetched = chunks;
      if (!status.ok()) return status;
    }
    if (!ValidateSources(begin, end).ok()) { batch_result_ = Result::kInvalidRequest; break; }
    // Read the next known-ready descriptor slice while refreshing/posting this
    // chunk. Each READ lands in its own stable slice of the registered mirror.
    for (std::uint32_t c = 0; c < chunks; ++c) {
      if (c == next || posted[c] || published[c % workers_] <= c / workers_) continue;
      const auto first = c * kChunkRows, last = std::min(count, first + kChunkRows);
      prefetch_start = p ? BatchGetProfileNowNs() : 0;
      auto status = BeginControlRead(layout_.descriptor + std::uint64_t{first} * 8, std::uint64_t{last - first} * 8);
      if (!status.ok()) return status;
      prefetched = c;
      break;
    }
    const auto plans = chunk_plans_[next];
    for (auto w = plans.begin; w < plans.end; ++w) {
      auto& plan = *prepared_[w];
      bool accepted = false;
      while (!accepted) {
        if (inflight < kWindow) {
          auto status = BeginPrepared(plan, w + 1, &accepted);
          if (!status.ok()) return status;
        }
        if (!accepted) {
          if (!inflight) return Error("chunk exceeds drained queue credits");
          auto status = poll(true);
          if (!status.ok()) return status;
        }
      }
      // Every plan owns independent WR/SGE storage, retained through its tail CQEs.
      {
        BatchGetProfileScope timer(p ? &p->chunk_submit_ns : nullptr);
        auto status = PostPrepared(plan, true, p);
        if (!status.ok()) return status;
      }
      ++inflight;
      if (p) p->max_inflight = std::max(p->max_inflight, inflight);
    }
    posted[next] = true;
    ++submitted_chunks;
    if (p) ++p->chunks;
  }
  // An invalid later chunk cannot retire imports or publish done over prior DMA.
  while (inflight) { auto status = poll(true); if (!status.ok()) return status; }
  return Status::OK();
}

Status Transport::WriteBatch(const std::atomic<bool>* stop) {
  BatchGetProfileScope timer(profile_.enabled ? &profile_.write_ns : nullptr);
  Status status;
  {
    BatchGetProfileScope stage((payload_only_ || pipeline_) && profile_.enabled ? &profile_.prepare_writes_ns : nullptr);
    status = PrepareWrites(request().count);
  }
  if (!status.ok()) return status;
  if (device_push_) return WritePushed(stop);
  if (payload_only_) return WritePayloadOnly();
  if (chunked_) return WriteChunks();
  if (pipeline_) return WritePipeline();
  // Match the ordered-WRITE probe: short submissions to all four queues from
  // their owner, then poll their tails together. Avoid waking an OpenMP team
  // and crossing several team barriers for each short request.
  for (std::size_t w = 0; w < prepared_.size(); ++w) {
    auto& plan = *prepared_[w];
    status = BeginPrepared(plan, w + 1);
    if (!status.ok()) return status;
    if (profile_.enabled) {
      profile_.cpu_threads = 1;
      ++profile_.waves;
      for (const auto& q : plan.queues) {
        profile_.ready_wrs += q.markers.size();
        profile_.data_wrs += q.wrs.size() - q.markers.size();
      }
    }
    status = PostPrepared(plan, true, profile_.enabled ? &profile_ : nullptr);
    if (!status.ok()) return status;
    const auto posted = profile_.enabled ? BatchGetProfileNowNs() : 0;
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
  if (control_queue_.pending) return Error("control READ is still in flight");
  // No imported output survives done. Host may unregister/release it as soon
  // as the kernel observes completion; cached plans are retargeted next round.
  if (output_remote_) {
    if (!tracker_.Empty()) return Error("output WRITE is still in flight");
    BatchGetProfileScope stage((payload_only_ || pipeline_) && profile_.enabled ? &profile_.output_unimport_ns : nullptr);
    if (urma_unimport_seg(output_remote_)) return Error("unimport output segment");
    output_remote_ = nullptr;
  }
  auto* done = reinterpret_cast<Completion*>(memory_ + layout_.done);
  *done = {Progress(sequence, static_cast<std::uint32_t>(result)), {}};
  if (payload_only_ && result == Result::kOk) done->cpu_sgl_write = payload_sample_;
  auto status = BeginPrepared(done_, UINT64_MAX);
  if (status.ok()) {
    BatchGetProfileScope stage((payload_only_ || pipeline_) && profile_.enabled ? &profile_.done_post_ns : nullptr);
    status = PostPrepared(done_);
  }
  if (status.ok()) {
    BatchGetProfileScope stage((payload_only_ || pipeline_) && profile_.enabled ? &profile_.done_cq_wait_ns : nullptr);
    status = PollPrepared(UINT64_MAX);
  }
  if ((payload_only_ || pipeline_) && profile_.enabled && profile_.cpu_start_ns)
    profile_.cpu_total_ns = BatchGetProfileNowNs() - profile_.cpu_start_ns;
  return status;
}

void Transport::PrintProfile() const {
  if (!profile_.enabled) return;
  if (device_push_) {
    const auto& p = profile_;
    std::fprintf(stderr, "CPU_SGL_PUSH_PROFILE sequence=%" PRIu64 " rows=%u descriptor_reads=0 ready_wrs=0 source_validation=disabled"
        " head_first_post_us=%.3f head_submit_us=%.3f suffix_wait_us=%.3f suffix_submit_us=%.3f"
        " output_import_us=%.3f address_validate_us=%.3f source_sge_update_us=%.3f cpu_total_us=%.3f"
        " data_wrs=%u tail_cqes=%u max_inflight=%u plan_reused=%u result=%u\n",
        request().sequence, request().count, p.first_post_ns ? (p.first_post_ns - p.cpu_start_ns) / 1000.0 : 0,
        p.head_submit_ns / 1000.0, p.suffix_wait_ns / 1000.0, p.suffix_submit_ns / 1000.0,
        p.output_import_ns / 1000.0, p.address_validate_ns / 1000.0, p.source_sge_update_ns / 1000.0,
        p.cpu_total_ns / 1000.0, p.data_wrs, p.tail_cqes, p.max_inflight, unsigned(p.plan_reused), unsigned(batch_result_));
    return;
  }
  if (chunked_) {
    const auto& p = profile_;
    std::fprintf(stderr,
        "CPU_SGL_CHUNK_PROFILE sequence=%" PRIu64 " rows=%u chunk_rows=%u chunks=%u"
        " notify_read_us=%.3f header_read_us=%.3f output_import_us=%.3f plan_us=%.3f"
        " first_post_us=%.3f chunk_ready_us=%.3f chunk_read_us=%.3f chunk_submit_us=%.3f chunk_cq_wait_us=%.3f"
        " address_validate_us=%.3f source_sge_update_us=%.3f target_update_us=%.3f"
        " output_unimport_us=%.3f done_post_us=%.3f done_cq_wait_us=%.3f cpu_total_us=%.3f"
        " progress_reads=%u max_inflight=%u data_wrs=%u ready_wrs=0 post_calls=%u waves=%u tail_cqes=%u plan_reused=%u result=%u\n",
        request().sequence, request().count, kChunkRows, p.chunks,
        p.notify_read_ns / 1000.0, p.header_read_ns / 1000.0, p.output_import_ns / 1000.0,
        (p.prepare_writes_ns - p.target_update_ns) / 1000.0,
        (p.first_post_ns ? p.first_post_ns - p.cpu_start_ns : 0) / 1000.0,
        p.chunk_ready_ns / 1000.0, p.chunk_read_ns / 1000.0, p.chunk_submit_ns / 1000.0, p.chunk_cq_wait_ns / 1000.0,
        p.address_validate_ns / 1000.0, p.source_sge_update_ns / 1000.0, p.target_update_ns / 1000.0,
        p.output_unimport_ns / 1000.0, p.done_post_ns / 1000.0, p.done_cq_wait_ns / 1000.0, p.cpu_total_ns / 1000.0,
        p.progress_reads, p.max_inflight, p.data_wrs, p.post_calls, p.waves, p.tail_cqes,
        unsigned(p.plan_reused), unsigned(batch_result_));
    return;
  }
  if (pipeline_) {
    const auto& p = profile_;
    std::fprintf(stderr,
        "CPU_SGL_PIPELINE_PROFILE sequence=%" PRIu64 " rows=%u head_rows=%u"
        " notify_read_us=%.3f head_read_us=%.3f plan_us=%.3f"
        " head_first_post_us=%.3f head_submit_us=%.3f suffix_wait_us=%.3f"
        " suffix_read_us=%.3f head_cq_wait_us=%.3f suffix_submit_us=%.3f suffix_cq_wait_us=%.3f"
        " header_read_us=%.3f output_desc_read_us=%.3f output_import_us=%.3f"
        " address_validate_us=%.3f source_sge_update_us=%.3f target_update_us=%.3f"
        " output_unimport_us=%.3f done_post_us=%.3f done_cq_wait_us=%.3f cpu_total_us=%.3f"
        " data_wrs=%u ready_wrs=0 post_calls=%u waves=%u tail_cqes=%u plan_reused=%u result=%u\n",
        request().sequence, request().count, head_count_,
        p.notify_read_ns / 1000.0, p.address_list_read_ns / 1000.0,
        (p.prepare_writes_ns - p.target_update_ns) / 1000.0,
        (p.first_post_ns ? p.first_post_ns - p.cpu_start_ns : 0) / 1000.0,
        p.head_submit_ns / 1000.0, p.suffix_wait_ns / 1000.0, p.suffix_read_ns / 1000.0,
        p.head_cq_wait_ns / 1000.0, p.suffix_submit_ns / 1000.0, p.suffix_cq_wait_ns / 1000.0,
        p.header_read_ns / 1000.0, p.output_desc_read_ns / 1000.0, p.output_import_ns / 1000.0,
        p.address_validate_ns / 1000.0, p.source_sge_update_ns / 1000.0, p.target_update_ns / 1000.0,
        p.output_unimport_ns / 1000.0, p.done_post_ns / 1000.0, p.done_cq_wait_ns / 1000.0,
        p.cpu_total_ns / 1000.0, p.data_wrs, p.post_calls, p.waves, p.tail_cqes,
        unsigned(p.plan_reused), unsigned(batch_result_));
    return;
  }
  if (payload_only_) {
    std::fprintf(stderr, "CPU_SGL_WRITE_SAMPLE sequence=%" PRIu64 " bytes=%" PRIu64
                 " elapsed_ns=%" PRIu64 " data_wrs=%u ready_wrs=0 waves=%u tail_cqes=%u\n",
                 request().sequence, std::uint64_t{request().count} * token_bytes_,
                 payload_sample_.elapsed_ns, payload_sample_.data_wrs,
                 unsigned(payload_sample_.waves), unsigned(payload_sample_.tail_cqes));
    const auto& p = profile_;
    std::fprintf(stderr,
        "CPU_SGL_WRITE_STAGES sequence=%" PRIu64 " rows=%u"
        " notify_read_us=%.3f header_read_us=%.3f address_list_read_us=%.3f address_validate_us=%.3f"
        " output_desc_read_us=%.3f output_import_us=%.3f plan_us=%.3f target_update_us=%.3f"
        " source_sge_update_us=%.3f data_begin_us=%.3f payload_post_us=%.3f payload_cq_wait_us=%.3f"
        " output_unimport_us=%.3f done_post_us=%.3f done_cq_wait_us=%.3f cpu_total_us=%.3f"
        " plan_reused=%u\n",
        request().sequence, request().count,
        p.notify_read_ns / 1000.0, p.header_read_ns / 1000.0,
        p.address_list_read_ns / 1000.0, p.address_validate_ns / 1000.0,
        p.output_desc_read_ns / 1000.0, p.output_import_ns / 1000.0,
        (p.prepare_writes_ns - p.target_update_ns) / 1000.0, p.target_update_ns / 1000.0,
        p.source_sge_update_ns / 1000.0, p.data_begin_ns / 1000.0,
        p.payload_post_ns / 1000.0, p.payload_cq_wait_ns / 1000.0,
        p.output_unimport_ns / 1000.0, p.done_post_ns / 1000.0,
        p.done_cq_wait_ns / 1000.0, p.cpu_total_ns / 1000.0, unsigned(p.plan_reused));
    return;
  }
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
  for (std::uint32_t i = 0; i <= kJettyCount; ++i) {
    auto& q = i == kJettyCount ? control_queue_ : queues_[i];
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
  if (output_remote_) {
    if (urma_unimport_seg(output_remote_)) return Error("unimport output");
    output_remote_ = nullptr;
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
  ResetRequests(control_queue_); control_queue_.poisoned = false; control_queue_.sequence = 0;
  std::free(memory_); memory_ = nullptr;
  context_ = nullptr;
  return Status::OK();
}
}  // namespace kvaf::cpu_push
