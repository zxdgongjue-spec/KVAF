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
#include <thread>

namespace kvaf::cpu_push {
namespace {
Status Error(const char* operation) { return Status::Internal(std::string("CPU push: ") + operation); }
// Reserve SQ headroom for the provider's multi-block encoding of 16-SGE WRs.
constexpr std::size_t kWaveWrs = 512;
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
        {16384, 1024, max_sge_, false, true}, c.local_jfc, c.local_jfr, c.local_jetty);
    if (error) return Error(error);
    // Keep slots and all pointers stable until a tail CQE (or queue teardown).
    queue.local.reserve(kWaveWrs * max_sge_);
    queue.remote.reserve(kWaveWrs);
    queue.wrs.reserve(kWaveWrs);
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

Status Transport::WriteBatch() {
  BatchGetProfileScope timer(profile_.enabled ? &profile_.write_ns : nullptr);
  const auto& r = request();
  const auto* addresses = reinterpret_cast<const std::uint64_t*>(memory_ + layout_.descriptor);
  auto* markers = reinterpret_cast<std::uint64_t*>(memory_ + layout_.staging);
  for (auto& q : queues_) {
    if (q.poisoned || q.pending) return Error("previous WRITE still live");
    ResetRequests(q);
  }
  const auto flush_wave = [&]() -> Status {
    if (profile_.enabled) ++profile_.waves;
    for (auto& q : queues_) {
      if (q.wrs.empty()) continue;
      const auto begin = profile_.enabled ? BatchGetProfileNowNs() : 0;
      if (profile_.enabled && !profile_.first_post_ns) profile_.first_post_ns = begin;
      auto status = Post(q);
      if (profile_.enabled) {
        profile_.post_ns += BatchGetProfileNowNs() - begin;
        profile_.post_calls += (q.wrs.size() + 31) / 32;
      }
      if (!status.ok()) return status;
    }
    for (auto& q : queues_) {
      if (q.pending) {
        const auto begin = profile_.enabled ? BatchGetProfileNowNs() : 0;
        auto status = Wait(q);
        if (profile_.enabled) {
          profile_.last_cqe_ns = BatchGetProfileNowNs();
          profile_.cq_wait_ns += profile_.last_cqe_ns - begin;
        }
        if (!status.ok()) return status;
      }
      ResetRequests(q);
    }
    return Status::OK();
  };
  const auto reserve_wr = [&](Queue& q) -> Status {
    return q.wrs.size() == kWaveWrs ? flush_wave() : Status::OK();
  };
  std::uint32_t chunks = 0;
  for (std::uint32_t w = 0; w < r.workers; ++w)
    chunks = std::max(chunks, ChunkCount(WorkerRange(r.count, token_bytes_, w, r.workers)));
  for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
    for (std::uint32_t worker = 0; worker < r.workers; ++worker) {
      const auto partition = WorkerRange(r.count, token_bytes_, worker, r.workers);
      if (chunk >= ChunkCount(partition)) continue;
      const auto range = ChunkRange(partition, chunk);
      std::uint32_t row = range.begin, offset = 0, wr_index = 0;
      while (row < range.end) {
        auto& q = queues_[wr_index++ % kJettyCount];
        auto status = reserve_wr(q);
        if (!status.ok()) return status;
        const auto first = static_cast<std::uint32_t>(q.local.size());
        const auto destination = hbm_base_ + layout_.staging + std::uint64_t{row} * token_bytes_ + offset;
        std::uint32_t bytes = 0, sges = 0;
        while (row < range.end && sges < max_sge_ && bytes < max_bytes_) {
          const auto length = std::min(token_bytes_ - offset, max_bytes_ - bytes);
          q.local.push_back({addresses[row] + offset, length, source_, nullptr});
          ++sges; bytes += length; offset += length;
          if (offset == token_bytes_) { ++row; offset = 0; }
        }
        Add(q, URMA_OPC_WRITE, destination, bytes, first, sges);
        if (profile_.enabled) ++profile_.data_wrs;
      }
      for (std::uint32_t index = 0; index < kJettyCount; ++index) {
        auto& q = queues_[index];
        auto status = reserve_wr(q);
        if (!status.ok()) return status;
        auto* marker = markers + (worker * kMaxChunks + chunk) * kJettyCount + index;
        *marker = Progress(r.sequence, chunk + 1);
        const auto first = static_cast<std::uint32_t>(q.local.size());
        q.local.push_back({reinterpret_cast<std::uint64_t>(marker), 8, control_, nullptr});
        Add(q, URMA_OPC_WRITE, hbm_base_ + layout_.ready + worker * sizeof(Ready) + index * 8,
            8, first, 1);
        q.wrs.back().flag.bs.place_order = URMA_STRONG_ORDER;
        if (profile_.enabled) ++profile_.ready_wrs;
      }
    }
  }
  // Chunks share one wave. Wait only before reusing full WR storage or finishing.
  return flush_wave();
}
Status Transport::Complete(std::uint64_t sequence, Result result) {
  BatchGetProfileScope timer(profile_.enabled ? &profile_.done_ns : nullptr);
  auto* done = reinterpret_cast<Completion*>(memory_ + layout_.done);
  *done = {Progress(sequence, static_cast<std::uint32_t>(result)), {}};
  return Transfer(true, layout_.done, sizeof(Completion));
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
      " data_wrs=%u ready_wrs=%u post_calls=%u waves=%u optimized=%u\n",
      r.sequence, r.workers, r.count, std::uint64_t{r.count} * token_bytes_,
      p.header_read_ns / 1000.0, p.address_read_ns / 1000.0, build_ns / 1000.0, p.post_ns / 1000.0,
      p.cq_wait_ns / 1000.0, p.write_ns / 1000.0, submit_ns / 1000.0, p.done_ns / 1000.0,
      p.data_wrs, p.ready_wrs, p.post_calls, p.waves, optimized);
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
  std::free(memory_); memory_ = nullptr;
  context_ = nullptr;
  return Status::OK();
}
}  // namespace kvaf::cpu_push
