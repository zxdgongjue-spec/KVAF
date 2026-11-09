#include "gather_progress_transport.h"
#include "cpu_gather_executor.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace kvaf::peer {
namespace {
Status Error(const char* action) { return Status::Internal(std::string("gather progress: ") + action); }
}
GatherProgressTransport::~GatherProgressTransport() {
  if (frames_ || source_ || token_ || remote_) {
    std::fprintf(stderr, "gather progress destroyed before quiesced Shutdown\n");
    std::terminate();
  }
}
Status GatherProgressTransport::Initialize(urma_context_t* context,
    const EnableGatherProgressRequest& request, uint32_t lane,
    urma_jetty_t* jetty, urma_target_jetty_t* target, urma_jfc_t* cq) {
  if (context_ || !context || !jetty || !target || !cq || lane >= request.lanes ||
      request.segment_bytes != sizeof(urma_seg_t) || !request.token_value)
    return Status::InvalidArgument("invalid gather progress transport");
  context_ = context; jetty_ = jetty; target_ = target; cq_ = cq;
  session_ = request.session_id; lane_ = lane;
  mailbox_ = request.mailbox_base + uint64_t(lane) * kGatherScratchBytes;
  urma_seg_t segment{};
  std::memcpy(&segment, request.segment, sizeof(segment));
  if (mailbox_ % 64 || !GatherRange(segment.ubva.va, segment.len, mailbox_, kGatherProgressBytes))
    return Status::InvalidArgument("progress mailbox outside exported HBM");
  urma_import_seg_flag_t flags{};
  flags.bs.cacheable = URMA_NON_CACHEABLE;
  flags.bs.access = 1U << 2; // remote WRITE only
  flags.bs.mapping = URMA_SEG_NOMAP;
  urma_token_t value{}; value.token = request.token_value;
  remote_ = urma_import_seg(context_, &segment, &value, 0, flags);
  if (!remote_) return Error("import HBM");
  void* storage = nullptr;
  if (posix_memalign(&storage, 4096, 4096)) return Error("allocate snapshots");
  frames_ = static_cast<uint64_t*>(storage);
  std::memset(frames_, 0, 4096);
  token_ = urma_alloc_token_id(context_);
  if (!token_) return Error("allocate snapshot token");
  urma_seg_cfg_t cfg{};
  cfg.va = reinterpret_cast<uint64_t>(frames_); cfg.len = 4096;
  cfg.token_id = token_; cfg.token_value.token = request.token_value;
  cfg.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
  cfg.flag.bs.access = 1U; // same pinned direct-host source contract as CPU WRITEs
  cfg.flag.bs.non_pin = 0; cfg.flag.bs.token_id_valid = 1;
  source_ = urma_register_seg(context_, &cfg);
  if (!source_) return Error("register snapshots");
  pump_ = std::make_unique<GatherProgressPump>(frames_, GatherProgressPump::Ops{this, Post, Poll});
  return Status::OK();
}
bool GatherProgressTransport::Post(void* context, uint32_t slot, uint64_t id, uint32_t bytes) {
  auto& self = *static_cast<GatherProgressTransport*>(context);
  self.local_[slot] = {reinterpret_cast<uint64_t>(self.frames_ + slot * kGatherMaxShards), bytes, self.source_, nullptr};
  self.destination_[slot] = {self.mailbox_, bytes, self.remote_, nullptr};
  auto& wr = self.wrs_[slot];
  wr = {};
  wr.opcode = URMA_OPC_WRITE;
  wr.flag.bs.place_order = URMA_STRONG_ORDER;
  wr.flag.bs.comp_order = 1;
  wr.flag.bs.complete_enable = 1;
  wr.tjetty = self.target_; wr.user_ctx = id;
  wr.rw.src = {reinterpret_cast<urma_sge_t*>(&self.local_[slot]), 1};
  wr.rw.dst = {reinterpret_cast<urma_sge_t*>(&self.destination_[slot]), 1};
  GatherBarrier();
  urma_jfs_wr_t* bad = nullptr;
  return urma_post_jetty_send_wr(self.jetty_, &wr, &bad) == 0;
}
int GatherProgressTransport::Poll(void* context, uint64_t* id) {
  auto& self = *static_cast<GatherProgressTransport*>(context);
  urma_cr_t cr{};
  const auto count = urma_poll_jfc(self.cq_, 1, &cr);
  if (count < 0 || (count && cr.status != URMA_CR_SUCCESS)) return -1;
  if (count) *id = cr.user_ctx;
  return count;
}
void GatherProgressTransport::Advance(uint64_t sequence, const uint64_t* progress, uint32_t shards) {
  sequence_ = sequence; progress_ = progress; shards_ = shards;
  if (pump_) (void)pump_->Advance(sequence, progress, shards);
}
bool GatherProgressTransport::Finish() {
  if (!pump_) return false;
  if (!progress_) return pump_->ok() && !pump_->Pending();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  uint32_t polls = 0;
  do {
    if (!pump_->Advance(sequence_, progress_, shards_)) return false;
    if (!pump_->Pending() && pump_->Latest()) {
      progress_ = nullptr; shards_ = 0;
      return true;
    }
    GatherSpinPause();
    if ((++polls & 255) == 0 && std::chrono::steady_clock::now() >= deadline) return false;
  } while (true);
}
Status GatherProgressTransport::Shutdown() {
  if (jetty_) return Error("initiator not quiesced; DMA sources retained");
  if (remote_) { if (urma_unimport_seg(remote_)) return Error("unimport HBM"); remote_ = nullptr; }
  if (source_) { if (urma_unregister_seg(source_)) return Error("unregister snapshots"); source_ = nullptr; }
  if (token_) { if (urma_free_token_id(token_)) return Error("free snapshot token"); token_ = nullptr; }
  if (pump_)
    std::fprintf(stderr, "[KVAF_CPU] GATHER_PROGRESS_STATS session=%llu lane=%u frames=%llu busy_checks=%llu\n",
        static_cast<unsigned long long>(session_), lane_,
        static_cast<unsigned long long>(pump_->posts()),
        static_cast<unsigned long long>(pump_->busy_checks()));
  pump_.reset(); std::free(frames_); frames_ = nullptr;
  context_ = nullptr;
  return Status::OK();
}
}  // namespace kvaf::peer
