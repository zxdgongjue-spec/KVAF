#pragma once
#include <memory>
#include "urma_transport.h"
#include "gather_progress_pump.h"
#include "peer_protocol.h"
#include "status.h"

namespace kvaf::peer {
// Borrows the CPU half of one gather control channel. CpuService deletes that
// initiator Jetty before Shutdown releases any registered DMA source storage.
class GatherProgressTransport {
 public:
  ~GatherProgressTransport();
  Status Initialize(urma_context_t*, const EnableGatherProgressRequest&, uint32_t lane,
                    urma_jetty_t*, urma_target_jetty_t*, urma_jfc_t*);
  void Advance(uint64_t sequence, const uint64_t* progress, uint32_t shards);
  bool Finish();
  bool ok() const { return !pump_ || pump_->ok(); }
  urma_jetty_t* jetty() const { return jetty_; }
  void Quiesced() { jetty_ = nullptr; }
  Status Shutdown();
 private:
  static bool Post(void*, uint32_t slot, uint64_t id, uint32_t bytes);
  static int Poll(void*, uint64_t* id);
  urma_context_t* context_{};
  urma_jetty_t* jetty_{};
  urma_target_jetty_t* target_{};
  urma_jfc_t* cq_{};
  urma_target_seg_t* remote_{}, *source_{};
  urma_token_id_t* token_{};
  uint64_t* frames_{};
  uint64_t mailbox_{}, sequence_{}, session_{};
  uint32_t lane_{};
  const uint64_t* progress_{};
  uint32_t shards_{};
  urma::Sge local_[2]{}, destination_[2]{};
  urma_jfs_wr_t wrs_[2]{};
  std::unique_ptr<GatherProgressPump> pump_;
};
}  // namespace kvaf::peer
