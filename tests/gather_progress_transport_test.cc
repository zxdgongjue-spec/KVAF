// Uses public URMA types with a deferred transport double; no device or liburma.
#include "gather_progress_transport.h"
#include <array>
#include <cassert>
#include <cstring>
#include <deque>
#include <iostream>
#include <vector>
using kvaf::peer::GatherProgressTransport;
namespace {
struct Pending { urma_jfs_wr_t* wr; uint64_t id, source, destination; std::vector<uint8_t> snapshot; };
std::deque<Pending> live;
int registrations{}, imports{}, tokens{}, delay{};
bool fail_import{}, fail_post{}, wrong_cq{};
}
extern "C" {
urma_target_seg_t* urma_import_seg(urma_context_t*, urma_seg_t* seg, urma_token_t*, uint64_t,
                                  urma_import_seg_flag_t flags) {
  assert(flags.bs.access == (1U << 2));
  if (fail_import) return nullptr;
  ++imports; auto* result = new urma_target_seg_t{}; result->seg = *seg; return result;
}
urma_status_t urma_unimport_seg(urma_target_seg_t* seg) { --imports; delete seg; return static_cast<urma_status_t>(0); }
urma_token_id_t* urma_alloc_token_id(urma_context_t*) { ++tokens; return new urma_token_id_t{}; }
urma_status_t urma_free_token_id(urma_token_id_t* id) { --tokens; delete id; return static_cast<urma_status_t>(0); }
urma_target_seg_t* urma_register_seg(urma_context_t*, urma_seg_cfg_t* cfg) {
  assert(cfg->len == 4096 && cfg->flag.bs.access == 1 && !cfg->flag.bs.non_pin);
  ++registrations; auto* result = new urma_target_seg_t{};
  result->seg.ubva.va = cfg->va; result->seg.len = cfg->len; return result;
}
urma_status_t urma_unregister_seg(urma_target_seg_t* seg) {
  assert(live.empty()); --registrations; delete seg; return static_cast<urma_status_t>(0);
}
urma_status_t urma_post_jetty_send_wr(urma_jetty_t*, urma_jfs_wr_t* wr, urma_jfs_wr_t**) {
  assert(wr->opcode == URMA_OPC_WRITE && !wr->next && wr->flag.bs.complete_enable);
  assert(wr->flag.bs.place_order == URMA_STRONG_ORDER && wr->flag.bs.comp_order);
  assert(wr->rw.src.num_sge == 1 && wr->rw.dst.num_sge == 1);
  const auto& src = wr->rw.src.sge[0]; const auto& dst = wr->rw.dst.sge[0];
  assert(src.len == dst.len && src.len <= 512);
  auto* ptr = reinterpret_cast<const uint8_t*>(src.addr);
  live.push_back({wr, wr->user_ctx, src.addr, dst.addr, {ptr, ptr + src.len}});
  return static_cast<urma_status_t>(fail_post ? -1 : 0);
}
int urma_poll_jfc(urma_jfc_t*, int count, urma_cr_t* cq) {
  assert(count == 1);
  if (live.empty() || delay-- > 0) return 0;
  const auto& p = live.front();
  assert(p.wr->user_ctx == p.id);
  assert(!std::memcmp(reinterpret_cast<const void*>(p.source), p.snapshot.data(), p.snapshot.size()));
  std::memcpy(reinterpret_cast<void*>(p.destination), p.snapshot.data(), p.snapshot.size());
  *cq = {}; cq->status = URMA_CR_SUCCESS; cq->user_ctx = p.id + wrong_cq;
  live.pop_front(); return 1;
}
}
int main() {
  urma_context_t context{}; urma_jetty_t jetty{}; urma_target_jetty_t target{}; urma_jfc_t cq{};
  alignas(64) uint64_t mailbox[64]{};
  kvaf::peer::EnableGatherProgressRequest request{};
  request.lanes = 1; request.token_value = 3; request.segment_bytes = sizeof(urma_seg_t);
  request.mailbox_base = reinterpret_cast<uint64_t>(mailbox);
  urma_seg_t segment{}; segment.ubva.va = request.mailbox_base; segment.len = sizeof(mailbox);
  std::memcpy(request.segment, &segment, sizeof(segment));
  {
    GatherProgressTransport transport;
    assert(transport.Initialize(&context, request, 0, &jetty, &target, &cq).ok());
    uint64_t progress[64]{};
    delay = 10;
    for (uint32_t n = 1; n <= 9; ++n) {
      progress[63] = kvaf::GatherShardProgress(1, n);
      transport.Advance(1, progress, 64);
      assert(live.size() <= 2);
    }
    assert(transport.Finish() && live.empty() && mailbox[63] == progress[63]);
    std::memset(progress, 0, sizeof(progress));
    progress[0] = kvaf::GatherShardProgress(2, 1);
    transport.Advance(2, progress, 1);
    assert(transport.Finish() && mailbox[0] == progress[0]);
    assert(!transport.Shutdown().ok()); // initiator must be stopped first
    transport.Quiesced(); assert(transport.Shutdown().ok());
  }
  for (int failure = 0; failure < 3; ++failure) {
    GatherProgressTransport transport;
    fail_import = failure == 0; fail_post = failure == 1; wrong_cq = failure == 2;
    const auto initialized = transport.Initialize(&context, request, 0, &jetty, &target, &cq);
    assert(initialized.ok() == !fail_import);
    if (initialized.ok()) {
      uint64_t progress[1]{kvaf::GatherShardProgress(3, 1)};
      transport.Advance(3, progress, 1);
      assert(!transport.Finish());
    }
    assert(!transport.Shutdown().ok());
    live.clear(); // models successful initiator deletion, including partial posts
    transport.Quiesced(); assert(transport.Shutdown().ok());
  }
  assert(!registrations && !imports && !tokens);
  std::cout << "gather progress URMA transport: PASS\n";
}
