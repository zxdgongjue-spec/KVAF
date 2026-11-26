// Exercise the real async UMDK post/wait storage with delayed provider reads.
#include "transport_internal.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>
#include <unistd.h>

using namespace kvaf::aicpu_urma;
using namespace kvaf::aicpu_urma::detail;
namespace {
auto session = std::make_shared<ExecutionSession>();
urma_context_t context{};
urma_target_seg_t local_segment{}, remote_segment{};
urma_jetty_t jetty{};
urma_target_jetty_t target{};
urma_jfc_t cq{};
bool reject_post{};
struct SavedWr {
  urma_jfs_wr_t* address;
  urma_jfs_wr_t wr;
  kvaf::urma::Sge source, destination;
  void Check() const {
    assert(!std::memcmp(address, &wr, sizeof(wr)));
    assert(!std::memcmp(wr.rw.src.sge, &source, sizeof(source)));
    assert(!std::memcmp(wr.rw.dst.sge, &destination, sizeof(destination)));
  }
};
std::deque<std::vector<SavedWr>> flights;
urma_target_seg_t* Import(urma_context_t*, urma_seg_t* segment, urma_token_t*,
                          uint64_t, urma_import_seg_flag_t) {
  remote_segment.seg = *segment; return &remote_segment;
}
urma_status_t Unimport(urma_target_seg_t*) { return URMA_SUCCESS; }
urma_status_t Post(urma_jetty_t* queue, urma_jfs_wr_t* wr, urma_jfs_wr_t**) {
  assert(queue == &jetty);
  std::vector<SavedWr> saved;
  for (; wr; wr = wr->next) {
    assert(wr->opcode == URMA_OPC_WRITE && wr->rw.src.num_sge == 1 && wr->rw.dst.num_sge == 1);
    saved.push_back({wr, *wr, *reinterpret_cast<kvaf::urma::Sge*>(wr->rw.src.sge),
                    *reinterpret_cast<kvaf::urma::Sge*>(wr->rw.dst.sge)});
  }
  assert(!saved.empty() && saved.back().wr.flag.bs.complete_enable);
  flights.push_back(std::move(saved));
  return reject_post ? URMA_FAIL : URMA_SUCCESS;
}
int Poll(urma_jfc_t* queue, int maximum, urma_cr_t* completion) {
  assert(queue == &cq && maximum == 1 && !flights.empty());
  for (const auto& saved : flights.front()) {
    saved.Check();
    std::memcpy(reinterpret_cast<void*>(saved.destination.addr),
                reinterpret_cast<void*>(saved.source.addr), saved.source.len);
  }
  *completion = {};
  completion->status = URMA_CR_SUCCESS;
  completion->user_ctx = flights.front().back().wr.user_ctx;
  flights.pop_front();
  return 1;
}
void CheckRetained() { for (const auto& flight : flights) for (const auto& wr : flight) wr.Check(); }
}  // namespace

namespace kvaf::aicpu_urma::detail {
thread_local ExecutionSession* active_session{};
std::shared_ptr<ExecutionSession> FindSession(uint64_t id) { return id == session->state.session ? session : nullptr; }
Queue* FindQueue(uint32_t index) { return index == kUmdkControlQueueIndex ? &CurrentState().control_queues[0] : nullptr; }
urma_target_seg_t* LocalSegment(uint64_t address, uint64_t bytes) {
  return UmdkRange(local_segment.seg.ubva.va, local_segment.seg.len, address, bytes) ? &local_segment : nullptr;
}
bool Fail(UmdkControlStatus& status, const char* stage, int result, const char*) {
  status.result = result; std::snprintf(status.stage, sizeof(status.stage), "%s", stage); return false;
}
}  // namespace kvaf::aicpu_urma::detail

int main() {
  std::vector<uint8_t> source(8 * 131072), destination(source.size());
  for (size_t i = 0; i < source.size(); ++i) source[i] = uint8_t(i * 7 + i / 131072);
  local_segment.seg.ubva.va = reinterpret_cast<uint64_t>(source.data());
  local_segment.seg.len = source.size();
  urma_seg_t remote{}; remote.ubva.va = reinterpret_cast<uint64_t>(destination.data()); remote.len = destination.size();
  auto& state = session->state; state.session = 7; state.worker_count = 1;
  assert(!state.remote.Import(&context, Import, Unimport, remote, 0, remote.ubva.va, state.session));
  state.api.urma_post_jetty_send_wr = Post; state.api.urma_poll_jfc = Poll;
  auto& q = state.control_queues[0]; q.jetty = &jetty; q.target = &target; q.jfc = &cq;
  UmdkContext args{state.session, uint64_t(getpid())};
  auto transfer = [&](uint32_t i) {
    return UmdkTransfer{remote.ubva.va + uint64_t(i) * 131072,
                       local_segment.seg.ubva.va + uint64_t(i) * 131072, 131072};
  };
  // Fill the window, retire its oldest slot and reuse that slot while the
  // other three still reference their original WR/SGE arrays.
  for (uint32_t i = 0; i < 4; ++i) { const auto t = transfer(i); assert(!UmdkTransferPost(&args, &t, 1, 0)); }
  auto t = transfer(4);
  assert(UmdkTransferPost(&args, &t, 1, 0) && q.control_count == 4);
  assert(UmdkTransferSync(&args, true, &t, 1, 0));
  CheckRetained();
  for (uint32_t i = 4; i < 8; ++i) {
    assert(!UmdkTransferWait(&args, 0));
    t = transfer(i); assert(!UmdkTransferPost(&args, &t, 1, 0)); CheckRetained();
  }
  while (q.control_count) assert(!UmdkTransferWait(&args, 0));
  assert(!q.control_pending && source == destination && flights.empty());
  assert(UmdkTransferWait(&args, 0));
  t = transfer(0); assert(!UmdkTransferSync(&args, true, &t, 1, 0));
  // A partially accepted post poisons the queue and retains all referenced
  // storage; a subsequent attempt must not rebuild or recycle that slot.
  reject_post = true;
  assert(UmdkTransferPost(&args, &t, 1, 0) && q.poisoned && q.control_count == 1);
  assert(UmdkTransferPost(&args, &t, 1, 0) && UmdkTransferWait(&args, 0));
  CheckRetained(); flights.clear();
  assert(!state.remote.Close());
  std::puts("AICPU native control-flight lifetime checks passed");
}
