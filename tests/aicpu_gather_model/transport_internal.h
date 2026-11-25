#pragma once
// Minimal transport boundary for the real gather_read.cc. No WR-building or
// window logic is copied into this model; the test supplies synchronous DMA.
#include "get/common/aicpu/include/batch_get_abi.h"
#include <array>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

struct urma_target_seg_t { struct { struct { std::uint64_t va{}; } ubva; } seg; };
struct urma_sge_t { std::uint64_t addr{}, len{}; urma_target_seg_t* segment{}; void* user{}; };
struct urma_jfs_wr_t {
  unsigned opcode{};
  struct { struct { unsigned comp_order{}, place_order{}, complete_enable{}; } bs; } flag;
  void* tjetty{};
  struct { struct { urma_sge_t* sge{}; unsigned num_sge{}; } src, dst; } rw;
  std::uint64_t user_ctx{};
  urma_jfs_wr_t* next{};
};
enum { URMA_OPC_READ, URMA_RELAX_ORDER, URMA_STRONG_ORDER };
namespace kvaf::aicpu_urma::detail {
using Sge = urma_sge_t;
struct RequestBuffer {
  std::vector<Sge> local_sges, remote_sges;
  std::vector<urma_jfs_wr_t> wrs;
  std::uint32_t wr_count{};
};
struct Queue {
  RequestBuffer requests;
  bool poisoned{};
  void* target{reinterpret_cast<void*>(1)};
  std::uint64_t sequence{}, completed{};
};
struct Remote {
  urma_target_seg_t segment;
  urma_target_seg_t* handle() { return &segment; }
  std::uint64_t canonical() const { return segment.seg.ubva.va; }
};
struct SessionState {
  Remote remote;
  std::uint64_t session{1};
  std::uint32_t worker_count{};
  UmdkQueueProfile queue_profile{UmdkQueueProfile::Gather};
  std::array<Queue, 8> data_queues;
};
struct ExecutionSession {
  SessionState state;
  std::shared_mutex mutex;
  std::array<std::mutex, 8> worker_mutexes;
};
extern thread_local ExecutionSession* active_session;
inline SessionState& CurrentState() { return active_session->state; }
struct SessionScope {
  ExecutionSession* previous;
  explicit SessionScope(ExecutionSession& s) : previous(active_session) { active_session = &s; }
  ~SessionScope() { active_session = previous; }
};
std::shared_ptr<ExecutionSession> FindSession(std::uint64_t);
std::uint64_t NowNs();
Queue& DataQueue(std::uint32_t);
bool Fail(UmdkControlStatus&, const char*, int result = -1, const char* detail = "");
bool PostBatch(Queue&, urma_jfs_wr_t*, urma_jfs_wr_t*, UmdkControlStatus&);
bool WaitCompletion(Queue&, std::uint64_t, std::uint64_t, std::uint32_t, UmdkControlStatus&);
urma_target_seg_t* LocalSegment(std::uint64_t, std::uint64_t);
bool TransferControl(bool, const UmdkTransfer*, std::uint32_t, UmdkControlStatus&, std::uint32_t = 0);
std::uint32_t TransferControlSync(bool, const UmdkTransfer*, std::uint32_t, std::uint32_t);
}
