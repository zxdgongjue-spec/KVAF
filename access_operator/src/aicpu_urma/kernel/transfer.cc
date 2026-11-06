#include "transport_internal.h"

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>

namespace kvaf::aicpu_urma::detail {


std::uint64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool PostBatch(Queue& worker, urma_jfs_wr_t* first, urma_jfs_wr_t* tail,
               UmdkControlStatus& status) {
  auto* saved_next = tail->next;
  tail->next = nullptr;
  urma_jfs_wr_t* bad = nullptr;
  errno = 0;
  const int result = CurrentState().api.urma_post_jetty_send_wr(worker.jetty, first, &bad);
  tail->next = saved_next;
  if (result == 0) return true;
  worker.poisoned = true;

  return Fail(status, "urma_post_jetty_send_wr", result);
}

bool WaitCompletion(Queue& worker, std::uint64_t context, std::uint64_t start,
                    std::uint32_t timeout_ms, UmdkControlStatus& status) {
  const auto deadline = start + static_cast<std::uint64_t>(timeout_ms) * 1000000;
  std::uint32_t polls = 0;
  auto now = start;
  while (now < deadline) {
    urma_cr_t completion{};
    const int count = CurrentState().api.urma_poll_jfc(worker.jfc, 1, &completion);
    if (count < 0) {
      Fail(status, "urma_poll_jfc", count);
      break;
    }
    if (count) {
      status.cq_status = completion.status;
      status.cq_context = completion.user_ctx;
      if (completion.status == URMA_CR_SUCCESS && completion.user_ctx == context) return true;
      Fail(status, "READ/WRITE CQE status or user_ctx mismatch");
      break;
    }
    if (++polls % 64U == 0U) now = NowNs();
  }
  worker.poisoned = true;
  if (!status.result) {
    errno = ETIMEDOUT;
    Fail(status, "waiting for CQE timed out", ETIMEDOUT);
  }
  return false;
}

bool SubmitQueue(Queue& worker, std::vector<urma_jfs_wr_t>& requests, std::uint32_t batch,
                 std::uint32_t timeout_ms, UmdkControlStatus& status) {
  const auto start = NowNs();
  for (std::size_t first = 0; first < requests.size(); first += batch) {
    const auto end = std::min(first + batch, requests.size());
    if (!PostBatch(worker, &requests[first], &requests[end - 1U], status)) return false;
  }
  return WaitCompletion(worker, requests.back().user_ctx, start, timeout_ms, status);
}

bool BuildTransfers(Queue& worker, bool write, const UmdkTransfer* transfers, std::uint32_t count,
                    std::uint32_t& transfer_index, std::uint64_t& offset,
                    UmdkControlStatus& status) {
  auto& requests = worker.requests;
  requests.local_sges.clear();
  requests.remote_sges.clear();
  requests.wrs.clear();
  while (transfer_index < count && requests.wrs.size() < 4096) {
    const auto& transfer = transfers[transfer_index];
    if (transfer.bytes == 0 || offset >= transfer.bytes ||
        !UmdkRange(CurrentState().remote->seg.ubva.va, CurrentState().remote->seg.len, transfer.remote,
                   transfer.bytes))
      return Fail(status, "UMDK transfer outside remote region");
    auto* segment = LocalSegment(transfer.local, transfer.bytes);
    if (segment == nullptr) return Fail(status, "UMDK transfer outside registered HBM");
    const auto bytes =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(65536, transfer.bytes - offset));
    requests.local_sges.push_back({transfer.local + offset, bytes, segment, nullptr});
    requests.remote_sges.push_back({transfer.remote + offset, bytes, CurrentState().remote, nullptr});
    urma_jfs_wr_t wr{};
    wr.opcode = write ? URMA_OPC_WRITE : URMA_OPC_READ;
    wr.flag.bs.comp_order = 1;
    wr.tjetty = worker.target;
    wr.user_ctx = ++worker.sequence;
    requests.wrs.push_back(wr);
    offset += bytes;
    if (offset == transfer.bytes) {
      ++transfer_index;
      offset = 0;
    }
  }
  for (std::size_t index = 0; index < requests.wrs.size(); ++index) {
    auto& wr = requests.wrs[index];
    const bool last = index + 1 == requests.wrs.size();
    wr.flag.bs.place_order = last ? URMA_STRONG_ORDER : URMA_RELAX_ORDER;
    wr.flag.bs.complete_enable = last;
    urma_sg_t local{reinterpret_cast<urma_sge_t*>(&requests.local_sges[index]), 1};
    urma_sg_t remote{reinterpret_cast<urma_sge_t*>(&requests.remote_sges[index]), 1};
    wr.rw.src = write ? local : remote;
    wr.rw.dst = write ? remote : local;
    wr.next = last ? nullptr : &requests.wrs[index + 1];
  }
  return true;
}

// Mailbox queries and short writes need one WR, not the segmented transfer builder.
// The caller holds the control lock and waits for the CQE before reusing these slots.
bool TransferOne(Queue& worker, bool write, const UmdkTransfer& transfer,
                 UmdkControlStatus& status) {
  if (!UmdkRange(CurrentState().remote->seg.ubva.va, CurrentState().remote->seg.len, transfer.remote,
                 transfer.bytes))
    return Fail(status, "UMDK transfer outside remote region");
  auto* segment = LocalSegment(transfer.local, transfer.bytes);
  if (!segment) return Fail(status, "UMDK transfer outside registered HBM");
  const auto bytes = static_cast<std::uint32_t>(transfer.bytes);
  worker.single_local = {transfer.local, bytes, segment, nullptr};
  worker.single_remote = {transfer.remote, bytes, CurrentState().remote, nullptr};
  urma_sg_t local{reinterpret_cast<urma_sge_t*>(&worker.single_local), 1};
  urma_sg_t remote{reinterpret_cast<urma_sge_t*>(&worker.single_remote), 1};
  auto& wr = worker.single_wr;
  wr = {};
  wr.opcode = write ? URMA_OPC_WRITE : URMA_OPC_READ;
  wr.flag.bs.comp_order = 1;
  wr.flag.bs.place_order = URMA_STRONG_ORDER;
  wr.flag.bs.complete_enable = 1;
  wr.tjetty = worker.target;
  wr.user_ctx = ++worker.sequence;
  wr.rw.src = write ? local : remote;
  wr.rw.dst = write ? remote : local;
  const auto start = NowNs();
  return PostBatch(worker, &wr, &wr, status) &&
         WaitCompletion(worker, wr.user_ctx, start, 1000, status);
}

bool TransferQueue(Queue& worker, bool write, const UmdkTransfer* transfers, std::uint32_t count,
                   UmdkControlStatus& status) {
  if (worker.poisoned || worker.target == nullptr || CurrentState().remote == nullptr ||
      transfers == nullptr || count == 0 || count > kUmdkMaxTransfers)
    return Fail(status, "invalid UMDK transfer queue");
  if (count == 1 && transfers[0].bytes > 0 && transfers[0].bytes <= 65536)
    return TransferOne(worker, write, transfers[0], status);
  std::uint32_t index = 0;
  std::uint64_t offset = 0;
  while (index < count) {
    if (!BuildTransfers(worker, write, transfers, count, index, offset, status)) return false;
    UmdkControlStatus submitted{};
    const bool success = SubmitQueue(worker, worker.requests.wrs, 8, 1000, submitted);
    if (!success) {
      status.cq_status = submitted.cq_status;
      status.cq_context = submitted.cq_context;
      errno = submitted.error_number;
      return Fail(status, submitted.stage, submitted.result);
    }
  }
  return true;
}

// The control Jetty has its own lock and never polls a data CQ.
bool TransferControl(bool write, const UmdkTransfer* transfers, std::uint32_t count,
                     UmdkControlStatus& status, std::uint32_t worker_index) {
  auto* control = FindQueue(kUmdkControlQueueIndex + worker_index);
  if (!control) return Fail(status, "control queue is unavailable");
  std::lock_guard<std::mutex> control_lock(active_session->control_mutexes[worker_index]);
  return TransferQueue(*control, write, transfers, count, status);
}

}  // namespace kvaf::aicpu_urma::detail

namespace kvaf::aicpu_urma {
using namespace detail;
std::uint32_t UmdkTransferSync(const UmdkContext* args, bool write, const UmdkTransfer* transfers,
                               std::uint32_t count, std::uint32_t worker_index) {
  if (!args) return 16;
  auto session = FindSession(args->session);
  if (!session) return 16;
  std::shared_lock<std::shared_mutex> lock(session->mutex);
  SessionScope scope(*session);
  if (args == nullptr || args->session != CurrentState().session ||
      args->owner_pid != static_cast<std::uint64_t>(getpid()) || worker_index >= CurrentState().worker_count)
    return 16;
  UmdkControlStatus status{};
  const bool success = TransferControl(write, transfers, count, status, worker_index);
  if (!success)
    std::fprintf(stderr,
                 "UMDK_CONTROL_ERROR worker=%u write=%u stage=%s result=%d errno=%d cq_status=%d\n",
                 worker_index, write ? 1U : 0U, status.stage, status.result, status.error_number,
                 status.cq_status);
  return success ? 0 : 16;
}

}  // namespace kvaf::aicpu_urma
