#include "transport_internal.h"
#include "cpu_gather_executor.h"
#include "get/gather/urma_aicpu/kernel/gather_read.h"
#include <unistd.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

using namespace kvaf;
using namespace kvaf::aicpu_urma;
namespace d = kvaf::aicpu_urma::detail;
#define CHECK(v) do { if (!(v)) { std::cerr << "line " << __LINE__ << ": " #v "\n"; std::abort(); } } while (false)

struct Fixture;
Fixture* current{};
namespace kvaf::aicpu_urma::detail {
thread_local ExecutionSession* active_session{};
}
struct Fixture {
  uint32_t capacity, workers, window, token;
  std::shared_ptr<d::ExecutionSession> session = std::make_shared<d::ExecutionSession>();
  std::vector<cpu_push::Word> local, remote;
  std::vector<uint8_t> source, output;
  std::vector<uint64_t> addresses;
  std::vector<int32_t> indices, offsets, rows;
  std::vector<GatherReadStatus> reports;
  std::vector<BatchGetWorkerProfile> profiles;
  std::vector<std::unique_ptr<peer::CpuGatherExecutor>> cpu;
  std::array<uint64_t, 8> last{}, calls{};
  std::array<uint32_t, 8> pending_wrs{}, peak_wrs{}, cq_waits{};
  GatherExecutionControl control{};
  UmdkBatchGetArgs args{};
  int32_t count{};
  uint64_t launches{}, fail_at{};
  Fixture(uint32_t n, uint32_t w, uint32_t c, uint32_t bytes = 7)
      : capacity(n), workers(w), window(c), token(bytes),
        local(GatherWorkspaceBytes(w, c) / 64), remote(GatherRemoteWorkspaceBytes(bytes, w, c) / 64),
        source(std::size_t(n) * bytes), output(source.size()), addresses(n), indices(n), offsets(n), rows(n),
        reports(w), profiles(w) {
    current = this;
    session->state.worker_count = w;
    session->state.remote.segment.seg.ubva.va = reinterpret_cast<uint64_t>(remote.data());
    for (uint32_t i = 0; i < n; ++i) {
      addresses[i] = reinterpret_cast<uint64_t>(source.data() + std::size_t(i) * token);
      indices[i] = i; rows[i] = n - 1 - i;
      for (uint32_t b = 0; b < token; ++b) source[std::size_t(i) * token + b] = (i * 19U + b) % 251;
    }
    for (uint32_t w = 0; w < workers; ++w)
      cpu.push_back(std::make_unique<peer::CpuGatherExecutor>(source.data(), reinterpret_cast<uint64_t>(source.data()),
          1, 1, peer::GatherWaitMode::kActive, remote.data(), reinterpret_cast<uint64_t>(remote.data())));
    auto& r = args.request;
    r.address_table = reinterpret_cast<uint64_t>(addresses.data()); r.indices = reinterpret_cast<uint64_t>(indices.data());
    r.offsets = reinterpret_cast<uint64_t>(offsets.data()); r.output_indices = reinterpret_cast<uint64_t>(rows.data());
    r.miss_count = reinterpret_cast<uint64_t>(&count); r.request_capacity = n; r.token_bytes = token;
    auto& g = args.gather;
    g.context = {1, uint64_t(getpid())}; g.worker_count = workers; g.window_capacity = window;
    g.output = reinterpret_cast<uint64_t>(output.data()); g.output_extent = output.size();
    g.core_status = reinterpret_cast<uint64_t>(reports.data()); g.profile = reinterpret_cast<uint64_t>(profiles.data());
    g.execution_control = reinterpret_cast<uint64_t>(&control); g.scratch = reinterpret_cast<uint64_t>(local.data());
    g.source_descriptor = g.scratch + kGatherDescriptorOffset;
    g.remote_staging = reinterpret_cast<uint64_t>(remote.data());
    g.remote_descriptor = g.remote_staging + GatherStagingBytes(token, workers, window);
    g.remote_notify = g.remote_descriptor + workers * GatherLaneDescriptorBytes(workers, window);
    g.remote_ack = g.remote_notify + 8; g.remote_command = g.remote_notify + 16;
  }
  void Gather(uint32_t worker) {
    auto* c = reinterpret_cast<peer::GatherControl*>(args.gather.remote_notify) + worker;
    CHECK(c->notify > last[worker]); last[worker] = c->notify; ++calls[worker];
    auto* descriptor = reinterpret_cast<peer::GatherDescriptorHeader*>(c->command);
    CHECK(descriptor->sequence == c->notify);
    CHECK(descriptor->span_count <= GatherLaneCapacity(workers, window));
    if (fail_at && worker == 0 && calls[worker] == fail_at) {
      descriptor->ready_bytes = peer::kGatherErrorBit;
      c->ack = c->notify | peer::kGatherErrorBit;
      return;
    }
    CHECK(cpu[worker]->PrepareDirect(remote.size() * 64, c->command, c->notify,
                                     GatherLaneCapacity(workers, window)) != 0);
    cpu[worker]->Copy();
    c->ack = c->notify;
  }
  void Run(int32_t n, bool success = true, bool direct = false) {
    count = n; ++launches;
    pending_wrs.fill(0); peak_wrs.fill(0); cq_waits.fill(0);
    args.request.indices = direct ? 0 : reinterpret_cast<uint64_t>(indices.data());
    std::fill(output.begin(), output.end(), 0xa5);
    std::vector<std::thread> tasks;
    for (uint32_t w = 0; w < workers; ++w) tasks.emplace_back([&] { (void)ExecuteGatherRead(&args); });
    for (auto& t : tasks) t.join();
    if (!success) { CHECK(control.failed); return; }
    CHECK(control.failed == 0 && control.claimed == launches * workers);
    uint64_t completed = 0, spans = 0;
    for (uint32_t w = 0; w < workers; ++w) {
      CHECK(reports[w].magic == kGatherStatusMagic && reports[w].sequence == launches && reports[w].result == 0);
      CHECK(reports[w].failed_request == UINT64_MAX);
      completed += reports[w].completed_requests; spans += profiles[w].spans;
      CHECK(pending_wrs[w] == 0 && peak_wrs[w] <= args.gather.read_wave_wrs);
    }
    CHECK(completed == uint32_t(n) && spans == uint32_t(n));
    for (uint32_t i = 0; i < capacity; ++i) for (uint32_t b = 0; b < token; ++b)
      CHECK(output[std::size_t(rows[i]) * token + b] ==
          (i < uint32_t(n) ? source[std::size_t(i) * token + b] : 0xa5));
  }
};
namespace kvaf::aicpu_urma::detail {
std::shared_ptr<ExecutionSession> FindSession(std::uint64_t id) { return id == 1 ? current->session : nullptr; }
std::uint64_t NowNs() { return BatchGetProfileNowNs(); }
Queue& DataQueue(std::uint32_t i) { return CurrentState().data_queues[i]; }
bool Fail(UmdkControlStatus& status, const char* stage, int result, const char*) {
  status.result = result; std::snprintf(status.stage, sizeof(status.stage), "%s", stage); return false;
}
urma_target_seg_t* LocalSegment(std::uint64_t address, std::uint64_t bytes) {
  static urma_target_seg_t segment;
  return peer::GatherRange(reinterpret_cast<uint64_t>(current->output.data()), current->output.size(), address, bytes)
      ? &segment : nullptr;
}
bool PostBatch(Queue& queue, urma_jfs_wr_t* first, urma_jfs_wr_t* tail, UmdkControlStatus&) {
  const auto worker = static_cast<std::size_t>(&queue - current->session->state.data_queues.data());
  CHECK(worker < current->workers);
  for (auto* wr = first;; wr = wr->next) {
    const auto live = ++current->pending_wrs[worker];
    current->peak_wrs[worker] = std::max(current->peak_wrs[worker], live);
    CHECK(live <= current->args.gather.read_wave_wrs);
    CHECK(wr && wr->rw.src.num_sge == 1 && wr->rw.dst.num_sge <= 6);
    auto source = wr->rw.src.sge[0].addr;
    const auto bytes = wr->rw.src.sge[0].len;
    CHECK(bytes <= 65536);
    CHECK(peer::GatherRange(reinterpret_cast<uint64_t>(current->remote.data()), current->remote.size() * 64, source, bytes));
    uint64_t copied = 0;
    for (uint32_t i = 0; i < wr->rw.dst.num_sge; ++i) {
      const auto& dst = wr->rw.dst.sge[i]; CHECK(LocalSegment(dst.addr, dst.len));
      std::memcpy(reinterpret_cast<void*>(dst.addr), reinterpret_cast<void*>(source + copied), dst.len); copied += dst.len;
    }
    CHECK(copied == bytes); queue.completed = wr->user_ctx;
    // Every wave must end at a signaled, strongly ordered checkpoint.
    if (wr->flag.bs.complete_enable) CHECK(wr == tail && wr->flag.bs.place_order == URMA_STRONG_ORDER);
    if (wr == tail) break;
  }
  return true;
}
bool WaitCompletion(Queue& q, std::uint64_t context, std::uint64_t, std::uint32_t, UmdkControlStatus&) {
  CHECK(q.completed == context);
  const auto worker = static_cast<std::size_t>(&q - current->session->state.data_queues.data());
  CHECK(current->pending_wrs[worker] && q.requests.wrs[q.requests.wr_count - 1].flag.bs.complete_enable);
  current->pending_wrs[worker] = 0;
  ++current->cq_waits[worker];
  return true;
}
std::uint32_t TransferControlSync(bool write, const UmdkTransfer* transfers, std::uint32_t count, std::uint32_t worker) {
  bool notify = false;
  for (uint32_t i = 0; i < count; ++i) {
    const auto& t = transfers[i];
    CHECK(peer::GatherRange(reinterpret_cast<uint64_t>(current->remote.data()), current->remote.size() * 64, t.remote, t.bytes));
    if (write) {
      std::memcpy(reinterpret_cast<void*>(t.remote), reinterpret_cast<void*>(t.local), t.bytes);
      notify |= t.remote == current->args.gather.remote_notify + worker * sizeof(peer::GatherControl);
    } else std::memcpy(reinterpret_cast<void*>(t.local), reinterpret_cast<void*>(t.remote), t.bytes);
  }
  if (notify) current->Gather(worker);
  return 0;
}
bool TransferControl(bool write, const UmdkTransfer* t, std::uint32_t n, UmdkControlStatus&, std::uint32_t w) {
  return TransferControlSync(write, t, n, w) == 0;
}
}

int main() {
  for (uint32_t workers : {1U, 5U, 8U}) for (uint32_t window : {4096U, 65536U, 131072U}) {
    Fixture f(270001, workers, window);
    for (int32_t n : {0, 1, 31, 32, 4095, 4096, 4097, 65537, 270001, 0, 1}) f.Run(n);
    f.Run(65537, true, true);
  }
  { Fixture f(1000000, 5, 4096); f.Run(1000000); }
  { Fixture f(8193, 5, 4096, 656); f.Run(8193); }
  { Fixture f(8193, 5, 4096, 4096); f.Run(8193); }
  // Replay the same retained transport buffers with smaller/larger wave limits.
  // Also exercise limits below/non-multiples of the 32-WR post batch.
  { Fixture f(20001, 5, 65536, 656);
    for (uint32_t limit : {4096U, 256U, 64U, 1U, 31U, 32U, 96U, 128U, 512U, 4096U}) {
      f.args.gather.read_wave_wrs = limit;
      f.Run(20001);
      CHECK(f.peak_wrs[0] == std::min(limit, 667U));
      CHECK(f.cq_waits[0] == (667U + limit - 1U) / limit);
      f.Run(8192, true, true);
      f.Run(0);
    }
  }
  { Fixture f(1000000, 5, 65536); f.args.gather.read_wave_wrs = 64; f.Run(1000000); }
  { Fixture f(20000, 5, 4096); f.args.gather.read_wave_wrs = 32; f.Run(20000); }
  { Fixture f(1, 1, 4096);
    for (uint32_t invalid : {0U, 4097U, UINT32_MAX}) {
      f.args.gather.read_wave_wrs = invalid;
      CHECK(ExecuteGatherRead(&f.args) != 0 && f.control.claimed == 0);
    }
  }
  { Fixture f(20000, 5, 4096); f.fail_at = 2; f.Run(20000, false);
    auto before = f.control.claimed; f.Run(1, false); CHECK(f.control.claimed == before); }
  { Fixture f(1, 5, 4096); f.Run(-1, false); }
  { Fixture f(1, 5, 4096); f.Run(2, false); }
  { Fixture f(5, 5, 4096); f.control.window_sequence[0] = peer::kGatherErrorBit - 1; f.Run(5, false); }
  std::cout << "AICPU Gather: single-launch windows, bounded workspaces, global outputs and replay sequences passed\n";
}
