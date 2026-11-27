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
  std::array<std::thread, 8> cpu_tasks;
  std::atomic<bool> stop{};
  std::array<uint32_t, 8> pending_wrs{}, peak_wrs{}, cq_waits{};
  std::array<uint32_t, 64> queue_wrs{};
  std::array<uint64_t, 8> max_read_bytes{};
  GatherExecutionControl control{};
  UmdkBatchGetArgs args{};
  int32_t count{};
  uint64_t launches{}, fail_at{};
  Fixture(uint32_t n, uint32_t w, uint32_t c, uint32_t bytes = 7, uint32_t jettys = 1)
      : capacity(n), workers(w), window(c), token(bytes),
        local(GatherWorkspaceBytes(w, c) / 64), remote(GatherRemoteWorkspaceBytes(bytes, w, c) / 64),
        source(std::size_t(n) * bytes), output(source.size()), addresses(n), indices(n), offsets(n), rows(n),
        reports(w), profiles(w) {
    current = this;
    session->state.worker_count = w;
    session->state.data_jettys = jettys;
    for (std::size_t i = 0; i < session->state.data_queues.size(); ++i) {
      auto& q = session->state.data_queues[i];
      q.route = &session->state.route;
      q.target = reinterpret_cast<void*>(i + 1);
    }
    for (auto& q : session->state.control_queues) q.route = &session->state.route;
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
    g.pull_groups = kGatherChunkedPullEnabled;
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
                                     GatherLaneCapacity(workers, window), c, &stop) != 0);
    const auto success = cpu[worker]->Copy();
    peer::FinishGatherCopy(*c, c->notify, success, descriptor);
  }
  void Run(int32_t n, bool success = true, bool direct = false) {
    count = n; ++launches;
    pending_wrs.fill(0); peak_wrs.fill(0); cq_waits.fill(0);
    queue_wrs.fill(0); max_read_bytes.fill(0);
    args.request.indices = direct ? 0 : reinterpret_cast<uint64_t>(indices.data());
    std::fill(output.begin(), output.end(), 0xa5);
    std::vector<std::thread> tasks;
    for (uint32_t w = 0; w < workers; ++w) tasks.emplace_back([&] { (void)ExecuteGatherRead(&args); });
    for (auto& t : tasks) t.join();
    if (control.failed) stop.store(true);
    for (auto& t : cpu_tasks) if (t.joinable()) t.join();
    if (!success) { CHECK(control.failed); return; }
    CHECK(control.failed == 0 && control.claimed == launches * workers);
    uint64_t completed = 0, spans = 0;
    for (uint32_t w = 0; w < workers; ++w) {
      CHECK(reports[w].magic == kGatherStatusMagic && reports[w].sequence == launches && reports[w].result == 0);
      CHECK(reports[w].failed_request == UINT64_MAX);
      completed += reports[w].completed_requests; spans += profiles[w].spans;
      CHECK(pending_wrs[w] == 0 &&
            peak_wrs[w] <= ResolveGatherReadWaveWrs(args.gather.read_wave_wrs, uint32_t(n)));
    }
    CHECK(completed == uint32_t(n) && spans == uint32_t(n));
    for (const auto& q : session->state.data_queues) CHECK(q.completions.empty());
    for (uint32_t i = 0; i < capacity; ++i) for (uint32_t b = 0; b < token; ++b)
      CHECK(output[std::size_t(rows[i]) * token + b] ==
          (i < uint32_t(n) ? source[std::size_t(i) * token + b] : 0xa5));
  }
};
namespace kvaf::aicpu_urma::detail {
std::shared_ptr<ExecutionSession> FindSession(std::uint64_t id) { return id == 1 ? current->session : nullptr; }
std::uint64_t NowNs() { return BatchGetProfileNowNs(); }
Queue& DataQueue(std::uint32_t i) { return CurrentState().data_queues[i]; }
Queue* FindQueue(std::uint32_t i) {
  return i >= kUmdkControlQueueIndex ? &CurrentState().control_queues[i - kUmdkControlQueueIndex]
                                    : &DataQueue(i);
}
bool Fail(UmdkControlStatus& status, const char* stage, int result, const char*) {
  status.result = result; std::snprintf(status.stage, sizeof(status.stage), "%s", stage); return false;
}
urma_target_seg_t* LocalSegment(std::uint64_t address, std::uint64_t bytes) {
  static urma_target_seg_t segment;
  return peer::GatherRange(reinterpret_cast<uint64_t>(current->output.data()), current->output.size(), address, bytes)
      ? &segment : nullptr;
}
bool PostBatch(Queue& queue, urma_jfs_wr_t* first, urma_jfs_wr_t* tail, UmdkControlStatus&) {
  const auto ordinal = static_cast<std::size_t>(&queue - current->session->state.data_queues.data());
  CHECK(ordinal < current->workers * current->session->state.data_jettys);
  const auto worker = ordinal % current->workers;
  for (auto* wr = first;; wr = wr->next) {
    ++current->queue_wrs[ordinal];
    const auto live = ++current->pending_wrs[worker];
    current->peak_wrs[worker] = std::max(current->peak_wrs[worker], live);
    CHECK(live <= ResolveGatherReadWaveWrs(current->args.gather.read_wave_wrs,
                                         uint32_t(current->count)));
    CHECK(wr && wr->rw.src.num_sge == 1 && wr->rw.dst.num_sge <= 6);
    CHECK(wr->tjetty == queue.target);
    auto source = wr->rw.src.sge[0].addr;
    const auto bytes = wr->rw.src.sge[0].len;
    current->max_read_bytes[worker] = std::max<uint64_t>(current->max_read_bytes[worker], bytes);
    CHECK(peer::GatherRange(reinterpret_cast<uint64_t>(current->remote.data()), current->remote.size() * 64, source, bytes));
    uint64_t copied = 0;
    for (uint32_t i = 0; i < wr->rw.dst.num_sge; ++i) {
      const auto& dst = wr->rw.dst.sge[i]; CHECK(LocalSegment(dst.addr, dst.len));
      std::memcpy(reinterpret_cast<void*>(dst.addr), reinterpret_cast<void*>(source + copied), dst.len); copied += dst.len;
    }
    CHECK(copied == bytes); queue.completed = wr->user_ctx;
    // Every wave must end at a signaled, strongly ordered checkpoint.
    if (wr->flag.bs.complete_enable) {
      CHECK(wr == tail && wr->flag.bs.place_order == URMA_STRONG_ORDER);
      queue.completions.push_back(wr->user_ctx);
    }
    if (wr == tail) break;
  }
  return true;
}
bool WaitCompletion(Queue& q, std::uint64_t context, std::uint64_t, std::uint32_t, UmdkControlStatus&) {
  for (auto& control : current->session->state.control_queues) if (&q == &control) {
    CHECK(!q.completions.empty() && q.completions.front() == context);
    q.completions.pop_front();
    return true;
  }
  CHECK(q.completed == context);
  CHECK(!q.completions.empty() && q.completions.front() == context);
  q.completions.pop_front();
  const auto ordinal = static_cast<std::size_t>(&q - current->session->state.data_queues.data());
  const auto worker = ordinal % current->workers;
  CHECK(current->queue_wrs[ordinal] && current->pending_wrs[worker] >= current->queue_wrs[ordinal]);
  current->pending_wrs[worker] -= current->queue_wrs[ordinal];
  current->queue_wrs[ordinal] = 0;
  ++current->cq_waits[worker];
  return true;
}
std::uint32_t TransferControlSync(bool write, const UmdkTransfer* transfers, std::uint32_t count, std::uint32_t worker, urma_order_type_t) {
  bool notify = false;
  for (uint32_t i = 0; i < count; ++i) {
    const auto& t = transfers[i];
    CHECK(peer::GatherRange(reinterpret_cast<uint64_t>(current->remote.data()), current->remote.size() * 64, t.remote, t.bytes));
    if (write) {
      if (t.bytes == 8) __atomic_store_n(reinterpret_cast<uint64_t*>(t.remote),
          *reinterpret_cast<const uint64_t*>(t.local), __ATOMIC_RELEASE);
      else std::memcpy(reinterpret_cast<void*>(t.remote), reinterpret_cast<void*>(t.local), t.bytes);
      notify |= t.remote == current->args.gather.remote_notify + worker * sizeof(peer::GatherControl);
    } else if (t.bytes == 8) *reinterpret_cast<uint64_t*>(t.local) =
        __atomic_load_n(reinterpret_cast<const uint64_t*>(t.remote), __ATOMIC_ACQUIRE);
    else std::memcpy(reinterpret_cast<void*>(t.local), reinterpret_cast<void*>(t.remote), t.bytes);
  }
  if (notify) {
    auto& task = current->cpu_tasks[worker];
    if (task.joinable()) task.join();
    task = std::thread([worker] { current->Gather(worker); });
  }
  return 0;
}
bool TransferControl(bool write, const UmdkTransfer* t, std::uint32_t n, UmdkControlStatus&, std::uint32_t w, urma_order_type_t placement) {
  return TransferControlSync(write, t, n, w, placement) == 0;
}
static std::uint32_t ControlWorker(Queue& q) {
  return static_cast<std::uint32_t>(&q - CurrentState().control_queues.data());
}
bool PostOne(Queue& q, bool write, const UmdkTransfer& transfer, UmdkControlStatus&, bool) {
  q.single_wr.user_ctx = ++q.sequence;
  TransferControlSync(write, &transfer, 1, ControlWorker(q));
  q.completed = q.single_wr.user_ctx;
  q.completions.push_back(q.completed);
  std::this_thread::yield();
  return true;
}
bool PostOrderedWrites(Queue& q, const UmdkTransfer* transfers, std::uint32_t count, UmdkControlStatus&) {
  q.requests.wrs.resize(count);
  for (auto& wr : q.requests.wrs) wr.user_ctx = ++q.sequence;
  TransferControlSync(true, transfers, count, ControlWorker(q));
  q.completed = q.requests.wrs.back().user_ctx;
  q.completions.push_back(q.completed);
  return true;
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
  // Cross the former 6*token <= 64 KiB gate, including a token larger than
  // 64 KiB. Retain full-token SGEs, arbitrary output rows and Jetty tail CQEs.
  for (uint32_t token : {10922U, 10923U, 16384U, 65536U, 131072U}) {
    Fixture f(769, 1, 4096, token, 4);
    for (uint32_t wave : {31U, 64U, 256U}) {
      f.args.gather.read_wave_wrs = wave;
      f.control.address_groups_per_post = wave == 64 ? UINT32_MAX : 3;
      f.Run(769);
      CHECK(f.max_read_bytes[0] == uint64_t{6} * token);
      CHECK(f.profiles[0].read_wqes == 129);
    }
    f.Run(7, true, true);
    f.Run(0);
  }
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
  // Auto is resolved anew from the total device count on every replay, even
  // when queues retain storage sized for the previous, larger wave.
  { Fixture f(20001, 5, 65536, 656);
    for (int32_t n : {8192, 8193, 8192, 0, 1, 20001, 8192}) {
      f.Run(n);
      CHECK(f.args.gather.read_wave_wrs == 0);
      if (n == 8192) CHECK(f.peak_wrs[0] == 273 && f.cq_waits[0] == 1);
      if (n == 8193) CHECK(f.peak_wrs[0] == 64 && f.cq_waits[0] == 5);
      if (n == 20001) CHECK(f.peak_wrs[0] == 64 && f.cq_waits[0] == 11);
    }
    f.args.gather.read_wave_wrs = 256;
    f.Run(20001);
    CHECK(f.peak_wrs[0] == 256 && f.cq_waits[0] == 3);
    f.args.gather.read_wave_wrs = 64;
    f.Run(8192);
    CHECK(f.peak_wrs[0] == 64 && f.cq_waits[0] == 5);
  }
  { Fixture f(1000000, 5, 65536);
    f.Run(1000000); CHECK(f.peak_wrs[0] == 64);
    f.Run(8192); CHECK(f.peak_wrs[0] == 273);
    f.Run(1000000); CHECK(f.peak_wrs[0] == 64);
  }
  // Small memory windows do not select the small-total-count policy.
  { Fixture f(20000, 5, 4096);
    f.Run(20000); CHECK(f.peak_wrs[0] == 64);
    f.Run(8192); CHECK(f.peak_wrs[0] == 137);
  }
  { Fixture f(20000, 5, 4096); f.args.gather.read_wave_wrs = 32; f.Run(20000); }
  { Fixture f(1, 1, 4096);
    for (uint32_t invalid : {4097U, UINT32_MAX}) {
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
