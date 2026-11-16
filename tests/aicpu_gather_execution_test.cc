// Execute the production Gather/transfer path against a synchronous URMA model.
// Rename only the registry entry point to count lookups without a production hook.
#include "transport_internal.h"
#define FindSession FindSessionUncounted
#include "../access_operator/transports/aicpu_urma/kernel/session.cc"
#undef FindSession
#include "gather_read.h"
#include "cpu_gather.h"

#include <deque>
#include <future>
#include <iostream>
#include <thread>

#define CHECK(c) do { if (!(c)) { std::cerr << "line " << __LINE__ << ": " #c "\n"; std::abort(); } } while (false)
using namespace kvaf;
using namespace kvaf::aicpu_urma;
using namespace kvaf::aicpu_urma::detail;
static std::atomic<unsigned> registry_lookups{0};
namespace kvaf::aicpu_urma::detail {
std::shared_ptr<ExecutionSession> FindSession(std::uint64_t id) {
  ++registry_lookups;
  return FindSessionUncounted(id);
}
}
namespace {
std::uint64_t Address(const void* p) { return reinterpret_cast<std::uint64_t>(p); }
urma_target_seg_t* Import(urma_context_t*, urma_seg_t* segment, urma_token_t*, uint64_t, urma_import_seg_flag_t) {
  auto* imported = new urma_target_seg_t{};
  imported->seg = *segment;
  return imported;
}
urma_status_t Unimport(urma_target_seg_t* segment) { delete segment; return URMA_SUCCESS; }
struct Fixture;
struct HardwareQueue {
  Fixture* fixture{};
  unsigned worker{};
  bool control{}, fail_post{}, bad_completion{};
  unsigned ack_reads{}, ready_reads{}, posts{};
  std::deque<std::uint64_t> completions;
};
urma_status_t Post(urma_jetty_t*, urma_jfs_wr_t*, urma_jfs_wr_t**);
int Poll(urma_jfc_t*, int, urma_cr_t*);
struct Fixture {
  std::shared_ptr<ExecutionSession> session = CreateSession();
  std::uint64_t canonical = 0x100000;
  unsigned rows, token, workers;
  std::vector<std::uint64_t> remote, workspace, addresses;
  std::vector<std::int32_t> indices, offsets, outputs;
  std::vector<std::uint8_t> output;
  std::vector<GatherReadStatus> statuses;
  std::vector<BatchGetWorkerProfile> profiles;
  std::array<HardwareQueue, kGatherMaxWorkers> controls{}, data{};
  urma_target_seg_t local{};
  std::int32_t active;
  UmdkBatchGetArgs launch{};
  GatherExecutionControl execution_control{};
  std::uint64_t expected_sequence{1};
  bool check_lock = false;
  std::uint64_t source_bytes{}, staging_offset{}, descriptor_offset{}, control_offset{};
  Fixture(unsigned n, unsigned bytes, unsigned w, bool pipeline, bool direct = false)
      : rows(n), token(bytes), workers(w), addresses(n), indices(n), offsets(n), outputs(n),
        output(n * bytes, 0xA5), statuses(w), profiles(w), active(n) {
    source_bytes = std::uint64_t(n) * (token + 8);
    staging_offset = (source_bytes + 63) & ~std::uint64_t{63};
    descriptor_offset = staging_offset + (w - 1) * GatherLaneStagingBytes(token, w) +
                        ((std::uint64_t(n) * token + 63) & ~std::uint64_t{63});
    control_offset = descriptor_offset + w * GatherLaneDescriptorBytes(w);
    remote.resize((control_offset + w * sizeof(peer::GatherControl)) / 8);
    workspace.resize(GatherWorkspaceBytes(w) / 8);
    for (unsigned i = 0; i < n; ++i) {
      for (unsigned j = 0; j < token + 8; ++j)
        reinterpret_cast<std::uint8_t*>(remote.data())[i * (token + 8) + j] = (i * 17 + j) % 251;
      addresses[i] = canonical + i * (token + 8);
      indices[i] = n - 1 - i;
      offsets[i] = (i % 2) * 8;
      outputs[i] = n - 1 - i;
    }
    if (direct)
      for (unsigned i = 0; i < n; ++i)
        addresses[i] = canonical + indices[i] * (token + 8) + offsets[i];
    auto& state = session->state;
    state.worker_count = w;
    state.api.urma_post_jetty_send_wr = Post;
    state.api.urma_poll_jfc = Poll;
    state.local = &local;
    state.buffer = Address(workspace.data()); state.buffer_bytes = workspace.size() * 8;
    state.buffers.push_back({Address(output.data()), output.size(), nullptr, &local});
    urma_seg_t segment{};
    segment.ubva.va = Address(remote.data()); segment.len = remote.size() * 8;
    CHECK(state.remote.Import(reinterpret_cast<urma_context_t*>(this), Import, Unimport,
                              segment, 1, canonical, state.session) == 0);
    for (unsigned i = 0; i < w; ++i) {
      for (bool control : {false, true}) {
        auto& hw = control ? controls[i] : data[i];
        hw.fixture = this; hw.worker = i; hw.control = control;
        auto& queue = control ? state.control_queues[i] : state.data_queues[i];
        queue.jetty = reinterpret_cast<urma_jetty_t*>(&hw);
        queue.jfc = reinterpret_cast<urma_jfc_t*>(&hw);
        queue.target = reinterpret_cast<urma_target_jetty_t*>(&hw);
      }
    }
    launch.request = {Address(addresses.data()), direct ? 0 : Address(indices.data()),
                      Address(offsets.data()), Address(&active), Address(outputs.data()), n, n, token};
    auto& a = launch.gather;
    a.context = {state.session, static_cast<std::uint64_t>(getpid())};
    a.core_status = Address(statuses.data()); a.profile = Address(profiles.data());
    a.output = Address(output.data()); a.output_extent = output.size();
    a.scratch = Address(workspace.data());
    a.source_descriptor = a.scratch + kGatherDescriptorOffset;
    a.remote_staging = canonical + staging_offset;
    a.remote_descriptor = canonical + descriptor_offset;
    a.remote_notify = canonical + control_offset;
    a.remote_ack = a.remote_notify + offsetof(peer::GatherControl, ack);
    a.remote_command = a.remote_notify + offsetof(peer::GatherControl, command);
    a.worker_count = w; a.execution_control = Address(&execution_control);
    a.pipeline_chunk_bytes = pipeline ? kGatherChunkBytes : 0;
  }
  ~Fixture() { CHECK(session->state.remote.Close() == 0); RemoveSession(session->state.session); }
  std::uint8_t* Remote(std::uint64_t address) {
    CHECK(address >= canonical && address - canonical < remote.size() * 8);
    return reinterpret_cast<std::uint8_t*>(remote.data()) + (address - canonical);
  }
  peer::GatherControl& Control(unsigned w) {
    return *reinterpret_cast<peer::GatherControl*>(Remote(canonical + control_offset + w * 64));
  }
  void Gather(unsigned w) {
    auto& control = Control(w);
    auto* header = reinterpret_cast<peer::GatherDescriptorHeader*>(Remote(control.command));
    CHECK(header->magic == peer::kGatherDescriptorMagic && control.notify == expected_sequence);
    auto* spans = reinterpret_cast<peer::GatherSpan*>(header + 1);
    auto* staging = Remote(header->staging_address);
    std::uint64_t bytes = 0;
    for (unsigned i = 0; i < header->span_count; ++i) {
      CHECK(spans[i].length == token);
      std::memcpy(staging + bytes, Remote(spans[i].address), spans[i].length);
      bytes += spans[i].length;
    }
    CHECK(bytes == header->total_bytes);
    header->ready_bytes = bytes;
    control.ack = control.notify;
  }
  void Run(bool success = true) {
    expected_sequence = execution_control.claimed / workers + 1;
    std::vector<std::future<unsigned>> executions;
    for (unsigned i = 0; i < workers; ++i)
      executions.push_back(std::async(std::launch::async, [&] {
        const auto result = ExecuteGatherRead(&launch);
        CHECK(active_session == nullptr); // execution scope always unwinds
        return result;
      }));
    for (auto& execution : executions) CHECK((execution.get() == 0) == success);
    if (!success) return;
    for (unsigned i = 0; i < static_cast<unsigned>(active); ++i)
      CHECK(std::memcmp(output.data() + outputs[i] * token,
                        Remote(canonical + indices[i] * (token + 8) + offsets[i]), token) == 0);
    for (unsigned w = 0; w < workers; ++w) {
      CHECK(statuses[w].magic == kGatherStatusMagic && statuses[w].result == 0);
      CHECK(statuses[w].ack_seen == expected_sequence);
      const auto spans = std::uint64_t(active) * (w + 1) / workers - std::uint64_t(active) * w / workers;
      CHECK(profiles[w].spans == spans);
      if (spans) {
        CHECK(controls[w].ack_reads >= 3);
        CHECK(!launch.gather.pipeline_chunk_bytes || controls[w].ready_reads >= 3);
        CHECK(data[w].posts > 0);
      }
    }
  }
};
urma_status_t Post(urma_jetty_t* jetty, urma_jfs_wr_t* first, urma_jfs_wr_t**) {
  auto& hw = *reinterpret_cast<HardwareQueue*>(jetty);
  auto& f = *hw.fixture;
  ++hw.posts;
  CHECK(active_session == f.session.get());
  if (f.check_lock && hw.posts == 1 && hw.control) {
    // A resource mutation on another thread must be excluded for the entire
    // execution, including the initial descriptor WRITE (before data READs).
    CHECK(!std::async(std::launch::async, [&] {
      std::unique_lock<std::shared_mutex> lock(f.session->mutex, std::try_to_lock);
      return lock.owns_lock();
    }).get());
  }
  if (hw.fail_post) return URMA_FAIL;
  for (auto* wr = first; wr; wr = wr->next) {
    const auto* src = reinterpret_cast<const Sge*>(wr->rw.src.sge);
    const auto* dst = reinterpret_cast<const Sge*>(wr->rw.dst.sge);
    CHECK(wr->rw.src.num_sge == 1);
    unsigned copied = 0;
    for (unsigned i = 0; i < wr->rw.dst.num_sge; ++i) {
      CHECK(copied + dst[i].len <= src->len);
      std::memcpy(reinterpret_cast<void*>(dst[i].addr), reinterpret_cast<const void*>(src->addr + copied), dst[i].len);
      copied += dst[i].len;
    }
    CHECK(copied == src->len);
    if (hw.control) {
      auto& control = f.Control(hw.worker);
      if (wr->opcode == URMA_OPC_WRITE && dst->addr == Address(&control.notify)) f.Gather(hw.worker);
      if (wr->opcode == URMA_OPC_READ) {
        const auto ready = Address(f.Remote(f.launch.gather.remote_descriptor +
             hw.worker * GatherLaneDescriptorBytes(f.workers))) + offsetof(peer::GatherDescriptorHeader, ready_bytes);
        if (src->addr == Address(&control.ack) && ++hw.ack_reads < 3)
          *reinterpret_cast<std::uint64_t*>(dst->addr) = 0;
        if (src->addr == ready && ++hw.ready_reads < 3)
          *reinterpret_cast<std::uint64_t*>(dst->addr) = 0;
      }
    }
    if (wr->flag.bs.complete_enable) hw.completions.push_back(wr->user_ctx);
  }
  return URMA_SUCCESS;
}
int Poll(urma_jfc_t* jfc, int count, urma_cr_t* cr) {
  auto& hw = *reinterpret_cast<HardwareQueue*>(jfc);
  CHECK(count == 1);
  if (hw.completions.empty()) return 0;
  cr->status = URMA_CR_SUCCESS;
  cr->user_ctx = hw.completions.front() + (hw.bad_completion ? 1 : 0);
  hw.completions.pop_front();
  return 1;
}
}  // namespace
int main() {
  for (bool pipeline : {false, true}) {
    for (bool direct : {false, true}) {
      Fixture f(8193, 16, 1, pipeline, direct);
      f.check_lock = true;
      f.launch.gather.sgl_size = 1; // exercise multiple 4096-WR waves
      registry_lookups = 0;
      f.Run();
      CHECK(registry_lookups == 1);
      f.Run();
      CHECK(registry_lookups == 2 && f.expected_sequence == 2);
    }
  }
  { // One token can cross a 64 KiB WR boundary.
    Fixture f(3, 70000, 1, true);
    registry_lookups = 0; f.Run(); CHECK(registry_lookups == 1);
  }
  { // Simultaneous sessions/workers must keep independent resource scopes.
    Fixture f(4097, 656, 5, true), g(3, 16, 8, false, true);
    registry_lookups = 0;
    auto first = std::async(std::launch::async, [&] { f.Run(); });
    g.Run(); first.get();
    CHECK(registry_lookups == 13);
  }
  { // A captured argument block remains unchanged while input data varies.
    Fixture f(9, 16, 5, true);
    f.Run();
    f.active = 0; f.Run();
    f.active = 3; f.Run();
    CHECK(f.execution_control.claimed == 15 && f.expected_sequence == 3);
    CHECK(f.launch.gather.sequence == 0); // device never patches captured Host arguments
  }
  for (bool post_error : {false, true}) {
    Fixture f(3, 16, 1, true);
    if (post_error) f.controls[0].fail_post = true;
    else f.data[0].bad_completion = true;
    registry_lookups = 0; f.Run(false); CHECK(registry_lookups == 1);
    CHECK(f.statuses[0].result == kGatherTransportFailed);
    CHECK(post_error ? f.session->state.control_queues[0].poisoned : f.session->state.data_queues[0].poisoned);
  }
  {
    Fixture f(3, 16, 1, true);
    f.launch.gather.context.owner_pid = 0;
    registry_lookups = 0; f.Run(false); CHECK(registry_lookups == 1);
    CHECK(f.controls[0].posts == 0 && f.data[0].posts == 0);
  }
  std::cout << "Gather single session lookup, address translation, waves, pipeline and failure handling passed\n";
}
