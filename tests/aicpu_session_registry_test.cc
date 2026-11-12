// Host-only test of the actual device-side registry. Requires URMA headers,
// but never loads liburma, creates a context, or accesses a device.
#include "transport_internal.h"
#include <future>
#include <unistd.h>
#include <iostream>
#include <stdexcept>
using namespace kvaf::aicpu_urma;
using namespace kvaf::aicpu_urma::detail;
#define CHECK(c) do { if (!(c)) throw std::runtime_error(#c); } while (false)
static urma_target_seg_t imported;
static bool fail_unimport;
static int imports, unimports, unregisters, token_frees;
static bool fail_token;
static urma_status_t Unregister(urma_target_seg_t*) { ++unregisters; return URMA_SUCCESS; }
static urma_status_t FreeToken(urma_token_id_t*) { ++token_frees; return fail_token ? URMA_FAIL : URMA_SUCCESS; }
static urma_target_seg_t* Import(urma_context_t*, urma_seg_t* segment, urma_token_t*, uint64_t, urma_import_seg_flag_t) {
  ++imports; imported.seg = *segment; return &imported;
}
static urma_status_t Unimport(urma_target_seg_t*) { ++unimports; return fail_unimport ? URMA_FAIL : URMA_SUCCESS; }
namespace kvaf::aicpu_urma::detail {
bool BuildTransfers(Queue&, bool, const UmdkTransfer*, std::uint32_t, std::uint32_t&,
                    std::uint64_t&, UmdkControlStatus&);
}
int main() {
  auto put = CreateSession(), get = CreateSession();
  const auto put_id = put->state.session, get_id = get->state.session;
  CHECK(put_id && get_id && put_id != get_id);
  CHECK(&put->state.data_queues[0] != &get->state.data_queues[0]);
  {
    SessionScope outer(*put);
    CurrentState().buffer = 123;
    { SessionScope inner(*get); CurrentState().buffer = 456; }
    CHECK(CurrentState().buffer == 123);
  }
  CHECK(active_session == nullptr && get->state.buffer == 456);
  std::unique_lock<std::shared_mutex> get_lock(get->mutex);
  auto independent = std::async(std::launch::async, [put] {
    std::unique_lock<std::shared_mutex> lock(put->mutex);
    SessionScope scope(*put);
    CHECK(CurrentState().buffer == 123);
    CurrentState().buffer = 789;
  });
  CHECK(independent.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
  independent.get(); get_lock.unlock();
  CHECK(put->state.buffer == 789);
  // Invalid create must remove its new entry without destroying either peer.
  UmdkControlStatus status{};
  UmdkControlArgs args{};
  args.op = UmdkControlOp::kPrepare;
  args.status = reinterpret_cast<std::uint64_t>(&status);
  CHECK(KvafAicpuUmdkControl(&args) == 0 && status.result && !status.session);
  CHECK(FindSession(put_id) == put && FindSession(get_id) == get);
  {
    SessionScope scope(*put);
    urma_target_seg_t remote{}, local{};
    remote.seg.ubva.va = 0x9000;
    constexpr std::uint64_t large_transfer = 4097ULL * 65536 + 13;
    remote.seg.len = large_transfer + 0x1000;
    auto* context = reinterpret_cast<urma_context_t*>(1);
    CHECK(CurrentState().remote.Import(context, Import, Unimport, remote.seg, 1, 0x1000, put_id) == 0);
    CHECK(CurrentState().remote.Import(context, Import, Unimport, remote.seg, 1, 0x1000, put_id) == 0);
    CHECK(imports == 1);
    CHECK(CurrentState().remote.Import(context, Import, Unimport, remote.seg, 2, 0x1000, put_id) != 0);
    CurrentState().local = &local;
    CurrentState().buffer = 0x3000;
    CurrentState().buffer_bytes = large_transfer + 0x1000;
    UmdkTransfer transfer{0x1040, 0x3080, 64};
    std::uint32_t index = 0;
    std::uint64_t offset = 0;
    auto& queue = CurrentState().data_queues[0];
    CHECK(BuildTransfers(queue, false, &transfer, 1, index, offset, status));
    CHECK(queue.requests.remote_sges[0].addr == 0x9040);
    // A single span crosses both 64 KiB fragments and the 4096-WR wave cap.
    // The second span must follow it without carrying over the first offset.
    UmdkTransfer fragments[] = {{0x1040, 0x3080, large_transfer}, {0x1080, 0x3000, 7}};
    index = 0; offset = 0;
    CHECK(BuildTransfers(queue, true, fragments, 2, index, offset, status));
    CHECK(index == 0 && offset == 4096ULL * 65536 && queue.requests.wrs.size() == 4096);
    for (unsigned i = 0; i < 4096; ++i) {
      CHECK(queue.requests.remote_sges[i].addr == 0x9040 + i * 65536ULL);
      CHECK(queue.requests.local_sges[i].addr == 0x3080 + i * 65536ULL);
      CHECK(queue.requests.remote_sges[i].len == 65536);
    }
    CHECK(queue.requests.wrs.back().next == nullptr && queue.requests.wrs.back().flag.bs.complete_enable);
    CHECK(BuildTransfers(queue, true, fragments, 2, index, offset, status));
    CHECK(index == 2 && offset == 0 && queue.requests.wrs.size() == 3);
    CHECK(queue.requests.remote_sges[0].addr == 0x9040 + 4096ULL * 65536);
    CHECK(queue.requests.remote_sges[1].addr == 0x9040 + 4097ULL * 65536);
    CHECK(queue.requests.remote_sges[1].len == 13);
    CHECK(queue.requests.remote_sges[2].addr == 0x9080 && queue.requests.remote_sges[2].len == 7);
    CHECK(queue.requests.wrs[0].rw.dst.sge == reinterpret_cast<urma_sge_t*>(&queue.requests.remote_sges[0]));
    index = 0;
    offset = 0;
    transfer.remote = 0xFFF;
    CHECK(!BuildTransfers(queue, false, &transfer, 1, index, offset, status));
    fail_unimport = true;
    CHECK(CurrentState().remote.Close() != 0 && CurrentState().remote.handle());
    fail_unimport = false;
    CHECK(CurrentState().remote.Close() == 0 && !CurrentState().remote.handle());
    CHECK(unimports == 2);
    CurrentState().local = nullptr;
    CurrentState().buffer = 789;
  }
  // Selective output teardown must not touch the session's initial workspace.
  urma_target_seg_t workspace{}, output_segment{};
  urma_token_id_t workspace_token{}, output_token{};
  get->state.local = &workspace; get->state.token = &workspace_token;
  get->state.buffer = 0x1000; get->state.buffer_bytes = 4096;
  get->state.buffers.push_back({0x3000, 128, &output_token, &output_segment});
  get->state.api.urma_unregister_seg = Unregister;
  get->state.api.urma_free_token_id = FreeToken;
  args.op = UmdkControlOp::kUnregisterBuffer; args.session = get_id;
  args.owner_pid = getpid(); args.local_buffer = 0x3000; args.buffer_bytes = 128;
  fail_token = true;
  CHECK(KvafAicpuUmdkControl(&args) == 0 && status.result);
  CHECK(unregisters == 1 && token_frees == 1 && get->state.buffers.size() == 1);
  CHECK(!get->state.buffers[0].segment && get->state.buffers[0].token == &output_token);
  fail_token = false;
  CHECK(KvafAicpuUmdkControl(&args) == 0 && !status.result);
  CHECK(unregisters == 1 && token_frees == 2 && get->state.buffers.empty());
  CHECK(get->state.local == &workspace && get->state.buffer == 0x1000);
  args.op = UmdkControlOp::kUnregisterBuffers;
  CHECK(KvafAicpuUmdkControl(&args) == 0 && !status.result);
  CHECK(unregisters == 2 && token_frees == 3 && !get->state.local && !get->state.token);
  RemoveSession(put_id);
  CHECK(!FindSession(put_id) && FindSession(get_id) == get);
  CHECK(put->state.buffer == 789); // in-flight ownership outlives index removal
  RemoveSession(get_id);
  std::cout << "Device session ownership, thread scope and isolation passed\n";
}
