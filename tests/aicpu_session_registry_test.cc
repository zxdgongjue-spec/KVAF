// Host-only test of the actual device-side registry. Requires URMA headers,
// but never loads liburma, creates a context, or accesses a device.
#include "transport_internal.h"
#include <future>
#include <iostream>
#include <stdexcept>
using namespace kvaf::aicpu_urma;
using namespace kvaf::aicpu_urma::detail;
#define CHECK(c) do { if (!(c)) throw std::runtime_error(#c); } while (false)
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
    remote.seg.len = 0x1000;
    CurrentState().remote = &remote;
    CurrentState().remote_address = 0x1000;
    CurrentState().local = &local;
    CurrentState().buffer = 0x3000;
    CurrentState().buffer_bytes = 0x1000;
    UmdkTransfer transfer{0x1040, 0x3080, 64};
    std::uint32_t index = 0;
    std::uint64_t offset = 0;
    auto& queue = CurrentState().data_queues[0];
    CHECK(BuildTransfers(queue, false, &transfer, 1, index, offset, status));
    CHECK(queue.requests.remote_sges[0].addr == 0x9040);
    index = 0;
    offset = 0;
    transfer.remote = 0xFFF;
    CHECK(!BuildTransfers(queue, false, &transfer, 1, index, offset, status));
    CurrentState().remote = nullptr;
    CurrentState().local = nullptr;
    CurrentState().buffer = 789;
  }
  RemoveSession(put_id);
  CHECK(!FindSession(put_id) && FindSession(get_id) == get);
  CHECK(put->state.buffer == 789); // in-flight ownership outlives index removal
  RemoveSession(get_id);
  std::cout << "Device session ownership, thread scope and isolation passed\n";
}
