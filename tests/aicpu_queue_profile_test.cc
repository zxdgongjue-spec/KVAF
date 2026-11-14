#include "transport_internal.h"

#include <cstdlib>
#include <iostream>

using namespace kvaf::aicpu_urma;
using namespace kvaf::aicpu_urma::detail;
#define CHECK(c) do { if (!(c)) { std::cerr << "line " << __LINE__ << ": " #c "\n"; std::abort(); } } while (false)
namespace {
unsigned creates = 0, deletes = 0, live_cqs = 0, live_rqs = 0;
unsigned fail_create = 0;
bool fail_delete = false;
urma_jfc_t* CreateCq(urma_context_t*, urma_jfc_cfg_t*) { ++live_cqs; return new urma_jfc_t{}; }
urma_jfr_t* CreateRq(urma_context_t*, urma_jfr_cfg_t*) { ++live_rqs; return new urma_jfr_t{}; }
urma_jetty_t* CreateJetty(urma_context_t*, urma_jetty_cfg_t*) {
  ++creates;
  return creates == fail_create ? nullptr : new urma_jetty_t{};
}
urma_status_t DeleteJetty(urma_jetty_t* value) {
  if (fail_delete) return URMA_FAIL;
  ++deletes; delete value; return URMA_SUCCESS;
}
urma_status_t DeleteRq(urma_jfr_t* value) { --live_rqs; delete value; return URMA_SUCCESS; }
urma_status_t DeleteCq(urma_jfc_t* value) { --live_cqs; delete value; return URMA_SUCCESS; }
void BindQueueApi(SessionState& state) {
  state.api.urma_create_jfc = CreateCq;
  state.api.urma_create_jfr = CreateRq;
  state.api.urma_create_jetty = CreateJetty;
  state.api.urma_delete_jetty = DeleteJetty;
  state.api.urma_delete_jfr = DeleteRq;
  state.api.urma_delete_jfc = DeleteCq;
}
}
// Control-kernel dispatch is linked for legacy/V5 ABI validation; no transfer is
// issued by these queue construction tests.
namespace kvaf::aicpu_urma::detail {
bool TransferControl(bool, const UmdkTransfer*, std::uint32_t, UmdkControlStatus&, std::uint32_t) {
  return false;
}
}
extern "C" std::uint32_t KvafAicpuUmdkControlSessionsV5(UmdkControlArgs*);

int main() {
  for (auto profile : {UmdkQueueProfile::Put, UmdkQueueProfile::Gather}) {
    auto session = CreateSession(); const auto id = session->state.session;
    SessionScope scope(*session);
    BindQueueApi(CurrentState());
    UmdkControlArgs args{}; args.worker_count = 5; args.queue_profile = profile;
    UmdkControlStatus status{};
    const auto old_creates = creates, old_deletes = deletes;
    CHECK(CreateSessionQueues(args, status));
    const auto expected = profile == UmdkQueueProfile::Put ? 1U : 10U;
    CHECK(creates - old_creates == expected && live_cqs == expected && live_rqs == expected);
    CHECK(FindQueue(kUmdkControlQueueIndex) != nullptr);
    CHECK((FindQueue(0) == nullptr) == (profile == UmdkQueueProfile::Put));
    CHECK((FindQueue(kUmdkControlQueueIndex + 1) == nullptr) == (profile == UmdkQueueProfile::Put));
    CHECK(FindQueue(5) == nullptr && FindQueue(kUmdkControlQueueIndex + 5) == nullptr);
    fail_delete = true;
    CHECK(!Cleanup(status));
    CHECK(live_cqs == expected && deletes == old_deletes);
    fail_delete = false;
    CHECK(Cleanup(status));
    CHECK(deletes - old_deletes == expected && live_cqs == 0 && live_rqs == 0);
    RemoveSession(id);
  }
  // A partial queue must keep both its receive queue and CQ until cleanup.
  auto partial = CreateSession(); const auto partial_id = partial->state.session;
  {
    SessionScope scope(*partial); BindQueueApi(CurrentState());
    UmdkControlArgs args{}; args.worker_count = 3; args.queue_profile = UmdkQueueProfile::Gather;
    UmdkControlStatus status{}; fail_create = creates + 3;
    const auto old_deletes = deletes;
    CHECK(!CreateSessionQueues(args, status));
    CHECK(live_cqs == 3 && live_rqs == 3);
    CHECK(Cleanup(status));
    CHECK(live_cqs == 0 && live_rqs == 0 && deletes == old_deletes + 2);
  }
  RemoveSession(partial_id);
  UmdkControlArgs args{}; UmdkControlStatus status{};
  args.status = reinterpret_cast<std::uint64_t>(&status);
  args.op = UmdkControlOp::kPrepare;
  args.queue_profile = static_cast<UmdkQueueProfile>(0xFFFF);
  CHECK(KvafAicpuUmdkControl(&args) == 0 && status.result && !status.session);
  CHECK(args.queue_profile == UmdkQueueProfile::Gather); // Old ABI padding ignored.
  args.queue_profile = static_cast<UmdkQueueProfile>(0xFFFF);
  CHECK(KvafAicpuUmdkControlSessionsV5(&args) == 0 && status.result && !status.session);
  CHECK(args.queue_profile != UmdkQueueProfile::Gather); // V5 rejects invalid profile.
  std::cout << "AICPU Put/Gather queue layouts, partial cleanup and V5 ABI passed\n";
}
