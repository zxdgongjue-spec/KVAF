// Exercise the production environment cache with a counted dynamic loader.
// No device/context API is invoked in this test.
#include "transport_internal.h"
#include <dlfcn.h>
#include <cstring>
#include <future>
#include <iostream>

#define CHECK(c) do { if (!(c)) { std::cerr << "line " << __LINE__ << ": " #c "\n"; std::abort(); } } while (false)
namespace {
std::atomic<unsigned> opens{0}, symbols{0}, closes{0}, initializations{0}, uninitializations{0};
bool fail_uninit = false, fail_close = false, fail_symbol = false, externally_initialized = false;
bool fail_context = false;
urma_status_t Init(urma_init_attr_t*) { ++initializations; return externally_initialized ? static_cast<urma_status_t>(EEXIST) : URMA_SUCCESS; }
urma_status_t Uninit() { ++uninitializations; return fail_uninit ? URMA_FAIL : URMA_SUCCESS; }
urma_status_t DeleteContext(urma_context_t*) { return fail_context ? URMA_FAIL : URMA_SUCCESS; }
int Unused() { return 0; }
void* Open(const char*, int) { ++opens; return reinterpret_cast<void*>(0x1234); }
void* Symbol(void*, const char* name) {
  ++symbols;
  if (!std::strcmp(name, "urma_init")) return reinterpret_cast<void*>(&Init);
  if (!std::strcmp(name, "urma_uninit")) return reinterpret_cast<void*>(&Uninit);
  if (fail_symbol) return nullptr;
  return reinterpret_cast<void*>(&Unused);
}
int Close(void*) { ++closes; return fail_close ? -1 : 0; }
char* Error() { static char message[] = "injected loader error"; return message; }
}
#define dlopen Open
#define dlsym Symbol
#define dlclose Close
#define dlerror Error
#include "../access_operator/transports/aicpu_urma/kernel/session.cc"
#undef dlopen
#undef dlsym
#undef dlclose
#undef dlerror
extern "C" std::uint32_t KvafAicpuUmdkControl(kvaf::aicpu_urma::UmdkControlArgs*) { return 0; }

int main() {
  using namespace kvaf::aicpu_urma;
  using namespace kvaf::aicpu_urma::detail;
  UmdkControlArgs args{};
  std::strcpy(args.library, "/fake/shared/liburma.so");
  auto first = CreateSession(), second = CreateSession();
  const auto first_id = first->state.session, second_id = second->state.session;
  std::promise<void> release;
  const auto start = release.get_future().share();
  auto acquire = [&](std::shared_ptr<ExecutionSession> session) {
    start.wait();
    SessionScope scope(*session);
    UmdkControlStatus status{};
    return LoadApi(args, status);
  };
  auto one = std::async(std::launch::async, acquire, first);
  auto two = std::async(std::launch::async, acquire, second);
  release.set_value();
  CHECK(one.get() && two.get());
  CHECK(opens == 1 && initializations == 1 && symbols > 20);
  CHECK(first->state.environment == second->state.environment);
  CHECK(first->state.environment->references == 2);
  const auto resolved = symbols.load();
  UmdkControlStatus status{};
  second->state.context = reinterpret_cast<urma_context_t*>(1);
  second->state.api.urma_delete_context = DeleteContext;
  fail_context = true;
  { SessionScope scope(*second); CHECK(!Cleanup(status)); }
  CHECK(second->state.environment->references == 2 && closes == 0);
  CHECK(second->state.context != nullptr); // Failed dependencies retain the API owner.
  { SessionScope scope(*first); CHECK(Cleanup(status)); }
  CHECK(!first->state.environment && second->state.environment->references == 1);
  CHECK(closes == 0 && uninitializations == 0);
  CHECK(second->state.api.urma_init == Init && symbols == resolved);
  // The final failed uninit retains the library, disallows reacquisition and
  // can be retried without unloading code still referenced by this session.
  fail_context = false;
  fail_uninit = true;
  { SessionScope scope(*second); CHECK(!Cleanup(status)); }
  CHECK(second->state.environment && second->state.environment->library);
  CHECK(second->state.environment->references == 1 && closes == 0);
  auto blocked = CreateSession();
  const auto blocked_id = blocked->state.session;
  { SessionScope scope(*blocked); CHECK(!LoadApi(args, status)); CHECK(Cleanup(status)); }
  CHECK(opens == 1 && initializations == 1 && symbols == resolved);
  fail_uninit = false;
  { SessionScope scope(*second); CHECK(Cleanup(status)); }
  CHECK(!second->state.environment && closes == 1 && uninitializations == 2);
  RemoveSession(first_id); RemoveSession(second_id); RemoveSession(blocked_id);
  // Distinct path spellings of one loaded object must share init and symbols.
  auto alias_one = CreateSession(), alias_two = CreateSession();
  const auto alias_one_id = alias_one->state.session, alias_two_id = alias_two->state.session;
  { SessionScope scope(*alias_one); CHECK(LoadApi(args, status)); }
  std::strcpy(args.library, "/fake/alias/liburma.so");
  { SessionScope scope(*alias_two); CHECK(LoadApi(args, status)); }
  CHECK(alias_one->state.environment == alias_two->state.environment);
  CHECK(initializations == 2 && symbols == resolved * 2 && closes == 2);
  { SessionScope scope(*alias_one); CHECK(Cleanup(status)); }
  fail_close = true;
  { SessionScope scope(*alias_two); CHECK(!Cleanup(status)); }
  CHECK(alias_two->state.environment && !alias_two->state.environment->initialized);
  const auto uninit_before_retry = uninitializations.load();
  fail_close = false;
  { SessionScope scope(*alias_two); CHECK(Cleanup(status)); }
  CHECK(uninitializations == uninit_before_retry);
  RemoveSession(alias_one_id); RemoveSession(alias_two_id);
  // A partial symbol lookup owns enough state for ordinary cleanup.
  auto partial = CreateSession(); const auto partial_id = partial->state.session;
  fail_symbol = true;
  { SessionScope scope(*partial); CHECK(!LoadApi(args, status)); }
  CHECK(partial->state.environment && partial->state.environment->library);
  const auto initializations_before_cleanup = initializations.load();
  { SessionScope scope(*partial); CHECK(Cleanup(status)); }
  CHECK(initializations == initializations_before_cleanup);
  RemoveSession(partial_id);
  fail_symbol = false;
  externally_initialized = true;
  auto borrowed = CreateSession(); const auto borrowed_id = borrowed->state.session;
  const auto prior_uninit = uninitializations.load();
  { SessionScope scope(*borrowed); CHECK(LoadApi(args, status)); CHECK(Cleanup(status)); }
  CHECK(uninitializations == prior_uninit); // Do not uninit an external provider owner.
  RemoveSession(borrowed_id);
  std::cout << "AICPU shared API environment, concurrent init, aliases and retryable cleanup passed\n";
}
