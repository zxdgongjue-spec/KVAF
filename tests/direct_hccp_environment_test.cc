// Compile the production loader against a counted dynamic-loader surface.
// No RA bootstrap, device memory, context or queue is created by this test.
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "peer_link.h"
#include <acl/acl.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <future>
#include <iostream>
#include <vector>

#define CHECK(c) do { if (!(c)) { \
  std::cerr << "line " << __LINE__ << ": " #c "\n"; std::abort(); \
} } while (false)

namespace {
std::atomic<unsigned> opens{}, symbols{}, closes{}, invocations{};
bool fail_open = true, fail_symbol = false;
int library;
void* Open(const char* name, int flags) {
  ++opens;
  CHECK(flags == (RTLD_NOW | RTLD_LOCAL));
  if (fail_open) return nullptr;
  CHECK(std::strcmp(name, "libra.so") == 0);
  return &library;
}
int Deinit(void* context) { CHECK(context == &library); ++invocations; return 0; }
int Unused() { return 0; }
void* Symbol(void* handle, const char* name) {
  CHECK(handle == &library);
  ++symbols;
  if (fail_symbol && !std::strcmp(name, "RaCtxQpImport")) return nullptr;
  if (!std::strcmp(name, "RaCtxDeinit")) return reinterpret_cast<void*>(&Deinit);
  // This optional symbol is deliberately absent on the successful provider.
  if (!std::strcmp(name, "RaGetSecRandom")) return nullptr;
  return reinterpret_cast<void*>(&Unused);
}
int Close(void* handle) { CHECK(handle == &library); ++closes; return 0; }
char* Error() { static char message[] = "injected loader failure"; return message; }
}  // namespace

aclError aclrtGetDevice(std::int32_t*) { return 1; }
namespace kvaf::peer {
Status PeerLink::LookupRegion(std::uint64_t, PeerRegion*) const {
  std::abort();  // Loader tests must never reach a peer control-plane operation.
}
Status PeerLink::CreateChannel(std::uint64_t, const std::uint8_t*, std::uint32_t,
    const std::uint8_t*, std::uint32_t, std::uint64_t, std::uint32_t,
    std::array<std::uint8_t, kQpKeyBytes>*, std::uint32_t*, std::uint64_t*, std::uint32_t*) {
  std::abort();
}
}  // namespace kvaf::peer

#define dlopen Open
#define dlsym Symbol
#define dlclose Close
#define dlerror Error
#include "../access_operator/src/direct_hccp/direct_hccp_remote_transport.cc"
#undef dlopen
#undef dlsym
#undef dlclose
#undef dlerror

int main() {
  using namespace kvaf::direct_hccp;
  CHECK(!LoadDirectHccp(nullptr));
  CHECK(opens == 0 && symbols == 0 && closes == 0);
  DirectHccpApi failed;
  CHECK(!LoadDirectHccp(&failed) && !failed.library && !failed.ctx_init);
  CHECK(opens == 4 && symbols == 0 && closes == 0);
  fail_open = false;
  fail_symbol = true;
  CHECK(!LoadDirectHccp(&failed) && !failed.library && !failed.ctx_init);
  CHECK(opens == 5 && symbols > 10 && closes == 1);
  CloseDirectHccp(&failed);
  CHECK(closes == 1);
  fail_symbol = false;
  const auto symbols_after_failure = symbols.load();
  std::promise<void> release;
  const auto start = release.get_future().share();
  std::vector<std::future<DirectHccpApi>> futures;
  for (unsigned i = 0; i < 16; ++i)
    futures.emplace_back(std::async(std::launch::async, [start] {
      start.wait();
      DirectHccpApi api;
      CHECK(LoadDirectHccp(&api));
      return api;
    }));
  release.set_value();
  std::vector<DirectHccpApi> apis;
  for (auto& future : futures) apis.push_back(future.get());
  CHECK(opens == 6 && symbols == symbols_after_failure + 18 && closes == 1);
  for (const auto& api : apis) {
    CHECK(api.library == &library && api.ctx_deinit == Deinit && !api.get_random);
    CHECK(api.get_async_req_result && api.qp_import && api.lmem_register);
  }
  // Clearing one transport cannot unload or mutate another transport's table.
  CloseDirectHccp(&apis.front());
  CHECK(!apis.front().library && !apis.front().ctx_deinit);
  CHECK(apis.back().ctx_deinit(&library) == 0 && invocations == 1 && closes == 1);
  const auto successful_symbols = symbols.load();
  fail_symbol = true; fail_open = true; // No loader calls after publication.
  for (unsigned i = 0; i < 100; ++i) {
    DirectHccpApi api;
    CHECK(LoadDirectHccp(&api));
    CHECK(api.ctx_deinit(&library) == 0);
    CloseDirectHccp(&api);
  }
  for (auto& api : apis) CloseDirectHccp(&api);
  CHECK(opens == 6 && symbols == successful_symbols && closes == 1 && invocations == 101);
  std::cout << "Direct HCCP process API cache, concurrent load and failed publication retry passed\n";
}
