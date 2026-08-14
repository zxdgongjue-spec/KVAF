#include <cstdio>
#include <dlfcn.h>

#include "direct_hccp_session.h"

namespace {

using CounterFn = int (*)();
using ResetFn = void (*)();
using SetResultFn = void (*)(int);

bool Check(bool condition, const char *message) {
  if (condition) {
    return true;
  }
  std::fprintf(stderr, "direct_hccp_session_test: %s\n", message);
  return false;
}

template <typename Function>
Function Load(void *library, const char *name) {
  return reinterpret_cast<Function>(dlsym(library, name));
}

}  // namespace

int main() {
  void *ra_library = dlopen("libra.so", RTLD_NOW | RTLD_LOCAL);
  if (!Check(ra_library != nullptr, "failed to load fake libra.so")) {
    return 1;
  }
  void *runtime_library = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
  if (!Check(runtime_library != nullptr, "failed to load fake libascendcl.so")) {
    dlclose(ra_library);
    return 1;
  }

  const auto reset = Load<ResetFn>(ra_library, "KvafFakeRaReset");
  const auto set_device_init_result =
      Load<SetResultFn>(ra_library, "KvafFakeRaSetDeviceInitResult");
  const auto host_init = Load<CounterFn>(ra_library, "KvafFakeRaHostInitCalls");
  const auto device_init = Load<CounterFn>(ra_library, "KvafFakeRaDeviceInitCalls");
  const auto host_deinit = Load<CounterFn>(ra_library, "KvafFakeRaHostDeinitCalls");
  const auto device_deinit = Load<CounterFn>(ra_library, "KvafFakeRaDeviceDeinitCalls");
  const auto service_calls =
      Load<CounterFn>(runtime_library, "KvafFakeRuntimeOpenNetServiceCalls");
  if (!Check(reset != nullptr && set_device_init_result != nullptr &&
                 host_init != nullptr && device_init != nullptr && host_deinit != nullptr &&
                 device_deinit != nullptr && service_calls != nullptr,
             "failed to resolve fake-library test hooks")) {
    dlclose(runtime_library);
    dlclose(ra_library);
    return 1;
  }

  reset();
  kvaf::direct_hccp::DirectHccpSession *session = nullptr;
  if (!Check(kvaf::direct_hccp::AcquireDirectHccpSession(4U, 18U, &session),
             "AcquireDirectHccpSession rejected repeated device RaInit") ||
      !Check(session != nullptr, "AcquireDirectHccpSession returned a null session") ||
      !Check(host_init() == 1, "host RA was not initialized exactly once") ||
      !Check(device_init() == 1, "device RA was not attempted exactly once") ||
      !Check(service_calls() == 1, "network service was not opened exactly once")) {
    if (session != nullptr) {
      kvaf::direct_hccp::ReleaseDirectHccpSession(session);
    }
    dlclose(runtime_library);
    dlclose(ra_library);
    return 1;
  }

  kvaf::direct_hccp::ReleaseDirectHccpSession(session);
  const bool reused_ra_preserved =
      Check(host_deinit() == 1, "owned host RA was not deinitialized") &&
      Check(device_deinit() == 0, "borrowed device RA must not be deinitialized");
  if (!reused_ra_preserved) {
    dlclose(runtime_library);
    dlclose(ra_library);
    return 1;
  }

  reset();
  set_device_init_result(0);
  session = nullptr;
  if (!Check(kvaf::direct_hccp::AcquireDirectHccpSession(5U, 18U, &session),
             "AcquireDirectHccpSession rejected a newly owned device RA") ||
      !Check(session != nullptr, "owned device RA returned a null session")) {
    if (session != nullptr) {
      kvaf::direct_hccp::ReleaseDirectHccpSession(session);
    }
    dlclose(runtime_library);
    dlclose(ra_library);
    return 1;
  }
  kvaf::direct_hccp::ReleaseDirectHccpSession(session);
  const bool owned_ra_released =
      Check(device_init() == 1, "owned device RA was not initialized exactly once") &&
      Check(device_deinit() == 1, "owned device RA was not deinitialized");
  dlclose(runtime_library);
  dlclose(ra_library);
  return owned_ra_released ? 0 : 1;
}
