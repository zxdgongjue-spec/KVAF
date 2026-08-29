#include "direct_hccp_session.h"

#include <chrono>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "direct_hccp_abi.h"

namespace kvaf::direct_hccp {

bool VerboseLoggingEnabled() {
  const char* value = std::getenv("KVAF_AIV_VERBOSE_DIRECT_HCCP");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

void VerboseLog(const char* format, ...) {
  if (!VerboseLoggingEnabled()) {
    return;
  }
  va_list arguments;
  va_start(arguments, format);
  std::vfprintf(stderr, format, arguments);
  va_end(arguments);
}

struct SessionKey {
  uint32_t physical_device_id = 0U;
  uint32_t hdc_type = 0U;

  bool operator==(const SessionKey &other) const {
    return physical_device_id == other.physical_device_id && hdc_type == other.hdc_type;
  }
};

struct DirectHccpSession {
  SessionKey key{};
  void *ra_library = nullptr;
  void *runtime_library = nullptr;
  RaInitFn ra_init = nullptr;
  RaDeinitFn ra_deinit = nullptr;
  RtOpenNetServiceFn rt_open_net_service = nullptr;
  std::string ra_library_name;
  uint32_t reference_count = 0U;
  bool host_ra_owned = false;
  bool device_ra_owned = false;
};

namespace {

constexpr uint32_t kPeerOnline = 0U;
constexpr uint32_t kOfflineHdc = 1U;
constexpr int kInitAgain = 128001;
// HCCL maps this RA return value to REPEAT_RAINIT_ERROR_CODE. It means a
// compatible RA instance already exists for this device in the process.
constexpr int kRepeatRaInit = 328002;
constexpr uint32_t kRetryMs = 10U;
constexpr uint32_t kTimeoutMs = 10000U;

struct SessionKeyHash {
  std::size_t operator()(const SessionKey &key) const {
    return (static_cast<std::size_t>(key.physical_device_id) << 32U) ^ key.hdc_type;
  }
};

std::mutex g_sessions_mutex;
std::unordered_map<SessionKey, std::unique_ptr<DirectHccpSession>, SessionKeyHash> g_sessions;

void CloseLibraries(DirectHccpSession *session) {
  if (session == nullptr) {
    return;
  }
  if (session->runtime_library != nullptr) {
    dlclose(session->runtime_library);
    session->runtime_library = nullptr;
  }
  if (session->ra_library != nullptr) {
    dlclose(session->ra_library);
    session->ra_library = nullptr;
  }
  session->ra_init = nullptr;
  session->ra_deinit = nullptr;
  session->rt_open_net_service = nullptr;
  session->ra_library_name.clear();
}

bool LoadSessionApi(DirectHccpSession *session) {
  if (session == nullptr) {
    return false;
  }
  for (const char *candidate : {"libra.so", "libra.so.0", "libhccp.so", "libhccp.so.0"}) {
    session->ra_library = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
    if (session->ra_library == nullptr) {
      continue;
    }
    session->ra_init = reinterpret_cast<RaInitFn>(dlsym(session->ra_library, "RaInit"));
    session->ra_deinit = reinterpret_cast<RaDeinitFn>(dlsym(session->ra_library, "RaDeinit"));
    if (session->ra_init != nullptr && session->ra_deinit != nullptr) {
      session->ra_library_name = candidate;
      break;
    }
    dlclose(session->ra_library);
    session->ra_library = nullptr;
    session->ra_init = nullptr;
    session->ra_deinit = nullptr;
  }
  if (session->ra_library == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_SESSION_API_FAILED reason=ra_symbols\n");
    return false;
  }

  session->rt_open_net_service =
      reinterpret_cast<RtOpenNetServiceFn>(dlsym(RTLD_DEFAULT, "rtOpenNetService"));
  if (session->rt_open_net_service == nullptr) {
    for (const char *candidate : {"libascendcl.so", "libruntime.so", "libruntime.so.1"}) {
      session->runtime_library = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
      if (session->runtime_library == nullptr) {
        continue;
      }
      session->rt_open_net_service =
          reinterpret_cast<RtOpenNetServiceFn>(dlsym(session->runtime_library, "rtOpenNetService"));
      if (session->rt_open_net_service != nullptr) {
        break;
      }
      dlclose(session->runtime_library);
      session->runtime_library = nullptr;
    }
  }
  if (session->rt_open_net_service == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_SESSION_API_FAILED reason=runtime_service\n");
    CloseLibraries(session);
    return false;
  }
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_HCCP_SESSION_API_READY ra_library=%s "
               "device_ra_policy=ra_init_ret_328002_reuse\n",
               session->ra_library_name.c_str());
  return true;
}

int CallRaInitWithRetry(DirectHccpSession *session, uint32_t mode,
                        const char *role) {
  RaInitConfig config{};
  config.phy_id = session->key.physical_device_id;
  config.nic_position = mode;
  config.hdc_type = static_cast<int32_t>(session->key.hdc_type);
  config.enable_hdc_async = true;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);
  int ret = 0;
  do {
    ret = session->ra_init(&config);
    if (ret == 0) {
      VerboseLog(
          "[KVAF_AIV] DIRECT_HCCP_SESSION_RA_READY role=%s physical=%u mode=%u hdc_type=%u\n",
          role, session->key.physical_device_id, mode, session->key.hdc_type);
      return 0;
    }
    if (ret != kInitAgain) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kRetryMs));
  } while (std::chrono::steady_clock::now() < deadline);
  return ret;
}

bool CallRaInit(DirectHccpSession *session, uint32_t mode, const char *role) {
  const int ret = CallRaInitWithRetry(session, mode, role);
  if (ret == 0) {
    return true;
  }
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_HCCP_SESSION_RA_INIT_FAILED role=%s physical=%u mode=%u ret=%d timeout_ms=%u\n",
               role, session->key.physical_device_id, mode, ret, kTimeoutMs);
  return false;
}

// vLLM/HCCL can initialize device-side PID/HDC RA before KVAF creates its
// Direct-HCCP transport. In that case RaInit returns 328002. Do not use
// RaIsFirstUsed to detect this: that API tracks a static reference count in the
// particular loaded RA DSO, so it does not see RA initialized through a
// different library image (the vLLM/HCCL integration case). RaInit's 328002 is
// the reliable compatibility signal used by HCCL itself. KVAF therefore owns
// and deinitializes RA only after a successful RaInit; on 328002 it borrows the
// existing device RA while retaining independent CTP/Jetty data-plane state.
bool EnsureDeviceRa(DirectHccpSession *session) {
  if (session == nullptr) {
    return false;
  }
  const int ret = CallRaInitWithRetry(session, kOfflineHdc, "device_aiv");
  if (ret == 0) {
    session->device_ra_owned = true;
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_HCCP_SESSION_RA_OWNED "
                 "role=device_aiv physical=%u hdc_type=%u\n",
                 session->key.physical_device_id, session->key.hdc_type);
    return true;
  }
  if (ret == kRepeatRaInit) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_HCCP_SESSION_RA_REUSED "
                 "role=device_aiv physical=%u hdc_type=%u "
                 "source=ra_init_ret_328002\n",
                 session->key.physical_device_id, session->key.hdc_type);
    session->device_ra_owned = false;
    return true;
  }
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_HCCP_SESSION_RA_INIT_FAILED "
               "role=device_aiv physical=%u mode=%u ret=%d timeout_ms=%u\n",
               session->key.physical_device_id, kOfflineHdc, ret, kTimeoutMs);
  return false;
}

void CallRaDeinitBestEffort(DirectHccpSession *session, uint32_t mode, const char *role) {
  if (session == nullptr || session->ra_deinit == nullptr) {
    return;
  }
  RaInitConfig config{};
  config.phy_id = session->key.physical_device_id;
  config.nic_position = mode;
  config.hdc_type = static_cast<int32_t>(session->key.hdc_type);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);
  int ret = 0;
  do {
    ret = session->ra_deinit(&config);
    if (ret == 0) {
      VerboseLog(
          "[KVAF_AIV] DIRECT_HCCP_SESSION_RA_RELEASED role=%s physical=%u mode=%u\n",
          role, session->key.physical_device_id, mode);
      return;
    }
    if (ret != kInitAgain) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kRetryMs));
  } while (std::chrono::steady_clock::now() < deadline);
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_HCCP_SESSION_RA_DEINIT_WARNING role=%s physical=%u mode=%u ret=%d\n",
               role, session->key.physical_device_id, mode, ret);
}

bool Bootstrap(DirectHccpSession *session) {
  if (!CallRaInit(session, kPeerOnline, "host_cpu")) {
    return false;
  }
  session->host_ra_owned = true;

  const std::string hdc_service_argument = "--hdcType=" + std::to_string(session->key.hdc_type);
  RtProcExtParam parameter{};
  parameter.param_info = hdc_service_argument.c_str();
  parameter.param_len = hdc_service_argument.size();
  RtNetServiceOpenArgs service{};
  service.ext_param_list = &parameter;
  service.ext_param_count = 1U;
  const int service_ret = session->rt_open_net_service(&service);
  if (service_ret != 0) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_HCCP_SESSION_SERVICE_FAILED physical=%u ret=%d hdc_type=%u\n",
                 session->key.physical_device_id, service_ret, session->key.hdc_type);
    CallRaDeinitBestEffort(session, kPeerOnline, "host_cpu");
    session->host_ra_owned = false;
    return false;
  }
  VerboseLog("[KVAF_AIV] DIRECT_HCCP_SESSION_SERVICE_READY physical=%u hdc_type=%u\n",
             session->key.physical_device_id, session->key.hdc_type);

  if (!EnsureDeviceRa(session)) {
    CallRaDeinitBestEffort(session, kPeerOnline, "host_cpu");
    session->host_ra_owned = false;
    return false;
  }
  return true;
}

void DestroySession(DirectHccpSession *session) {
  if (session == nullptr) {
    return;
  }
  if (session->device_ra_owned) {
    CallRaDeinitBestEffort(session, kOfflineHdc, "device_aiv");
  }
  if (session->host_ra_owned) {
    CallRaDeinitBestEffort(session, kPeerOnline, "host_cpu");
  }
  CloseLibraries(session);
}

}  // namespace

bool AcquireDirectHccpSession(uint32_t physical_device_id, uint32_t hdc_type,
                              DirectHccpSession **session) {
  if (session == nullptr || hdc_type == 0U) {
    return false;
  }
  *session = nullptr;
  const SessionKey key{physical_device_id, hdc_type};
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  const auto existing = g_sessions.find(key);
  if (existing != g_sessions.end()) {
    ++existing->second->reference_count;
    *session = existing->second.get();
    VerboseLog("[KVAF_AIV] DIRECT_HCCP_SESSION_REUSED physical=%u hdc_type=%u refs=%u\n",
               physical_device_id, hdc_type, existing->second->reference_count);
    return true;
  }

  auto created = std::make_unique<DirectHccpSession>();
  created->key = key;
  if (!LoadSessionApi(created.get()) || !Bootstrap(created.get())) {
    DestroySession(created.get());
    return false;
  }
  created->reference_count = 1U;
  *session = created.get();
  g_sessions.emplace(key, std::move(created));
  VerboseLog("[KVAF_AIV] DIRECT_HCCP_SESSION_READY physical=%u hdc_type=%u refs=1\n",
             physical_device_id, hdc_type);
  return true;
}

void ReleaseDirectHccpSession(DirectHccpSession *session) {
  if (session == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(g_sessions_mutex);
  const auto found = g_sessions.find(session->key);
  if (found == g_sessions.end() || found->second.get() != session || session->reference_count == 0U) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_SESSION_RELEASE_INVALID\n");
    return;
  }
  --session->reference_count;
  if (session->reference_count != 0U) {
    VerboseLog("[KVAF_AIV] DIRECT_HCCP_SESSION_RELEASE physical=%u hdc_type=%u refs=%u\n",
               session->key.physical_device_id, session->key.hdc_type, session->reference_count);
    return;
  }
  // `rtOpenNetService` has no symmetric close primitive.  CANN can keep
  // service/TLS callbacks alive after all Direct-HCCP CQ work has retired;
  // calling RaDeinit or dlclose here therefore leaves a use-after-unload
  // window at Python process exit.  Keep only this process-level bootstrap
  // resident.  Every transport still destroys its own RaCtx, MR, token, CQ,
  // Jetty and local-buffer resources before reaching this function.
  //
  // The map entry deliberately remains at refs=0 so the next NpuRuntime on
  // the same physical NPU reuses the compatible service without another
  // RaInit/rtOpenNetService round.  The OS reclaims the process-owned service
  // state at process exit.
  VerboseLog(
      "[KVAF_AIV] DIRECT_HCCP_SESSION_IDLE physical=%u hdc_type=%u "
      "refs=0 lifetime=process\n",
      session->key.physical_device_id, session->key.hdc_type);
}

}  // namespace kvaf::direct_hccp
