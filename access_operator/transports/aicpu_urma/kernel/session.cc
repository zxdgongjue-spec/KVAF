#include "transport_internal.h"

#include <dlfcn.h>
#include <unistd.h>
#if defined(__linux__)
#include <link.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <limits.h>
#include <cstdlib>
#include <unordered_map>

namespace kvaf::aicpu_urma::detail {

thread_local ExecutionSession* active_session = nullptr;
namespace {
std::mutex registry_mutex, environment_mutex;
std::uint64_t generation = 0;
std::unordered_map<std::uint64_t, std::shared_ptr<ExecutionSession>> sessions;
std::unordered_map<std::string, std::weak_ptr<ApiEnvironment>> environments;
std::unordered_map<void*, std::weak_ptr<ApiEnvironment>> loaded_environments;
}  // namespace
std::shared_ptr<ExecutionSession> CreateSession() {
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (generation == UINT64_MAX) return {};
  auto session = std::make_shared<ExecutionSession>();
  session->state.session = ++generation;
  sessions.emplace(generation, session);
  return session;
}
std::shared_ptr<ExecutionSession> FindSession(std::uint64_t id) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto it = sessions.find(id);
  return it == sessions.end() ? nullptr : it->second;
}
void RemoveSession(std::uint64_t id) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  sessions.erase(id);
}

Queue& DataQueue(std::uint32_t index) {
  return CurrentState().data_queues[index];
}

Queue* FindQueue(std::uint32_t index) {
  auto& state = CurrentState();
  if (index < kUmdkControlQueueIndex || index >= 2 * kUmdkControlQueueIndex) {
    const auto ordinal = index < kUmdkControlQueueIndex ? index : index - kUmdkControlQueueIndex;
    return ordinal < state.data_queues.size() ? &DataQueue(ordinal) : nullptr;
  }
  if (index >= kUmdkControlQueueIndex &&
      index < kUmdkControlQueueIndex + UmdkControlQueueCount(state.queue_profile, state.worker_count))
    return &state.control_queues[index - kUmdkControlQueueIndex];
  return nullptr;
}

void Stage(UmdkControlStatus& status, const char* name) {
  std::snprintf(status.stage, sizeof(status.stage), "%s", name);
  errno = 0;
}

bool Fail(UmdkControlStatus& status, const char* stage, int result, const char* detail) {
  status.result = result == 0 ? -1 : result;
  status.error_number = errno;
  std::snprintf(status.stage, sizeof(status.stage), "%s", stage);
  std::snprintf(status.detail, sizeof(status.detail), "%s", detail == nullptr ? "" : detail);
  return false;
}

bool CleanupQueue(Queue& worker, UmdkControlStatus& status) {
#define KVAF_RELEASE_WORKER(member, function)                \
  if (worker.member != nullptr) {                            \
    errno = 0;                                               \
    const int result = CurrentState().api.function(worker.member);    \
    if (result != 0) return Fail(status, #function, result); \
    worker.member = nullptr;                                 \
  }
  KVAF_RELEASE_WORKER(jetty, urma_delete_jetty)
  KVAF_RELEASE_WORKER(target, urma_unimport_jetty)
  KVAF_RELEASE_WORKER(jfr, urma_delete_jfr)
  KVAF_RELEASE_WORKER(jfc, urma_delete_jfc)
#undef KVAF_RELEASE_WORKER
  return true;
}

bool UnregisterBuffers(UmdkControlStatus& status) {
  while (!CurrentState().buffers.empty()) {
    auto& buffer = CurrentState().buffers.back();
    if (buffer.segment) {
      if (CurrentState().api.urma_unregister_seg(buffer.segment) != 0) return Fail(status, "urma_unregister_seg");
      buffer.segment = nullptr;
    }
    if (CurrentState().api.urma_free_token_id(buffer.token) != 0) return Fail(status, "urma_free_token_id");
    CurrentState().buffers.pop_back();
  }
  if (CurrentState().local) {
    if (CurrentState().api.urma_unregister_seg(CurrentState().local) != 0) return Fail(status, "urma_unregister_seg");
    CurrentState().local = nullptr;
  }
  if (CurrentState().token) {
    if (CurrentState().api.urma_free_token_id(CurrentState().token) != 0) return Fail(status, "urma_free_token_id");
    CurrentState().token = nullptr;
  }
  CurrentState().buffer = CurrentState().buffer_bytes = 0;
  return true;
}
bool UnregisterBuffer(const UmdkControlArgs& args, UmdkControlStatus& status) {
  auto& s = CurrentState();
  for (auto it = s.buffers.begin(); it != s.buffers.end(); ++it) {
    if (it->base != args.local_buffer || it->bytes != args.buffer_bytes) continue;
    if (it->segment) {
      if (s.api.urma_unregister_seg(it->segment)) return Fail(status, "urma_unregister_seg");
      it->segment = nullptr;
    }
    if (it->token && s.api.urma_free_token_id(it->token)) return Fail(status, "urma_free_token_id");
    s.buffers.erase(it);
    return true;
  }
  if (s.buffer == args.local_buffer && s.buffer_bytes == args.buffer_bytes) {
    if (s.local) {
      if (s.api.urma_unregister_seg(s.local)) return Fail(status, "urma_unregister_seg");
      s.local = nullptr;
    }
    if (s.token && s.api.urma_free_token_id(s.token)) return Fail(status, "urma_free_token_id");
    s.token = nullptr;
    s.buffer = s.buffer_bytes = 0;
    return true;
  }
  return Fail(status, "buffer registration was not found");
}

bool ReleaseApiEnvironment(UmdkControlStatus& status) {
  auto& reference = CurrentState().environment;
  if (!reference) return true;
  const auto environment = reference;
  std::lock_guard<std::mutex> lock(environment->mutex);
  if (environment->references > 1) {
    --environment->references;
    reference.reset();
    return true;
  }
  // Failed final retirement remains attached to this session. Other sessions
  // cannot acquire a half-uninitialized provider while cleanup is retried.
  environment->retiring = true;
  if (environment->initialized && environment->owns_init) {
    const int result = environment->api.urma_uninit();
    if (result != 0) return Fail(status, "urma_uninit", result);
    environment->owns_init = false;
  }
  environment->initialized = false;
  if (environment->library) {
    if (dlclose(environment->library)) return Fail(status, "dlclose liburma", -1, dlerror());
    environment->library = nullptr;
  }
  if (environment->common_library) {
    if (dlclose(environment->common_library)) return Fail(status, "dlclose liburma_common", -1, dlerror());
    environment->common_library = nullptr;
  }
  environment->references = 0;
  reference.reset();
  return true;
}

bool Cleanup(UmdkControlStatus& status) {
  const auto profile = CurrentState().queue_profile;
  const auto workers = CurrentState().worker_count;
  for (auto& queue : CurrentState().data_queues)
    if (!CleanupQueue(queue, status)) return false;
  for (std::uint32_t index = 0; index < UmdkControlQueueCount(profile, workers); ++index)
    if (!CleanupQueue(CurrentState().control_queues[index], status)) return false;
#define KVAF_RELEASE(member, function)                       \
  if (CurrentState().member != nullptr) {                             \
    errno = 0;                                               \
    const int result = CurrentState().api.function(CurrentState().member);     \
    if (result != 0) return Fail(status, #function, result); \
    CurrentState().member = nullptr;                                  \
  }
  for (auto& entry : CurrentState().routes) {
    const auto result = entry.second.remote.Close();
    if (result != 0) return Fail(status, "urma_unimport_seg", result);
  }
  while (!CurrentState().buffers.empty()) {
    auto& buffer = CurrentState().buffers.back();
    if (buffer.segment != nullptr) {
      if (CurrentState().api.urma_unregister_seg(buffer.segment) != 0)
        return Fail(status, "urma_unregister_seg");
      buffer.segment = nullptr;
    }
    if (CurrentState().api.urma_free_token_id(buffer.token) != 0) return Fail(status, "urma_free_token_id");
    CurrentState().buffers.pop_back();
  }
  KVAF_RELEASE(local, urma_unregister_seg)
  KVAF_RELEASE(token, urma_free_token_id)
  KVAF_RELEASE(context, urma_delete_context)
#undef KVAF_RELEASE
  if (!ReleaseApiEnvironment(status)) return false;
  CurrentState() = {};
  return true;
}

std::string ResidentUrmaPath() {
  std::string path;
#if defined(__linux__)
  dl_iterate_phdr(
      [](struct dl_phdr_info* info, std::size_t, void* output) {
        if (info->dlpi_name == nullptr) return 0;
        const char* name = std::strrchr(info->dlpi_name, '/');
        name = name == nullptr ? info->dlpi_name : name + 1;
        if (std::strncmp(name, "liburma.so", 10) != 0 || (name[10] != '\0' && name[10] != '.') ||
            (std::strstr(info->dlpi_name, "/umdk/") != nullptr ||
             std::strstr(info->dlpi_name, "/umdk_runtime/") != nullptr))
          return 0;
        *static_cast<std::string*>(output) = info->dlpi_name;
        return 1;
      },
      &path);
#else
  auto* symbol = dlsym(RTLD_DEFAULT, "urma_init");
  Dl_info info{};
  if (symbol != nullptr && dladdr(symbol, &info) != 0 && info.dli_fname != nullptr &&
      std::strstr(info.dli_fname, "/umdk/") == nullptr &&
      std::strstr(info.dli_fname, "/umdk_runtime/") == nullptr)
    path = info.dli_fname;
#endif
  return path;
}

bool OpenApiLibraries(const UmdkControlArgs& args, UmdkControlStatus& status, ApiEnvironment& environment) {
  Stage(status, "dlopen liburma (inside AICPU)");
  const auto resident = args.library[0] == '\0' ? ResidentUrmaPath() : std::string{};
  if (args.library[0] != '\0') {
    environment.library = dlopen(args.library, RTLD_NOW | RTLD_GLOBAL);
  } else if (!resident.empty()) {
    environment.library = dlopen(resident.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  } else {
    Dl_info info{};
    std::string bundled;
    if (dladdr(reinterpret_cast<void*>(&KvafAicpuUmdkControl), &info) != 0 &&
        info.dli_fname != nullptr) {
      const std::string path = info.dli_fname;
      const auto slash = path.find_last_of('/');
      if (slash != std::string::npos) bundled = path.substr(0, slash) + "/umdk_runtime/";
    }
    auto core = bundled + "liburma.so.0";
    if (access(core.c_str(), R_OK) != 0) core = bundled + "liburma.so";
    if (!bundled.empty() && access(core.c_str(), R_OK) == 0) {
      auto common = bundled + "liburma_common.so.0";
      if (access(common.c_str(), R_OK) != 0) common = bundled + "liburma_common.so";
      if (access(common.c_str(), R_OK) == 0) {
        environment.common_library = dlopen(common.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (environment.common_library == nullptr)
          return Fail(status, "dlopen bundled liburma_common", -1, dlerror());
      }
      environment.library = dlopen(core.c_str(), RTLD_NOW | RTLD_GLOBAL);
    } else {
      environment.library = dlopen("liburma.so.0", RTLD_NOW | RTLD_GLOBAL);
      if (environment.library == nullptr) environment.library = dlopen("liburma.so", RTLD_NOW | RTLD_GLOBAL);
    }
  }
  if (environment.library == nullptr) {
    const char* error = dlerror();
    return Fail(status, "dlopen liburma (inside AICPU)", -1, error == nullptr ? "" : error);
  }
  return true;
}

bool ResolveApi(ApiEnvironment& environment, UmdkControlStatus& status) {
#define KVAF_LOAD_API(name)                                                                 \
  environment.api.name = reinterpret_cast<decltype(environment.api.name)>(dlsym(environment.library, #name)); \
  if (environment.api.name == nullptr) return Fail(status, "dlsym " #name, -1, dlerror());
  KVAF_UMDK_APIS(KVAF_LOAD_API)
#undef KVAF_LOAD_API
  environment.api.get_tp_list = reinterpret_cast<GetTpList>(dlsym(environment.library, "urma_get_tp_list"));
  if (environment.api.get_tp_list == nullptr) return Fail(status, "dlsym urma_get_tp_list");
  environment.api.import_jetty_ex =
      reinterpret_cast<ImportJettyEx>(dlsym(environment.library, "urma_import_jetty_ex"));
  if (environment.api.import_jetty_ex == nullptr) return Fail(status, "dlsym urma_import_jetty_ex");

  return true;
}

std::string EnvironmentKey(const UmdkControlArgs& args) {
  if (!args.library[0]) return "default";
  char path[PATH_MAX];
  return std::string("explicit:") + (realpath(args.library, path) ? path : args.library);
}

bool AttachApiEnvironment(std::shared_ptr<ApiEnvironment> environment,
                           const std::string& key, const UmdkControlArgs& args,
                           UmdkControlStatus& status) {
  std::unique_lock<std::mutex> lock(environment->mutex);
  if (environment->redirect) {
    const auto redirect = environment->redirect;
    lock.unlock();
    return AttachApiEnvironment(redirect, key, args, status);
  }
  if (environment->failed || environment->retiring)
    return Fail(status, "URMA environment awaits cleanup");
  if (!environment->loaded) {
    // The per-environment mutex admits one creator. The cache-index lock is
    // released before dynamic loading and provider initialization.
    const bool opened = OpenApiLibraries(args, status, *environment);
    if (opened) {
      std::shared_ptr<ApiEnvironment> existing;
      {
        std::lock_guard<std::mutex> cache_lock(environment_mutex);
        existing = loaded_environments[environment->library].lock();
        if (!existing) loaded_environments[environment->library] = environment;
      }
      if (existing && existing != environment) {
        // dlopen aliases (SONAME, absolute path, or symlink) resolve to the same
        // loaded object. Drop only these extra load references before sharing.
        if (dlclose(environment->library) == 0) {
          environment->library = nullptr;
          if (!environment->common_library || dlclose(environment->common_library) == 0) {
            environment->common_library = nullptr;
            environment->redirect = existing;
            { std::lock_guard<std::mutex> cache_lock(environment_mutex); environments[key] = existing; }
            lock.unlock();
            return AttachApiEnvironment(existing, key, args, status);
          }
        }
        environment->failed = true;
        ++environment->references;
        CurrentState().environment = environment;
        return Fail(status, "close duplicate URMA library reference", -1, dlerror());
      }
    }
    environment->loaded = true;
    ++environment->references;
    CurrentState().environment = environment;
    if (!opened || !ResolveApi(*environment, status)) {
      CurrentState().api = environment->api;
      environment->failed = true;
      return false;
    }
    Stage(status, "urma_init");
    const int result = environment->api.urma_init(nullptr);
    if (result != 0 && result != EEXIST && result != -EEXIST) {
      CurrentState().api = environment->api;
      environment->failed = true;
      return Fail(status, "urma_init", result);
    }
    environment->owns_init = result == 0;
    environment->initialized = true;
  } else {
    ++environment->references;
    CurrentState().environment = environment;
  }
  CurrentState().api = environment->api;
  return true;
}

bool LoadApi(const UmdkControlArgs& args, UmdkControlStatus& status) {
  const auto key = EnvironmentKey(args);
  std::shared_ptr<ApiEnvironment> environment;
  {
    std::lock_guard<std::mutex> lock(environment_mutex);
    environment = environments[key].lock();
    if (!environment) {
      environment = std::make_shared<ApiEnvironment>();
      environments[key] = environment;
    }
  }
  return AttachApiEnvironment(environment, key, args, status);
}

bool CreateQueue(const UmdkControlArgs& args, UmdkControlStatus& status, Queue& worker) {
  const auto* error =
      kvaf::urma::CreateQueues(CurrentState().api, CurrentState().context, args.token, {64U, 64U, 6U, true},
                               worker.jfc, worker.jfr, worker.jetty);
  return error ? Fail(status, error) : true;
}

bool CreateSessionQueues(const UmdkControlArgs& args, UmdkControlStatus& status) {
  auto& state = CurrentState();
  state.worker_count = args.worker_count;
  state.queue_profile = args.queue_profile;
  state.data_jettys = args.queue_profile == UmdkQueueProfile::GatherStriped ? args.data_jettys : 1;
  state.data_queues.resize(UmdkDataQueueCount(state.queue_profile, state.worker_count) * state.data_jettys);
  // Gather preserves its existing order; Put only posts through control[0].
  if (!CreateQueue(args, status, state.control_queues[0])) return false;
  for (std::uint32_t index = 0; index < state.data_queues.size(); ++index) {
    if (!CreateQueue(args, status, state.data_queues[index])) return false;
    if (index != 0 && index < state.worker_count && !CreateQueue(args, status, state.control_queues[index])) return false;
  }
  return true;
}

bool DescribeWorker(const UmdkControlArgs& args, std::uint32_t index, UmdkControlStatus& status) {
  auto* selected = FindQueue(index);
  if (selected == nullptr) return Fail(status, "invalid worker index");
  auto& worker = *selected;
  if (worker.jetty == nullptr) return Fail(status, "worker Jetty is unavailable");
  std::array<std::uint8_t, 16> eid{};
  std::copy_n(args.route_eid, eid.size(), eid.begin());
  auto& route = CurrentState().routes[eid];
  if (worker.route && worker.route != &route) return Fail(status, "Jetty peer EID changed");
  worker.route = &route;
  if (!route.tp) {
    Stage(status, "urma_get_tp_list");
    if (!kvaf::urma::QueryTp(CurrentState().api.get_tp_list, CurrentState().context, eid.data(), &route.tp))
      return Fail(status, "urma_get_tp_list");
  }
  worker.tp = route.tp;
  status.tp = worker.tp;
  std::memcpy(status.eid, CurrentState().context->eid.raw, 16);
  status.key_bytes = sizeof(urma_jetty_id_t) + sizeof(urma_transport_mode_t);
  std::memcpy(status.key, &worker.jetty->jetty_id, sizeof(urma_jetty_id_t));
  const urma_transport_mode_t mode = URMA_TM_RM;
  std::memcpy(status.key + sizeof(urma_jetty_id_t), &mode, sizeof(mode));
  return true;
}

static bool CreateContext(const UmdkControlArgs& args, UmdkControlStatus& status) {
  if (std::all_of(args.local_eid, args.local_eid + 16, [](auto b) { return b == 0; }))
    return Fail(status, "Host-selected EID is zero");
  bool matched = false;
  const bool enumerated = kvaf::urma::EnumerateEndpoints(
      CurrentState().api, "", UINT32_MAX, [&](auto* device, const auto& eid) {
        if (std::memcmp(args.local_eid, eid.eid.raw, 16) != 0) return false;
        matched = true;
        CurrentState().context = CurrentState().api.urma_create_context(device, eid.eid_index);
        if (!CurrentState().context) {
          Fail(status, "urma_create_context");
          // The same EID can be exposed at another device/index.
          return false;
        }
        status.result = status.error_number = 0;
        status.detail[0] = '\0';
        std::memcpy(status.eid, CurrentState().context->eid.raw, 16);
        std::fprintf(stderr, "[KVAF_AICPU] UMDK_EID_SELECTED source=host device=%s eid_index=%u eid=",
                     device->name, eid.eid_index);
        for (auto byte : status.eid) std::fprintf(stderr, "%02x", byte);
        std::fprintf(stderr, "\n");
        return true;
      });
  if (!enumerated) return Fail(status, "urma_get_device_list");
  if (CurrentState().context) return true;
  if (matched) return false;
  char expected[33]{};
  for (unsigned i = 0; i < 16; ++i) std::snprintf(expected + i * 2, 3, "%02x", args.local_eid[i]);
  errno = ENODEV;
  return Fail(status, "Host EID not exposed by UMDK", -1, expected);
}

bool Prepare(const UmdkControlArgs& args, UmdkControlStatus& status) {
  status.session = CurrentState().session;

  if (!LoadApi(args, status)) return false;
  if (!CreateContext(args, status)) return false;
  if (!CreateSessionQueues(args, status)) return false;
  Stage(status, "urma_alloc_token_id");
  CurrentState().token = CurrentState().api.urma_alloc_token_id(CurrentState().context);
  if (CurrentState().token == nullptr) return Fail(status, "urma_alloc_token_id");
  urma_seg_cfg_t cfg{};
  cfg.va = args.local_buffer & ~std::uint64_t{4095};
  cfg.len = args.buffer_bytes + args.local_buffer - cfg.va;
  cfg.token_id = CurrentState().token;
  cfg.token_value.token = args.token;
  cfg.flag.value = MemRegFlags(true);

  Stage(status, "urma_register_seg (local HBM)");
  CurrentState().local = CurrentState().api.urma_register_seg(CurrentState().context, &cfg);
  if (CurrentState().local == nullptr) {
    Fail(status, "urma_register_seg (local HBM)");

    return false;
  }
  CurrentState().buffer = args.local_buffer;
  CurrentState().buffer_bytes = args.buffer_bytes;
  if (!DescribeWorker(args, kUmdkControlQueueIndex, status)) return false;
  if (args.queue_profile != UmdkQueueProfile::Put && !DescribeWorker(args, 0, status)) return false;
  return true;
}

bool Connect(const UmdkControlArgs& args, UmdkControlStatus& status) {
  auto* selected = FindQueue(args.worker_index);
  if (selected == nullptr) return Fail(status, "invalid worker index");
  auto& worker = *selected;
  if (CurrentState().local == nullptr || worker.target != nullptr || worker.tp == 0)
    return Fail(status, "invalid connect CurrentState()");
  if (args.segment_bytes < sizeof(urma_seg_t) || args.key_bytes != 28 || args.peer_tp == 0)
    return Fail(status, "invalid peer descriptor");
  if (!worker.route) return Fail(status, "queue has no peer route");
  auto& route = *worker.route;
  if (route.peer_tp != 0 &&
      (args.peer_tp != route.peer_tp || args.peer_psn != route.peer_psn))
    return Fail(status, "shared TP peer binding changed", -1,
                "all Jettys must use the same peer TP and PSN");
  urma_seg_t segment{};
  std::memcpy(&segment, args.segment, sizeof(segment));
  if (!args.remote_address || segment.len > UINT64_MAX - args.remote_address ||
      segment.len > UINT64_MAX - segment.ubva.va)
    return Fail(status, "remote range outside segment");
  Stage(status, "urma_import_seg");
  if (route.remote.Import(CurrentState().context, CurrentState().api.urma_import_seg,
          CurrentState().api.urma_unimport_seg, segment, args.token, args.remote_address,
          CurrentState().session) != 0) return Fail(status, "urma_import_seg");
  Stage(status, "urma_import_jetty_ex");
  worker.target = kvaf::urma::ImportJetty(CurrentState().api.import_jetty_ex, CurrentState().context, args.peer_key,
                                          args.token, worker.tp, args.peer_tp, 1U, args.peer_psn);
  if (worker.target == nullptr) return Fail(status, "urma_import_jetty_ex");
  route.peer_tp = args.peer_tp;
  route.peer_psn = args.peer_psn;
  return DescribeWorker(args, args.worker_index, status);
}

urma_target_seg_t* LocalSegment(std::uint64_t address, std::uint64_t bytes) {
  auto& state = CurrentState();
  if (state.local != nullptr && UmdkRange(state.buffer, state.buffer_bytes, address, bytes)) {
    return state.local;
  }
  for (const auto& buffer : state.buffers)
    if (UmdkRange(buffer.base, buffer.bytes, address, bytes)) {
      return buffer.segment;
    }
  return nullptr;
}

bool RegisterBuffer(const UmdkControlArgs& args, UmdkControlStatus& status) {
  if (args.local_buffer == 0 || args.buffer_bytes == 0 ||
      args.local_buffer > UINT64_MAX - args.buffer_bytes)
    return Fail(status, "invalid UMDK HBM registration");
  if (LocalSegment(args.local_buffer, args.buffer_bytes) != nullptr) return true;
  Stage(status, "urma_alloc_token_id");
  auto* token = CurrentState().api.urma_alloc_token_id(CurrentState().context);
  if (token == nullptr) return Fail(status, "urma_alloc_token_id");
  urma_seg_cfg_t cfg{};
  cfg.va = args.local_buffer & ~std::uint64_t{4095};
  cfg.len = args.buffer_bytes + args.local_buffer - cfg.va;
  cfg.token_id = token;
  cfg.token_value.token = args.token;
  cfg.flag.value = MemRegFlags(true);
  Stage(status, "urma_register_seg (local HBM)");
  auto* segment = CurrentState().api.urma_register_seg(CurrentState().context, &cfg);
  if (segment == nullptr) {
    Fail(status, "urma_register_seg (local HBM)");
    (void)CurrentState().api.urma_free_token_id(token);
    return false;
  }
  CurrentState().buffers.push_back({args.local_buffer, args.buffer_bytes, token, segment});
  return true;
}

}  // namespace kvaf::aicpu_urma::detail
