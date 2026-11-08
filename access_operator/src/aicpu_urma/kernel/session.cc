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
#include <unordered_map>

namespace kvaf::aicpu_urma::detail {

thread_local ExecutionSession* active_session = nullptr;
namespace {
std::mutex registry_mutex, init_mutex;
std::uint64_t generation = 0;
std::unordered_map<std::uint64_t, std::shared_ptr<ExecutionSession>> sessions;
std::size_t init_references = 0;
bool owns_provider_init = false;
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
  if (index < CurrentState().worker_count) return &DataQueue(index);
  if (index >= kUmdkControlQueueIndex && index < kUmdkControlQueueIndex + CurrentState().worker_count)
    return &CurrentState().control_queues[index - kUmdkControlQueueIndex];
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

bool Cleanup(UmdkControlStatus& status) {
  for (std::uint32_t index = 0; index < CurrentState().worker_count; ++index)
    if (!CleanupQueue(CurrentState().data_queues[index], status)) return false;
  for (std::uint32_t index = 0; index < CurrentState().worker_count; ++index)
    if (!CleanupQueue(CurrentState().control_queues[index], status)) return false;
#define KVAF_RELEASE(member, function)                       \
  if (CurrentState().member != nullptr) {                             \
    errno = 0;                                               \
    const int result = CurrentState().api.function(CurrentState().member);     \
    if (result != 0) return Fail(status, #function, result); \
    CurrentState().member = nullptr;                                  \
  }
  KVAF_RELEASE(remote, urma_unimport_seg)
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
  if (CurrentState().owns_init) {
    std::lock_guard<std::mutex> lock(init_mutex);
    if (init_references == 1 && owns_provider_init) {
      const int result = CurrentState().api.urma_uninit();
      if (result != 0) return Fail(status, "urma_uninit", result);
      owns_provider_init = false;
    }
    --init_references;
    CurrentState().owns_init = false;
  }
  if (CurrentState().library != nullptr) dlclose(CurrentState().library);
  if (CurrentState().common_library != nullptr) dlclose(CurrentState().common_library);
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

bool LoadApi(const UmdkControlArgs& args, UmdkControlStatus& status) {
  Stage(status, "dlopen liburma (inside AICPU)");
  const auto resident = args.library[0] == '\0' ? ResidentUrmaPath() : std::string{};
  if (args.library[0] != '\0') {
    CurrentState().library = dlopen(args.library, RTLD_NOW | RTLD_GLOBAL);
  } else if (!resident.empty()) {
    CurrentState().library = dlopen(resident.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
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
        CurrentState().common_library = dlopen(common.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (CurrentState().common_library == nullptr)
          return Fail(status, "dlopen bundled liburma_common", -1, dlerror());
      }
      CurrentState().library = dlopen(core.c_str(), RTLD_NOW | RTLD_GLOBAL);
    } else {
      CurrentState().library = dlopen("liburma.so.0", RTLD_NOW | RTLD_GLOBAL);
      if (CurrentState().library == nullptr) CurrentState().library = dlopen("liburma.so", RTLD_NOW | RTLD_GLOBAL);
    }
  }
  if (CurrentState().library == nullptr) {
    const char* error = dlerror();
    return Fail(status, "dlopen liburma (inside AICPU)", -1, error == nullptr ? "" : error);
  }
#define KVAF_LOAD_API(name)                                                                 \
  CurrentState().api.name = reinterpret_cast<decltype(CurrentState().api.name)>(dlsym(CurrentState().library, #name)); \
  if (CurrentState().api.name == nullptr) return Fail(status, "dlsym " #name, -1, dlerror());
  KVAF_UMDK_APIS(KVAF_LOAD_API)
#undef KVAF_LOAD_API
  CurrentState().api.get_tp_list = reinterpret_cast<GetTpList>(dlsym(CurrentState().library, "urma_get_tp_list"));
  if (CurrentState().api.get_tp_list == nullptr) return Fail(status, "dlsym urma_get_tp_list");
  CurrentState().api.import_jetty_ex =
      reinterpret_cast<ImportJettyEx>(dlsym(CurrentState().library, "urma_import_jetty_ex"));
  if (CurrentState().api.import_jetty_ex == nullptr) return Fail(status, "dlsym urma_import_jetty_ex");

  return true;
}

bool CreateQueue(const UmdkControlArgs& args, UmdkControlStatus& status, Queue& worker) {
  const auto* error =
      kvaf::urma::CreateQueues(CurrentState().api, CurrentState().context, args.token, {64U, 64U, 6U, true},
                               worker.jfc, worker.jfr, worker.jetty);
  return error ? Fail(status, error) : true;
}

bool DescribeWorker(const UmdkControlArgs& args, std::uint32_t index, UmdkControlStatus& status) {
  auto* selected = FindQueue(index);
  if (selected == nullptr) return Fail(status, "invalid worker index");
  auto& worker = *selected;
  if (worker.jetty == nullptr) return Fail(status, "worker Jetty is unavailable");
  if (worker.tp == 0 && CurrentState().shared_tp != 0) worker.tp = CurrentState().shared_tp;
  if (worker.tp == 0) {
    Stage(status, "urma_get_tp_list");
    if (!kvaf::urma::QueryTp(CurrentState().api.get_tp_list, CurrentState().context, CurrentState().peer_eid.data(),
                             &worker.tp))
      return Fail(status, "urma_get_tp_list");
    CurrentState().shared_tp = worker.tp;
  }
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
  Stage(status, "urma_init");
  {
    std::lock_guard<std::mutex> lock(init_mutex);
    if (init_references == 0) {
      const int result = CurrentState().api.urma_init(nullptr);
      if (result != 0 && result != EEXIST && result != -EEXIST)
        return Fail(status, "urma_init", result);
      owns_provider_init = result == 0;
    }
    ++init_references;
    CurrentState().owns_init = true;  // one reference, not exclusive ownership
  }
  if (!CreateContext(args, status)) return false;
  std::copy_n(args.route_eid, 16, CurrentState().peer_eid.begin());
  CurrentState().worker_count = args.worker_count;
  // Preserve queue creation order while storing every worker uniformly.
  if (!CreateQueue(args, status, CurrentState().control_queues[0])) return false;
  for (std::uint32_t index = 0; index < CurrentState().worker_count; ++index) {
    if (!CreateQueue(args, status, CurrentState().data_queues[index])) return false;
    if (index != 0 && !CreateQueue(args, status, CurrentState().control_queues[index])) return false;
  }
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
  if (!DescribeWorker(args, 0, status)) return false;
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
  if (CurrentState().shared_peer_tp != 0 &&
      (args.peer_tp != CurrentState().shared_peer_tp || args.peer_psn != CurrentState().shared_peer_psn))
    return Fail(status, "shared TP peer binding changed", -1,
                "all Jettys must use the same peer TP and PSN");
  urma_seg_t segment{};
  std::memcpy(&segment, args.segment, sizeof(segment));
  if (!args.remote_address || segment.len > UINT64_MAX - args.remote_address ||
      segment.len > UINT64_MAX - segment.ubva.va)
    return Fail(status, "remote range outside segment");
  urma_token_t token{};
  token.token = args.token;
  urma_import_seg_flag_t flag{};
  flag.bs.cacheable = URMA_NON_CACHEABLE;
  flag.bs.access = (1U << 1U) | (1U << 2U);
  flag.bs.mapping = URMA_SEG_NOMAP;
  Stage(status, "urma_import_seg");
  if (CurrentState().remote == nullptr)
    CurrentState().remote = CurrentState().api.urma_import_seg(CurrentState().context, &segment, &token, 0, flag);
  if (CurrentState().remote == nullptr) return Fail(status, "urma_import_seg");
  Stage(status, "urma_import_jetty_ex");
  worker.target = kvaf::urma::ImportJetty(CurrentState().api.import_jetty_ex, CurrentState().context, args.peer_key,
                                          args.token, worker.tp, args.peer_tp, 1U, args.peer_psn);
  if (worker.target == nullptr) return Fail(status, "urma_import_jetty_ex");
  CurrentState().shared_peer_tp = args.peer_tp;
  CurrentState().shared_peer_psn = args.peer_psn;
  CurrentState().remote_address = args.remote_address;
  return DescribeWorker(args, args.worker_index, status);
}

urma_target_seg_t* LocalSegment(std::uint64_t address, std::uint64_t bytes) {
  if (CurrentState().local != nullptr && UmdkRange(CurrentState().buffer, CurrentState().buffer_bytes, address, bytes)) {
    return CurrentState().local;
  }
  for (const auto& buffer : CurrentState().buffers)
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
