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

namespace kvaf::aicpu_urma::detail {

SessionState state;
std::shared_mutex state_mutex;
std::array<std::mutex, 8> worker_mutexes;
namespace {
std::uint64_t generation = 0;
}  // namespace

Queue& DataQueue(std::uint32_t index) {
  return state.data_queues[index];
}

Queue* FindQueue(std::uint32_t index) {
  if (index < state.worker_count) return &DataQueue(index);
  if (index >= kUmdkControlQueueIndex && index < kUmdkControlQueueIndex + state.worker_count)
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
    const int result = state.api.function(worker.member);    \
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

bool Cleanup(UmdkControlStatus& status) {
  for (std::uint32_t index = 0; index < state.worker_count; ++index)
    if (!CleanupQueue(state.data_queues[index], status)) return false;
  for (std::uint32_t index = 0; index < state.worker_count; ++index)
    if (!CleanupQueue(state.control_queues[index], status)) return false;
#define KVAF_RELEASE(member, function)                       \
  if (state.member != nullptr) {                             \
    errno = 0;                                               \
    const int result = state.api.function(state.member);     \
    if (result != 0) return Fail(status, #function, result); \
    state.member = nullptr;                                  \
  }
  KVAF_RELEASE(remote, urma_unimport_seg)
  while (!state.buffers.empty()) {
    auto& buffer = state.buffers.back();
    if (buffer.segment != nullptr) {
      if (state.api.urma_unregister_seg(buffer.segment) != 0)
        return Fail(status, "urma_unregister_seg");
      buffer.segment = nullptr;
    }
    if (state.api.urma_free_token_id(buffer.token) != 0) return Fail(status, "urma_free_token_id");
    state.buffers.pop_back();
  }
  KVAF_RELEASE(local, urma_unregister_seg)
  KVAF_RELEASE(token, urma_free_token_id)
  KVAF_RELEASE(context, urma_delete_context)
#undef KVAF_RELEASE
  if (state.owns_init) {
    const int result = state.api.urma_uninit();
    if (result != 0) return Fail(status, "urma_uninit", result);
    state.owns_init = false;
  }
  if (state.library != nullptr) dlclose(state.library);
  if (state.common_library != nullptr) dlclose(state.common_library);
  state = {};
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
    state.library = dlopen(args.library, RTLD_NOW | RTLD_GLOBAL);
  } else if (!resident.empty()) {
    state.library = dlopen(resident.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
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
        state.common_library = dlopen(common.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (state.common_library == nullptr)
          return Fail(status, "dlopen bundled liburma_common", -1, dlerror());
      }
      state.library = dlopen(core.c_str(), RTLD_NOW | RTLD_GLOBAL);
    } else {
      state.library = dlopen("liburma.so.0", RTLD_NOW | RTLD_GLOBAL);
      if (state.library == nullptr) state.library = dlopen("liburma.so", RTLD_NOW | RTLD_GLOBAL);
    }
  }
  if (state.library == nullptr) {
    const char* error = dlerror();
    return Fail(status, "dlopen liburma (inside AICPU)", -1, error == nullptr ? "" : error);
  }
#define KVAF_LOAD_API(name)                                                                 \
  state.api.name = reinterpret_cast<decltype(state.api.name)>(dlsym(state.library, #name)); \
  if (state.api.name == nullptr) return Fail(status, "dlsym " #name, -1, dlerror());
  KVAF_UMDK_APIS(KVAF_LOAD_API)
#undef KVAF_LOAD_API
  state.api.get_tp_list = reinterpret_cast<GetTpList>(dlsym(state.library, "urma_get_tp_list"));
  if (state.api.get_tp_list == nullptr) return Fail(status, "dlsym urma_get_tp_list");
  state.api.import_jetty_ex =
      reinterpret_cast<ImportJettyEx>(dlsym(state.library, "urma_import_jetty_ex"));
  if (state.api.import_jetty_ex == nullptr) return Fail(status, "dlsym urma_import_jetty_ex");

  return true;
}

bool CreateQueue(const UmdkControlArgs& args, UmdkControlStatus& status, Queue& worker) {
  const auto* error =
      kvaf::urma::CreateQueues(state.api, state.context, args.token, {64U, 64U, 6U, true},
                               worker.jfc, worker.jfr, worker.jetty);
  return error ? Fail(status, error) : true;
}

bool DescribeWorker(const UmdkControlArgs& args, std::uint32_t index, UmdkControlStatus& status) {
  auto* selected = FindQueue(index);
  if (selected == nullptr) return Fail(status, "invalid worker index");
  auto& worker = *selected;
  if (worker.jetty == nullptr) return Fail(status, "worker Jetty is unavailable");
  if (worker.tp == 0 && state.shared_tp != 0) worker.tp = state.shared_tp;
  if (worker.tp == 0) {
    Stage(status, "urma_get_tp_list");
    if (!kvaf::urma::QueryTp(state.api.get_tp_list, state.context, state.peer_eid.data(),
                             &worker.tp))
      return Fail(status, "urma_get_tp_list");
    state.shared_tp = worker.tp;
  }
  status.tp = worker.tp;
  std::memcpy(status.eid, state.context->eid.raw, 16);
  status.key_bytes = sizeof(urma_jetty_id_t) + sizeof(urma_transport_mode_t);
  std::memcpy(status.key, &worker.jetty->jetty_id, sizeof(urma_jetty_id_t));
  const urma_transport_mode_t mode = URMA_TM_RM;
  std::memcpy(status.key + sizeof(urma_jetty_id_t), &mode, sizeof(mode));
  return true;
}

bool CreateContext(const UmdkControlArgs& args, UmdkControlStatus& status) {
  if (std::all_of(args.local_eid, args.local_eid + 16, [](auto b) { return b == 0; }))
    return Fail(status, "Host-selected EID is zero");
  bool matched = false;
  const bool enumerated = kvaf::urma::EnumerateEndpoints(
      state.api, "", UINT32_MAX, [&](auto* device, const auto& eid) {
        if (std::memcmp(args.local_eid, eid.eid.raw, 16) != 0) return false;
        matched = true;
        state.context = state.api.urma_create_context(device, eid.eid_index);
        if (!state.context) {
          Fail(status, "urma_create_context");
          // The same EID can be exposed at another device/index.
          return false;
        }
        status.result = status.error_number = 0;
        status.detail[0] = '\0';
        std::memcpy(status.eid, state.context->eid.raw, 16);
        std::fprintf(stderr, "[KVAF_AICPU] UMDK_EID_SELECTED source=host device=%s eid_index=%u eid=",
                     device->name, eid.eid_index);
        for (auto byte : status.eid) std::fprintf(stderr, "%02x", byte);
        std::fprintf(stderr, "\n");
        return true;
      });
  if (!enumerated) return Fail(status, "urma_get_device_list");
  if (state.context) return true;
  if (matched) return false;
  char expected[33]{};
  for (unsigned i = 0; i < 16; ++i) std::snprintf(expected + i * 2, 3, "%02x", args.local_eid[i]);
  errno = ENODEV;
  return Fail(status, "Host EID not exposed by UMDK", -1, expected);
}

bool Prepare(const UmdkControlArgs& args, UmdkControlStatus& status) {
  if (state.session != 0) return Fail(status, "UMDK session already active");
  state.session = ++generation;
  status.session = state.session;

  if (!LoadApi(args, status)) return false;
  Stage(status, "urma_init");
  const int init_result = state.api.urma_init(nullptr);
  const int init_errno = errno;

  errno = init_errno;
  if (init_result != 0 && init_result != EEXIST && init_result != -EEXIST)
    return Fail(status, "urma_init", init_result);
  state.owns_init = init_result == 0;
  if (!CreateContext(args, status)) return false;
  std::copy_n(args.route_eid, 16, state.peer_eid.begin());
  state.worker_count = args.worker_count;
  // Preserve queue creation order while storing every worker uniformly.
  if (!CreateQueue(args, status, state.control_queues[0])) return false;
  for (std::uint32_t index = 0; index < state.worker_count; ++index) {
    if (!CreateQueue(args, status, state.data_queues[index])) return false;
    if (index != 0 && !CreateQueue(args, status, state.control_queues[index])) return false;
  }
  Stage(status, "urma_alloc_token_id");
  state.token = state.api.urma_alloc_token_id(state.context);
  if (state.token == nullptr) return Fail(status, "urma_alloc_token_id");
  urma_seg_cfg_t cfg{};
  cfg.va = args.local_buffer & ~std::uint64_t{4095};
  cfg.len = args.buffer_bytes + args.local_buffer - cfg.va;
  cfg.token_id = state.token;
  cfg.token_value.token = args.token;
  cfg.flag.value = MemRegFlags(true);

  Stage(status, "urma_register_seg (local HBM)");
  state.local = state.api.urma_register_seg(state.context, &cfg);
  if (state.local == nullptr) {
    Fail(status, "urma_register_seg (local HBM)");

    return false;
  }
  state.buffer = args.local_buffer;
  state.buffer_bytes = args.buffer_bytes;
  if (!DescribeWorker(args, kUmdkControlQueueIndex, status)) return false;
  if (!DescribeWorker(args, 0, status)) return false;
  return true;
}

bool Connect(const UmdkControlArgs& args, UmdkControlStatus& status) {
  auto* selected = FindQueue(args.worker_index);
  if (selected == nullptr) return Fail(status, "invalid worker index");
  auto& worker = *selected;
  if (state.local == nullptr || worker.target != nullptr || worker.tp == 0)
    return Fail(status, "invalid connect state");
  if (args.segment_bytes < sizeof(urma_seg_t) || args.key_bytes != 28 || args.peer_tp == 0)
    return Fail(status, "invalid peer descriptor");
  if (state.shared_peer_tp != 0 &&
      (args.peer_tp != state.shared_peer_tp || args.peer_psn != state.shared_peer_psn))
    return Fail(status, "shared TP peer binding changed", -1,
                "all Jettys must use the same peer TP and PSN");
  urma_seg_t segment{};
  std::memcpy(&segment, args.segment, sizeof(segment));
  if (args.remote_address < segment.ubva.va || args.remote_address - segment.ubva.va >= segment.len)
    return Fail(status, "remote range outside segment");
  urma_token_t token{};
  token.token = args.token;
  urma_import_seg_flag_t flag{};
  flag.bs.cacheable = URMA_NON_CACHEABLE;
  flag.bs.access = (1U << 1U) | (1U << 2U);
  flag.bs.mapping = URMA_SEG_NOMAP;
  Stage(status, "urma_import_seg");
  if (state.remote == nullptr)
    state.remote = state.api.urma_import_seg(state.context, &segment, &token, 0, flag);
  if (state.remote == nullptr) return Fail(status, "urma_import_seg");
  Stage(status, "urma_import_jetty_ex");
  worker.target = kvaf::urma::ImportJetty(state.api.import_jetty_ex, state.context, args.peer_key,
                                          args.token, worker.tp, args.peer_tp, 1U, args.peer_psn);
  if (worker.target == nullptr) return Fail(status, "urma_import_jetty_ex");
  state.shared_peer_tp = args.peer_tp;
  state.shared_peer_psn = args.peer_psn;
  state.remote_address = args.remote_address;
  return DescribeWorker(args, args.worker_index, status);
}

urma_target_seg_t* LocalSegment(std::uint64_t address, std::uint64_t bytes) {
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
  auto* token = state.api.urma_alloc_token_id(state.context);
  if (token == nullptr) return Fail(status, "urma_alloc_token_id");
  urma_seg_cfg_t cfg{};
  cfg.va = args.local_buffer & ~std::uint64_t{4095};
  cfg.len = args.buffer_bytes + args.local_buffer - cfg.va;
  cfg.token_id = token;
  cfg.token_value.token = args.token;
  cfg.flag.value = MemRegFlags(true);
  Stage(status, "urma_register_seg (local HBM)");
  auto* segment = state.api.urma_register_seg(state.context, &cfg);
  if (segment == nullptr) {
    Fail(status, "urma_register_seg (local HBM)");
    (void)state.api.urma_free_token_id(token);
    return false;
  }
  state.buffers.push_back({args.local_buffer, args.buffer_bytes, token, segment});
  return true;
}

}  // namespace kvaf::aicpu_urma::detail
