#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <vector>

#include "cpu_gather.h"
#include "gather_cache.h"
#include "umdk_abi.h"
#include "umdk_control.h"
#include "umdk_diagnostics.h"
#include "umdk_gather.h"

namespace {
using namespace kvaf::aicpu_urma;
using namespace kvaf::aicpu_urma::umdk;

std::uint64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

#define KVAF_UMDK_APIS(APPLY) \
  APPLY(urma_init) \
  APPLY(urma_uninit) \
  APPLY(urma_get_device_list) \
  APPLY(urma_free_device_list) \
  APPLY(urma_get_eid_list) \
  APPLY(urma_free_eid_list) \
  APPLY(urma_create_context) \
  APPLY(urma_delete_context) \
  APPLY(urma_create_jfc) \
  APPLY(urma_delete_jfc) \
  APPLY(urma_create_jfr) \
  APPLY(urma_delete_jfr) \
  APPLY(urma_create_jetty) \
  APPLY(urma_delete_jetty) \
  APPLY(urma_alloc_token_id) \
  APPLY(urma_free_token_id) \
  APPLY(urma_register_seg) \
  APPLY(urma_unregister_seg) \
  APPLY(urma_import_seg) \
  APPLY(urma_unimport_seg) \
  APPLY(urma_unimport_jetty) \
  APPLY(urma_post_jetty_send_wr) \
  APPLY(urma_poll_jfc)

struct Api {
#define KVAF_DECLARE_API(name) decltype(&::name) name = nullptr;
  KVAF_UMDK_APIS(KVAF_DECLARE_API)
#undef KVAF_DECLARE_API
  GetTpList get_tp_list = nullptr;
  ImportJettyEx import_jetty_ex = nullptr;
  decltype(&::urma_register_log_func) register_log = nullptr;
  decltype(&::urma_unregister_log_func) unregister_log = nullptr;
  decltype(&::urma_log_get_level) get_log_level = nullptr;
  decltype(&::urma_log_set_level) set_log_level = nullptr;
};

struct WrQueue {
  std::vector<Sge> local_sges;
  std::vector<Sge> remote_sges;
  std::vector<urma_jfs_wr_t> wrs;
};

struct GatherQueue : WrQueue {
  // Vectors own constructed workspace slots; wr_count is the active chain size.
  std::vector<kvaf::GatherOutputSpan> targets;
  std::uint32_t wr_count = 0;
  std::uint64_t builds = 0;
};

template <typename T>
void EnsureSlots(std::vector<T>& slots, std::size_t count) {
  if (count > slots.size()) slots.resize(std::max(count, slots.size() * 2U));
}

void SnapshotGatherTargets(GatherQueue& queue, const GatherReadArgs& args, std::uint32_t first,
                           std::uint32_t last) {
  const auto count = last - first;
  EnsureSlots(queue.targets, count);
  std::memcpy(queue.targets.data(),
              reinterpret_cast<const kvaf::GatherOutputSpan*>(args.output_spans) + first,
              static_cast<std::size_t>(count) * sizeof(kvaf::GatherOutputSpan));
}

struct Worker : WrQueue {
  bool poisoned = false;
  urma_jfc_t* jfc = nullptr;
  urma_jfr_t* jfr = nullptr;
  urma_jetty_t* jetty = nullptr;
  urma_target_jetty_t* target = nullptr;
  std::uint64_t tp = 0;
  std::uint64_t sequence = 0;
  GatherQueue read_queue;
};

struct RegisteredBuffer {
  std::uint64_t base, bytes;
  urma_token_id_t* token;
  urma_target_seg_t* segment;
};

struct State : Worker {
  Api api;
  void* library = nullptr;
  void* common_library = nullptr;
  bool owns_init = false;
  bool log_registered = false;
  urma_vlog_level_t old_log_level = URMA_VLOG_LEVEL_INFO;
  urma_context_t* context = nullptr;
  urma_token_id_t* token = nullptr;
  urma_target_seg_t* local = nullptr;
  urma_target_seg_t* remote = nullptr;
  std::uint64_t session = 0;
  std::uint64_t shared_tp = 0, shared_peer_tp = 0;
  std::uint32_t shared_peer_psn = 0;
  std::array<std::uint8_t, 16> peer_eid{};
  std::uint64_t buffer = 0, buffer_bytes = 0, remote_address = 0;
  std::uint32_t bytes = 0;
  char library_path[256]{};
  char device[64]{};
  std::uint32_t eid_index = 0, worker_count = 1;
  std::array<std::unique_ptr<Worker>, 7> extra_workers;
  std::unique_ptr<Worker> control;
  std::vector<RegisteredBuffer> buffers;
};

State state;
std::shared_mutex state_mutex;
std::array<std::mutex, 8> worker_mutexes;
std::array<std::mutex, 8> control_mutexes;
std::uint64_t generation = 0;
DiagnosticLog diagnostic_log;

Worker& GetWorker(std::uint32_t index) {
  return index == 0 ? static_cast<Worker&>(state) : *state.extra_workers[index - 1];
}

Worker* SetupQueue(std::uint32_t index) {
  if (index < state.worker_count) return &GetWorker(index);
  if (index == kUmdkControlQueueIndex) return state.control.get();
  return nullptr;
}

void CaptureUrmaLog(int level, char* message) { diagnostic_log.NativeLog(level, message); }

void EnableLogging() {
  const auto& api = state.api;
  if (api.register_log == nullptr || api.unregister_log == nullptr ||
      api.get_log_level == nullptr || api.set_log_level == nullptr) {
    diagnostic_log.Append("native_log_capture=unavailable (optional APIs missing)\n");
    return;
  }
  state.old_log_level = api.get_log_level();
  const int result = api.register_log(CaptureUrmaLog);
  state.log_registered = result == 0;
  diagnostic_log.Append("native_log_capture_register=%d\n", result);
  if (state.log_registered) api.set_log_level(URMA_VLOG_LEVEL_DEBUG);
}

void Stage(UmdkControlStatus& status, const char* name) {
  std::snprintf(status.stage, sizeof(status.stage), "%s", name);
  errno = 0;
}

bool Fail(UmdkControlStatus& status, const char* stage, int result = -1, const char* detail = "") {
  status.result = result == 0 ? -1 : result;
  status.error_number = errno;
  std::snprintf(status.stage, sizeof(status.stage), "%s", stage);
  std::snprintf(status.detail, sizeof(status.detail), "%s", detail == nullptr ? "" : detail);
  return false;
}

bool CleanupWorker(Worker& worker, UmdkControlStatus& status) {
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
  for (std::uint32_t index = 0; index < state.worker_count; ++index) {
    if (index != 0 && !state.extra_workers[index - 1]) continue;
    if (!CleanupWorker(GetWorker(index), status)) return false;
  }
  if (state.control && !CleanupWorker(*state.control, status)) return false;
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
  if (state.log_registered) {
    state.api.set_log_level(state.old_log_level);
    const int result = state.api.unregister_log();
    if (result != 0) return Fail(status, "urma_unregister_log_func", result);
    state.log_registered = false;
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
            std::strstr(info->dlpi_name, "/umdk/") != nullptr)
          return 0;
        *static_cast<std::string*>(output) = info->dlpi_name;
        return 1;
      },
      &path);
#else
  auto* symbol = dlsym(RTLD_DEFAULT, "urma_init");
  Dl_info info{};
  if (symbol != nullptr && dladdr(symbol, &info) != 0 && info.dli_fname != nullptr &&
      std::strstr(info.dli_fname, "/umdk/") == nullptr)
    path = info.dli_fname;
#endif
  return path;
}

bool LoadApi(const UmdkControlArgs& args, UmdkControlStatus& status) {
  Stage(status, "dlopen liburma (inside AICPU)");
  const auto resident = args.library[0] == '\0' ? ResidentUrmaPath() : std::string{};
  if (args.library[0] != '\0') {
    diagnostic_log.Append("core_selection=explicit requested=%s\n", args.library);
    state.library = dlopen(args.library, RTLD_NOW | RTLD_GLOBAL);
  } else if (!resident.empty()) {
    diagnostic_log.Append("core_selection=reuse_loaded path=%s\n", resident.c_str());
    state.library = dlopen(resident.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  } else {
    Dl_info info{};
    std::string bundled;
    if (dladdr(reinterpret_cast<void*>(&KvafAicpuUmdkControl), &info) != 0 &&
        info.dli_fname != nullptr) {
      const std::string path = info.dli_fname;
      const auto slash = path.find_last_of('/');
      if (slash != std::string::npos) bundled = path.substr(0, slash) + "/umdk/";
    }
    auto core = bundled + "liburma.so.0";
    if (access(core.c_str(), R_OK) != 0) core = bundled + "liburma.so";
    if (!bundled.empty() && access(core.c_str(), R_OK) == 0) {
      diagnostic_log.Append("core_selection=bundled no_resident_core=1 path=%s\n", core.c_str());
      auto common = bundled + "liburma_common.so.0";
      if (access(common.c_str(), R_OK) != 0) common = bundled + "liburma_common.so";
      if (access(common.c_str(), R_OK) == 0) {
        state.common_library = dlopen(common.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (state.common_library == nullptr)
          return Fail(status, "dlopen bundled liburma_common", -1, dlerror());
      }
      state.library = dlopen(core.c_str(), RTLD_NOW | RTLD_GLOBAL);
    } else {
      diagnostic_log.Append("core_selection=loader_search no_resident_core=1\n");
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
  Dl_info library_info{};
  if (dladdr(reinterpret_cast<void*>(state.api.urma_init), &library_info) != 0 &&
      library_info.dli_fname != nullptr) {
    std::snprintf(status.detail, sizeof(status.detail), "%s", library_info.dli_fname);
    std::snprintf(status.library, sizeof(status.library), "%s", library_info.dli_fname);
    std::snprintf(state.library_path, sizeof(state.library_path), "%s", library_info.dli_fname);
    diagnostic_log.Append("core_library=%s\n", library_info.dli_fname);
    const std::string path = library_info.dli_fname;
    const auto slash = path.find_last_of('/');
  }
  state.api.register_log = reinterpret_cast<decltype(state.api.register_log)>(
      dlsym(state.library, "urma_register_log_func"));
  state.api.unregister_log = reinterpret_cast<decltype(state.api.unregister_log)>(
      dlsym(state.library, "urma_unregister_log_func"));
  state.api.get_log_level = reinterpret_cast<decltype(state.api.get_log_level)>(
      dlsym(state.library, "urma_log_get_level"));
  state.api.set_log_level = reinterpret_cast<decltype(state.api.set_log_level)>(
      dlsym(state.library, "urma_log_set_level"));
  EnableLogging();
  return true;
}

bool CreateWorker(const UmdkControlArgs& args, UmdkControlStatus& status, Worker& worker) {
  urma_jfc_cfg_t jfc_cfg{};
  jfc_cfg.depth = 64;
  // Data/control paths hold their queue's outer mutex across post + CQ wait;
  // setup/teardown holds state_mutex exclusively. No provider lock is needed.
  jfc_cfg.flag.bs.lock_free = 1;
  Stage(status, "urma_create_jfc");
  worker.jfc = state.api.urma_create_jfc(state.context, &jfc_cfg);
  if (worker.jfc == nullptr) return Fail(status, "urma_create_jfc");
  urma_jfr_cfg_t jfr_cfg{};
  jfr_cfg.depth = 64;
  jfr_cfg.trans_mode = URMA_TM_RM;
  jfr_cfg.max_sge = 1;
  jfr_cfg.jfc = worker.jfc;
  jfr_cfg.token_value.token = args.token;
  Stage(status, "urma_create_jfr");
  worker.jfr = state.api.urma_create_jfr(state.context, &jfr_cfg);
  if (worker.jfr == nullptr) return Fail(status, "urma_create_jfr");
  urma_jetty_cfg_t jetty_cfg{};
  jetty_cfg.flag.bs.share_jfr = URMA_SHARE_JFR;
  jetty_cfg.jfs_cfg.depth = 8192;
  jetty_cfg.jfs_cfg.trans_mode = URMA_TM_RM;
  jetty_cfg.jfs_cfg.flag.bs.lock_free = 1;
  jetty_cfg.jfs_cfg.priority = URMA_MAX_PRIORITY;
  jetty_cfg.jfs_cfg.max_sge = 6U;
  jetty_cfg.jfs_cfg.max_rsge = 1;
  jetty_cfg.jfs_cfg.rnr_retry = URMA_TYPICAL_RNR_RETRY;
  jetty_cfg.jfs_cfg.err_timeout = URMA_TYPICAL_ERR_TIMEOUT;
  jetty_cfg.jfs_cfg.jfc = worker.jfc;
  jetty_cfg.shared.jfr = worker.jfr;
  jetty_cfg.shared.jfc = worker.jfc;
  Stage(status, "urma_create_jetty (normal UMDK)");
  worker.jetty = state.api.urma_create_jetty(state.context, &jetty_cfg);
  if (worker.jetty == nullptr) return Fail(status, "urma_create_jetty (normal UMDK)");
  diagnostic_log.Append("normal_jetty_created id=%u depth=%u\n", worker.jetty->jetty_id.id,
                        jetty_cfg.jfs_cfg.depth);
  return true;
}

bool DescribeWorker(const UmdkControlArgs& args, std::uint32_t index, UmdkControlStatus& status) {
  auto* selected = SetupQueue(index);
  if (selected == nullptr) return Fail(status, "invalid worker index");
  auto& worker = *selected;
  if (worker.jetty == nullptr)
    return Fail(status, "builder has no Jetty; worker 0 owns submission");
  if (worker.tp == 0 && state.shared_tp != 0) worker.tp = state.shared_tp;
  if (worker.tp == 0) {
    TpQuery query{};
    query.flag = 1;
    query.trans_mode = URMA_TM_RM;
    std::memcpy(query.local_eid, state.context->eid.raw, 16);
    std::memcpy(query.peer_eid, state.peer_eid.data(), 16);
    TpInfo info{};
    std::uint32_t count = 1;
    Stage(status, "urma_get_tp_list");
    const int result = state.api.get_tp_list(state.context, &query, &count, &info);
    if (result != 0 || count == 0 || info.tp_handle == 0)
      return Fail(status, "urma_get_tp_list", result);
    worker.tp = info.tp_handle;
    state.shared_tp = worker.tp;
  }
  status.tp = worker.tp;
  status.wr_builds = worker.read_queue.builds;
  status.local_jetty_id = worker.jetty->jetty_id.id;
  status.local_cq_id = worker.jfc->jfc_id.id;
  status.local_jetty_address = reinterpret_cast<std::uint64_t>(worker.jetty);
  status.local_cq_address = reinterpret_cast<std::uint64_t>(worker.jfc);
  if (worker.target != nullptr) {
    status.peer_jetty_id = worker.target->id.id;
    status.target_address = reinterpret_cast<std::uint64_t>(worker.target);
  }
  status.eid_index = state.eid_index;
  std::memcpy(status.eid, state.context->eid.raw, 16);
  std::snprintf(status.device, sizeof(status.device), "%s", state.device);
  status.key_bytes = sizeof(urma_jetty_id_t) + sizeof(urma_transport_mode_t);
  std::memcpy(status.key, &worker.jetty->jetty_id, sizeof(urma_jetty_id_t));
  const urma_transport_mode_t mode = URMA_TM_RM;
  std::memcpy(status.key + sizeof(urma_jetty_id_t), &mode, sizeof(mode));
  return true;
}

bool Prepare(const UmdkControlArgs& args, UmdkControlStatus& status) {
  if (state.session != 0) return Fail(status, "UMDK session already active");
  state.session = ++generation;
  status.session = state.session;
  diagnostic_log.Append(
      "selection requested_device='%s' name_bytes=%zu eid_index=%u workers=%u "
      "expected_ub_type=%u\n",
      args.device, std::strlen(args.device), args.eid_index, std::max(1U, args.worker_count),
      static_cast<unsigned>(URMA_TRANSPORT_UB));
  if (!LoadApi(args, status)) return false;
  Stage(status, "urma_init");
  const int init_result = state.api.urma_init(nullptr);
  const int init_errno = errno;
  status.init_result = init_result;
  status.init_errno = init_errno;
  diagnostic_log.Append("urma_init result=%d errno=%d\n", init_result, init_errno);
  errno = init_errno;
  if (init_result != 0 && init_result != EEXIST && init_result != -EEXIST)
    return Fail(status, "urma_init", init_result);
  state.owns_init = init_result == 0;
  int device_count = 0;
  Stage(status, "urma_get_device_list");
  auto** devices = state.api.urma_get_device_list(&device_count);
  const int device_errno = errno;
  status.device_count = device_count;
  diagnostic_log.Append("urma_get_device_list null=%d count=%d errno=%d\n", devices == nullptr,
                        device_count, device_errno);
  if (devices == nullptr) {
    errno = device_errno;
    Fail(status, "urma_get_device_list", -1,
         "See AICPU diagnostics: sysfs visibility, provider logs and loaded modules");
    diagnostic_log.LastError(status.detail, sizeof(status.detail));
    return false;
  }
  std::uint32_t matching_names = 0, matching_ub_devices = 0;
  int eid_error = 0;
  for (int device_index = 0; device_index < device_count && state.context == nullptr;
       ++device_index) {
    auto* device = devices[device_index];
    const bool name_match = args.device[0] == '\0' || std::strcmp(args.device, device->name) == 0;
    const bool ub_match = device->type == URMA_TRANSPORT_UB;
    diagnostic_log.Append("urma_device name=%s type=%u path=%s name_match=%u ub_match=%u\n",
                          device->name, static_cast<unsigned>(device->type), device->path,
                          name_match, ub_match);
    if (name_match) ++matching_names;
    if (!ub_match || !name_match) continue;
    ++matching_ub_devices;
    std::uint32_t eid_count = 0;
    Stage(status, "urma_get_eid_list");
    auto* eids = state.api.urma_get_eid_list(device, &eid_count);
    if (eids == nullptr) eid_error = errno;
    diagnostic_log.Append("urma_get_eid_list device=%s null=%d count=%u errno=%d\n", device->name,
                          eids == nullptr, eid_count, errno);
    if (eids == nullptr) continue;
    for (std::uint32_t index = 0; index < eid_count; ++index) {
      diagnostic_log.Append("urma_eid device=%s index=%u eid=", device->name,
                            eids[index].eid_index);
      for (auto byte : eids[index].eid.raw) diagnostic_log.Append("%02x", byte);
      diagnostic_log.Append("\n");
      if (args.eid_index != UINT32_MAX && eids[index].eid_index != args.eid_index) continue;
      const std::uint8_t zero[16]{};
      if (std::memcmp(eids[index].eid.raw, zero, 16) == 0) continue;
      Stage(status, "urma_create_context");
      state.context = state.api.urma_create_context(device, eids[index].eid_index);
      if (state.context == nullptr) {
        Fail(status, "urma_create_context");
        break;
      }
      status.eid_index = eids[index].eid_index;
      state.eid_index = status.eid_index;
      std::snprintf(state.device, sizeof(state.device), "%s", device->name);
      std::memcpy(status.eid, state.context->eid.raw, 16);
      std::snprintf(status.device, sizeof(status.device), "%s", device->name);
      break;
    }
    state.api.urma_free_eid_list(eids);
    if (status.result != 0) break;
  }
  state.api.urma_free_device_list(devices);
  if (state.context == nullptr) {
    if (status.result != 0) return false;
    char detail[256]{};
    std::snprintf(detail, sizeof(detail),
                  "requested_device='%s' name_bytes=%zu eid_index=%u expected_ub_type=%u "
                  "name_matches=%u ub_matches=%u",
                  args.device, std::strlen(args.device), args.eid_index,
                  static_cast<unsigned>(URMA_TRANSPORT_UB), matching_names, matching_ub_devices);
    if (matching_names == 0) {
      errno = ENODEV;
      return Fail(status, "requested UMDK device not found", -1, detail);
    }
    if (matching_ub_devices == 0) {
      errno = EPROTONOSUPPORT;
      return Fail(status, "selected device is not UB transport", -1, detail);
    }
    errno = eid_error != 0 ? eid_error : ENOENT;
    return Fail(status, "no usable EID on selected UB device", -1, detail);
  }
  std::copy_n(args.route_eid, 16, state.peer_eid.begin());
  state.worker_count = args.worker_count;
  // Keep the control queue's creation order independent of the data workers.
  {
    state.control = std::make_unique<Worker>();
    if (!CreateWorker(args, status, *state.control)) return false;
  }
  for (std::uint32_t index = 0; index < state.worker_count; ++index) {
    if (index != 0) state.extra_workers[index - 1] = std::make_unique<Worker>();
    if (!CreateWorker(args, status, GetWorker(index))) return false;
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
  diagnostic_log.Append(
      "local_hbm_register va=0x%llx bytes=%llu flags=0x%x non_pin=%u access=0x%x\n",
      static_cast<unsigned long long>(cfg.va), static_cast<unsigned long long>(cfg.len),
      cfg.flag.value, cfg.flag.bs.non_pin, cfg.flag.bs.access);
  Stage(status, "urma_register_seg (local HBM)");
  state.local = state.api.urma_register_seg(state.context, &cfg);
  if (state.local == nullptr) {
    Fail(status, "urma_register_seg (local HBM)");
    diagnostic_log.LastError(status.detail, sizeof(status.detail));
    return false;
  }
  state.buffer = args.local_buffer;
  state.buffer_bytes = args.buffer_bytes;
  state.bytes = args.bytes;
  if (state.control && !DescribeWorker(args, kUmdkControlQueueIndex, status)) return false;
  if (!DescribeWorker(args, 0, status)) return false;
  if (state.log_registered) state.api.set_log_level(URMA_VLOG_LEVEL_ERR);
  return true;
}

bool Connect(const UmdkControlArgs& args, UmdkControlStatus& status) {
  auto* selected = SetupQueue(args.worker_index);
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
  if (args.remote_address < segment.ubva.va || state.bytes > segment.len ||
      args.remote_address - segment.ubva.va > segment.len - state.bytes)
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
  RemoteJettyEx remote{};
  std::memcpy(&remote.jetty_id, args.peer_key, sizeof(remote.jetty_id));
  std::memcpy(&remote.trans_mode, args.peer_key + sizeof(remote.jetty_id),
              sizeof(remote.trans_mode));
  remote.policy = URMA_JETTY_GRP_POLICY_RR;
  remote.type = URMA_JETTY;
  urma_import_jetty_flag_t jetty_flag{};
  jetty_flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
  remote.flag.value = jetty_flag.value;
  remote.tp_type = 1;
  ActiveTp active{};
  active.tp_handle = worker.tp;
  active.peer_tp_handle = args.peer_tp;
  active.tx_psn = 1;
  active.rx_psn = args.peer_psn;
  Stage(status, "urma_import_jetty_ex");
  worker.target = state.api.import_jetty_ex(state.context, &remote, &token, &active);
  if (worker.target == nullptr) return Fail(status, "urma_import_jetty_ex");
  state.shared_peer_tp = args.peer_tp;
  state.shared_peer_psn = args.peer_psn;
  state.remote_address = args.remote_address;
  return DescribeWorker(args, args.worker_index, status);
}

bool PostBatch(Worker& worker, urma_jfs_wr_t* first, urma_jfs_wr_t* tail, std::uint32_t first_index,
               std::uint32_t count, UmdkControlStatus& status) {
  auto* saved_next = tail->next;
  tail->next = nullptr;
  urma_jfs_wr_t* bad = nullptr;
  errno = 0;
  ++status.post_calls;
  const int result = state.api.urma_post_jetty_send_wr(worker.jetty, first, &bad);
  tail->next = saved_next;
  if (result == 0) return true;
  worker.poisoned = true;
  diagnostic_log.Append("post_failed first_wr=%u batch_wqes=%u result=%d errno=%d bad_wr=%p\n",
                        first_index, count, result, errno, static_cast<void*>(bad));
  return Fail(status, "urma_post_jetty_send_wr", result);
}

bool WaitCompletion(Worker& worker, std::uint64_t expected_context, std::uint64_t start,
                    std::uint32_t timeout_ms, UmdkControlStatus& status) {
  const auto posted = status.last_post_ns;
  const auto deadline = start + static_cast<std::uint64_t>(timeout_ms) * 1000000;
  bool success = false;
  auto poll_time = posted;
  while (poll_time < deadline) {
    urma_cr_t completion{};
    const int polled = state.api.urma_poll_jfc(worker.jfc, 1, &completion);
    ++status.cq_poll_calls;
    if (polled < 0) {
      Fail(status, "urma_poll_jfc", polled);
      break;
    }
    if (polled == 0) {
      if (status.cq_poll_calls % 64U == 0U) poll_time = NowNs();
      continue;
    }
    status.cq_status = completion.status;
    status.cq_context = completion.user_ctx;
    if (completion.status != URMA_CR_SUCCESS || completion.user_ctx != expected_context) {
      Fail(status, "READ/WRITE CQE status or user_ctx mismatch");
      break;
    }
    success = true;
    break;
  }
  status.completion_ns = NowNs();
  status.total_ns = status.completion_ns - status.first_post_ns;
  status.cq_ns = status.completion_ns - posted;
  if (!success) {
    worker.poisoned = true;
    if (status.result == 0) {
      errno = ETIMEDOUT;
      Fail(status, "waiting for CQE timed out", ETIMEDOUT);
    }
  }
  return success;
}

template <typename RequestAt>
bool SubmitRequests(Worker& worker, std::uint32_t count, std::uint32_t batch,
                    std::uint32_t timeout_ms, UmdkControlStatus& status, RequestAt request_at) {
  status.read_wqes = count;
  Stage(status, "provider post and CQ wait");
  const auto start = NowNs();
  status.first_post_ns = start;
  for (std::uint32_t first = 0; first < count; first += batch) {
    const auto end = std::min(first + batch, count);
    if (!PostBatch(worker, request_at(first), request_at(end - 1U), first, end - first, status)) {
      status.post_ns = NowNs() - start;
      return false;
    }
  }
  status.last_post_ns = NowNs();
  status.post_ns = status.last_post_ns - start;
  return WaitCompletion(worker, request_at(count - 1U)->user_ctx, start, timeout_ms, status);
}

bool SubmitQueue(Worker& worker, std::vector<urma_jfs_wr_t>& requests, std::uint32_t batch,
                 std::uint32_t timeout_ms, UmdkControlStatus& status) {
  return SubmitRequests(worker, requests.size(), batch, timeout_ms, status,
                        [&](std::uint32_t index) { return &requests[index]; });
}

urma_target_seg_t* LocalSegment(std::uint64_t address, std::uint64_t bytes,
                                std::uint64_t* available = nullptr) {
  if (state.local != nullptr &&
      kvaf::peer::GatherRange(state.buffer, state.buffer_bytes, address, bytes)) {
    if (available != nullptr) *available = state.buffer_bytes - (address - state.buffer);
    return state.local;
  }
  for (const auto& buffer : state.buffers)
    if (kvaf::peer::GatherRange(buffer.base, buffer.bytes, address, bytes)) {
      if (available != nullptr) *available = buffer.bytes - (address - buffer.base);
      return buffer.segment;
    }
  return nullptr;
}

bool RegisterGatherBuffer(const UmdkControlArgs& args, UmdkControlStatus& status) {
  if (args.local_buffer == 0 || args.buffer_bytes == 0 ||
      args.local_buffer > UINT64_MAX - args.buffer_bytes)
    return Fail(status, "invalid GatherRead HBM registration");
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

bool BuildTransfers(Worker& worker, bool write, const UmdkTransfer* transfers, std::uint32_t count,
                    std::uint32_t& transfer_index, std::uint64_t& offset,
                    UmdkControlStatus& status) {
  worker.local_sges.clear();
  worker.remote_sges.clear();
  worker.wrs.clear();
  while (transfer_index < count && worker.wrs.size() < 4096) {
    const auto& transfer = transfers[transfer_index];
    if (transfer.bytes == 0 || offset >= transfer.bytes ||
        !kvaf::peer::GatherRange(state.remote->seg.ubva.va, state.remote->seg.len, transfer.remote,
                                 transfer.bytes))
      return Fail(status, "UMDK transfer outside remote region");
    auto* segment = LocalSegment(transfer.local, transfer.bytes);
    if (segment == nullptr) return Fail(status, "UMDK transfer outside registered HBM");
    const auto bytes =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(65536, transfer.bytes - offset));
    worker.local_sges.push_back({transfer.local + offset, bytes, segment, nullptr});
    worker.remote_sges.push_back({transfer.remote + offset, bytes, state.remote, nullptr});
    urma_jfs_wr_t wr{};
    wr.opcode = write ? URMA_OPC_WRITE : URMA_OPC_READ;
    wr.flag.bs.comp_order = 1;
    wr.tjetty = worker.target;
    wr.user_ctx = ++worker.sequence;
    worker.wrs.push_back(wr);
    offset += bytes;
    if (offset == transfer.bytes) {
      ++transfer_index;
      offset = 0;
    }
  }
  for (std::size_t index = 0; index < worker.wrs.size(); ++index) {
    auto& wr = worker.wrs[index];
    const bool last = index + 1 == worker.wrs.size();
    wr.flag.bs.place_order = last ? URMA_STRONG_ORDER : URMA_RELAX_ORDER;
    wr.flag.bs.complete_enable = last;
    urma_sg_t local{reinterpret_cast<urma_sge_t*>(&worker.local_sges[index]), 1};
    urma_sg_t remote{reinterpret_cast<urma_sge_t*>(&worker.remote_sges[index]), 1};
    wr.rw.src = write ? local : remote;
    wr.rw.dst = write ? remote : local;
    wr.next = last ? nullptr : &worker.wrs[index + 1];
  }
  return true;
}

bool TransferGather(Worker& worker, bool write, const UmdkTransfer* transfers, std::uint32_t count,
                    UmdkControlStatus& status) {
  if (worker.poisoned || worker.target == nullptr || state.remote == nullptr ||
      transfers == nullptr || count == 0 || count > kGatherMaxSpans)
    return Fail(status, "invalid UMDK transfer queue");
  std::uint32_t index = 0;
  std::uint64_t offset = 0;
  while (index < count) {
    if (!BuildTransfers(worker, write, transfers, count, index, offset, status)) return false;
    UmdkControlStatus submitted{};
    const bool success = SubmitQueue(worker, worker.wrs, 8, 1000, submitted);
    status.post_ns += submitted.post_ns;
    status.cq_ns += submitted.cq_ns;
    status.read_wqes += submitted.read_wqes;
    status.post_calls += submitted.post_calls;
    status.cq_poll_calls += submitted.cq_poll_calls;
    if (!success) {
      status.cq_status = submitted.cq_status;
      status.cq_context = submitted.cq_context;
      errno = submitted.error_number;
      return Fail(status, submitted.stage, submitted.result);
    }
  }
  return true;
}

// The caller holds its data-worker lock when using the original shared queue.
// A dedicated control queue has its own lock and never polls a data CQ.
bool TransferControl(Worker& shared_worker, bool write, const UmdkTransfer* transfers,
                     std::uint32_t count, UmdkControlStatus& status, std::uint32_t lane = 0U) {
  if (!state.control) return TransferGather(shared_worker, write, transfers, count, status);
  auto* control = SetupQueue(kUmdkControlQueueIndex);
  if (!control) return Fail(status, "control queue is unavailable");
  std::lock_guard<std::mutex> control_lock(control_mutexes[0]);
  return TransferGather(*control, write, transfers, count, status);
}

constexpr std::uint32_t kGatherWaveWrs = 4096U;

void BeginGatherWave(GatherQueue& queue, std::uint32_t sgl_size) {
  EnsureSlots(queue.local_sges, kGatherWaveWrs * std::min(sgl_size, kvaf::kGatherMaxSglSize));
  EnsureSlots(queue.remote_sges, kGatherWaveWrs);
  EnsureSlots(queue.wrs, kGatherWaveWrs);
  queue.wr_count = 0U;
  ++queue.builds;
}

// Resolve this request's fixed addresses and registration once, before batches.
// Only allocation capacity survives a request; no WR/layout contents are cached.
struct GatherBuildLayout {
  std::uint64_t output, staging;
  std::uint32_t sgl_size;
  urma_target_seg_t* local_segment;
  urma_target_seg_t* remote_segment;
  urma_jfs_wr_t prototype{};

  GatherBuildLayout(Worker& worker, const GatherReadArgs& args,
                    urma_target_jetty_t* target = nullptr)
      : output(args.output),
        staging(args.remote_staging),
        sgl_size(std::min(args.sgl_size, kvaf::kGatherMaxSglSize)),
        local_segment(LocalSegment(args.output, args.output_extent)),
        remote_segment(state.remote) {
    prototype.opcode = URMA_OPC_READ;
    prototype.flag.bs.comp_order = 1;
    prototype.flag.bs.place_order = URMA_RELAX_ORDER;
    prototype.tjetty = target == nullptr ? worker.target : target;
    prototype.rw.src.num_sge = 1U;
  }
};

void AppendGatherBatch(Worker& worker, const GatherBuildLayout& layout, std::uint32_t& span_index,
                       std::uint32_t last, std::uint32_t& span_offset, std::uint64_t& source_offset,
                       std::uint32_t snapshot_first, std::uint32_t batch_wrs,
                       std::uint32_t& local_index) {
  auto& queue = worker.read_queue;
  const auto* spans = queue.targets.data();
  auto index = span_index - snapshot_first;
  auto offset = span_offset;
  auto source_position = source_offset;
  const auto limit = last - snapshot_first;
  auto* local = queue.local_sges.data();
  auto* remote = queue.remote_sges.data();
  auto* requests = queue.wrs.data();
  auto sequence = worker.sequence;
  auto wr_count = queue.wr_count;
  const auto wr_limit = wr_count + std::min(batch_wrs, kGatherWaveWrs - wr_count);
  while (index < limit && wr_count < wr_limit) {
    const auto wr_index = wr_count++;
    const auto local_begin = local_index;
    std::uint32_t bytes = 0U, local_count = 0U;
    const auto source = layout.staging + source_position;
    while (index < limit && local_count < layout.sgl_size && bytes < 65536U) {
      const auto& span = spans[index];
      const auto length = std::min(span.length - offset, 65536U - bytes);
      local[local_index++] = {layout.output + span.offset + offset, length, layout.local_segment,
                              nullptr};
      ++local_count;
      bytes += length;
      source_position += length;
      offset += length;
      if (offset == span.length) {
        ++index;
        offset = 0U;
      }
    }
    remote[wr_index] = {source, bytes, layout.remote_segment, nullptr};
    auto wr = layout.prototype;
    wr.user_ctx = ++sequence;
    wr.rw.src.sge = reinterpret_cast<urma_sge_t*>(&remote[wr_index]);
    wr.rw.dst.sge = reinterpret_cast<urma_sge_t*>(local + local_begin);
    wr.rw.dst.num_sge = local_count;
    wr.next = requests + wr_index + 1U;
    requests[wr_index] = wr;
  }
  if (wr_count != 0U) {
    auto& tail = requests[wr_count - 1U];
    tail.next = nullptr;
    const bool final = index == limit || wr_count == kGatherWaveWrs;
    tail.flag.bs.place_order = final ? URMA_STRONG_ORDER : URMA_RELAX_ORDER;
    tail.flag.bs.complete_enable = final;
  }
  queue.wr_count = wr_count;
  worker.sequence = sequence;
  span_index = snapshot_first + index;
  span_offset = offset;
  source_offset = source_position;
}

struct GatherPrefix {
  Worker& worker;
  const GatherReadArgs& args;
  GatherReadStatus& gather;
  std::uint32_t worker_index;
  std::uint64_t ready_bytes = 0;
};

bool WaitBatchPrefix(GatherPrefix& prefix, const urma_jfs_wr_t& tail, UmdkControlStatus& submitted,
                     std::uint64_t& ready_wait_ns) {
  const auto& args = prefix.args;
  const auto& source = *reinterpret_cast<const Sge*>(tail.rw.src.sge);
  const auto required = source.addr + source.len - args.remote_staging;
  if (required <= prefix.ready_bytes) return true;
  auto& gather = prefix.gather;
  gather.phase = 5;
  const auto start = NowNs();
  const auto scratch =
      args.output + ((args.output_extent + 63U) & ~std::uint64_t{63U}) + prefix.worker_index * 64U;
  const UmdkTransfer transfer{
      args.remote_descriptor + offsetof(kvaf::peer::GatherDescriptorHeader, ready_bytes), scratch,
      8};
  bool success = false;
  for (std::uint32_t poll = 0; poll < 10000; ++poll) {
    FlushGatherCache(reinterpret_cast<const void*>(scratch), 8);
    UmdkControlStatus query{};
    // The dedicated control Jetty can complete this query while data READs
    // remain outstanding; controls use their own WR storage.
    if (!TransferControl(prefix.worker, false, &transfer, 1, query)) {
      errno = query.error_number;
      Fail(submitted, query.stage, query.result);
      gather.result = 16;
      break;
    }
    FlushGatherCache(reinterpret_cast<const void*>(scratch), 8);
    prefix.ready_bytes = *reinterpret_cast<const volatile std::uint64_t*>(scratch);
    if ((prefix.ready_bytes & kvaf::peer::kGatherErrorBit) != 0 ||
        prefix.ready_bytes > args.bytes) {
      gather.result = 10;
      Fail(submitted, "CPU ready prefix error");
      break;
    }
    if (prefix.ready_bytes >= required) {
      success = true;
      break;
    }
  }
  const auto elapsed = NowNs() - start;
  gather.ready_wait_ns += elapsed;
  ready_wait_ns += elapsed;
  if (!success) {
    gather.ack_seen = prefix.ready_bytes;
    if (gather.result == 0) {
      gather.result = 9;
      errno = ETIMEDOUT;
      Fail(submitted, "waiting for CPU ready prefix timed out", ETIMEDOUT);
    }
  } else
    gather.phase = 4;
  return success;
}

bool ScatterQueue(Worker& worker, const GatherReadArgs& args, GatherReadStatus& status,
                  std::uint32_t first, std::uint32_t last, std::uint64_t remote_offset,
                  UmdkControlStatus& timeline, std::uint32_t timeout_ms,
                  std::uint32_t worker_index = 0) {
  if (first == last) return true;
  const auto snapshot_start = NowNs();
  const auto snapshot_first = first;
  SnapshotGatherTargets(worker.read_queue, args, first, last);
  const GatherBuildLayout layout(worker, args);
  const auto batch_wrs = args.sq_batch_wqes;
  bool first_wave = true;
  GatherPrefix prefix{worker, args, status, worker_index};
  const bool pipeline = args.pipeline_chunk_bytes != 0;
  std::uint32_t span_offset = 0;
  while (first < last) {
    const auto source_begin = remote_offset;
    auto plan_start = first_wave ? snapshot_start : NowNs();
    first_wave = false;
    if (timeline.build_begin_ns == 0) timeline.build_begin_ns = plan_start;
    auto& queue = worker.read_queue;
    BeginGatherWave(queue, layout.sgl_size);
    ++timeline.wr_builds;
    std::uint32_t local_index = 0U;
    UmdkControlStatus submitted{};
    Stage(submitted, "provider post and CQ wait");
    std::uint64_t submit_start = 0U, build_ns = 0U, last_build_ns = 0U;
    bool success = true;
    do {
      const auto batch_first = queue.wr_count;
      AppendGatherBatch(worker, layout, first, last, span_offset, remote_offset, snapshot_first,
                        batch_wrs, local_index);
      const auto build_end = NowNs();
      last_build_ns = build_end;
      build_ns += build_end - plan_start;
      submitted.read_wqes = queue.wr_count;
      if (submit_start == 0U) submit_start = build_end;
      auto* tail = &queue.wrs[queue.wr_count - 1U];
      std::uint64_t ready_wait_ns = 0U;
      if (pipeline && !WaitBatchPrefix(prefix, *tail, submitted, ready_wait_ns)) {
        worker.poisoned = true;
        submitted.post_ns += NowNs() - build_end - ready_wait_ns;
        success = false;
        break;
      }
      if (submitted.first_post_ns == 0U) submitted.first_post_ns = NowNs();
      success = PostBatch(worker, &queue.wrs[batch_first], tail, batch_first,
                          queue.wr_count - batch_first, submitted);
      const auto post_end = NowNs();
      submitted.post_ns += post_end - build_end - ready_wait_ns;
      submitted.last_post_ns = post_end;
      if (!success) break;
      plan_start = post_end;
    } while (first < last && queue.wr_count < kGatherWaveWrs);
    // Only the last WR of this wave requests completion. Retain all workspace
    // slots until that CQE arrives; a later batch never overwrites an earlier one.
    if (success)
      success = WaitCompletion(worker, queue.wrs[queue.wr_count - 1U].user_ctx, submit_start,
                               timeout_ms, submitted);
    timeline.build_end_ns = last_build_ns;
    status.scatter_plan_ns += build_ns;
    status.read_wqes += submitted.read_wqes;
    status.scatter_post_ns += submitted.post_ns;
    status.scatter_cq_ns += submitted.cq_ns;
    status.provider_post_calls += submitted.post_calls;
    status.cq_poll_calls += submitted.cq_poll_calls;
    if (timeline.first_post_ns == 0) timeline.first_post_ns = submitted.first_post_ns;
    timeline.last_post_ns = submitted.last_post_ns;
    timeline.completion_ns = submitted.completion_ns;
    timeline.cq_status = submitted.cq_status;
    timeline.cq_context = submitted.cq_context;
    if (success)
      status.read_bytes += remote_offset - source_begin;
    else {
      errno = submitted.error_number;
      return Fail(timeline, submitted.stage, submitted.result);
    }
  }
  return true;
}

}  // namespace

extern "C" std::uint32_t KvafAicpuUmdkGatherMemory(const GatherReadArgs* args,
                                                   UmdkGatherMemory* memory) {
  std::shared_lock<std::shared_mutex> lock(state_mutex);
  if (args == nullptr || memory == nullptr || args->umdk_session != state.session ||
      args->umdk_owner_pid != static_cast<std::uint64_t>(getpid()))
    return 16;
  std::uint64_t available = 0;
  auto* local = LocalSegment(args->output, args->output_extent, &available);
  if (local == nullptr || state.remote == nullptr) return 16;
  *memory = {args->output, available, state.remote->seg.ubva.va, state.remote->seg.len};
  return 0;
}

extern "C" std::uint32_t KvafAicpuUmdkGatherTransfer(const GatherReadArgs* args, bool write,
                                                     const UmdkTransfer* transfers,
                                                     std::uint32_t count,
                                                     std::uint32_t worker_index,
                                                     std::uint64_t* post_ns, std::uint64_t* cq_ns) {
  std::shared_lock<std::shared_mutex> lock(state_mutex);
  if (args == nullptr || args->umdk_session != state.session ||
      args->umdk_owner_pid != static_cast<std::uint64_t>(getpid()) ||
      worker_index >= state.worker_count)
    return 16;
  if (worker_index >= state.worker_count) return 16;
  std::unique_lock<std::mutex> worker_lock(worker_mutexes[worker_index], std::defer_lock);
  if (!state.control) worker_lock.lock();
  UmdkControlStatus status{};
  const bool success =
      TransferControl(GetWorker(worker_index), write, transfers, count, status, 0U);
  if (post_ns != nullptr) *post_ns = status.post_ns;
  if (cq_ns != nullptr) *cq_ns = status.cq_ns;
  if (!success)
    std::fprintf(
        stderr,
        "GATHER_UMDK_CONTROL_ERROR worker=%u write=%u stage=%s result=%d errno=%d cq_status=%d\n",
        worker_index, write ? 1U : 0U, status.stage, status.result, status.error_number,
        status.cq_status);
  return success ? 0 : 16;
}

extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuUmdkGatherScatter(
    const kvaf::aicpu_urma::GatherReadArgs* args, kvaf::aicpu_urma::GatherReadStatus* status,
    std::uint32_t first, std::uint32_t last, std::uint64_t remote_offset,
    std::uint32_t worker_index) {
  std::shared_lock<std::shared_mutex> lock(state_mutex);
  if (args == nullptr || status == nullptr || worker_index >= state.worker_count ||
      std::max(1U, args->worker_count) != state.worker_count ||
      args->umdk_owner_pid != static_cast<std::uint64_t>(getpid()) ||
      args->umdk_session != state.session ||
      LocalSegment(args->output, args->output_extent) == nullptr || first > last ||
      last > args->output_span_count || args->sgl_size == 0 || args->sq_batch_wqes == 0)
    return 16;
  std::lock_guard<std::mutex> worker_lock(worker_mutexes[worker_index]);
  auto& worker = GetWorker(worker_index);
  if (worker.poisoned || worker.target == nullptr) return 16;
  UmdkControlStatus timeline{};
  if (!ScatterQueue(worker, *args, *status, first, last, remote_offset, timeline, 1000,
                    worker_index)) {
    std::fprintf(stderr,
                 "GATHER_UMDK_READ_ERROR worker=%u stage=%s result=%d errno=%d cq_status=%d "
                 "cq_context=%llu\n",
                 worker_index, timeline.stage, timeline.result, timeline.error_number,
                 timeline.cq_status, static_cast<unsigned long long>(timeline.cq_context));
    status->cqe_word = timeline.cq_status != 0 ? timeline.cq_status : timeline.error_number;
    return status->result != 0 ? static_cast<std::uint32_t>(status->result) : 16;
  }
  return 0;
}

extern "C" __attribute__((visibility("default"))) std::uint32_t KvafAicpuUmdkControl(
    UmdkControlArgs* args) {
  if (!args || !args->status) return 1;
  std::unique_lock<std::shared_mutex> lock(state_mutex);
  auto& status = *reinterpret_cast<UmdkControlStatus*>(args->status);
  status = {};
  status.magic = kUmdkControlMagic;
  status.owner_pid = getpid();
  status.session = state.session;
  diagnostic_log.Clear();
  errno = 0;
  if (args->op == UmdkControlOp::kPrepareGather) {
    if (!args->worker_count || args->worker_count > 8 || !args->blocks ||
        args->blocks > kGatherMaxSpans || !args->bytes || !args->local_buffer ||
        !args->buffer_bytes || args->local_buffer > UINT64_MAX - args->buffer_bytes ||
        args->device[63] || args->library[255])
      Fail(status, "invalid UMDK session arguments");
    else if (state.session != 0) {
      status.session = 0;
      Fail(status, "UMDK session already active");
    } else if (!Prepare(*args, status)) {
      UmdkControlStatus cleanup{};
      if (Cleanup(cleanup)) status.session = 0;
    }
  } else if (args->owner_pid != status.owner_pid || !args->session ||
             args->session != state.session)
    Fail(status, "AICPU process/session changed");
  else
    switch (args->op) {
      case UmdkControlOp::kConnect:
        Connect(*args, status);
        break;
      case UmdkControlOp::kQueryGatherWorker:
        DescribeWorker(*args, args->worker_index, status);
        break;
      case UmdkControlOp::kRegisterGatherBuffer:
        RegisterGatherBuffer(*args, status);
        break;
      case UmdkControlOp::kWriteGather:
        TransferControl(GetWorker(0), true,
                        reinterpret_cast<const UmdkTransfer*>(args->transfer_spans),
                        args->transfer_count, status);
        break;
      case UmdkControlOp::kDestroy:
        Cleanup(status);
        break;
      default:
        Fail(status, "unknown UMDK operation");
        break;
    }
  diagnostic_log.Publish(*args, status);
  if (!status.result) std::snprintf(status.stage, sizeof(status.stage), "OK");
  return 0;
}
