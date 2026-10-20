#include <dlfcn.h>
#include <unistd.h>
#if defined(__linux__)
#include <link.h>
#endif

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
#include "umdk_control.h"
#include "umdk_gather.h"
#include "urma_transport.h"

namespace {
using namespace kvaf::aicpu_urma;
using namespace kvaf::urma;

std::uint64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

#define KVAF_UMDK_APIS(APPLY)    \
  APPLY(urma_init)               \
  APPLY(urma_uninit)             \
  APPLY(urma_get_device_list)    \
  APPLY(urma_free_device_list)   \
  APPLY(urma_get_eid_list)       \
  APPLY(urma_free_eid_list)      \
  APPLY(urma_create_context)     \
  APPLY(urma_delete_context)     \
  APPLY(urma_create_jfc)         \
  APPLY(urma_delete_jfc)         \
  APPLY(urma_create_jfr)         \
  APPLY(urma_delete_jfr)         \
  APPLY(urma_create_jetty)       \
  APPLY(urma_delete_jetty)       \
  APPLY(urma_alloc_token_id)     \
  APPLY(urma_free_token_id)      \
  APPLY(urma_register_seg)       \
  APPLY(urma_unregister_seg)     \
  APPLY(urma_import_seg)         \
  APPLY(urma_unimport_seg)       \
  APPLY(urma_unimport_jetty)     \
  APPLY(urma_post_jetty_send_wr) \
  APPLY(urma_poll_jfc)

struct Api {
#define KVAF_DECLARE_API(name) decltype(&::name) name = nullptr;
  KVAF_UMDK_APIS(KVAF_DECLARE_API)
#undef KVAF_DECLARE_API
  GetTpList get_tp_list = nullptr;
  ImportJettyEx import_jetty_ex = nullptr;
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
  urma_context_t* context = nullptr;
  urma_token_id_t* token = nullptr;
  urma_target_seg_t* local = nullptr;
  urma_target_seg_t* remote = nullptr;
  std::uint64_t session = 0;
  std::uint64_t shared_tp = 0, shared_peer_tp = 0;
  std::uint32_t shared_peer_psn = 0;
  std::array<std::uint8_t, 16> peer_eid{};
  std::uint64_t buffer = 0, buffer_bytes = 0, remote_address = 0;
  std::uint32_t worker_count = 1;
  std::array<std::unique_ptr<Worker>, 7> extra_workers;
  std::unique_ptr<Worker> control;
  std::vector<RegisteredBuffer> buffers;
};

State state;
std::shared_mutex state_mutex;
std::array<std::mutex, 8> worker_mutexes;
std::mutex control_mutex;
std::uint64_t generation = 0;

Worker& GetWorker(std::uint32_t index) {
  return index == 0 ? static_cast<Worker&>(state) : *state.extra_workers[index - 1];
}

Worker* SetupQueue(std::uint32_t index) {
  if (index < state.worker_count) return &GetWorker(index);
  if (index == kUmdkControlQueueIndex) return state.control.get();
  return nullptr;
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
      if (slash != std::string::npos) bundled = path.substr(0, slash) + "/umdk/";
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

bool CreateWorker(const UmdkControlArgs& args, UmdkControlStatus& status, Worker& worker) {
  const auto* error =
      kvaf::urma::CreateQueues(state.api, state.context, args.token, {64U, 64U, 6U, true},
                               worker.jfc, worker.jfr, worker.jetty);
  return error ? Fail(status, error) : true;
}

bool DescribeWorker(const UmdkControlArgs& args, std::uint32_t index, UmdkControlStatus& status) {
  auto* selected = SetupQueue(index);
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
  const bool enumerated = kvaf::urma::EnumerateEndpoints(
      state.api, args.device, args.eid_index, [&](auto* device, const auto& eid) {
        state.context = state.api.urma_create_context(device, eid.eid_index);
        if (!state.context) {
          Fail(status, "urma_create_context");
          return true;
        }
        std::memcpy(status.eid, state.context->eid.raw, 16);
        return true;
      });
  if (!enumerated) return Fail(status, "urma_get_device_list");
  if (!state.context) return status.result ? false : Fail(status, "no usable UB EID inside AICPU");
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

  Stage(status, "urma_register_seg (local HBM)");
  state.local = state.api.urma_register_seg(state.context, &cfg);
  if (state.local == nullptr) {
    Fail(status, "urma_register_seg (local HBM)");

    return false;
  }
  state.buffer = args.local_buffer;
  state.buffer_bytes = args.buffer_bytes;
  if (state.control && !DescribeWorker(args, kUmdkControlQueueIndex, status)) return false;
  if (!DescribeWorker(args, 0, status)) return false;
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
  if (args.remote_address < segment.ubva.va ||
      args.remote_address - segment.ubva.va >= segment.len)
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

bool PostBatch(Worker& worker, urma_jfs_wr_t* first, urma_jfs_wr_t* tail, std::uint32_t first_index,
               std::uint32_t count, UmdkControlStatus& status) {
  auto* saved_next = tail->next;
  tail->next = nullptr;
  urma_jfs_wr_t* bad = nullptr;
  errno = 0;
  const int result = state.api.urma_post_jetty_send_wr(worker.jetty, first, &bad);
  tail->next = saved_next;
  if (result == 0) return true;
  worker.poisoned = true;

  return Fail(status, "urma_post_jetty_send_wr", result);
}

bool WaitCompletion(Worker& worker, std::uint64_t context, std::uint64_t start,
                    std::uint32_t timeout_ms, UmdkControlStatus& status) {
  const auto deadline = start + static_cast<std::uint64_t>(timeout_ms) * 1000000;
  std::uint32_t polls = 0;
  auto now = start;
  while (now < deadline) {
    urma_cr_t completion{};
    const int count = state.api.urma_poll_jfc(worker.jfc, 1, &completion);
    if (count < 0) {
      Fail(status, "urma_poll_jfc", count);
      break;
    }
    if (count) {
      status.cq_status = completion.status;
      status.cq_context = completion.user_ctx;
      if (completion.status == URMA_CR_SUCCESS && completion.user_ctx == context) return true;
      Fail(status, "READ/WRITE CQE status or user_ctx mismatch");
      break;
    }
    if (++polls % 64U == 0U) now = NowNs();
  }
  worker.poisoned = true;
  if (!status.result) {
    errno = ETIMEDOUT;
    Fail(status, "waiting for CQE timed out", ETIMEDOUT);
  }
  return false;
}

template <typename RequestAt>
bool SubmitRequests(Worker& worker, std::uint32_t count, std::uint32_t batch,
                    std::uint32_t timeout_ms, UmdkControlStatus& status, RequestAt request_at) {
  Stage(status, "provider post and CQ wait");
  const auto start = NowNs();
  for (std::uint32_t first = 0; first < count; first += batch) {
    const auto end = std::min(first + batch, count);
    if (!PostBatch(worker, request_at(first), request_at(end - 1U), first, end - first, status)) {
      return false;
    }
  }
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
    if (!success) {
      status.cq_status = submitted.cq_status;
      status.cq_context = submitted.cq_context;
      errno = submitted.error_number;
      return Fail(status, submitted.stage, submitted.result);
    }
  }
  return true;
}

// The control Jetty has its own lock and never polls a data CQ.
bool TransferControl(bool write, const UmdkTransfer* transfers, std::uint32_t count,
                     UmdkControlStatus& status) {
  auto* control = SetupQueue(kUmdkControlQueueIndex);
  if (!control) return Fail(status, "control queue is unavailable");
  std::lock_guard<std::mutex> control_lock(control_mutex);
  return TransferGather(*control, write, transfers, count, status);
}

constexpr std::uint32_t kGatherWaveWrs = 4096U;

void BeginGatherWave(GatherQueue& queue, std::uint32_t sgl_size) {
  EnsureSlots(queue.local_sges, kGatherWaveWrs * std::min(sgl_size, kvaf::kGatherMaxSglSize));
  EnsureSlots(queue.remote_sges, kGatherWaveWrs);
  EnsureSlots(queue.wrs, kGatherWaveWrs);
  queue.wr_count = 0U;
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
  const GatherReadArgs& args;
  GatherReadStatus& gather;
  std::uint32_t worker_index;
  std::uint64_t ready_bytes = 0;
};

bool WaitBatchPrefix(GatherPrefix& prefix, const urma_jfs_wr_t& tail,
                     UmdkControlStatus& submitted) {
  const auto& args = prefix.args;
  const auto& source = *reinterpret_cast<const Sge*>(tail.rw.src.sge);
  const auto required = source.addr + source.len - args.remote_staging;
  if (required <= prefix.ready_bytes) return true;
  auto& gather = prefix.gather;
  gather.phase = 5;
  const auto scratch = args.scratch + prefix.worker_index * 64U;
  const UmdkTransfer transfer{
      args.remote_descriptor + offsetof(kvaf::peer::GatherDescriptorHeader, ready_bytes), scratch,
      8};
  bool success = false;
  for (std::uint32_t poll = 0; poll < 10000; ++poll) {
    FlushGatherCache(reinterpret_cast<const void*>(scratch), 8);
    UmdkControlStatus query{};
    // The dedicated control Jetty can complete this query while data READs
    // remain outstanding; controls use their own WR storage.
    if (!TransferControl(false, &transfer, 1, query)) {
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
                  UmdkControlStatus& error, std::uint32_t timeout_ms,
                  std::uint32_t worker_index = 0) {
  if (first == last) return true;
  const auto snapshot_first = first;
  SnapshotGatherTargets(worker.read_queue, args, first, last);
  const GatherBuildLayout layout(worker, args);
  GatherPrefix prefix{args, status, worker_index};
  std::uint32_t span_offset = 0;
  while (first < last) {
    auto& queue = worker.read_queue;
    BeginGatherWave(queue, layout.sgl_size);
    std::uint32_t local_index = 0;
    const auto start = NowNs();  // Deadline only; no per-stage profiling.
    do {
      const auto batch_first = queue.wr_count;
      AppendGatherBatch(worker, layout, first, last, span_offset, remote_offset, snapshot_first,
                        args.sq_batch_wqes, local_index);
      auto* tail = &queue.wrs[queue.wr_count - 1U];
      if ((args.pipeline_chunk_bytes && !WaitBatchPrefix(prefix, *tail, error)) ||
          !PostBatch(worker, &queue.wrs[batch_first], tail, batch_first,
                     queue.wr_count - batch_first, error)) {
        worker.poisoned = true;
        return false;
      }
    } while (first < last && queue.wr_count < kGatherWaveWrs);
    // Retain every WR/SGL slot until the final ordered CQE for this wave.
    if (!WaitCompletion(worker, queue.wrs[queue.wr_count - 1U].user_ctx, start, timeout_ms, error))
      return false;
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
                                                     std::uint32_t worker_index) {
  std::shared_lock<std::shared_mutex> lock(state_mutex);
  if (args == nullptr || args->umdk_session != state.session ||
      args->umdk_owner_pid != static_cast<std::uint64_t>(getpid()) ||
      worker_index >= state.worker_count)
    return 16;
  UmdkControlStatus status{};
  const bool success = TransferControl(write, transfers, count, status);
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

  errno = 0;
  if (args->op == UmdkControlOp::kPrepareGather) {
    if (!args->worker_count || args->worker_count > 8 || !args->local_buffer ||
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
        TransferControl(true, reinterpret_cast<const UmdkTransfer*>(args->transfer_spans),
                        args->transfer_count, status);
        break;
      case UmdkControlOp::kDestroy:
        Cleanup(status);
        break;
      default:
        Fail(status, "unknown UMDK operation");
        break;
    }

  if (!status.result) std::snprintf(status.stage, sizeof(status.stage), "OK");
  return 0;
}
