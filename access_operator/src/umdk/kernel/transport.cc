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

#include "transport_internal.h"
#include "umdk/cache.h"
#include "urma_transport.h"

namespace kvaf::aicpu_urma::detail {

std::uint64_t NowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

State state;
std::shared_mutex state_mutex;
std::array<std::mutex, 8> worker_mutexes;
std::array<std::mutex, 8> control_mutexes;
std::uint64_t generation = 0;

Worker& GetWorker(std::uint32_t index) {
  return index == 0 ? static_cast<Worker&>(state) : *state.extra_workers[index - 1];
}

Worker* SetupQueue(std::uint32_t index) {
  if (index < state.worker_count) return &GetWorker(index);
  if (index == kUmdkControlQueueIndex) return state.control.get();
  if (index > kUmdkControlQueueIndex && index < kUmdkControlQueueIndex + state.worker_count)
    return state.extra_controls[index - kUmdkControlQueueIndex - 1U].get();
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
  for (auto& control : state.extra_controls)
    if (control && !CleanupWorker(*control, status)) return false;
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
    if (index != 0) {
      state.extra_controls[index - 1] = std::make_unique<Worker>();
      if (!CreateWorker(args, status, *state.extra_controls[index - 1])) return false;
    }
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
                                std::uint64_t* available) {
  if (state.local != nullptr && UmdkRange(state.buffer, state.buffer_bytes, address, bytes)) {
    if (available != nullptr) *available = state.buffer_bytes - (address - state.buffer);
    return state.local;
  }
  for (const auto& buffer : state.buffers)
    if (UmdkRange(buffer.base, buffer.bytes, address, bytes)) {
      if (available != nullptr) *available = buffer.bytes - (address - buffer.base);
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

bool BuildTransfers(Worker& worker, bool write, const UmdkTransfer* transfers, std::uint32_t count,
                    std::uint32_t& transfer_index, std::uint64_t& offset,
                    UmdkControlStatus& status) {
  worker.local_sges.clear();
  worker.remote_sges.clear();
  worker.wrs.clear();
  while (transfer_index < count && worker.wrs.size() < 4096) {
    const auto& transfer = transfers[transfer_index];
    if (transfer.bytes == 0 || offset >= transfer.bytes ||
        !UmdkRange(state.remote->seg.ubva.va, state.remote->seg.len, transfer.remote,
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

bool TransferQueue(Worker& worker, bool write, const UmdkTransfer* transfers, std::uint32_t count,
                   UmdkControlStatus& status) {
  if (worker.poisoned || worker.target == nullptr || state.remote == nullptr ||
      transfers == nullptr || count == 0 || count > kUmdkMaxTransfers)
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
                     UmdkControlStatus& status, std::uint32_t worker_index) {
  auto* control = SetupQueue(kUmdkControlQueueIndex + worker_index);
  if (!control) return Fail(status, "control queue is unavailable");
  std::lock_guard<std::mutex> control_lock(control_mutexes[worker_index]);
  return TransferQueue(*control, write, transfers, count, status);
}

}  // namespace kvaf::aicpu_urma::detail

namespace kvaf::aicpu_urma {
using namespace detail;
std::uint32_t UmdkGetMemory(const UmdkContext* args, std::uint64_t address, std::uint64_t bytes,
                            UmdkMemory* memory) {
  std::shared_lock<std::shared_mutex> lock(state_mutex);
  if (args == nullptr || memory == nullptr || args->session != state.session ||
      args->owner_pid != static_cast<std::uint64_t>(getpid()))
    return 16;
  std::uint64_t available = 0;
  auto* local = LocalSegment(address, bytes, &available);
  if (local == nullptr || state.remote == nullptr) return 16;
  *memory = {address, available, state.remote->seg.ubva.va, state.remote->seg.len};
  return 0;
}

std::uint32_t UmdkTransferSync(const UmdkContext* args, bool write, const UmdkTransfer* transfers,
                               std::uint32_t count, std::uint32_t worker_index) {
  std::shared_lock<std::shared_mutex> lock(state_mutex);
  if (args == nullptr || args->session != state.session ||
      args->owner_pid != static_cast<std::uint64_t>(getpid()) || worker_index >= state.worker_count)
    return 16;
  UmdkControlStatus status{};
  const bool success = TransferControl(write, transfers, count, status, worker_index);
  if (!success)
    std::fprintf(stderr,
                 "UMDK_CONTROL_ERROR worker=%u write=%u stage=%s result=%d errno=%d cq_status=%d\n",
                 worker_index, write ? 1U : 0U, status.stage, status.result, status.error_number,
                 status.cq_status);
  return success ? 0 : 16;
}

}  // namespace kvaf::aicpu_urma

using namespace kvaf::aicpu_urma;
using namespace kvaf::aicpu_urma::detail;
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
  if (args->op == UmdkControlOp::kPrepare) {
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
      case UmdkControlOp::kQueryWorker:
        DescribeWorker(*args, args->worker_index, status);
        break;
      case UmdkControlOp::kRegisterBuffer:
        RegisterBuffer(*args, status);
        break;
      case UmdkControlOp::kWrite:
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
