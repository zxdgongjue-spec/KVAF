#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include "aicpu_urma/umdk_abi.h"

namespace {
using namespace kvaf::aicpu_urma::umdk;
urma_device_t device{};
urma_device_t* devices[] = {&device};
urma_eid_info_t eid{};
urma_context_t context{};
struct MockCq {
  urma_jfc_t handle{};
  urma_cr_t pending{};
  bool has_pending = false;
  int empty_polls = 0;
  std::atomic<bool> active{false};
};
// A lock-free provider requires callers to serialize access to each queue.
struct QueueAccess {
  MockCq& cq;
  bool acquired;
  explicit QueueAccess(MockCq& queue) : cq(queue), acquired(!cq.active.exchange(true)) {}
  ~QueueAccess() {
    if (acquired) cq.active.store(false);
  }
};
struct MockJetty {
  urma_jetty_t handle{};
  std::uint64_t barrier_generation = 0;
};
struct MockTarget {
  urma_target_jetty_t handle{};
  std::uint64_t tp = 0;
};
urma_target_seg_t remote{};
std::vector<urma_target_seg_t*> local_segments;
std::atomic<int> live{0};
std::atomic<int> post_calls{0}, submitted_wqes{0}, completion_requests{0}, next_jetty_id{17};
std::uint64_t next_tp = 42;
bool shared_tp = false;
int next_cq_id = 101, empty_polls = 0;
std::mutex barrier_mutex;
std::condition_variable barrier_condition;
std::uint64_t barrier_generation = 0;
int barrier_workers = 0, barrier_arrivals = 0;
std::uint64_t remote_base = 0, remote_bytes = 0;
urma_log_cb_t log_callback = nullptr;
urma_vlog_level_t log_level = URMA_VLOG_LEVEL_INFO;
void (*read_observer)(std::uint64_t, std::uint32_t, bool) = nullptr;
void (*post_observer)(std::uint32_t, std::uint32_t, const urma_jfs_wr_t*) = nullptr;

void Log(int level, const char* message) {
  if (log_callback != nullptr) log_callback(level, const_cast<char*>(message));
}

bool Failure(const char* stage) {
  const auto* requested = std::getenv("KVAF_UMDK_MOCK_FAIL");
  if (requested != nullptr && std::strcmp(requested, stage) == 0) {
    errno = EACCES;
    return true;
  }
  return false;
}
template <typename Value>
Value* Create(Value& value, const char* stage) {
  if (Failure(stage)) return nullptr;
  ++live;
  return &value;
}

bool CheckSgl(const urma_sg_t& list, urma_target_seg_t* segment, std::uint64_t base,
              std::uint64_t capacity, std::uint64_t& total) {
  const auto* entries = reinterpret_cast<const Sge*>(list.sge);
  if (entries == nullptr || list.num_sge == 0 || list.num_sge > 6) return false;
  for (std::uint32_t index = 0; index < list.num_sge; ++index) {
    const auto& entry = entries[index];
    const auto* expected = segment;
    if (expected == nullptr) {
      if (std::find(local_segments.begin(), local_segments.end(), entry.tseg) ==
          local_segments.end())
        return false;
      expected = entry.tseg;
      base = expected->seg.ubva.va;
      capacity = expected->seg.len;
    }
    if (entry.tseg != expected || entry.addr < base || entry.len == 0 || entry.len > capacity ||
        entry.addr - base > capacity - entry.len)
      return false;
    total += entry.len;
  }
  return true;
}

bool CopyWr(urma_jetty_t* jetty, urma_jfs_wr_t& wr) {
  const bool read = wr.opcode == URMA_OPC_READ;
  if (!read && wr.opcode != URMA_OPC_WRITE) return false;
  const auto& local_sg = read ? wr.rw.dst : wr.rw.src;
  const auto& remote_sg = read ? wr.rw.src : wr.rw.dst;
  std::uint64_t source_bytes = 0, destination_bytes = 0;
  if (wr.tjetty == nullptr || !wr.flag.bs.comp_order || remote_sg.num_sge != 1 ||
      local_sg.num_sge > jetty->jetty_cfg.jfs_cfg.max_sge ||
      (wr.flag.bs.complete_enable && wr.flag.bs.place_order != URMA_STRONG_ORDER) ||
      !CheckSgl(local_sg, nullptr, 0, 0, source_bytes) ||
      !CheckSgl(remote_sg, &remote, remote_base, remote_bytes, destination_bytes) ||
      source_bytes != destination_bytes)
    return false;
  const auto tp = reinterpret_cast<MockTarget*>(wr.tjetty)->tp;
  if (tp < 42U || tp >= next_tp) return false;
  const auto* source = reinterpret_cast<const Sge*>(wr.rw.src.sge);
  const auto* destination = reinterpret_cast<const Sge*>(wr.rw.dst.sge);
  if (read && read_observer != nullptr)
    read_observer(source[0].addr, source[0].len, wr.flag.bs.complete_enable != 0);
  std::uint32_t source_index = 0, destination_index = 0, source_offset = 0, destination_offset = 0;
  while (source_index < wr.rw.src.num_sge && destination_index < wr.rw.dst.num_sge) {
    const auto bytes = std::min(source[source_index].len - source_offset,
                                destination[destination_index].len - destination_offset);
    const auto source_address = source[source_index].addr + source_offset;
    const auto destination_address = destination[destination_index].addr + destination_offset;
    if (!read && bytes == 8 && destination_address % 8 == 0) {
      std::uint64_t value = 0;
      std::memcpy(&value, reinterpret_cast<const void*>(source_address), 8);
      __atomic_store_n(reinterpret_cast<std::uint64_t*>(destination_address), value,
                       __ATOMIC_RELEASE);
    } else if (read && bytes >= 8 && source_address % 8 == 0) {
      const auto value =
          __atomic_load_n(reinterpret_cast<const std::uint64_t*>(source_address), __ATOMIC_ACQUIRE);
      std::memcpy(reinterpret_cast<void*>(destination_address), &value, 8);
      std::memcpy(reinterpret_cast<void*>(destination_address + 8),
                  reinterpret_cast<const void*>(source_address + 8), bytes - 8);
    } else {
      std::memcpy(reinterpret_cast<void*>(destination_address),
                  reinterpret_cast<const void*>(source_address), bytes);
    }
    source_offset += bytes;
    destination_offset += bytes;
    if (source_offset == source[source_index].len) {
      ++source_index;
      source_offset = 0;
    }
    if (destination_offset == destination[destination_index].len) {
      ++destination_index;
      destination_offset = 0;
    }
  }
  return true;
}
}  // namespace

extern "C" {
int kvaf_umdk_mock_live_objects() { return live; }
int kvaf_umdk_mock_logging_active() { return log_callback != nullptr; }
int kvaf_umdk_mock_post_calls() { return post_calls; }
int kvaf_umdk_mock_submitted_wqes() { return submitted_wqes; }
int kvaf_umdk_mock_completion_requests() { return completion_requests; }
int kvaf_umdk_mock_created_jettys() { return next_jetty_id - 17; }
int kvaf_umdk_mock_tp_queries() { return next_tp - 42; }
void kvaf_umdk_mock_read_observer(void (*observer)(std::uint64_t, std::uint32_t, bool)) {
  read_observer = observer;
}
void kvaf_umdk_mock_post_observer(void (*observer)(std::uint32_t, std::uint32_t,
                                                   const urma_jfs_wr_t*)) {
  post_observer = observer;
}
void kvaf_umdk_mock_shared_tp(int enabled) { shared_tp = enabled != 0; }
void kvaf_umdk_mock_empty_polls(int count) { empty_polls = count; }
void kvaf_umdk_mock_parallel_barrier(int workers) {
  std::lock_guard<std::mutex> lock(barrier_mutex);
  ++barrier_generation;
  barrier_workers = workers;
  barrier_arrivals = 0;
}
urma_status_t urma_register_log_func(urma_log_cb_t callback) {
  log_callback = callback;
  return URMA_SUCCESS;
}
urma_status_t urma_unregister_log_func() {
  log_callback = nullptr;
  return URMA_SUCCESS;
}
urma_vlog_level_t urma_log_get_level() { return log_level; }
void urma_log_set_level(urma_vlog_level_t level) { log_level = level; }
urma_status_t urma_init(urma_init_attr_t*) {
  Log(6, "MOCK_PROVIDER_INIT");
  if (Failure("borrowed_init")) return static_cast<urma_status_t>(EEXIST);
  if (Failure("init")) return static_cast<urma_status_t>(EACCES);
  ++live;
  return URMA_SUCCESS;
}
urma_status_t urma_uninit() {
  --live;
  return URMA_SUCCESS;
}
urma_device_t** urma_get_device_list(int* count) {
  if (Failure("devices_enodev")) {
    *count = 0;
    Log(3, "MOCK_NO_MATCHING_PROVIDER");
    errno = ENODEV;
    return nullptr;
  }
  if (Failure("empty_list")) {
    *count = 0;
    return devices;
  }
  *count = 1;
  std::strcpy(device.name, "mock_ub");
  device.type = Failure("non_ub_device") ? URMA_TRANSPORT_IB : URMA_TRANSPORT_UB;
  std::strcpy(device.path, "/dev/uburma/mock_ub");
  errno = EINVAL;
  return Failure("devices") ? nullptr : devices;
}
void urma_free_device_list(urma_device_t**) {}
urma_device_t* urma_get_device_by_name(char* name) {
  return std::strcmp(name, device.name) == 0 ? &device : nullptr;
}
urma_eid_info_t* urma_get_eid_list(urma_device_t*, std::uint32_t* count) {
  *count = 1;
  eid.eid_index = 3;
  eid.eid.raw[15] = 7;
  return &eid;
}
void urma_free_eid_list(urma_eid_info_t*) {}
urma_context_t* urma_create_context(urma_device_t*, std::uint32_t index) {
  if (index != 3) return nullptr;
  context.eid = eid.eid;
  post_calls = submitted_wqes = completion_requests = 0;
  next_jetty_id = 17;
  next_tp = 42;
  next_cq_id = 101;
  return Create(context, "context");
}
urma_status_t urma_delete_context(urma_context_t*) {
  --live;
  return URMA_SUCCESS;
}
urma_jfc_t* urma_create_jfc(urma_context_t*, urma_jfc_cfg_t* cfg) {
  if (Failure("jfc")) return nullptr;
  ++live;
  auto* cq = new MockCq;
  cq->handle.jfc_cfg = *cfg;
  cq->handle.jfc_id.id = next_cq_id++;
  return &cq->handle;
}
urma_status_t urma_delete_jfc(urma_jfc_t* cq) {
  delete reinterpret_cast<MockCq*>(cq);
  --live;
  return URMA_SUCCESS;
}
urma_jfr_t* urma_create_jfr(urma_context_t*, urma_jfr_cfg_t*) {
  if (Failure("jfr")) return nullptr;
  ++live;
  return new urma_jfr_t{};
}
urma_status_t urma_delete_jfr(urma_jfr_t* jfr) {
  delete jfr;
  --live;
  return URMA_SUCCESS;
}
urma_jetty_t* urma_create_jetty(urma_context_t*, urma_jetty_cfg_t* cfg) {
  if (Failure("jetty") || (next_jetty_id == 18 && Failure("second_jetty")) ||
      (next_jetty_id == 19 && Failure("third_jetty")))
    return nullptr;
  auto* jetty = new MockJetty;
  jetty->handle.jetty_cfg = *cfg;
  jetty->handle.jetty_id.id = next_jetty_id++;
  ++live;
  return &jetty->handle;
}
urma_status_t urma_delete_jetty(urma_jetty_t* jetty) {
  delete reinterpret_cast<MockJetty*>(jetty);
  --live;
  return URMA_SUCCESS;
}
urma_token_id_t* urma_alloc_token_id(urma_context_t*) {
  if (Failure("token")) return nullptr;
  ++live;
  return new urma_token_id_t{};
}
urma_status_t urma_free_token_id(urma_token_id_t* token) {
  delete token;
  --live;
  return URMA_SUCCESS;
}
urma_target_seg_t* urma_register_seg(urma_context_t*, urma_seg_cfg_t* cfg) {
  if (cfg->flag.value != 0x29c1U) {
    Log(3, "MOCK_LOCAL_HBM_REJECTED_REGISTRATION_FLAGS");
    errno = EINVAL;
    return nullptr;
  }
  if (Failure("register")) {
    Log(3, "MOCK_LOCAL_HBM_REGISTRATION_DRIVER_ERROR");
    return nullptr;
  }
  auto* segment = new urma_target_seg_t{};
  segment->seg.ubva.va = cfg->va;
  segment->seg.len = cfg->len;
  local_segments.push_back(segment);
  ++live;
  return segment;
}
urma_status_t urma_unregister_seg(urma_target_seg_t* segment) {
  local_segments.erase(std::find(local_segments.begin(), local_segments.end(), segment));
  delete segment;
  --live;
  return URMA_SUCCESS;
}
urma_target_seg_t* urma_import_seg(urma_context_t*, urma_seg_t* segment, urma_token_t*,
                                   std::uint64_t, urma_import_seg_flag_t) {
  remote_base = segment->ubva.va;
  remote_bytes = segment->len;
  remote.seg = *segment;
  return Create(remote, "import_seg");
}
urma_status_t urma_unimport_seg(urma_target_seg_t*) {
  --live;
  return URMA_SUCCESS;
}
urma_status_t urma_unimport_jetty(urma_target_jetty_t* target) {
  delete reinterpret_cast<MockTarget*>(target);
  --live;
  return URMA_SUCCESS;
}
int urma_get_tp_list(urma_context_t*, TpQuery* query, std::uint32_t* count, TpInfo* info) {
  if (Failure("tp")) return -EACCES;
  if (next_tp == 43 && Failure("second_tp")) return -EACCES;
  if (next_tp == 44 && Failure("third_tp")) return -EACCES;
  if (query->flag != 1 || query->trans_mode != URMA_TM_RM) return -EINVAL;
  *count = 1;
  info->tp_handle = shared_tp ? 42 : next_tp;
  ++next_tp;
  return 0;
}
urma_target_jetty_t* urma_import_jetty_ex(urma_context_t*, RemoteJettyEx* imported, urma_token_t*,
                                          ActiveTp* active) {
  if (imported->tp_type != 1 || active->tp_handle < 42 || active->tp_handle >= next_tp ||
      active->tx_psn != 1 || active->peer_tp_handle != 43 || active->rx_psn != 5)
    return nullptr;
  if (Failure("import_jetty")) return nullptr;
  if (active->tp_handle == 43 && Failure("second_import_jetty")) return nullptr;
  ++live;
  auto* target = new MockTarget;
  target->tp = active->tp_handle;
  target->handle.id = imported->jetty_id;
  return &target->handle;
}
urma_status_t urma_post_jetty_send_wr(urma_jetty_t* jetty, urma_jfs_wr_t* wr, urma_jfs_wr_t** bad) {
  ++post_calls;
  {
    std::unique_lock<std::mutex> lock(barrier_mutex);
    auto& queue_generation = reinterpret_cast<MockJetty*>(jetty)->barrier_generation;
    if (barrier_workers != 0 && queue_generation != barrier_generation) {
      queue_generation = barrier_generation;
      ++barrier_arrivals;
      barrier_condition.notify_all();
      if (!barrier_condition.wait_for(lock, std::chrono::seconds(2),
                                      [] { return barrier_arrivals == barrier_workers; }))
        return static_cast<urma_status_t>(EDEADLK);
    }
  }
  if (Failure("post")) return static_cast<urma_status_t>(EACCES);
  auto& cq = *reinterpret_cast<MockCq*>(jetty->jetty_cfg.jfs_cfg.jfc);
  QueueAccess access(cq);
  if (!access.acquired) {
    errno = EBUSY;
    return static_cast<urma_status_t>(EBUSY);
  }
  if (post_observer != nullptr) post_observer(jetty->jetty_id.id, cq.handle.jfc_id.id, wr);
  for (std::uint32_t index = 0; wr != nullptr; wr = wr->next, ++index) {
    if (index >= 8192 || !CopyWr(jetty, *wr)) {
      *bad = wr;
      return static_cast<urma_status_t>(EINVAL);
    }
    ++submitted_wqes;
    if (Failure("post_partial")) {
      *bad = wr->next;
      return static_cast<urma_status_t>(EACCES);
    }
    if (wr->flag.bs.complete_enable) {
      ++completion_requests;
      cq.pending = {};
      cq.pending.status = URMA_CR_SUCCESS;
      cq.pending.user_ctx = wr->user_ctx;
      cq.has_pending = true;
      cq.empty_polls = empty_polls;
    }
  }
  return URMA_SUCCESS;
}
int urma_poll_jfc(urma_jfc_t* handle, int count, urma_cr_t* completion) {
  auto& cq = *reinterpret_cast<MockCq*>(handle);
  QueueAccess access(cq);
  if (!access.acquired) return -EBUSY;
  if (Failure("poll")) return -EACCES;
  if (Failure("timeout") || !cq.has_pending) return 0;
  if (cq.empty_polls > 0) {
    --cq.empty_polls;
    return 0;
  }
  if (count != 1) return -EINVAL;
  *completion = cq.pending;
  cq.has_pending = false;
  if (Failure("cq_status")) completion->status = static_cast<urma_cr_status_t>(1);
  if (Failure("cq_context")) ++completion->user_ctx;
  return 1;
}
}
