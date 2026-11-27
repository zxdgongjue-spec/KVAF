#pragma once
#include <array>
#include <atomic>
#include <mutex>
#include <map>
#include <memory>
#include <shared_mutex>
#include <vector>

#include "aicpu_urma/transport.h"
#include "urma_transport.h"
#include "remote_memory_binding.h"
namespace kvaf::aicpu_urma::detail {
using namespace kvaf::urma;
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

// Process-local provider environment. Sessions borrow its immutable function
// table and keep it alive until every queue, registration and context is gone.
struct ApiEnvironment {
  std::mutex mutex;
  Api api;
  void* library = nullptr;
  void* common_library = nullptr;
  std::shared_ptr<ApiEnvironment> redirect;
  std::size_t references = 0;
  bool loaded = false, owns_init = false, initialized = false;
  bool failed = false, retiring = false;
};

struct RequestBuffer {
  std::vector<Sge> local_sges;
  std::vector<Sge> remote_sges;
  std::vector<urma_jfs_wr_t> wrs;
  std::uint32_t wr_count = 0;
};

struct PeerRoute {
  UmdkRemoteMemoryBinding remote;
  std::uint64_t tp{}, peer_tp{};
  std::uint32_t peer_psn{};
};

struct Queue {
  PeerRoute* route{};
  RequestBuffer requests;
  // Stable storage for one-WR control transfers, including a failed in-flight WR.
  Sge single_local{}, single_remote{};
  urma_jfs_wr_t single_wr{};
  bool poisoned = false;
  urma_jfc_t* jfc = nullptr;
  urma_jfr_t* jfr = nullptr;
  urma_jetty_t* jetty = nullptr;
  urma_target_jetty_t* target = nullptr;
  std::uint64_t tp = 0;
  std::uint64_t sequence = 0;
};

struct RegisteredBuffer {
  std::uint64_t base, bytes;
  urma_token_id_t* token;
  urma_target_seg_t* segment;
};

struct SessionState {
  // A cold-path copy avoids adding a pointer chase to each WR/CQ operation.
  // Ownership and dynamic symbol resolution belong only to environment.
  Api api;
  std::shared_ptr<ApiEnvironment> environment;
  urma_context_t* context = nullptr;
  urma_token_id_t* token = nullptr;
  urma_target_seg_t* local = nullptr;
  std::map<std::array<std::uint8_t, 16>, PeerRoute> routes;
  std::uint64_t session = 0;
  std::uint32_t data_jettys = 1;
  std::uint64_t buffer = 0, buffer_bytes = 0;
  std::uint32_t worker_count = 1;
  UmdkQueueProfile queue_profile{UmdkQueueProfile::Gather};
  std::vector<Queue> data_queues;
  std::array<Queue, kUmdkControlQueueIndex> control_queues;
  std::vector<RegisteredBuffer> buffers;
};

struct ExecutionSession {
  SessionState state;
  std::shared_mutex mutex;
  std::array<std::mutex, 8> worker_mutexes, control_mutexes;
  std::atomic<unsigned> next_worker{0};
};
std::shared_ptr<ExecutionSession> CreateSession();
std::shared_ptr<ExecutionSession> FindSession(std::uint64_t id);
void RemoveSession(std::uint64_t id);
extern thread_local ExecutionSession* active_session;
inline SessionState& CurrentState() { return active_session->state; }
struct SessionScope {
  ExecutionSession* previous;
  explicit SessionScope(ExecutionSession& session) : previous(active_session) { active_session = &session; }
  ~SessionScope() { active_session = previous; }
};
std::uint64_t NowNs();
Queue& DataQueue(std::uint32_t index);
Queue* FindQueue(std::uint32_t index);
bool CreateSessionQueues(const UmdkControlArgs&, UmdkControlStatus&);
bool Prepare(const UmdkControlArgs&, UmdkControlStatus&);
bool Connect(const UmdkControlArgs&, UmdkControlStatus&);
bool DescribeWorker(const UmdkControlArgs&, std::uint32_t index, UmdkControlStatus&);
bool RegisterBuffer(const UmdkControlArgs&, UmdkControlStatus&);
bool Cleanup(UmdkControlStatus&);
bool UnregisterBuffers(UmdkControlStatus&);
bool UnregisterBuffer(const UmdkControlArgs&, UmdkControlStatus&);
void Stage(UmdkControlStatus&, const char*);
bool Fail(UmdkControlStatus&, const char*, int result = -1, const char* detail = "");
bool PostBatch(Queue&, urma_jfs_wr_t* first, urma_jfs_wr_t* tail, UmdkControlStatus&);
// Caller owns the control queue until the signaled chain tail has been retired.
bool PostOrderedWrites(Queue&, const UmdkTransfer*, std::uint32_t count, UmdkControlStatus&);
// Caller holds the queue lock until the CQE for single_wr.user_ctx is consumed.
bool PostOne(Queue&, bool write, const UmdkTransfer&, UmdkControlStatus&, bool fence = false);
bool WaitCompletion(Queue&, std::uint64_t context, std::uint64_t start, std::uint32_t timeout_ms,
                    UmdkControlStatus&);
urma_target_seg_t* LocalSegment(std::uint64_t address, std::uint64_t bytes);
bool TransferControl(bool write, const UmdkTransfer*, std::uint32_t count, UmdkControlStatus&,
                     std::uint32_t worker_index = 0U, urma_order_type_t placement = URMA_RELAX_ORDER);
// Internal synchronous transfer. Caller owns the session, holds its mutex
// (shared or exclusive), and has installed SessionScope for the entire operation.
// Session/context validation belongs at the execution entry, not each message.
std::uint32_t TransferControlSync(bool write, const UmdkTransfer*, std::uint32_t count,
                                  std::uint32_t worker_index, urma_order_type_t placement = URMA_RELAX_ORDER);
}  // namespace kvaf::aicpu_urma::detail
