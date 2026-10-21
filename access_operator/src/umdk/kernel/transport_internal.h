#pragma once
#include <array>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "umdk/kernel_transport.h"
#include "urma_transport.h"
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

struct WrQueue {
  virtual ~WrQueue() = default;
  std::vector<Sge> local_sges;
  std::vector<Sge> remote_sges;
  std::vector<urma_jfs_wr_t> wrs;
};

struct Worker : WrQueue {
  bool poisoned = false;
  urma_jfc_t* jfc = nullptr;
  urma_jfr_t* jfr = nullptr;
  urma_jetty_t* jetty = nullptr;
  urma_target_jetty_t* target = nullptr;
  std::uint64_t tp = 0;
  std::uint64_t sequence = 0;
  std::unique_ptr<WrQueue> data_workspace;
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

extern State state;
extern std::shared_mutex state_mutex;
extern std::array<std::mutex, 8> worker_mutexes;
std::uint64_t NowNs();
Worker& GetWorker(std::uint32_t index);
void Stage(UmdkControlStatus&, const char*);
bool Fail(UmdkControlStatus&, const char*, int result = -1, const char* detail = "");
bool PostBatch(Worker&, urma_jfs_wr_t* first, urma_jfs_wr_t* tail, std::uint32_t first_index,
               std::uint32_t count, UmdkControlStatus&);
bool WaitCompletion(Worker&, std::uint64_t context, std::uint64_t start, std::uint32_t timeout_ms,
                    UmdkControlStatus&);
urma_target_seg_t* LocalSegment(std::uint64_t address, std::uint64_t bytes,
                                std::uint64_t* available = nullptr);
bool TransferControl(bool write, const UmdkTransfer*, std::uint32_t count, UmdkControlStatus&);
}  // namespace kvaf::aicpu_urma::detail
