#include "kvaf_aiv_batch_put.h"

#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "acl/acl.h"
#include "batch_put_internal.h"
#include "direct_hccp_abi.h"
#include "direct_hccp_session.h"

extern "C" int32_t KvafExecuteBatchPutLaunchInternal(
    const KvafExecuteBatchPutLaunchArgs *arguments, aclrtStream stream);

namespace hccp = kvaf::direct_hccp;

constexpr char kUbDevIdPath[] = "/proc/asdrv_ub/dev_id";
constexpr char kUbPairInfoPath[] = "/proc/asdrv_ub/pair_info";
constexpr char kUbSelectorLockPath[] = "/tmp/kvaf_aiv_batch_put_asdrv_ub.lock";
constexpr uint64_t kPageBytes = 4096U;
constexpr uint32_t kDirectEntityMagic = 0x4b564155U;  // "KVAU"
constexpr uint32_t kDirectEntityVersion = 1U;
constexpr uint32_t kHostStandardSqDepth = 8192U;
constexpr uint32_t kDeviceUserCtlSqDepth = 8192U;
constexpr uint32_t kWqeBytes = 64U;
constexpr uint32_t kWqebbPerWqe = 4U;
constexpr uint32_t kDeviceUserCtlCqDepth = 16384U;
constexpr uint32_t kDeviceRqDepth = 256U;
constexpr uint32_t kCtpTpCount = 1U;
constexpr uint32_t kTokenIdShift = 8U;
constexpr uint32_t kPidHdcType = 18U;
constexpr uint32_t kCtpQueryPollMs = 1U;
constexpr uint32_t kCtpQueryTimeoutMs = 10000U;
// A5 HCCP converts EAGAIN differently for initialization and for the
// asynchronous request-result API (HCCP_EAGAIN vs OTHERS_EAGAIN).
constexpr int kHccpAsyncAgain = 128301;
constexpr uint32_t kMaxDirectAivWorkers =
    KVAF_AIV_BATCH_PUT_MAX_AIV_CORES * KVAF_AIV_BATCH_PUT_MAX_JETTYS_PER_CORE;

struct DeviceIdentity {
  uint32_t logical_device_id = 0;
  uint32_t physical_device_id = 0;
};

struct EidPair {
  std::array<uint8_t, hccp::kEidBytes> host_eid{};
  std::array<uint8_t, hccp::kEidBytes> device_eid{};
  uint32_t channel = 0;
};

struct DirectHccpApi {
  void *library = nullptr;
  hccp::CtxInitFn ctx_init = nullptr;
  hccp::CtxDeinitFn ctx_deinit = nullptr;
  hccp::TokenAllocFn token_alloc = nullptr;
  hccp::TokenFreeFn token_free = nullptr;
  hccp::LmemRegisterFn lmem_register = nullptr;
  hccp::LmemUnregisterFn lmem_unregister = nullptr;
  hccp::RmemImportFn rmem_import = nullptr;
  hccp::RmemUnimportFn rmem_unimport = nullptr;
  hccp::CqCreateFn cq_create = nullptr;
  hccp::CqDestroyFn cq_destroy = nullptr;
  hccp::QpCreateFn qp_create = nullptr;
  hccp::QpDestroyFn qp_destroy = nullptr;
  hccp::QpImportFn qp_import = nullptr;
  hccp::QpUnimportFn qp_unimport = nullptr;
  hccp::GetTpInfoFn get_tp_info = nullptr;
  hccp::GetTpInfoAsyncFn get_tp_info_async = nullptr;
  hccp::AsyncReqResultFn get_async_req_result = nullptr;
  hccp::GetRandomFn get_random = nullptr;  // Optional on older CANN builds.
};

struct DirectMemory {
  void *handle = nullptr;
  uint64_t registered_address = 0;
  uint64_t registered_bytes = 0;
  uint32_t token_id = 0;
  uint32_t token_value = 0;
  uint8_t key[hccp::kMemKeyBytes]{};
  uint32_t key_bytes = 0;
};

struct DirectJetty {
  void *handle = nullptr;
  hccp::QpCreateInfo info{};
};

// A USER_CTL SQ/CQ is strictly single-producer/single-consumer from the AIV
// kernel's point of view. Each core/Jetty slot therefore owns one complete
// Host/Device Jetty pair rather than sharing producer/consumer indices.
struct DirectAivWorker {
  void *host_normal_cq = nullptr;
  hccp::CqInfo host_cq_info{};
  DirectJetty host_standard_jetty{};
  void *device_user_ctl_cq = nullptr;
  hccp::CqInfo device_cq_info{};
  DirectJetty device_user_ctl_jetty{};
  void *host_device_target_jetty = nullptr;
  hccp::QpImportInfoT host_device_import_info{};
  void *device_host_target_jetty = nullptr;
  hccp::QpImportInfoT device_host_import_info{};
};

// The two queue users in this demo have different HCCP contracts.  Keep the
// distinction explicit rather than treating every mode-3 queue as equivalent:
// the AIV producer must use the same DEV_USED USER_CTL contract as HCCL's
// AivUrmaChannel, while the host endpoint uses a regular URMA Jetty.
enum class DirectJettyProfile : uint8_t {
  kHostStandard,
  kAivDevUsed,
};

const char *JettyProfileName(DirectJettyProfile profile) {
  switch (profile) {
    case DirectJettyProfile::kHostStandard:
      return "host_standard";
    case DirectJettyProfile::kAivDevUsed:
      return "aiv_dev_used";
  }
  return "invalid";
}

struct KvafBatchPutTransport;

struct KvafBatchPutLocalBuffer {
  KvafBatchPutTransport *transport = nullptr;
  void *token_handle = nullptr;
  DirectMemory memory{};
  uint64_t input_hbm_address = 0U;
  uint64_t input_hbm_bytes = 0U;
  void *device_entities = nullptr;
  uint64_t device_channel_entity = 0U;
  uint32_t reference_count = 0U;
};

struct KvafBatchPutTransport {
  DirectHccpApi api{};
  hccp::DirectHccpSession *session = nullptr;
  uint32_t physical_device_id = 0;
  uint32_t aiv_core_count = 0U;
  uint32_t jettys_per_core = 0U;
  uint32_t worker_count = 0U;
  void *host_context = nullptr;
  void *device_context = nullptr;
  void *host_token_handle = nullptr;
  void *device_transport_token_handle = nullptr;
  uint32_t host_token_id = 0U;
  uint32_t host_token_value = 0U;
  uint32_t device_transport_token_value = 0U;
  std::array<uint8_t, hccp::kEidBytes> wqe_remote_eid{};
  uint64_t host_ddr_address = 0U;
  uint64_t host_ddr_bytes = 0U;
  DirectMemory host_memory{};
  void *device_host_memory_import = nullptr;
  std::array<DirectAivWorker, kMaxDirectAivWorkers> workers{};
  std::array<KvafBatchPutDirectUrmaEntity, kMaxDirectAivWorkers> entity_templates{};
  void *device_queue_state = nullptr;
  std::vector<std::unique_ptr<KvafBatchPutLocalBuffer>> local_buffers{};
  std::mutex mutex{};
  std::mutex launch_mutex{};
};

using AclrtGetPhyDevIdByLogicDevId = int32_t (*)(int32_t, int32_t *);

bool ParseEid(const char *text, std::array<uint8_t, hccp::kEidBytes> *eid) {
  if (text == nullptr || eid == nullptr) {
    return false;
  }
  std::string compact;
  compact.reserve(hccp::kEidBytes * 2U);
  for (const unsigned char value : std::string(text)) {
    if (std::isxdigit(value)) {
      compact.push_back(static_cast<char>(value));
    } else if (value != ':') {
      return false;
    }
  }
  if (compact.size() != hccp::kEidBytes * 2U) {
    return false;
  }
  const auto hex = [](unsigned char value) -> uint8_t {
    if (value >= '0' && value <= '9') {
      return static_cast<uint8_t>(value - '0');
    }
    value = static_cast<unsigned char>(std::tolower(value));
    return static_cast<uint8_t>(value - 'a' + 10U);
  };
  for (uint32_t index = 0; index < hccp::kEidBytes; ++index) {
    (*eid)[index] = static_cast<uint8_t>((hex(static_cast<unsigned char>(compact[index * 2U])) << 4U) |
                                         hex(static_cast<unsigned char>(compact[index * 2U + 1U])));
  }
  return true;
}

std::string EidToHex(const std::array<uint8_t, hccp::kEidBytes> &eid) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(hccp::kEidBytes * 2U);
  for (const uint8_t byte : eid) {
    result.push_back(kHex[byte >> 4U]);
    result.push_back(kHex[byte & 0x0fU]);
  }
  return result;
}

std::array<uint8_t, hccp::kEidBytes> ReverseEidForAivWqe(
    const std::array<uint8_t, hccp::kEidBytes> &ra_host_eid) {
  std::array<uint8_t, hccp::kEidBytes> wqe_remote_eid{};
  for (uint32_t index = 0U; index < hccp::kEidBytes; ++index) {
    wqe_remote_eid[index] = ra_host_eid[hccp::kEidBytes - 1U - index];
  }
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_AIV_WQE_EID_ENCODING format=reverse ra_host_eid=%s wqe_remote_eid=%s\n",
      EidToHex(ra_host_eid).c_str(), EidToHex(wqe_remote_eid).c_str());
  return wqe_remote_eid;
}

std::string Trim(std::string text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1U);
}

bool SelectEidPair(uint32_t physical_device_id, EidPair *pair) {
  if (pair == nullptr) {
    return false;
  }
  const int lock_fd = open(kUbSelectorLockPath, O_CREAT | O_RDWR, 0600);
  if (lock_fd < 0 || flock(lock_fd, LOCK_EX) != 0) {
    std::fprintf(stderr, "[KVAF_AIV] cannot lock %s for EID selection\n", kUbSelectorLockPath);
    if (lock_fd >= 0) {
      close(lock_fd);
    }
    return false;
  }

  bool success = false;
  {
    std::ofstream selector(kUbDevIdPath);
    if (!selector.is_open()) {
      std::fprintf(stderr, "[KVAF_AIV] cannot open %s for device selection\n", kUbDevIdPath);
    } else {
      selector << physical_device_id << '\n';
      selector.close();
      std::ifstream pair_info(kUbPairInfoPath);
      if (!pair_info.is_open()) {
        std::fprintf(stderr, "[KVAF_AIV] cannot open %s\n", kUbPairInfoPath);
      } else {
        const std::string selected_marker = "dev_id=" + std::to_string(physical_device_id) + ";";
        bool selected_device = false;
        bool active_channel = false;
        uint32_t active_channels = 0;
        uint32_t host_eids = 0;
        uint32_t device_eids = 0;
        std::string line;
        while (std::getline(pair_info, line)) {
          if (line.find("Dev id info.") != std::string::npos) {
            selected_device = line.find(selected_marker) != std::string::npos;
            active_channel = false;
            continue;
          }
          if (!selected_device) {
            continue;
          }
          const size_t channel_pos = line.find("chan_id=");
          if (channel_pos != std::string::npos) {
            active_channel = line.find("flag=1") != std::string::npos;
            if (active_channel) {
              std::istringstream channel_stream(line.substr(channel_pos + std::strlen("chan_id=")));
              if (!(channel_stream >> pair->channel)) {
                active_channel = false;
              } else {
                ++active_channels;
              }
            }
            continue;
          }
          if (!active_channel) {
            continue;
          }
          constexpr char kLocalPrefix[] = "local_eid:";
          constexpr char kRemotePrefix[] = "remote_eid:";
          const size_t local_pos = line.find(kLocalPrefix);
          if (local_pos != std::string::npos) {
            if (ParseEid(Trim(line.substr(local_pos + sizeof(kLocalPrefix) - 1U)).c_str(), &pair->host_eid)) {
              ++host_eids;
            }
            continue;
          }
          const size_t remote_pos = line.find(kRemotePrefix);
          if (remote_pos != std::string::npos &&
              ParseEid(Trim(line.substr(remote_pos + sizeof(kRemotePrefix) - 1U)).c_str(), &pair->device_eid)) {
            ++device_eids;
          }
        }
        success = selected_device && active_channels == 1U && host_eids == 1U && device_eids == 1U;
        if (!success) {
          std::fprintf(stderr,
                       "[KVAF_AIV] selected dev_id=%u must have one active Host/Device EID pair; "
                       "active=%u host_eids=%u device_eids=%u\n",
                       physical_device_id, active_channels, host_eids, device_eids);
        }
      }
    }
  }
  (void)flock(lock_fd, LOCK_UN);
  close(lock_fd);
  if (success) {
    hccp::VerboseLog(
        "[KVAF_AIV] HOST_DEVICE_UB_EID_PAIR_SELECTED asdrv_ub_dev_id=%u channel=%u\n",
        physical_device_id, pair->channel);
  }
  return success;
}

bool GetDeviceIdentity(DeviceIdentity *identity) {
  if (identity == nullptr) {
    return false;
  }
  int32_t current_device = -1;
  if (aclrtGetDevice(&current_device) != ACL_SUCCESS || current_device < 0) {
    return false;
  }
  auto get_physical = reinterpret_cast<AclrtGetPhyDevIdByLogicDevId>(
      dlsym(RTLD_DEFAULT, "aclrtGetPhyDevIdByLogicDevId"));
  if (get_physical == nullptr) {
    void *ascendcl = dlopen("libascendcl.so", RTLD_NOW | RTLD_NOLOAD);
    if (ascendcl == nullptr) {
      ascendcl = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (ascendcl != nullptr) {
      get_physical = reinterpret_cast<AclrtGetPhyDevIdByLogicDevId>(
          dlsym(ascendcl, "aclrtGetPhyDevIdByLogicDevId"));
    }
  }
  int32_t physical_device = -1;
  if (get_physical == nullptr || get_physical(current_device, &physical_device) != 0 || physical_device < 0) {
    return false;
  }
  identity->logical_device_id = static_cast<uint32_t>(current_device);
  identity->physical_device_id = static_cast<uint32_t>(physical_device);
  return true;
}

template <typename Function>
bool LoadSymbol(void *library, const char *name, Function *function) {
  *function = reinterpret_cast<Function>(dlsym(library, name));
  if (*function == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] direct libhccp ABI missing symbol %s: %s\n", name, dlerror());
    return false;
  }
  return true;
}

bool LoadDirectHccp(DirectHccpApi *api) {
  if (api == nullptr) {
    return false;
  }
  // On A5 the RaCtx* C ABI is exported by CANN's libra.so.  HCCP names the
  // protocol/source subsystem, not the installed shared-object name.  Keep
  // the libhccp candidates as a compatibility fallback for other releases,
  // but never require a separately installed libhccp.so on this platform.
  for (const char *candidate : {"libra.so", "libra.so.0", "libhccp.so", "libhccp.so.0"}) {
    api->library = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
    if (api->library != nullptr) {
      break;
    }
  }
  if (api->library == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] dlopen direct RA ABI (libra.so) failed: %s\n", dlerror());
    return false;
  }
  const bool loaded =
      LoadSymbol(api->library, "RaCtxInit", &api->ctx_init) &&
      LoadSymbol(api->library, "RaCtxDeinit", &api->ctx_deinit) &&
      LoadSymbol(api->library, "RaCtxTokenIdAlloc", &api->token_alloc) &&
      LoadSymbol(api->library, "RaCtxTokenIdFree", &api->token_free) &&
      LoadSymbol(api->library, "RaCtxLmemRegister", &api->lmem_register) &&
      LoadSymbol(api->library, "RaCtxLmemUnregister", &api->lmem_unregister) &&
      LoadSymbol(api->library, "RaCtxRmemImport", &api->rmem_import) &&
      LoadSymbol(api->library, "RaCtxRmemUnimport", &api->rmem_unimport) &&
      LoadSymbol(api->library, "RaCtxCqCreate", &api->cq_create) &&
      LoadSymbol(api->library, "RaCtxCqDestroy", &api->cq_destroy) &&
      LoadSymbol(api->library, "RaCtxQpCreate", &api->qp_create) &&
      LoadSymbol(api->library, "RaCtxQpDestroy", &api->qp_destroy) &&
      LoadSymbol(api->library, "RaCtxQpImport", &api->qp_import) &&
      LoadSymbol(api->library, "RaCtxQpUnimport", &api->qp_unimport) &&
      LoadSymbol(api->library, "RaCtxGetTpInfoList", &api->get_tp_info) &&
      LoadSymbol(api->library, "RaGetTpInfoListAsync", &api->get_tp_info_async) &&
      LoadSymbol(api->library, "RaGetAsyncReqResult", &api->get_async_req_result);
  // These observability symbols are optional because older releases may not
  // export them.  Their absence must never change the data-plane setup path.
  // Secure random is a convenience API.  A deterministic fallback remains
  // valid for the plaintext token policy when an older CANN omits it.
  api->get_random = reinterpret_cast<hccp::GetRandomFn>(dlsym(api->library, "RaGetSecRandom"));
  if (!loaded) {
    dlclose(api->library);
    *api = {};
    return false;
  }
  hccp::VerboseLog("[KVAF_AIV] DIRECT_HCCP_TRANSPORT_API_READY backend=RaCtx_no_HCOMM\n");
  return true;
}

void CloseDirectHccp(DirectHccpApi *api) {
  if (api == nullptr) {
    return;
  }
  // `libra` backs Direct-HCCP contexts as well as RA/HDC service/TLS state.
  // The session bootstrap is process-owned after a successful transport
  // create, and CANN has no close counterpart for rtOpenNetService.  Dropping
  // this DSO at a Runtime boundary can therefore leave an exit callback
  // pointing into unloaded code.  The LoadDirectHccp failure path still
  // dlcloses its never-published handle; a successfully published transport
  // only clears these per-transport function pointers here.
  *api = {};
}

uint32_t MakeTokenValue(const DirectHccpApi &api, uint32_t physical_device_id) {
  uint32_t token = static_cast<uint32_t>(
      std::chrono::steady_clock::now().time_since_epoch().count() ^ static_cast<int64_t>(getpid()));
  if (token == 0U) {
    token = 1U;
  }
  if (api.get_random != nullptr) {
    hccp::RaInfo info{};
    info.mode = hccp::kOfflineHdc;
    info.phy_id = physical_device_id;
    const int ret = api.get_random(&info, &token);
    if (ret != 0 || token == 0U) {
      hccp::VerboseLog("[KVAF_AIV] RaGetSecRandom fallback ret=%d token=0x%x\n", ret,
                       token);
      token = static_cast<uint32_t>(
          std::chrono::steady_clock::now().time_since_epoch().count() ^ static_cast<int64_t>(getpid()));
      if (token == 0U) {
        token = 1U;
      }
    }
  }
  return token;
}

bool InitContext(const DirectHccpApi &api, int32_t mode, uint32_t physical_device_id,
                 const std::array<uint8_t, hccp::kEidBytes> &eid, const char *role, void **context) {
  hccp::CtxInitCfg cfg{};
  cfg.mode = mode;
  hccp::CtxInitAttr attr{};
  attr.phy_id = physical_device_id;
  attr.ub.eid_index = 0U;
  std::memcpy(attr.ub.eid.raw, eid.data(), eid.size());
  const int ret = api.ctx_init(&cfg, &attr, context);
  if (ret != 0 || *context == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_CONTEXT_FAILED role=%s mode=%d physical=%u ret=%d context=%p\n",
                 role, mode, physical_device_id, ret, *context);
    return false;
  }
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_HCCP_CONTEXT_READY role=%s mode=%d physical=%u context=%p\n",
      role, mode, physical_device_id, *context);
  return true;
}

bool AllocateToken(const DirectHccpApi &api, void *context, const char *role, void **token_handle,
                   uint32_t *token_id) {
  hccp::TokenId raw{};
  const int ret = api.token_alloc(context, &raw, token_handle);
  if (ret != 0 || *token_handle == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_TOKEN_ALLOC_FAILED role=%s ret=%d handle=%p\n", role, ret,
                 *token_handle);
    return false;
  }
  *token_id = raw.token_id >> kTokenIdShift;
  hccp::VerboseLog("[KVAF_AIV] DIRECT_TOKEN_READY role=%s token_id=0x%x\n", role,
                   *token_id);
  return true;
}

bool AlignRange(uint64_t address, uint64_t bytes, uint64_t *aligned_address, uint64_t *aligned_bytes) {
  if (bytes == 0U || aligned_address == nullptr || aligned_bytes == nullptr) {
    return false;
  }
  const uint64_t base = address & ~(kPageBytes - 1U);
  const uint64_t delta = address - base;
  if (bytes > std::numeric_limits<uint64_t>::max() - delta) {
    return false;
  }
  *aligned_address = base;
  *aligned_bytes = bytes + delta;
  return true;
}

bool RegisterMemory(const DirectHccpApi &api, void *context, void *token_handle, uint32_t token_id,
                    uint32_t token_value, void *address, uint64_t bytes, const char *role, DirectMemory *memory) {
  uint64_t aligned_address = 0U;
  uint64_t aligned_bytes = 0U;
  if (memory == nullptr || address == nullptr || !AlignRange(reinterpret_cast<uint64_t>(address), bytes,
                                                              &aligned_address, &aligned_bytes)) {
    return false;
  }
  hccp::MrRegInfo info{};
  info.in.mem.addr = aligned_address;
  info.in.mem.size = aligned_bytes;
  info.in.ub.flags = hccp::MemRegFlags(true);
  info.in.ub.token_value = token_value;
  info.in.ub.token_id_handle = token_handle;
  const int ret = api.lmem_register(context, &info, &memory->handle);
  if (ret != 0 || memory->handle == nullptr || info.out.key.size == 0U || info.out.key.size > hccp::kMemKeyBytes) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_MEMORY_REGISTER_FAILED role=%s ret=%d handle=%p key_bytes=%u addr=0x%llx "
                 "bytes=%llu\n",
                 role, ret, memory->handle, info.out.key.size, static_cast<unsigned long long>(aligned_address),
                 static_cast<unsigned long long>(aligned_bytes));
    return false;
  }
  memory->registered_address = aligned_address;
  memory->registered_bytes = aligned_bytes;
  memory->token_id = token_id;
  memory->token_value = token_value;
  memory->key_bytes = info.out.key.size;
  std::memcpy(memory->key, info.out.key.value, memory->key_bytes);
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_MEMORY_REGISTERED role=%s token_id=0x%x host_or_hbm_va=0x%llx bytes=%llu\n",
      role, memory->token_id, static_cast<unsigned long long>(aligned_address),
      static_cast<unsigned long long>(aligned_bytes));
  return true;
}

bool ImportRemoteMemory(const DirectHccpApi &api, void *context, const DirectMemory &remote, void **import_handle) {
  hccp::MrImportInfo info{};
  std::memcpy(info.in.key.value, remote.key, remote.key_bytes);
  info.in.key.size = static_cast<uint8_t>(remote.key_bytes);
  info.in.ub.flags = hccp::MemImportFlags();
  info.in.ub.mapping_addr = 0U;
  info.in.ub.token_value = remote.token_value;
  const int ret = api.rmem_import(context, &info, import_handle);
  if (ret != 0 || *import_handle == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_REMOTE_DDR_IMPORT_FAILED ret=%d handle=%p\n", ret, *import_handle);
    return false;
  }
  hccp::VerboseLog("[KVAF_AIV] DIRECT_REMOTE_DDR_IMPORTED target_seg=0x%llx\n",
                   static_cast<unsigned long long>(info.out.ub.target_seg_handle));
  return true;
}

bool CreateCq(const DirectHccpApi &api, void *context, uint32_t mode, uint32_t depth, const char *role,
              hccp::CqInfo *info, void **handle) {
  *info = {};
  info->in.channel_handle = nullptr;
  info->in.depth = depth;
  info->in.ub.user_ctx = 0U;
  info->in.ub.mode = mode;
  info->in.ub.ceqn = 0U;
  info->in.ub.flag = 0U;
  const int ret = api.cq_create(context, info, handle);
  if (ret != 0 || *handle == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_CQ_CREATE_FAILED role=%s mode=%u depth=%u ret=%d handle=%p\n", role,
                 mode, depth, ret, *handle);
    return false;
  }
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_CQ_READY role=%s mode=%u jfc_id=%u cq_va=0x%llx cqe_size=%u db_va=0x%llx\n",
      role, mode, info->out.id, static_cast<unsigned long long>(info->out.buffer_addr),
      info->out.cqe_size, static_cast<unsigned long long>(info->out.swdb_addr));
  return true;
}

bool CreateJetty(const DirectHccpApi &api, void *context, void *cq_handle, void *token_handle,
                  uint32_t token_value, DirectJettyProfile profile, uint32_t sq_depth, const char *role,
                  DirectJetty *jetty) {
  const uint32_t mode = profile == DirectJettyProfile::kAivDevUsed ? hccp::kJettyUserCtlNormal
                                                                     : hccp::kJettyStandard;
  hccp::QpCreateAttr attr{};
  attr.scq_handle = cq_handle;
  attr.rcq_handle = cq_handle;
  attr.srq_handle = cq_handle;
  attr.sq_depth = sq_depth;
  attr.rq_depth = kDeviceRqDepth;
  attr.transport_mode = hccp::kReliableMessage;
  attr.ub.mode = mode;
  attr.ub.jetty_id = 0U;
  attr.ub.flag = hccp::JettyFlagShareJfr();
  attr.ub.jfs_flag = hccp::JfsFlagErrorSuspend();
  attr.ub.token_id_handle = token_handle;
  attr.ub.token_value = token_value;
  attr.ub.priority = 2U;
  attr.ub.rnr_retry = 7U;
  attr.ub.err_timeout = 16U;
  attr.ub.ext_mode.sqebb_num = sq_depth;
  if (profile == DirectJettyProfile::kAivDevUsed) {
    // Match the DEV_USED branch used by the A5 HCOMM legacy adapter.  On the
    // 950 HCCP product path sqCstm is valid only for CCU Jettys; setting it for
    // USER_CTL_NORMAL is rejected and later surfaced by HDC as -ENOMEM.  The
    // requested size still describes the four-WQEBB storage per SQ entry,
    // while HCCP owns and returns the actual SQ mapping.
    const uint64_t sq_bytes = static_cast<uint64_t>(sq_depth) * kWqebbPerWqe * kWqeBytes;
    if (sq_bytes == 0U || sq_bytes > UINT32_MAX) {
      std::fprintf(stderr,
                   "[KVAF_AIV] DIRECT_AIV_DEV_USED_QUEUE_INVALID role=%s sq_depth=%u requested_sq_bytes=%llu\n",
                   role, sq_depth, static_cast<unsigned long long>(sq_bytes));
      return false;
    }
    attr.ub.ext_mode.sq.buffer_size = static_cast<uint32_t>(sq_bytes);
    attr.ub.ext_mode.sq.buffer_va = 0U;
    attr.ub.ext_mode.cstm_flag = 0U;
  }
  jetty->info = {};
  jetty->handle = nullptr;
  const int ret = api.qp_create(context, &attr, &jetty->info, &jetty->handle);
  if (ret != 0 || jetty->handle == nullptr || jetty->info.key.size == 0U ||
      jetty->info.key.size > hccp::kQpKeyBytes) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_JETTY_CREATE_FAILED role=%s profile=%s mode=%u sq_depth=%u sqebb_num=%u "
                 "cstm_flag=%u requested_sq_bytes=%u ret=%d handle=%p key_bytes=%u\n",
                 role, JettyProfileName(profile), mode, sq_depth, attr.ub.ext_mode.sqebb_num,
                 attr.ub.ext_mode.cstm_flag, attr.ub.ext_mode.sq.buffer_size, ret, jetty->handle,
                 jetty->info.key.size);
    return false;
  }
  if (profile == DirectJettyProfile::kAivDevUsed &&
      (jetty->info.ub.sq_buffer_va == 0U || jetty->info.ub.db_addr == 0U ||
       jetty->info.ub.wqebb_size != kWqeBytes)) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_USER_CTL_QUEUE_CONTRACT_INVALID role=%s sq_va=0x%llx db_va=0x%llx "
                 "returned_wqebb_bytes=%llu expected_wqebb_bytes=%u\n",
                 role, static_cast<unsigned long long>(jetty->info.ub.sq_buffer_va),
                 static_cast<unsigned long long>(jetty->info.ub.db_addr),
                 static_cast<unsigned long long>(jetty->info.ub.wqebb_size), kWqeBytes);
    return false;
  }
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_JETTY_READY role=%s profile=%s mode=%u cstm_flag=%u jetty_id=%u sq_va=0x%llx "
      "db_va=0x%llx returned_wqebb_bytes=%llu\n",
      role, JettyProfileName(profile), mode, attr.ub.ext_mode.cstm_flag, jetty->info.ub.id,
      static_cast<unsigned long long>(jetty->info.ub.sq_buffer_va),
      static_cast<unsigned long long>(jetty->info.ub.db_addr),
      static_cast<unsigned long long>(jetty->info.ub.wqebb_size));
  if (profile == DirectJettyProfile::kAivDevUsed) {
    // QpCreateInfo.ub.db_addr is already the UDMA post-doorbell address
    // (including the driver's internal 0x80 offset).  The AIV kernel must
    // write that returned address directly, not add a second offset.
    hccp::VerboseLog(
        "[KVAF_AIV] DIRECT_USER_CTL_QUEUE_CONTRACT role=%s profile=aiv_dev_used "
        "sq_depth_entries=%u sqebb_num=%u requested_sq_buffer_bytes=%u "
        "returned_sq_va=0x%llx returned_wqebb_bytes=%llu db_addr_is_post_offset=1 sq_cstm=0\n",
        role, sq_depth, attr.ub.ext_mode.sqebb_num, attr.ub.ext_mode.sq.buffer_size,
        static_cast<unsigned long long>(jetty->info.ub.sq_buffer_va),
        static_cast<unsigned long long>(jetty->info.ub.wqebb_size));
  }
  return true;
}

bool GetCtp(const DirectHccpApi &api, void *context, const std::array<uint8_t, hccp::kEidBytes> &local,
            const std::array<uint8_t, hccp::kEidBytes> &peer, const char *role, bool use_hdc_async,
            uint64_t *tp_handle) {
  hccp::GetTpCfg cfg{};
  cfg.flag = hccp::CtpFlag();
  cfg.transport_mode = hccp::kReliableMessage;
  std::memcpy(cfg.local_eid.raw, local.data(), local.size());
  std::memcpy(cfg.peer_eid.raw, peer.data(), peer.size());
  hccp::TpInfo info{};
  uint32_t count = kCtpTpCount;
  int ret = 0;
  if (use_hdc_async) {
    void *request = nullptr;
    ret = api.get_tp_info_async(context, &cfg, &info, &count, &request);
    if (ret != 0 || request == nullptr) {
      std::fprintf(stderr,
                   "[KVAF_AIV] DIRECT_CTP_ASYNC_SUBMIT_FAILED role=%s ret=%d count=%u request=%p\n",
                   role, ret, count, request);
      return false;
    }
    hccp::VerboseLog("[KVAF_AIV] DIRECT_CTP_ASYNC_SUBMITTED role=%s request=%p\n", role,
                     request);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kCtpQueryTimeoutMs);
    while (true) {
      int request_result = 0;
      ret = api.get_async_req_result(request, &request_result);
      if (ret == kHccpAsyncAgain) {
        if (std::chrono::steady_clock::now() >= deadline) {
          std::fprintf(stderr, "[KVAF_AIV] DIRECT_CTP_ASYNC_TIMEOUT role=%s timeout_ms=%u\n", role,
                       kCtpQueryTimeoutMs);
          return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kCtpQueryPollMs));
        continue;
      }
      // RaGetAsyncReqResult consumes the request handle on every terminal
      // return.  Never query or free it after this point.
      request = nullptr;
      if (ret != 0 || request_result != 0) {
        std::fprintf(stderr,
                     "[KVAF_AIV] DIRECT_CTP_ASYNC_COMPLETE_FAILED role=%s api_ret=%d request_ret=%d\n",
                     role, ret, request_result);
        return false;
      }
      break;
    }
  } else {
    ret = api.get_tp_info(context, &cfg, &info, &count);
  }
  if (ret != 0 || count == 0U || info.tp_handle == 0U) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_CTP_GET_FAILED role=%s mode=%s ret=%d count=%u tp=0x%llx\n", role,
                 use_hdc_async ? "hdc_async" : "peer_sync", ret, count,
                 static_cast<unsigned long long>(info.tp_handle));
    return false;
  }
  *tp_handle = info.tp_handle;
  hccp::VerboseLog("[KVAF_AIV] DIRECT_CTP_READY role=%s mode=%s tp_handle=0x%llx\n", role,
                   use_hdc_async ? "hdc_async" : "peer_sync",
                   static_cast<unsigned long long>(*tp_handle));
  return true;
}

uint32_t MakeCtpPsn(const DirectHccpApi &api, uint32_t physical_device_id, const char *role) {
  // A CTP import needs the same PSN pair on both endpoints: local tx_psn
  // must be the peer import's rx_psn, and vice versa.  HCOMM obtains a
  // random value once per endpoint and exchanges it before either import.
  // Reuse the exact RA secure-random source used for token generation, with
  // the existing nonzero fallback for older CANN releases.
  const uint32_t psn = MakeTokenValue(api, physical_device_id);
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_CTP_PSN_READY role=%s psn=0x%x source=ra_secure_or_fallback\n",
      role, psn);
  return psn;
}

bool ImportJetty(const DirectHccpApi &api, void *context, const DirectJetty &remote_jetty,
                 uint32_t remote_token_value, uint64_t local_tp, uint64_t peer_tp, uint32_t local_psn,
                 uint32_t remote_psn, const char *role, hccp::QpImportInfoT *import_info,
                 void **target_jetty) {
  if (context == nullptr || remote_jetty.handle == nullptr || remote_jetty.info.key.size == 0U ||
      remote_jetty.info.key.size > hccp::kQpKeyBytes || import_info == nullptr || target_jetty == nullptr) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_JETTY_IMPORT_INVALID role=%s context=%p jetty=%p key_bytes=%u\n", role,
                 context, remote_jetty.handle, remote_jetty.info.key.size);
    return false;
  }
  *import_info = {};
  *target_jetty = nullptr;
  std::memcpy(import_info->in.key.value, remote_jetty.info.key.value, remote_jetty.info.key.size);
  import_info->in.key.size = remote_jetty.info.key.size;
  import_info->in.ub.mode = hccp::kJettyImportExp;
  import_info->in.ub.token_value = remote_token_value;
  import_info->in.ub.policy = hccp::kJettyGroupRoundRobin;
  import_info->in.ub.type = hccp::kTargetJetty;
  import_info->in.ub.flag = hccp::ImportFlagPlainTextToken();
  import_info->in.ub.exp_import_cfg.tp_handle = local_tp;
  import_info->in.ub.exp_import_cfg.peer_tp_handle = peer_tp;
  import_info->in.ub.exp_import_cfg.tag = 0U;
  import_info->in.ub.exp_import_cfg.tx_psn = local_psn;
  import_info->in.ub.exp_import_cfg.rx_psn = remote_psn;
  import_info->in.ub.tp_type = hccp::CtpProtocolType();
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_JETTY_IMPORT_ATTEMPT role=%s local_tp=0x%llx peer_tp=0x%llx tx_psn=0x%x "
      "rx_psn=0x%x\n",
      role, static_cast<unsigned long long>(local_tp), static_cast<unsigned long long>(peer_tp), local_psn,
      remote_psn);
  const int ret = api.qp_import(context, import_info, target_jetty);
  if (ret != 0 || *target_jetty == nullptr || import_info->out.ub.tpn == 0U) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_JETTY_IMPORT_FAILED role=%s ret=%d target=%p tpn=%u\n", role, ret,
                 *target_jetty, import_info->out.ub.tpn);
    return false;
  }
  hccp::VerboseLog("[KVAF_AIV] DIRECT_JETTY_IMPORT_READY role=%s target=0x%llx tpn=%u\n",
                   role,
                   static_cast<unsigned long long>(import_info->out.ub.target_jetty_handle),
                   import_info->out.ub.tpn);
  return true;
}

void ReleaseTransport(KvafBatchPutTransport *transport) {
  if (transport == nullptr) {
    return;
  }
  for (const auto &local : transport->local_buffers) {
    if (local->device_entities != nullptr) {
      (void)aclrtFree(local->device_entities);
    }
    if (transport->api.lmem_unregister != nullptr && transport->device_context != nullptr &&
        local->memory.handle != nullptr) {
      (void)transport->api.lmem_unregister(transport->device_context, local->memory.handle);
    }
    if (transport->api.token_free != nullptr && transport->device_context != nullptr &&
        local->token_handle != nullptr) {
      (void)transport->api.token_free(transport->device_context, local->token_handle);
    }
  }
  transport->local_buffers.clear();
  if (transport->device_queue_state != nullptr) {
    (void)aclrtFree(transport->device_queue_state);
    transport->device_queue_state = nullptr;
  }
  if (transport->api.qp_unimport != nullptr) {
    for (uint32_t worker_index = 0U; worker_index < transport->worker_count; ++worker_index) {
      DirectAivWorker &worker = transport->workers[worker_index];
      if (transport->host_context != nullptr && worker.host_device_target_jetty != nullptr) {
        (void)transport->api.qp_unimport(transport->host_context, worker.host_device_target_jetty);
      }
      if (transport->device_context != nullptr && worker.device_host_target_jetty != nullptr) {
        (void)transport->api.qp_unimport(transport->device_context, worker.device_host_target_jetty);
      }
    }
  }
  if (transport->api.qp_destroy != nullptr) {
    for (uint32_t worker_index = 0U; worker_index < transport->worker_count; ++worker_index) {
      DirectAivWorker &worker = transport->workers[worker_index];
      if (worker.device_user_ctl_jetty.handle != nullptr) {
        (void)transport->api.qp_destroy(worker.device_user_ctl_jetty.handle);
      }
      if (worker.host_standard_jetty.handle != nullptr) {
        (void)transport->api.qp_destroy(worker.host_standard_jetty.handle);
      }
    }
  }
  if (transport->api.cq_destroy != nullptr) {
    for (uint32_t worker_index = 0U; worker_index < transport->worker_count; ++worker_index) {
      DirectAivWorker &worker = transport->workers[worker_index];
      if (transport->device_context != nullptr && worker.device_user_ctl_cq != nullptr) {
        (void)transport->api.cq_destroy(transport->device_context, worker.device_user_ctl_cq);
      }
      if (transport->host_context != nullptr && worker.host_normal_cq != nullptr) {
        (void)transport->api.cq_destroy(transport->host_context, worker.host_normal_cq);
      }
    }
  }
  if (transport->api.rmem_unimport != nullptr && transport->device_context != nullptr &&
      transport->device_host_memory_import != nullptr) {
    (void)transport->api.rmem_unimport(transport->device_context, transport->device_host_memory_import);
  }
  if (transport->api.lmem_unregister != nullptr) {
    if (transport->host_context != nullptr && transport->host_memory.handle != nullptr) {
      (void)transport->api.lmem_unregister(transport->host_context, transport->host_memory.handle);
    }
  }
  if (transport->api.token_free != nullptr) {
    if (transport->device_context != nullptr && transport->device_transport_token_handle != nullptr) {
      (void)transport->api.token_free(transport->device_context, transport->device_transport_token_handle);
    }
    if (transport->host_context != nullptr && transport->host_token_handle != nullptr) {
      (void)transport->api.token_free(transport->host_context, transport->host_token_handle);
    }
  }
  if (transport->api.ctx_deinit != nullptr) {
    if (transport->device_context != nullptr) {
      (void)transport->api.ctx_deinit(transport->device_context);
    }
    if (transport->host_context != nullptr) {
      (void)transport->api.ctx_deinit(transport->host_context);
    }
  }
  if (transport->session != nullptr) {
    hccp::VerboseLog("[KVAF_AIV] DIRECT_HCCP_TEARDOWN phase=release_session\n");
    hccp::ReleaseDirectHccpSession(transport->session);
    transport->session = nullptr;
  }
  CloseDirectHccp(&transport->api);
  delete transport;
}

extern "C" int32_t KvafBatchPutTransportCreate(void *host_ddr, uint64_t host_ddr_bytes,
                                                uint32_t aiv_core_count, uint32_t jettys_per_core,
                                                KvafBatchPutTransport **transport) {
  if (host_ddr == nullptr || host_ddr_bytes == 0U || aiv_core_count == 0U ||
      aiv_core_count > KVAF_AIV_BATCH_PUT_MAX_AIV_CORES || jettys_per_core == 0U ||
      jettys_per_core > KVAF_AIV_BATCH_PUT_MAX_JETTYS_PER_CORE || transport == nullptr) {
    return KVAF_BATCH_PUT_INVALID_ARGUMENT;
  }
  *transport = nullptr;

  auto *state = new (std::nothrow) KvafBatchPutTransport();
  if (state == nullptr) {
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  state->aiv_core_count = aiv_core_count;
  state->jettys_per_core = jettys_per_core;
  state->worker_count = aiv_core_count * jettys_per_core;
  state->host_ddr_address = reinterpret_cast<uint64_t>(host_ddr);
  state->host_ddr_bytes = host_ddr_bytes;
  DeviceIdentity identity{};
  EidPair pair{};
  int32_t result = KVAF_BATCH_PUT_DIRECT_HCCP_ERROR;
  uint32_t device_transport_token_id = 0U;
  const uint64_t queue_bytes =
      static_cast<uint64_t>(state->worker_count) * sizeof(KvafBatchPutQueueState);

  if (!GetDeviceIdentity(&identity)) {
    std::fprintf(stderr, "[KVAF_AIV] unable to resolve physical device id from selected ACL device\n");
    result = KVAF_BATCH_PUT_EID_ERROR;
    goto failed;
  }
  state->physical_device_id = identity.physical_device_id;
  if (!SelectEidPair(identity.physical_device_id, &pair)) {
    goto failed;
  }
  state->wqe_remote_eid = ReverseEidForAivWqe(pair.host_eid);
  if (!LoadDirectHccp(&state->api)) goto failed;
  if (!hccp::AcquireDirectHccpSession(identity.physical_device_id, kPidHdcType, &state->session) ||
      !InitContext(state->api, hccp::kPeerOnline, identity.physical_device_id, pair.host_eid, "host_cpu",
                   &state->host_context) ||
      !InitContext(state->api, hccp::kOfflineHdc, identity.physical_device_id, pair.device_eid, "device_aiv",
                   &state->device_context) ||
      !AllocateToken(state->api, state->host_context, "host_cpu", &state->host_token_handle,
                     &state->host_token_id) ||
      !AllocateToken(state->api, state->device_context, "device_transport",
                     &state->device_transport_token_handle, &device_transport_token_id)) {
    goto failed;
  }
  state->host_token_value = MakeTokenValue(state->api, identity.physical_device_id);
  state->device_transport_token_value = state->host_token_value;
  if (!RegisterMemory(state->api, state->host_context, state->host_token_handle, state->host_token_id,
                      state->host_token_value, host_ddr, host_ddr_bytes, "host_ddr", &state->host_memory) ||
      !ImportRemoteMemory(state->api, state->device_context, state->host_memory,
                          &state->device_host_memory_import)) {
    goto failed;
  }
  for (uint32_t entity_index = 0U; entity_index < state->worker_count; ++entity_index) {
    const uint32_t core = entity_index / jettys_per_core;
    const uint32_t jetty_slot = entity_index % jettys_per_core;
    DirectAivWorker &worker = state->workers[entity_index];
    const std::string suffix =
        "core=" + std::to_string(core) + "_jetty=" + std::to_string(jetty_slot);
    const std::string host_cq_role = "host_normal_" + suffix;
    const std::string host_jetty_role = "host_standard_" + suffix;
    const std::string device_cq_role = "device_user_ctl_" + suffix;
    const std::string device_jetty_role = "device_aiv_dev_used_" + suffix;
    if (!CreateCq(state->api, state->host_context, hccp::kJfcNormal, kDeviceUserCtlCqDepth, host_cq_role.c_str(),
                  &worker.host_cq_info, &worker.host_normal_cq) ||
        !CreateJetty(state->api, state->host_context, worker.host_normal_cq, state->host_token_handle,
                      state->host_memory.token_value, DirectJettyProfile::kHostStandard, kHostStandardSqDepth,
                      host_jetty_role.c_str(), &worker.host_standard_jetty) ||
        !CreateCq(state->api, state->device_context, hccp::kJfcUserCtlNormal, kDeviceUserCtlCqDepth,
                  device_cq_role.c_str(), &worker.device_cq_info, &worker.device_user_ctl_cq) ||
        !CreateJetty(state->api, state->device_context, worker.device_user_ctl_cq,
                      state->device_transport_token_handle, state->device_transport_token_value,
                      DirectJettyProfile::kAivDevUsed, kDeviceUserCtlSqDepth,
                      device_jetty_role.c_str(), &worker.device_user_ctl_jetty)) {
      goto failed;
    }

    uint64_t host_tp = 0U;
    uint64_t device_tp = 0U;
    const std::string host_ctp_role = "host_cpu_" + suffix;
    const std::string device_ctp_role = "device_aiv_" + suffix;
    if (!GetCtp(state->api, state->host_context, pair.host_eid, pair.device_eid, host_ctp_role.c_str(), false,
                &host_tp) ||
        !GetCtp(state->api, state->device_context, pair.device_eid, pair.host_eid, device_ctp_role.c_str(), true,
                &device_tp)) {
      goto failed;
    }
    const uint32_t host_psn = MakeCtpPsn(state->api, identity.physical_device_id, host_ctp_role.c_str());
    const uint32_t device_psn = MakeCtpPsn(state->api, identity.physical_device_id, device_ctp_role.c_str());
    const std::string host_import_role = "host_import_device_" + suffix;
    const std::string device_import_role = "device_import_host_" + suffix;
    // Each core/Jetty slot owns a matched reciprocal Jetty pair.
    if (!ImportJetty(state->api, state->host_context, worker.device_user_ctl_jetty,
                     state->device_transport_token_value, host_tp, device_tp, host_psn, device_psn,
                     host_import_role.c_str(), &worker.host_device_import_info, &worker.host_device_target_jetty) ||
        !ImportJetty(state->api, state->device_context, worker.host_standard_jetty,
                     state->host_memory.token_value, device_tp, host_tp, device_psn, host_psn,
                     device_import_role.c_str(), &worker.device_host_import_info, &worker.device_host_target_jetty)) {
      goto failed;
    }
  }
  for (uint32_t entity_index = 0U; entity_index < state->worker_count; ++entity_index) {
    const DirectAivWorker &worker = state->workers[entity_index];
    KvafBatchPutDirectUrmaEntity &entity = state->entity_templates[entity_index];
    entity.magic = kDirectEntityMagic;
    entity.version = kDirectEntityVersion;
    entity.remote_host_ddr_address = state->host_ddr_address;
    entity.remote_host_ddr_size = state->host_ddr_bytes;
    entity.remote_token_id = state->host_memory.token_id;
    entity.remote_token_value = state->host_memory.token_value;
    entity.sq_va = worker.device_user_ctl_jetty.info.ub.sq_buffer_va;
    entity.sq_db_va = worker.device_user_ctl_jetty.info.ub.db_addr;
    entity.jetty_id = worker.device_user_ctl_jetty.info.ub.id;
    entity.sq_depth = kDeviceUserCtlSqDepth * kWqebbPerWqe;
    entity.tp_id = worker.device_host_import_info.out.ub.tpn;
    entity.wqe_size = kWqeBytes;
    std::memcpy(entity.remote_eid, state->wqe_remote_eid.data(), sizeof(entity.remote_eid));
    entity.cq_va = worker.device_cq_info.out.buffer_addr;
    entity.cq_db_va = worker.device_cq_info.out.swdb_addr;
    entity.jfc_id = worker.device_cq_info.out.id;
    entity.cq_depth = kDeviceUserCtlCqDepth;
    entity.cqe_size = worker.device_cq_info.out.cqe_size;
    if (entity.sq_va == 0U || entity.sq_db_va == 0U || entity.cq_va == 0U ||
        entity.cq_db_va == 0U || entity.cqe_size < 32U || entity.tp_id == 0U) {
      goto failed;
    }
    for (uint32_t previous = 0U; previous < entity_index; ++previous) {
      const KvafBatchPutDirectUrmaEntity &other = state->entity_templates[previous];
      if (entity.sq_va == other.sq_va || entity.sq_db_va == other.sq_db_va ||
          entity.cq_va == other.cq_va || entity.cq_db_va == other.cq_db_va ||
          entity.jetty_id == other.jetty_id) {
        goto failed;
      }
    }
  }
  if (aclrtMalloc(&state->device_queue_state, queue_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      state->device_queue_state == nullptr ||
      aclrtMemset(state->device_queue_state, queue_bytes, 0, queue_bytes) != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "[KVAF_AIV] DIRECT_QUEUE_STATE_ALLOC_FAILED cores=%u jettys_per_core=%u\n",
                 aiv_core_count, jettys_per_core);
    goto failed;
  }

  *transport = state;
  return KVAF_BATCH_PUT_SUCCESS;

failed:
  ReleaseTransport(state);
  return result;
}

extern "C" void KvafBatchPutTransportDestroy(KvafBatchPutTransport *transport) {
  ReleaseTransport(transport);
}

extern "C" int32_t KvafBatchPutLocalBufferAcquire(KvafBatchPutTransport *transport,
                                                    void *input_device, uint64_t input_device_bytes,
                                                    KvafBatchPutLocalBuffer **local_buffer) {
  if (transport == nullptr || input_device == nullptr || input_device_bytes == 0U || local_buffer == nullptr) {
    return KVAF_BATCH_PUT_INVALID_ARGUMENT;
  }
  *local_buffer = nullptr;
  std::lock_guard<std::mutex> lock(transport->mutex);
  const uint64_t input_address = reinterpret_cast<uint64_t>(input_device);
  for (const auto &existing : transport->local_buffers) {
    if (existing->input_hbm_address == input_address &&
        existing->input_hbm_bytes == input_device_bytes) {
      ++existing->reference_count;
      *local_buffer = existing.get();
      hccp::VerboseLog(
          "[KVAF_AIV] DIRECT_LOCAL_BUFFER_CACHE_HIT input=%p bytes=%llu refs=%u\n",
          input_device, static_cast<unsigned long long>(input_device_bytes),
          existing->reference_count);
      return KVAF_BATCH_PUT_SUCCESS;
    }
  }

  auto candidate = std::make_unique<KvafBatchPutLocalBuffer>();
  candidate->transport = transport;
  candidate->input_hbm_address = input_address;
  candidate->input_hbm_bytes = input_device_bytes;
  uint32_t token_id = 0U;
  if (!AllocateToken(transport->api, transport->device_context, "device_local_hbm",
                     &candidate->token_handle, &token_id) ||
      !RegisterMemory(transport->api, transport->device_context, candidate->token_handle,
                      token_id, transport->device_transport_token_value, input_device,
                      input_device_bytes, "device_hbm", &candidate->memory)) {
    if (candidate->memory.handle != nullptr && transport->api.lmem_unregister != nullptr) {
      (void)transport->api.lmem_unregister(transport->device_context, candidate->memory.handle);
    }
    if (candidate->token_handle != nullptr && transport->api.token_free != nullptr) {
      (void)transport->api.token_free(transport->device_context, candidate->token_handle);
    }
    return KVAF_BATCH_PUT_DIRECT_HCCP_ERROR;
  }

  auto entities = transport->entity_templates;
  for (uint32_t entity_index = 0U; entity_index < transport->worker_count; ++entity_index) {
    entities[entity_index].local_hbm_address = input_address;
    entities[entity_index].local_hbm_size = input_device_bytes;
    entities[entity_index].local_token_id = candidate->memory.token_id;
    entities[entity_index].local_token_value = candidate->memory.token_value;
  }
  const uint64_t entity_bytes =
      static_cast<uint64_t>(transport->worker_count) * sizeof(entities.front());
  if (aclrtMalloc(&candidate->device_entities, entity_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
      candidate->device_entities == nullptr ||
      aclrtMemcpy(candidate->device_entities, entity_bytes, entities.data(), entity_bytes,
                  ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
    if (candidate->device_entities != nullptr) (void)aclrtFree(candidate->device_entities);
    if (candidate->memory.handle != nullptr && transport->api.lmem_unregister != nullptr) {
      (void)transport->api.lmem_unregister(transport->device_context, candidate->memory.handle);
    }
    if (candidate->token_handle != nullptr && transport->api.token_free != nullptr) {
      (void)transport->api.token_free(transport->device_context, candidate->token_handle);
    }
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  candidate->device_channel_entity = reinterpret_cast<uint64_t>(candidate->device_entities);
  candidate->reference_count = 1U;
  *local_buffer = candidate.get();
  transport->local_buffers.push_back(std::move(candidate));
  hccp::VerboseLog(
      "[KVAF_AIV] DIRECT_LOCAL_BUFFER_REGISTERED input=%p bytes=%llu cached_mrs=%zu\n",
      input_device, static_cast<unsigned long long>(input_device_bytes),
      transport->local_buffers.size());
  return KVAF_BATCH_PUT_SUCCESS;
}

extern "C" void KvafBatchPutLocalBufferRelease(KvafBatchPutLocalBuffer *local_buffer) {
  if (local_buffer == nullptr) {
    return;
  }
  KvafBatchPutTransport *transport = local_buffer->transport;
  if (transport == nullptr) return;
  std::lock_guard<std::mutex> lock(transport->mutex);
  if (local_buffer->reference_count == 0U) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_LOCAL_BUFFER_RELEASE_WARNING reason=reference_underflow\n");
    return;
  }
  --local_buffer->reference_count;
}

extern "C" int32_t KvafExecuteBatchPutSync(KvafBatchPutLocalBuffer *local_buffer,
                                            const KvafExecuteBatchPutKernelArgs *arguments,
                                            aclrtStream stream,
                                            KvafBatchPutCoreStatus *core_status,
                                            uint32_t core_status_count) {
  if (local_buffer == nullptr || local_buffer->reference_count == 0U ||
      local_buffer->transport == nullptr || arguments == nullptr || stream == nullptr || core_status == nullptr ||
      arguments->input == 0U || arguments->input_offsets == 0U || arguments->lengths == 0U ||
      arguments->destination_addresses == 0U || arguments->core_status == 0U ||
      arguments->input_bytes == 0U || arguments->row_count == 0U ||
      arguments->aiv_core_count == 0U || arguments->aiv_core_count != core_status_count ||
      arguments->aiv_core_count != local_buffer->transport->aiv_core_count ||
      arguments->input != local_buffer->input_hbm_address ||
      arguments->input_bytes > local_buffer->input_hbm_bytes) {
    return KVAF_BATCH_PUT_INVALID_ARGUMENT;
  }

  KvafBatchPutTransport *transport = local_buffer->transport;
  std::lock_guard<std::mutex> launch_lock(transport->launch_mutex);
  const uint64_t status_bytes = static_cast<uint64_t>(core_status_count) * sizeof(*core_status);
  const aclError clear_result =
      aclrtMemset(reinterpret_cast<void *>(arguments->core_status), status_bytes, 0, status_bytes);
  if (clear_result != ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_BATCH_PUT_RUNTIME_ERROR phase=clear_status acl=%d\n",
                 static_cast<int>(clear_result));
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  KvafExecuteBatchPutLaunchArgs launch_args{};
  launch_args.request = *arguments;
  launch_args.channel_entity = local_buffer->device_channel_entity;
  launch_args.queue_state = reinterpret_cast<uint64_t>(transport->device_queue_state);
  launch_args.jettys_per_core = transport->jettys_per_core;
  int32_t result = KvafExecuteBatchPutLaunchInternal(&launch_args, stream);
  if (result != KVAF_BATCH_PUT_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_BATCH_PUT_RUNTIME_ERROR phase=launch result=%d\n", result);
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  const aclError synchronize_result = aclrtSynchronizeStream(stream);
  if (synchronize_result != ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_BATCH_PUT_RUNTIME_ERROR phase=synchronize acl=%d\n",
                 static_cast<int>(synchronize_result));
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  const aclError status_copy_result =
      aclrtMemcpy(core_status, status_bytes, reinterpret_cast<void *>(arguments->core_status), status_bytes,
                  ACL_MEMCPY_DEVICE_TO_HOST);
  if (status_copy_result != ACL_SUCCESS) {
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_BATCH_PUT_RUNTIME_ERROR phase=copy_status acl=%d\n",
                 static_cast<int>(status_copy_result));
    return KVAF_BATCH_PUT_RUNTIME_ERROR;
  }
  for (uint32_t core = 0U; core < core_status_count; ++core) {
    const uint64_t begin = static_cast<uint64_t>(arguments->row_count) * core / core_status_count;
    const uint64_t end = static_cast<uint64_t>(arguments->row_count) * (core + 1U) / core_status_count;
    if (core_status[core].magic != KVAF_AIV_BATCH_PUT_STATUS_MAGIC ||
        core_status[core].result != KVAF_BATCH_PUT_CORE_SUCCESS ||
        core_status[core].processed_request_count != end - begin ||
        core_status[core].failed_request_index != UINT64_MAX) {
      return KVAF_BATCH_PUT_KERNEL_ERROR;
    }
  }
  return KVAF_BATCH_PUT_SUCCESS;
}
