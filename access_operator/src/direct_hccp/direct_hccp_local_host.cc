// Local Host DDR RA setup derived from KVAF 720ed3ec; V9 transport and device
// ABIs remain authoritative. No PeerLink or TCP service is used on this path.
#include "direct_hccp/direct_hccp_local_host.h"
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <thread>
#include <unistd.h>

namespace kvaf::direct_hccp {
namespace hccp = kvaf::direct_hccp;
namespace {
constexpr char kUbDevIdPath[] = "/proc/asdrv_ub/dev_id";
constexpr char kUbPairInfoPath[] = "/proc/asdrv_ub/pair_info";
constexpr char kUbSelectorLockPath[] = "/tmp/kvaf_aiv_batch_get_asdrv_ub.lock";
constexpr uint64_t kPageBytes = 4096U;
constexpr uint32_t kTokenIdShift = 8U;
constexpr uint32_t kDeviceRqDepth = 256U;
constexpr uint32_t kWqebbPerWqe = 4U;
constexpr uint32_t kWqeBytes = 64U;
constexpr uint32_t kCtpTpCount = 1U;
constexpr uint32_t kCtpQueryPollMs = 1U;
constexpr uint32_t kCtpQueryTimeoutMs = 10000U;
constexpr int kHccpAsyncAgain = 128301;
using DirectJetty = DeviceAivJetty;
struct EidPair {
  std::array<uint8_t, kEidBytes> host_eid{};
  std::array<uint8_t, kEidBytes> device_eid{};
  uint32_t channel{0};
};
enum class DirectJettyProfile : uint8_t { kHostStandard, kAivDevUsed };
const char* JettyProfileName(DirectJettyProfile p) {
  return p == DirectJettyProfile::kHostStandard ? "host_standard" : "aiv_dev_used";
}
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
    std::fprintf(stderr, "[KVAF_AIV] HOST_DEVICE_UB_EID_PAIR_SELECTED asdrv_ub_dev_id=%u channel=%u\n",
                 physical_device_id, pair->channel);
  }
  return success;
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
      std::fprintf(stderr, "[KVAF_AIV] RaGetSecRandom fallback ret=%d token=0x%x\n", ret, token);
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
  std::fprintf(stderr, "[KVAF_AIV] DIRECT_HCCP_CONTEXT_READY role=%s mode=%d physical=%u context=%p\n",
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
  std::fprintf(stderr, "[KVAF_AIV] DIRECT_TOKEN_READY role=%s token_id=0x%x\n", role, *token_id);
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
  std::fprintf(stderr,
               "[KVAF_AIV] DIRECT_MEMORY_REGISTERED role=%s token_id=0x%x host_or_hbm_va=0x%llx bytes=%llu\n",
               role, memory->token_id, static_cast<unsigned long long>(aligned_address),
               static_cast<unsigned long long>(aligned_bytes));
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
  std::fprintf(stderr,
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
  std::fprintf(stderr,
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
    std::fprintf(stderr,
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
    std::fprintf(stderr, "[KVAF_AIV] DIRECT_CTP_ASYNC_SUBMITTED role=%s request=%p\n", role, request);
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
  std::fprintf(stderr, "[KVAF_AIV] DIRECT_CTP_READY role=%s mode=%s tp_handle=0x%llx\n", role,
               use_hdc_async ? "hdc_async" : "peer_sync", static_cast<unsigned long long>(*tp_handle));
  return true;
}
}  // namespace

bool CreateLocalHostEndpoint(const DirectHccpApi& api, uint32_t physical_device_id,
                             void* ddr, uint64_t bytes, LocalHostEndpoint* endpoint) {
  if (!endpoint || !ddr || !bytes || endpoint->context) return false;
  EidPair pair{};
  if (!SelectEidPair(physical_device_id, &pair)) return false;
  endpoint->host_eid = pair.host_eid;
  endpoint->device_eid = pair.device_eid;
  uint32_t token_id = 0;
  if (!InitContext(api, kPeerOnline, physical_device_id, pair.host_eid,
                   "local_host_ddr", &endpoint->context) ||
      !AllocateToken(api, endpoint->context, "local_host_ddr", &endpoint->token_handle,
                     &token_id)) return false;
  return RegisterMemory(api, endpoint->context, endpoint->token_handle, token_id,
                         MakeTokenValue(api, physical_device_id), ddr, bytes,
                         "local_host_ddr", &endpoint->memory);
}

bool CreateLocalHostChannel(const DirectHccpApi& api, uint32_t physical_device_id,
                            void* device_context, uint32_t device_token_value,
                            const QpCreateInfo& device_jetty, uint32_t queue_index,
                            LocalHostEndpoint* endpoint, PeerChannel* channel) {
  if (!endpoint || !endpoint->context || !channel || queue_index >= endpoint->workers.size())
    return false;
  auto& host = endpoint->workers[queue_index];
  if (!CreateCq(api, endpoint->context, kJfcNormal, 16384U, "local_host_ddr",
                 &host.cq_info, &host.cq) ||
      !CreateJetty(api, endpoint->context, host.cq, endpoint->token_handle,
                    endpoint->memory.token_value, DirectJettyProfile::kHostStandard,
                    8192U, "local_host_ddr", &host.jetty)) return false;
  uint64_t host_tp = 0, device_tp = 0;
  if (!GetCtp(api, endpoint->context, endpoint->host_eid, endpoint->device_eid,
                "local_host_ddr", false, &host_tp) ||
      !GetCtp(api, device_context, endpoint->device_eid, endpoint->host_eid,
                "local_device_aiv", true, &device_tp)) return false;
  const auto host_psn = MakeTokenValue(api, physical_device_id);
  const auto device_psn = MakeTokenValue(api, physical_device_id);
  // Reciprocal imports must agree on the same PSN pair and token values.
  return ImportPeerJetty(api, endpoint->context, device_jetty.key.value,
                           device_jetty.key.size, device_token_value, host_tp, device_tp,
                           host_psn, device_psn, "local_host_import_device",
                           &host.device_import, &host.device_target) &&
         ImportPeerJetty(api, device_context, host.jetty.info.key.value,
                           host.jetty.info.key.size, endpoint->memory.token_value,
                           device_tp, host_tp, device_psn, host_psn,
                           "local_device_import_host", &channel->local_import_info,
                           &channel->local_target_jetty);
}

bool RetireLocalAndDeviceQueues(const DirectHccpApi& api, void* device_context,
                                DeviceAivWorkerResources* workers, uint32_t count,
                                LocalHostEndpoint* endpoint) {
  if (count && !workers) return false;
  if (endpoint && count > endpoint->workers.size()) return false;
  bool ok = true;
  // Never destroy a local queue while its reciprocal import is still live.
  for (uint32_t i = 0; i < count; ++i) {
    auto& device = workers[i];
    if (device.device_peer_target_jetty) {
      if (!api.qp_unimport || api.qp_unimport(device_context, device.device_peer_target_jetty)) ok = false;
      else device.device_peer_target_jetty = nullptr;
    }
    if (endpoint && endpoint->workers[i].device_target) {
      auto& host = endpoint->workers[i];
      if (!api.qp_unimport || api.qp_unimport(endpoint->context, host.device_target)) ok = false;
      else host.device_target = nullptr;
    }
  }
  if (!ok) return false;
  for (uint32_t i = 0; i < count; ++i) {
    auto& device = workers[i];
    if (device.device_user_ctl_jetty.handle) {
      if (!api.qp_destroy || api.qp_destroy(device.device_user_ctl_jetty.handle)) ok = false;
      else device.device_user_ctl_jetty.handle = nullptr;
    }
    if (endpoint && endpoint->workers[i].jetty.handle) {
      auto& host = endpoint->workers[i];
      if (!api.qp_destroy || api.qp_destroy(host.jetty.handle)) ok = false;
      else host.jetty.handle = nullptr;
    }
  }
  if (!ok) return false;
  for (uint32_t i = 0; i < count; ++i) {
    auto& device = workers[i];
    if (device.device_user_ctl_cq) {
      if (!api.cq_destroy || api.cq_destroy(device_context, device.device_user_ctl_cq)) ok = false;
      else device.device_user_ctl_cq = nullptr;
    }
    if (endpoint && endpoint->workers[i].cq) {
      auto& host = endpoint->workers[i];
      if (!api.cq_destroy || api.cq_destroy(endpoint->context, host.cq)) ok = false;
      else host.cq = nullptr;
    }
  }
  return ok;
}

bool ReleaseLocalHostEndpoint(const DirectHccpApi& api, LocalHostEndpoint* endpoint) {
  if (!endpoint) return true;
  if (endpoint->memory.handle) {
    if (!api.lmem_unregister || api.lmem_unregister(endpoint->context, endpoint->memory.handle))
      return false;
    endpoint->memory.handle = nullptr;
  }
  if (endpoint->token_handle) {
    if (!api.token_free || api.token_free(endpoint->context, endpoint->token_handle)) return false;
    endpoint->token_handle = nullptr;
  }
  if (endpoint->context) {
    if (!api.ctx_deinit || api.ctx_deinit(endpoint->context)) return false;
    endpoint->context = nullptr;
  }
  return true;
}
}  // namespace kvaf::direct_hccp
