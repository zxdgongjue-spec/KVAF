#include "npu_service.h"

#include <acl/acl.h>
#include <urma_api.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/file.h>
#include <unistd.h>
#include <fcntl.h>

#include <array>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include "peer_protocol.h"

namespace kvaf {
namespace {

using namespace peer;
using direct_hccp::DirectHccpApi;
using direct_hccp::DirectHccpSession;

constexpr std::uint32_t kHdcType = 18U;
constexpr std::uint32_t kCqDepth = 16384U;
constexpr std::uint32_t kSqDepth = 256U;
constexpr std::uint32_t kRqDepth = 256U;

bool ReadAll(int fd, void* data, std::size_t bytes) {
  auto* p = static_cast<std::uint8_t*>(data);
  for (std::size_t done = 0; done < bytes;) {
    const ssize_t n = recv(fd, p + done, bytes - done, 0);
    if (n <= 0) return false;
    done += static_cast<std::size_t>(n);
  }
  return true;
}

bool WriteAll(int fd, const void* data, std::size_t bytes) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t done = 0; done < bytes;) {
    const ssize_t n = send(fd, p + done, bytes - done, MSG_NOSIGNAL);
    if (n <= 0) return false;
    done += static_cast<std::size_t>(n);
  }
  return true;
}

void SendStatus(int fd, Command command, Result result) {
  StatusResponse response{};
  response.header = MakeHeader(command, sizeof(response));
  response.header.command = ResponseCommand(command);
  response.result = static_cast<std::int32_t>(result);
  (void)WriteAll(fd, &response, sizeof(response));
}

struct HbmRegion {
  std::uint64_t session_id{0U};
  void* memory{nullptr};
  std::uint64_t bytes{0U};
  void* lmem_handle{nullptr};
  void* token_id_handle{nullptr};
  std::uint32_t token_id{0U};
  std::uint32_t token_value{0U};
  std::array<std::uint8_t, kSegKeyBytes> segment_key{};
  std::uint32_t segment_key_bytes{0U};
  std::array<std::uint8_t, kQpKeyBytes> jetty_key{};
  std::array<std::uint8_t, kEidBytes> eid{};
  std::uint32_t jetty_key_bytes{0U};
  void* cq_handle{nullptr};
  void* qp_handle{nullptr};
  direct_hccp::CqInfo cq_info{};
  direct_hccp::QpCreateInfo qp_info{};
  void* peer_target_jetty{nullptr};
};

bool ParseEidHex(const std::string& text, direct_hccp::Eid* eid) {
  if (eid == nullptr || text.size() != 32U) return false;
  auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  for (std::size_t i = 0; i < 16U; ++i) {
    const int hi = hex(text[i * 2U]);
    const int lo = hex(text[i * 2U + 1U]);
    if (hi < 0 || lo < 0) return false;
    eid->raw[i] = static_cast<std::uint8_t>((hi << 4U) | lo);
  }
  return true;
}

bool ReadPairInfoEid(std::uint32_t physical, direct_hccp::Eid* eid) {
  constexpr char kDevPath[] = "/proc/asdrv_ub/dev_id";
  constexpr char kPairPath[] = "/proc/asdrv_ub/pair_info";
  constexpr char kLockPath[] = "/tmp/kvaf_npu_runtime_asdrv_ub.lock";
  const int lock_fd = open(kLockPath, O_CREAT | O_RDWR, 0600);
  if (lock_fd < 0 || flock(lock_fd, LOCK_EX) != 0) {
    if (lock_fd >= 0) close(lock_fd);
    return false;
  }
  bool found = false;
  { std::ofstream selector(kDevPath); if (selector.is_open()) {
      selector << physical << '\n'; selector.close(); std::ifstream input(kPairPath);
      std::string line; bool selected = false, active = false; std::uint32_t active_count = 0, eid_count = 0;
      while (std::getline(input, line)) {
        if (line.find("Dev id info.") != std::string::npos) {
          selected = line.find("dev_id=" + std::to_string(physical) + ";") != std::string::npos;
          active = false; continue;
        }
        if (!selected) continue;
        if (line.find("chan_id=") != std::string::npos) {
          active = line.find("flag=1") != std::string::npos; if (active) ++active_count; continue;
        }
        const std::size_t pos = line.find("remote_eid:");
        if (!active || pos == std::string::npos) continue;
        std::string compact;
        for (unsigned char c : line.substr(pos + 11U)) if (std::isxdigit(c)) compact.push_back(static_cast<char>(c));
        if (ParseEidHex(compact, eid)) ++eid_count;
      }
      found = selected && active_count == 1U && eid_count == 1U;
    }}
  (void)flock(lock_fd, LOCK_UN); close(lock_fd); return found;
}

bool DiscoverNpuEid(std::uint32_t physical, std::uint32_t index,
                    direct_hccp::Eid* eid) {
  const char* configured = std::getenv("KVAF_HCCL_ROOTINFO_PATH");
  const std::string path = configured != nullptr && configured[0] != '\0'
                               ? configured : "/etc/hccl_rootinfo.json";
  std::ifstream input(path);
  if (!input.is_open() || eid == nullptr) return false;
  bool selected = false, layer_one = false, clos = false, expect_addr = false;
  std::uint32_t found_device = 0U;
  std::string line;
  std::vector<direct_hccp::Eid> candidates;
  while (std::getline(input, line)) {
    const auto field = [&](const char* name, std::uint32_t* value) {
      const std::string marker = std::string{"\""} + name + "\"";
      const std::size_t pos = line.find(marker);
      const std::size_t colon = pos == std::string::npos ? std::string::npos : line.find(':', pos);
      if (colon == std::string::npos) return false;
      const std::size_t first = line.find_first_of("0123456789", colon + 1U);
      if (first == std::string::npos) return false;
      *value = static_cast<std::uint32_t>(std::strtoul(line.c_str() + first, nullptr, 10));
      return true;
    };
    std::uint32_t value = 0U;
    if (field("device_id", &value)) {
      selected = value == physical; found_device = found_device || selected;
      layer_one = clos = expect_addr = false; continue;
    }
    if (!selected) continue;
    if (field("net_layer", &value)) { layer_one = value == 1U; clos = expect_addr = false; continue; }
    if (!layer_one) continue;
    if (line.find("\"net_type\"") != std::string::npos) { clos = line.find("\"CLOS\"") != std::string::npos; continue; }
    if (clos && line.find("\"addr_type\"") != std::string::npos) { expect_addr = line.find("\"EID\"") != std::string::npos; continue; }
    if (!expect_addr) continue;
    const std::size_t pos = line.find("\"addr\"");
    const std::size_t first = pos == std::string::npos ? std::string::npos : line.find('"', line.find(':', pos) + 1U);
    const std::size_t second = first == std::string::npos ? std::string::npos : line.find('"', first + 1U);
    if (second == std::string::npos) continue;
    direct_hccp::Eid candidate{};
    if (ParseEidHex(line.substr(first + 1U, second - first - 1U), &candidate)) candidates.push_back(candidate);
    expect_addr = false;
  }
  if (found_device && index < candidates.size()) { *eid = candidates[index]; return true; }
  return ReadPairInfoEid(physical, eid);
}

}  // namespace

struct NpuService::Impl {
  ~Impl() { (void)Shutdown(); }

  Status Initialize(const NpuServiceOptions& options) {
    if (initialized) return Status::InvalidState("NpuService already initialized");
    const auto fail = [this](Status status) {
      (void)Shutdown();
      return status;
    };
    if (aclrtSetDevice(options.device_id) != ACL_SUCCESS)
      return fail(Status::Internal("aclrtSetDevice failed"));
    if (!direct_hccp::LoadDirectHccp(&api))
      return fail(Status::NotImplemented("Direct-HCCP RA symbols are unavailable"));
    if (!direct_hccp::GetCurrentNpuPhysicalDeviceId(&physical_device_id))
      return fail(Status::Internal("cannot resolve NPU physical device id"));
    if (!direct_hccp::AcquireDirectHccpSession(physical_device_id, kHdcType, &session))
      return fail(Status::Internal("failed to acquire Direct-HCCP session"));

    direct_hccp::CtxInitCfg cfg{};
    cfg.mode = direct_hccp::kOfflineHdc;
    direct_hccp::CtxInitAttr attr{};
    attr.phy_id = physical_device_id;
    // Match the existing AIV host InitContext() exactly: the context EID
    // index is always zero; only the discovered EID value varies by device.
    attr.ub.eid_index = 0U;
    bool eid_selected = false;
    if (api.get_dev_eid_info_num != nullptr && api.get_dev_eid_info_list != nullptr) {
      direct_hccp::Eid route_eid{};
      if (ReadPairInfoEid(physical_device_id, &route_eid)) {
        direct_hccp::RaInfo ra_info{direct_hccp::kOfflineHdc, physical_device_id};
        std::uint32_t count = 0U;
        if (api.get_dev_eid_info_num(ra_info, &count) == 0 && count != 0U) {
          std::vector<direct_hccp::DevEidInfo> list(count);
          if (api.get_dev_eid_info_list(ra_info, list.data(), &count) == 0) {
            std::uint32_t selected = UINT32_MAX;
            if (options.has_eid_index && options.eid_index < count) {
              selected = options.eid_index;
            } else {
              for (std::uint32_t i = 0; i < count; ++i) {
                if (std::memcmp(list[i].eid.raw, route_eid.raw, sizeof(route_eid.raw)) == 0) {
                  selected = i;
                  break;
                }
              }
            }
            if (selected != UINT32_MAX) {
              attr.ub.eid = list[selected].eid;
              eid_selected = true;
              std::fprintf(stderr, "[KVAF_NPU] EID_SELECTED source=%s list_index=%u eid_index=%u\n",
                           options.has_eid_index ? "explicit" : "pair_info.remote_eid", selected,
                           list[selected].eid_index);
            }
          }
        }
      }
    }
    if (!eid_selected && !DiscoverNpuEid(physical_device_id, options.eid_index, &attr.ub.eid))
      return fail(Status::NotFound("NPU service could not select an EID from HCCL rootinfo"));
    std::memcpy(local_eid.data(), attr.ub.eid.raw, local_eid.size());
    const int ctx_rc = api.ctx_init(&cfg, &attr, &context);
    if (ctx_rc != 0 || context == nullptr) {
      std::fprintf(stderr, "[KVAF_NPU] RaCtxInit failed rc=%d physical=%u ctx_eid_index=%u eid_list_index=%u eid=",
                   ctx_rc, physical_device_id, attr.ub.eid_index, options.eid_index);
      for (std::uint8_t byte : local_eid) std::fprintf(stderr, "%02x", byte);
      std::fprintf(stderr, "\n");
      return fail(Status::Internal("RaCtxInit failed"));
    }
    initialized = true;
    return Status::OK();
  }

  Status Shutdown() {
    for (auto& entry : regions) Release(entry.second.get());
    regions.clear();
    if (context != nullptr && api.ctx_deinit != nullptr) {
      (void)api.ctx_deinit(context);
      context = nullptr;
    }
    if (session != nullptr) {
      direct_hccp::ReleaseDirectHccpSession(session);
      session = nullptr;
    }
    if (api.library != nullptr) direct_hccp::CloseDirectHccp(&api);
    initialized = false;
    return Status::OK();
  }

  Result Allocate(const AllocateHbmRegionRequest& request,
                  AllocateHbmRegionResponse* response) {
    if (!initialized || response == nullptr || request.hbm_bytes == 0U ||
        request.slot_size == 0U || request.hbm_bytes % request.slot_size != 0U)
      return Result::kInvalidRequest;
    auto region = std::make_unique<HbmRegion>();
    region->bytes = request.hbm_bytes;
    if (aclrtMalloc(&region->memory, request.hbm_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
      return Result::kResourceError;
    direct_hccp::TokenId token_id{};
    if (api.token_alloc == nullptr ||
        api.token_alloc(context, &token_id, &region->token_id_handle) != 0 ||
        region->token_id_handle == nullptr) {
      Release(region.get());
      return Result::kResourceError;
    }
    region->token_id = token_id.token_id;
    direct_hccp::MrRegInfo reg{};
    reg.in.mem.addr = reinterpret_cast<std::uint64_t>(region->memory);
    reg.in.mem.size = request.hbm_bytes;
    reg.in.ub.flags = direct_hccp::MemRegFlags(true);
    reg.in.ub.token_value = 0xABCFU;
    reg.in.ub.token_id_handle = region->token_id_handle;
    if (api.lmem_register(context, &reg, &region->lmem_handle) != 0 ||
        region->lmem_handle == nullptr) {
      Release(region.get());
      return Result::kResourceError;
    }
    region->token_value = reg.in.ub.token_value;
    region->segment_key_bytes = sizeof(urma_seg_t);
    std::memcpy(region->segment_key.data(), reg.out.key.value, region->segment_key_bytes);
    urma_seg_t exported_segment{};
    std::memcpy(&exported_segment, reg.out.key.value, sizeof(exported_segment));
    std::memcpy(region->eid.data(), exported_segment.ubva.eid.raw, region->eid.size());

    // Export a normal UMDK-visible Jetty key. The CPU peer uses this key as
    // the target Jetty for one-sided READ/WRITE operations.
    region->cq_info.in.depth = kCqDepth;
    region->cq_info.in.ub.mode = direct_hccp::kJfcNormal;
    if (api.cq_create(context, &region->cq_info, &region->cq_handle) != 0 ||
        region->cq_handle == nullptr) {
      Release(region.get());
      return Result::kResourceError;
    }
    direct_hccp::QpCreateAttr qp_attr{};
    qp_attr.scq_handle = region->cq_handle;
    qp_attr.rcq_handle = region->cq_handle;
    qp_attr.srq_handle = region->cq_handle;
    qp_attr.sq_depth = kSqDepth;
    qp_attr.rq_depth = kRqDepth;
    qp_attr.transport_mode = direct_hccp::kReliableMessage;
    qp_attr.ub.mode = direct_hccp::kJettyStandard;
    qp_attr.ub.flag = direct_hccp::JettyFlagShareJfr();
    qp_attr.ub.token_id_handle = region->token_id_handle;
    qp_attr.ub.jfs_flag = direct_hccp::JfsFlagErrorSuspend();
    qp_attr.ub.priority = 2U;
    qp_attr.ub.rnr_retry = 7U;
    qp_attr.ub.err_timeout = 16U;
    qp_attr.ub.token_value = region->token_value;
    if (api.qp_create(context, &qp_attr, &region->qp_info, &region->qp_handle) != 0 ||
        region->qp_handle == nullptr || region->qp_info.key.size == 0U) {
      Release(region.get());
      return Result::kResourceError;
    }
    region->jetty_key_bytes = region->qp_info.key.size;
    if (region->jetty_key_bytes > region->jetty_key.size()) {
      Release(region.get());
      return Result::kResourceError;
    }
    std::memcpy(region->jetty_key.data(), region->qp_info.key.value,
                region->jetty_key_bytes);
    region->session_id = next_session++;
    response->result = static_cast<std::int32_t>(Result::kOk);
    response->session_id = region->session_id;
    response->hbm_bytes = region->bytes;
    response->hbm_base = exported_segment.ubva.va;
    response->token_id = region->token_id;
    response->token_value = region->token_value;
    response->segment_key_bytes = region->segment_key_bytes;
    response->jetty_key_bytes = region->jetty_key_bytes;
    std::memcpy(response->segment_key, region->segment_key.data(), kSegKeyBytes);
    std::memcpy(response->jetty_key, region->jetty_key.data(), kQpKeyBytes);
    std::memcpy(response->peer_eid, exported_segment.ubva.eid.raw, kEidBytes);
    regions.emplace(region->session_id, std::move(region));
    return Result::kOk;
  }

  Result Release(std::uint64_t id) {
    auto it = regions.find(id);
    if (it == regions.end()) return Result::kNotFound;
    Release(it->second.get());
    regions.erase(it);
    return Result::kOk;
  }

  Result CreateChannel(const CreateChannelRequest& request,
                       CreateChannelResponse* response) {
    if (response == nullptr || request.session_id == 0U) return Result::kInvalidRequest;
    auto it = regions.find(request.session_id);
    if (it == regions.end()) return Result::kNotFound;
    HbmRegion& region = *it->second;
    if (region.jetty_key_bytes == 0U || request.local_qp_key_bytes == 0U) {
      std::fprintf(stderr, "[KVAF_NPU] CHANNEL_FAILED stage=descriptor session=%llu local_key_bytes=%u jetty_key_bytes=%u\n",
                   static_cast<unsigned long long>(request.session_id), request.local_qp_key_bytes,
                   region.jetty_key_bytes);
      return Result::kResourceError;
    }
    // Import the CPU Jetty into the NPU context so the provider can route
    // one-sided operations to the HBM owner. CTP lookup is asynchronous in
    // the HDC ABI and returns an opaque TP handle.
    std::uint64_t local_tp = 0U;
    if (api.get_tp_info_async != nullptr && api.get_async_req_result != nullptr) {
      direct_hccp::GetTpCfg cfg{};
      cfg.flag = direct_hccp::CtpFlag();
      cfg.transport_mode = direct_hccp::kReliableMessage;
      std::memcpy(cfg.local_eid.raw, local_eid.data(), local_eid.size());
      std::memcpy(cfg.peer_eid.raw, request.local_eid, kEidBytes);
      direct_hccp::TpInfo info{};
      std::uint32_t count = 1U;
      void* async_request = nullptr;
      const int submit_rc = api.get_tp_info_async(context, &cfg, &info, &count, &async_request);
      if (submit_rc == 0 && async_request != nullptr) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        for (int attempt = 0; std::chrono::steady_clock::now() < deadline; ++attempt) {
          int result = 0;
          const int rc = api.get_async_req_result(async_request, &result);
          if (rc == 128301) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
          }
          std::fprintf(stderr,
                       "[KVAF_NPU] CHANNEL_CTP_RESULT api_rc=%d result=%d count=%u tp=0x%llx local_eid=",
                       rc, result, count, static_cast<unsigned long long>(info.tp_handle));
          for (std::uint8_t byte : local_eid) std::fprintf(stderr, "%02x", byte);
          std::fprintf(stderr, " peer_eid=");
          for (std::uint8_t byte : request.local_eid) std::fprintf(stderr, "%02x", byte);
          std::fprintf(stderr, "\n");
          if (rc == 0 && result == 0 && count != 0U) local_tp = info.tp_handle;
          break;
        }
      } else {
        std::fprintf(stderr, "[KVAF_NPU] CHANNEL_FAILED stage=get_tp_info submit_rc=%d count=%u request=%p errno=%d\n",
                     submit_rc, count, async_request, errno);
      }
    }
    if (local_tp == 0U || request.local_tp_handle == 0U) {
      std::fprintf(stderr, "[KVAF_NPU] CHANNEL_FAILED stage=get_tp_info local_tp=0x%llx peer_tp=0x%llx\n",
                   static_cast<unsigned long long>(local_tp),
                   static_cast<unsigned long long>(request.local_tp_handle));
      return Result::kResourceError;
    }
    {
      direct_hccp::QpImportInfoT import_info{};
      if (!direct_hccp::ImportPeerJetty(api, context, request.local_qp_key,
                                        request.local_qp_key_bytes, region.token_value,
                                        local_tp, request.local_tp_handle, 1U,
                                        request.local_psn, "npu_service", &import_info,
                                        &region.peer_target_jetty)) {
        std::fprintf(stderr, "[KVAF_NPU] CHANNEL_FAILED stage=import_cpu_jetty\n");
        return Result::kResourceError;
      }
    }
    response->peer_qp_key_bytes = region.jetty_key_bytes;
    std::memcpy(response->peer_qp_key, region.jetty_key.data(), kQpKeyBytes);
    response->peer_tp_handle = local_tp;
    response->peer_psn = 1U;
    return Result::kOk;
  }

  void Release(HbmRegion* region) {
    if (region == nullptr) return;
    if (region->peer_target_jetty != nullptr && api.qp_unimport != nullptr) {
      (void)api.qp_unimport(context, region->peer_target_jetty);
      region->peer_target_jetty = nullptr;
    }
    if (region->qp_handle != nullptr && api.qp_destroy != nullptr) {
      (void)api.qp_destroy(region->qp_handle);
      region->qp_handle = nullptr;
    }
    if (region->cq_handle != nullptr && api.cq_destroy != nullptr) {
      (void)api.cq_destroy(context, region->cq_handle);
      region->cq_handle = nullptr;
    }
    if (region->lmem_handle != nullptr && api.lmem_unregister != nullptr) {
      (void)api.lmem_unregister(context, region->lmem_handle);
      region->lmem_handle = nullptr;
    }
    if (region->token_id_handle != nullptr && api.token_free != nullptr) {
      (void)api.token_free(context, region->token_id_handle);
      region->token_id_handle = nullptr;
    }
    if (region->memory != nullptr) {
      (void)aclrtFree(region->memory);
      region->memory = nullptr;
    }
  }

  void Handle(int fd) {
    MessageHeader header{};
    if (!ReadAll(fd, &header, sizeof(header)) || header.magic != kProtocolMagic ||
        header.version != kProtocolVersion) return;
    const Command command = static_cast<Command>(header.command);
    if (command == Command::kGetCandidateCount) {
      if (!IsValidHeader(header, command, sizeof(CandidateCountRequest))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      CandidateCountResponse response{};
      response.header = MakeHeader(command, sizeof(response));
      response.header.command = ResponseCommand(command);
      response.result = static_cast<std::int32_t>(initialized ? Result::kOk : Result::kResourceError);
      response.candidate_count = initialized ? 1U : 0U;
      (void)WriteAll(fd, &response, sizeof(response));
      return;
    }
    if (command == Command::kAllocateHbmRegion) {
      AllocateHbmRegionRequest request{};
      request.header = header;
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<std::uint8_t*>(&request) + sizeof(header),
                   sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      AllocateHbmRegionResponse response{};
      response.header = MakeHeader(command, sizeof(response));
      response.header.command = ResponseCommand(command);
      response.result = static_cast<std::int32_t>(Allocate(request, &response));
      (void)WriteAll(fd, &response, sizeof(response));
      return;
    }
    if (command == Command::kReleaseHbmRegion) {
      ReleaseHbmRegionRequest request{};
      request.header = header;
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<std::uint8_t*>(&request) + sizeof(header),
                   sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      SendStatus(fd, command, Release(request.session_id));
      return;
    }
    if (command == Command::kCreateChannel) {
      CreateChannelRequest request{};
      request.header = header;
      if (!IsValidHeader(header, command, sizeof(request)) ||
          !ReadAll(fd, reinterpret_cast<std::uint8_t*>(&request) + sizeof(header),
                   sizeof(request) - sizeof(header))) {
        SendStatus(fd, command, Result::kInvalidRequest);
        return;
      }
      CreateChannelResponse response{};
      response.header = MakeHeader(command, sizeof(response));
      response.header.command = ResponseCommand(command);
      response.result = static_cast<std::int32_t>(CreateChannel(request, &response));
      (void)WriteAll(fd, &response, sizeof(response));
      return;
    }
    SendStatus(fd, command, Result::kInvalidRequest);
  }

  DirectHccpApi api{};
  DirectHccpSession* session{nullptr};
  void* context{nullptr};
  std::uint32_t physical_device_id{0U};
  std::array<std::uint8_t, peer::kEidBytes> local_eid{};
  std::uint64_t next_session{1U};
  bool initialized{false};
  std::unordered_map<std::uint64_t, std::unique_ptr<HbmRegion>> regions;
};

NpuService::NpuService() : impl_(std::make_unique<Impl>()) {}
NpuService::~NpuService() = default;
Status NpuService::Initialize(const NpuServiceOptions& options) { return impl_->Initialize(options); }
Status NpuService::LinkTo(const LinkTarget&) { return Status::NotImplemented("NpuService is a server"); }
Status NpuService::AcceptLink(const LinkTarget&) { return initialized() ? Status::OK() : Status::NotInitialized("NpuService is not initialized"); }
Status NpuService::ReleaseLink(LinkId link_id) { return impl_->Release(link_id) == Result::kOk ? Status::OK() : Status::NotFound("NPU HBM session not found"); }
Status NpuService::ServeConnection(int fd) { if (!initialized()) return Status::NotInitialized("NpuService is not initialized"); impl_->Handle(fd); return Status::OK(); }
Status NpuService::Shutdown() { return impl_->Shutdown(); }
bool NpuService::initialized() const { return impl_ != nullptr && impl_->initialized; }
RuntimeCapabilities NpuService::capabilities() const { return RuntimeCapabilities{false, false, true, false}; }

}  // namespace kvaf
