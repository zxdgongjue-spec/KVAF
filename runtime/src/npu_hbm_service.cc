#include "npu_hbm_service.h"

#include <acl/acl.h>
#include <urma_api.h>
#include <sys/socket.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <unordered_map>
#include "direct_hccp/direct_hccp_remote_transport.h"
#include "direct_hccp/direct_hccp_session.h"
#include "npu_peer_eids.h"
#include "npu_target_channel.h"
#include "peer_protocol.h"

namespace kvaf {
namespace {
namespace h = direct_hccp;
using namespace peer;
using npu_service_detail::TargetChannel;

bool ReadAll(int fd, void* data, std::size_t bytes) {
  auto* cursor = static_cast<char*>(data);
  while (bytes != 0U) {
    const auto n = recv(fd, cursor, bytes, 0);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    cursor += n; bytes -= static_cast<std::size_t>(n);
  }
  return true;
}
bool WriteAll(int fd, const void* data, std::size_t bytes) {
  auto* cursor = static_cast<const char*>(data);
  while (bytes != 0U) {
    const auto n = send(fd, cursor, bytes, MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    cursor += n; bytes -= static_cast<std::size_t>(n);
  }
  return true;
}
template<class Request>
bool ReadRequest(int fd, const MessageHeader& header, Command command, Request* request) {
  request->header = header;
  return IsValidHeader(header, command, sizeof(Request)) &&
      ReadAll(fd, reinterpret_cast<char*>(request) + sizeof(header),
              sizeof(Request) - sizeof(header));
}
template<class Response>
Response ResponseFor(Command command) {
  Response response{};
  response.header = MakeHeader(command, sizeof(response));
  response.header.command = ResponseCommand(command);
  response.result = static_cast<std::int32_t>(Result::kInvalidRequest);
  return response;
}

struct Path {
  const h::DirectHccpApi& api;
  h::DirectHccpSession* session{nullptr};
  void* context{nullptr};
  std::array<std::uint8_t, kEidBytes> eid{};
  explicit Path(const h::DirectHccpApi& value) : api(value) {}
  ~Path() {
    if (context != nullptr) (void)api.ctx_deinit(context);
    if (session != nullptr) h::ReleaseDirectHccpSession(session);
  }
};
struct Region {
  const h::DirectHccpApi& api;
  Path& path;
  std::uint32_t device;
  void* memory{nullptr};
  void* token{nullptr};
  void* mr{nullptr};
  std::uint32_t token_value{0xABCFU};
  std::vector<std::unique_ptr<TargetChannel>> channels;
  Region(const h::DirectHccpApi& value, Path& owner, std::uint32_t device_id)
      : api(value), path(owner), device(device_id) {}
  ~Region() {
    (void)aclrtSetDevice(device);
    channels.clear();
    if (mr != nullptr) (void)api.lmem_unregister(path.context, mr);
    if (token != nullptr) (void)api.token_free(path.context, token);
    if (memory != nullptr) (void)aclrtFree(memory);
  }
};
}  // namespace

struct NpuHbmService::Impl {
  mutable std::mutex mutex;
  NpuHbmServiceOptions options;
  h::DirectHccpApi api{};
  std::vector<std::array<std::uint8_t, kEidBytes>> eids;
  std::unordered_map<std::uint32_t, std::unique_ptr<Path>> paths;
  std::unordered_map<std::uint64_t, std::unique_ptr<Region>> regions;
  std::uint64_t next_session{1U};
  bool ready{false};

  ~Impl() { Reset(); }
  void Reset() {
    // Used on initialization failure/destruction as well as graceful shutdown.
    // Paths and their shared RA sessions outlive all registered HBM/Jettys.
    if (!paths.empty() || !regions.empty()) (void)aclrtSetDevice(options.device_id);
    regions.clear();
    paths.clear();
    eids.clear();
    if (api.library != nullptr) h::CloseDirectHccp(&api);
    ready = false;
  }
  Path* EnsurePath(std::uint32_t candidate) {
    if (candidate >= eids.size()) return nullptr;
    const auto found = paths.find(candidate);
    if (found != paths.end()) return found->second.get();
    auto path = std::make_unique<Path>(api);
    path->eid = eids[candidate];
    std::uint32_t physical = 0U;
    if (!h::GetCurrentNpuPhysicalDeviceId(&physical) ||
        !h::AcquireDirectHccpSession(physical, 18U, &path->session)) return nullptr;
    h::CtxInitCfg config{};
    config.mode = h::kOfflineHdc;
    h::CtxInitAttr attr{};
    attr.phy_id = physical;
    std::copy(path->eid.begin(), path->eid.end(), attr.ub.eid.raw);
    if (api.ctx_init(&config, &attr, &path->context) != 0 || path->context == nullptr)
      return nullptr;
    auto* result = path.get();
    paths.emplace(candidate, std::move(path));
    return result;
  }
  Result Allocate(const AllocateRegionRequest& request, AllocateRegionResponse* response) {
    if (request.ddr_bytes == 0U || request.slot_size == 0U ||
        request.ddr_bytes % request.slot_size != 0U || request.reserved != 0U ||
        request.ddr_bytes > std::numeric_limits<std::size_t>::max() ||
        request.candidate_id >= eids.size()) return Result::kInvalidRequest;
    if (regions.size() >= options.max_sessions || next_session == 0U)
      return Result::kResourceError;
    Path* path = EnsurePath(request.candidate_id);
    if (path == nullptr) return Result::kResourceError;
    auto region = std::make_unique<Region>(api, *path, options.device_id);
    if (aclrtMalloc(&region->memory, request.ddr_bytes,
          options.huge_first ? ACL_MEM_MALLOC_HUGE_FIRST : ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS ||
        region->memory == nullptr) return Result::kResourceError;
    h::TokenId token{};
    if (api.token_alloc(path->context, &token, &region->token) != 0 ||
        region->token == nullptr || (token.token_id >> 8U) == 0U) return Result::kResourceError;
    h::MrRegInfo reg{};
    reg.in.mem.addr = reinterpret_cast<std::uint64_t>(region->memory);
    reg.in.mem.size = request.ddr_bytes;
    reg.in.ub.flags = h::MemRegFlags(true);
    reg.in.ub.token_value = region->token_value;
    reg.in.ub.token_id_handle = region->token;
    if (api.lmem_register(path->context, &reg, &region->mr) != 0 || region->mr == nullptr ||
        reg.out.key.size < sizeof(urma_seg_t) || reg.out.key.size > kSegKeyBytes)
      return Result::kResourceError;
    static_assert(sizeof(urma_seg_t) <= kSegKeyBytes, "HBM segment key exceeds wire capacity");
    urma_seg_t segment{};
    std::memcpy(&segment, reg.out.key.value, sizeof(segment));
    if (segment.ubva.va == 0U || segment.len < request.ddr_bytes ||
        segment.ubva.va > UINT64_MAX - request.ddr_bytes ||
        std::all_of(segment.ubva.eid.raw, segment.ubva.eid.raw + kEidBytes,
                    [](std::uint8_t b) { return b == 0U; })) return Result::kResourceError;
    response->session_id = next_session++;
    response->peer_base = segment.ubva.va;
    response->peer_length = request.ddr_bytes;
    // TokenId from RA is raw, just like the CPU descriptor. The AIV adapter
    // converts it to the WQE format exactly once (raw >> 8).
    response->token_id = token.token_id;
    response->token_value = region->token_value;
    response->segment_key_bytes = reg.out.key.size;
    std::copy_n(segment.ubva.eid.raw, kEidBytes, response->peer_eid);
    std::copy_n(reg.out.key.value, reg.out.key.size, response->segment_key);
    regions.emplace(response->session_id, std::move(region));
    std::fprintf(stderr, "[KVAF_NPU] HBM_ALLOCATED device=%u session=%llu bytes=%llu\n",
        options.device_id, static_cast<unsigned long long>(response->session_id),
        static_cast<unsigned long long>(request.ddr_bytes));
    return Result::kOk;
  }
  Result CreateChannel(const CreateChannelRequest& request, CreateChannelResponse* response) {
    if (request.local_qp_key_bytes == 0U || request.local_qp_key_bytes > kQpKeyBytes ||
        request.local_tp_handle == 0U || request.local_psn == 0U || request.peer_token_value == 0U ||
        std::all_of(request.local_eid, request.local_eid + kEidBytes,
                    [](std::uint8_t b) { return b == 0U; })) return Result::kInvalidRequest;
    auto found = regions.find(request.session_id);
    if (found == regions.end()) return Result::kNotFound;
    Region& region = *found->second;
    if (region.channels.size() >= options.max_channels_per_session) return Result::kResourceError;
    std::array<std::uint8_t, kEidBytes> peer_eid{};
    std::copy_n(request.local_eid, kEidBytes, peer_eid.begin());
    std::uint64_t local_tp = 0U;
    if (!h::QueryPeerCtp(api, region.path.context, region.path.eid, peer_eid,
                         "npu_hbm_target", &local_tp)) return Result::kTransportError;
    auto channel = std::make_unique<TargetChannel>(api, region.path.context);
    if (!channel->Initialize(region.token, region.token_value, 1024U, 1024U, 16384U).ok())
      return Result::kResourceError;
    h::QpImportInfoT imported{};
    if (!h::ImportPeerJetty(api, region.path.context, request.local_qp_key,
                            request.local_qp_key_bytes, request.peer_token_value,
                            local_tp, request.local_tp_handle, 1U, request.local_psn,
                            "npu_hbm_target", &imported, channel->peer_jetty_output()))
      return Result::kTransportError;
    const auto& key = channel->jetty_info().key;
    response->peer_qp_key_bytes = key.size;
    std::copy_n(key.value, key.size, response->peer_qp_key);
    response->peer_tp_handle = local_tp;
    response->peer_psn = 1U;
    region.channels.push_back(std::move(channel));
    return Result::kOk;
  }
  Result Release(std::uint64_t id) {
    if (regions.erase(id) == 0U) return Result::kNotFound;
    std::fprintf(stderr, "[KVAF_NPU] HBM_RELEASED session=%llu remaining=%zu\n",
                 static_cast<unsigned long long>(id), regions.size());
    return Result::kOk;
  }
  void Handle(int fd) {
    MessageHeader header{};
    if (!ReadAll(fd, &header, sizeof(header)) || header.magic != kProtocolMagic ||
        header.version != kProtocolVersion) return;
    const auto command = static_cast<Command>(header.command);
    if (command == Command::kGetCandidateCount) {
      auto response = ResponseFor<CandidateCountResponse>(command);
      if (IsValidHeader(header, command, sizeof(CandidateCountRequest))) {
        response.result = static_cast<std::int32_t>(Result::kOk);
        response.candidate_count = static_cast<std::uint32_t>(eids.size());
        response.device_count = 1U;
        response.device_filter_applied = 1U;
      }
      (void)WriteAll(fd, &response, sizeof(response));
    } else if (command == Command::kGetCandidateInfo) {
      CandidateInfoRequest request{};
      auto response = ResponseFor<CandidateInfoResponse>(command);
      if (ReadRequest(fd, header, command, &request) && request.candidate_id < eids.size()) {
        response.result = static_cast<std::int32_t>(Result::kOk);
        response.candidate_id = request.candidate_id;
        response.device_id = 0U; // Dense device ordinal, not the ACL device ID.
        response.eid_index = request.candidate_id;
        std::snprintf(response.device_name, sizeof(response.device_name), "npu:%u", options.device_id);
      }
      (void)WriteAll(fd, &response, sizeof(response));
    } else if (command == Command::kAllocateHbmRegion || command == Command::kAllocateRegion) {
      AllocateRegionRequest request{};
      auto response = ResponseFor<AllocateRegionResponse>(command);
      if (ReadRequest(fd, header, command, &request) && command == Command::kAllocateHbmRegion)
        response.result = static_cast<std::int32_t>(Allocate(request, &response));
      // A disconnected allocator must not leak a session it never received.
      if (!WriteAll(fd, &response, sizeof(response)) && response.session_id != 0U)
        (void)Release(response.session_id);
    } else if (command == Command::kCreateChannel) {
      CreateChannelRequest request{};
      auto response = ResponseFor<CreateChannelResponse>(command);
      if (ReadRequest(fd, header, command, &request))
        response.result = static_cast<std::int32_t>(CreateChannel(request, &response));
      (void)WriteAll(fd, &response, sizeof(response));
    } else {
      auto response = ResponseFor<StatusResponse>(command);
      ReleaseRegionRequest request{};
      if (command == Command::kReleaseRegion && ReadRequest(fd, header, command, &request))
        response.result = static_cast<std::int32_t>(Release(request.session_id));
      (void)WriteAll(fd, &response, sizeof(response));
    }
  }
};

NpuHbmService::NpuHbmService() : impl_(std::make_unique<Impl>()) {}
NpuHbmService::~NpuHbmService() = default;
Status NpuHbmService::Initialize(const NpuHbmServiceOptions& options) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->ready) return Status::InvalidState("NPU HBM service already initialized");
  if (options.max_sessions == 0U || options.max_channels_per_session == 0U)
    return Status::InvalidArgument("HBM session/channel limits must be nonzero");
  if (aclrtSetDevice(options.device_id) != ACL_SUCCESS)
    return Status::InvalidArgument("cannot select HBM provider device");
  impl_->options = options;
  const Status discovered = DiscoverNpuPeerEids(options.device_id, options.npu_eids, &impl_->eids);
  if (!discovered.ok()) return discovered;
  if (impl_->eids.empty() || !h::LoadDirectHccp(&impl_->api)) {
    impl_->Reset();
    return Status::Internal("cannot initialize HBM provider RA API/EIDs");
  }
  impl_->ready = true;
  return Status::OK();
}
std::size_t NpuHbmService::active_sessions() const {
  std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->regions.size();
}
Status NpuHbmService::LinkTo(const LinkTarget&) { return Status::NotImplemented("HBM service only accepts links"); }
Status NpuHbmService::AcceptLink(const LinkTarget&) {
  return initialized() ? Status::OK() : Status::NotInitialized("HBM service is not initialized");
}
Status NpuHbmService::ReleaseLink(LinkId id) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->ready) return Status::NotInitialized("HBM service is not initialized");
  return impl_->Release(id) == Result::kOk ? Status::OK() : Status::NotFound("HBM session not found");
}
Status NpuHbmService::ServeConnection(int fd) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->ready) return Status::NotInitialized("HBM service is not initialized");
  if (fd < 0) return Status::InvalidArgument("invalid HBM service socket");
  if (aclrtSetDevice(impl_->options.device_id) != ACL_SUCCESS)
    return Status::Internal("cannot select HBM provider device");
  timeval timeout{15, 0};
  (void)setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  (void)setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  impl_->Handle(fd);
  return Status::OK();
}
Status NpuHbmService::Shutdown() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->regions.empty()) return Status::Busy("NPU readers must release their HBM sessions first");
  impl_->Reset();
  return Status::OK();
}
bool NpuHbmService::initialized() const {
  std::lock_guard<std::mutex> lock(impl_->mutex); return impl_->ready;
}
RuntimeCapabilities NpuHbmService::capabilities() const { return {false, false, true, false}; }
}  // namespace kvaf
