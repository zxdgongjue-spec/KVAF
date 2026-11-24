#include "peer_link.h"
#include "execution_protocol.h"
#include "gather_workspace.h"
#include <atomic>
#include <random>

#include <arpa/inet.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <charconv>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace kvaf::peer {
namespace {

void SetError(std::string* error, std::string text) {
  if (error != nullptr) *error = std::move(text);
}

Status ToStatus(const std::string& operation, std::string error) {
  if (error.empty()) error = "peer request failed";
  return Status::Internal(operation + ": " + error);
}

bool WriteAll(int fd, const void* data, size_t bytes, std::string* error) {
  const auto* cursor = static_cast<const uint8_t*>(data);
  for (size_t written = 0U; written < bytes;) {
    const ssize_t result = send(fd, cursor + written, bytes - written, MSG_NOSIGNAL);
    if (result <= 0) {
      SetError(error, "TCP write failed: " + std::string(std::strerror(errno)));
      return false;
    }
    written += static_cast<size_t>(result);
  }
  return true;
}

bool ReadAll(int fd, void* data, size_t bytes, std::string* error) {
  auto* cursor = static_cast<uint8_t*>(data);
  for (size_t read_bytes = 0U; read_bytes < bytes;) {
    const ssize_t result = recv(fd, cursor + read_bytes, bytes - read_bytes, 0);
    if (result <= 0) {
      SetError(error, "TCP read failed: " +
                          std::string(result == 0 ? "peer closed" : std::strerror(errno)));
      return false;
    }
    read_bytes += static_cast<size_t>(result);
  }
  return true;
}

bool SplitEndpoint(const std::string& endpoint, std::string* host, std::string* port,
                   std::string* error) {
  const size_t separator = endpoint.rfind(':');
  if (separator == std::string::npos || separator == 0U ||
      separator + 1U == endpoint.size()) {
    SetError(error, "peer service address must be host:port");
    return false;
  }
  *host = endpoint.substr(0U, separator);
  if (host->size() > 1 && host->front() == '[' && host->back() == ']')
    *host = host->substr(1, host->size() - 2);
  *port = endpoint.substr(separator + 1U);
  return true;
}

int Connect(const std::string& endpoint, std::string* error) {
  std::string host;
  std::string port;
  if (!SplitEndpoint(endpoint, &host, &port, error)) return -1;
  addrinfo hints{};
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_family = AF_UNSPEC;
  addrinfo* addresses = nullptr;
  const int lookup = getaddrinfo(host.c_str(), port.c_str(), &hints, &addresses);
  if (lookup != 0) {
    SetError(error, "cannot resolve peer service: " + std::string(gai_strerror(lookup)));
    return -1;
  }
  int fd = -1;
  for (addrinfo* current = addresses; current != nullptr; current = current->ai_next) {
    fd = socket(current->ai_family, current->ai_socktype, current->ai_protocol);
    if (fd >= 0 && connect(fd, current->ai_addr, current->ai_addrlen) == 0) break;
    if (fd >= 0) close(fd);
    fd = -1;
  }
  freeaddrinfo(addresses);
  if (fd < 0) {
    SetError(error, "cannot connect to peer service: " + std::string(std::strerror(errno)));
  }
  return fd;
}

bool ResultOk(Result result, std::string* error) {
  if (result == Result::kOk) return true;
  SetError(error, "peer service rejected request, result=" +
                      std::to_string(static_cast<int32_t>(result)));
  return false;
}

// Normalize IPv4-mapped IPv6 addresses before comparing with local interfaces.
in6_addr IpAddress(const sockaddr* addr) {
  if (addr->sa_family == AF_INET6) return reinterpret_cast<const sockaddr_in6*>(addr)->sin6_addr;
  in6_addr ip{};
  ip.s6_addr[10] = ip.s6_addr[11] = 0xff;
  std::memcpy(ip.s6_addr + 12, &reinterpret_cast<const sockaddr_in*>(addr)->sin_addr, 4);
  return ip;
}

struct ConnectedPeer {
  bool local{};
  std::string endpoint;
};

bool IdentifyPeer(int fd, ConnectedPeer* peer, std::string* error) {
  sockaddr_storage storage{};
  socklen_t bytes = sizeof(storage);
  auto* addr = reinterpret_cast<sockaddr*>(&storage);
  char host[NI_MAXHOST], port[NI_MAXSERV];
  if (getpeername(fd, addr, &bytes) != 0 ||
      (addr->sa_family != AF_INET && addr->sa_family != AF_INET6) ||
      getnameinfo(addr, bytes, host, sizeof(host), port, sizeof(port),
                  NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
    SetError(error, "cannot identify connected CPU service address");
    return false;
  }
  peer->endpoint = std::string(host) + ":" + port;
  const auto ip = IpAddress(addr);
  peer->local = IN6_IS_ADDR_LOOPBACK(&ip) || (IN6_IS_ADDR_V4MAPPED(&ip) && ip.s6_addr[12] == 127);
  if (peer->local) return true;
  ifaddrs* interfaces = nullptr;
  if (getifaddrs(&interfaces) != 0) {
    SetError(error, "cannot enumerate local network interfaces");
    return false;
  }
  for (auto* it = interfaces; it; it = it->ifa_next) {
    if (!it->ifa_addr || (it->ifa_addr->sa_family != AF_INET && it->ifa_addr->sa_family != AF_INET6))
      continue;
    const auto local = IpAddress(it->ifa_addr);
    if (std::memcmp(&local, &ip, sizeof(ip)) != 0) continue;
    if (IN6_IS_ADDR_LINKLOCAL(&ip) &&
        reinterpret_cast<const sockaddr_in6*>(addr)->sin6_scope_id != if_nametoindex(it->ifa_name))
      continue;
    peer->local = true;
    break;
  }
  freeifaddrs(interfaces);
  return true;
}

template <typename Request, typename Response>
bool Exchange(const std::string& endpoint, const Request& request, Response* response,
              std::string* error, ConnectedPeer* peer = nullptr) {
  if (response == nullptr) {
    SetError(error, "response must not be null");
    return false;
  }
  const int fd = Connect(endpoint, error);
  if (fd < 0) return false;
  const bool transferred = WriteAll(fd, &request, sizeof(request), error) &&
                           ReadAll(fd, response, sizeof(*response), error) &&
                           (!peer || IdentifyPeer(fd, peer, error));
  close(fd);
  return transferred;
}

template <class Request, class Response>
bool Exchange(const PeerLink& link, const Request& request, Response* response,
              std::string* error) {
  return link.Control(&request, sizeof(request), response, sizeof(*response), error);
}

bool GetCandidateCountAt(const std::string& endpoint, uint32_t* candidate_count,
                         uint32_t* device_count,
                         bool* device_filter_applied,
                         std::string* error, ConnectedPeer* peer = nullptr,
                         Command command = Command::kGetCandidateCount) {
  if (candidate_count == nullptr) {
    SetError(error, "candidate count output must not be null");
    return false;
  }
  CandidateCountRequest request{};
  request.header = MakeHeader(command, sizeof(request));
  CandidateCountResponse response{};
  if (!Exchange(endpoint, request, &response, error, peer) ||
      !IsValidResponseHeader(response.header, command, sizeof(response)) ||
      !ResultOk(static_cast<Result>(response.result), error) || response.candidate_count == 0U) {
    if (error != nullptr && error->empty()) *error = "peer service returned no usable UMDK/EID candidates";
    return false;
  }
  if (response.device_count == 0U || response.device_count > response.candidate_count ||
      response.device_filter_applied > 1U) {
    SetError(error, "peer service returned invalid device count");
    return false;
  }
  *candidate_count = response.candidate_count;
  if (device_count != nullptr) *device_count = response.device_count;
  if (device_filter_applied != nullptr) {
    *device_filter_applied = response.device_filter_applied != 0U;
  }
  return true;
}

bool GetCandidateInfoAt(const std::string& endpoint, uint32_t candidate_id,
                        PeerCandidateInfo* info, bool with_eid, std::string* error) {
  if (info == nullptr) {
    SetError(error, "candidate info output must not be null");
    return false;
  }
  CandidateEidResponse record{};
  auto& response = record.info;
  bool received;
  if (with_eid) {
    CandidateEidRequest request{};
    request.info.header = MakeHeader(Command::kGetCandidateInfo, sizeof(request));
    request.info.candidate_id = candidate_id;
    request.response_bytes = sizeof(record);
    received = Exchange(endpoint, request, &record, error);
  } else {
    CandidateInfoRequest request{};
    request.header = MakeHeader(Command::kGetCandidateInfo, sizeof(request));
    request.candidate_id = candidate_id;
    received = Exchange(endpoint, request, &response, error);
  }
  if (!received || !IsValidResponseHeader(response.header, Command::kGetCandidateInfo,
                                          with_eid ? sizeof(record) : sizeof(response)) ||
      !ResultOk(static_cast<Result>(response.result), error) ||
      response.candidate_id != candidate_id) {
    if (error != nullptr && error->empty()) {
      *error = "peer service returned invalid candidate metadata";
    }
    return false;
  }
  const void* name_end = std::memchr(response.device_name, '\0', kDeviceNameBytes);
  if (response.device_name[0] == '\0' || name_end == nullptr) {
    SetError(error, "peer service returned an invalid candidate device name");
    return false;
  }
  PeerCandidateInfo candidate{};
  candidate.candidate_id = response.candidate_id;
  candidate.device_id = response.device_id;
  candidate.eid_index = response.eid_index;
  candidate.device_name.assign(
      response.device_name,
      static_cast<const char*>(name_end) - response.device_name);
  std::copy_n(record.eid, candidate.eid.size(), candidate.eid.begin());
  if (with_eid && candidate.eid == decltype(candidate.eid){}) {
    SetError(error, "CPU candidate has zero EID");
    return false;
  }
  *info = std::move(candidate);
  return true;
}

bool AllocateRegionAt(const std::string& endpoint, uint64_t ddr_bytes, uint64_t slot_size,
                      uint32_t candidate_id, PeerRegion* region, std::string* error) {
  if (region == nullptr || ddr_bytes == 0U || slot_size == 0U ||
      ddr_bytes % slot_size != 0U) {
    SetError(error, "invalid peer region allocation request");
    return false;
  }
  AllocateRegionRequest request{};
  request.header = MakeHeader(Command::kAllocateRegion, sizeof(request));
  request.ddr_bytes = ddr_bytes;
  request.slot_size = slot_size;
  request.candidate_id = candidate_id;
  AllocateRegionResponse response{};
  if (!Exchange(endpoint, request, &response, error) ||
      !IsValidResponseHeader(response.header, Command::kAllocateRegion, sizeof(response)) ||
      !ResultOk(static_cast<Result>(response.result), error) || response.session_id == 0U ||
      response.peer_base == 0U || response.peer_length != ddr_bytes ||
      response.token_id == 0U || response.token_value == 0U ||
      response.segment_key_bytes == 0U || response.segment_key_bytes > kSegKeyBytes) {
    if (error != nullptr && error->empty()) *error = "peer service returned an invalid region descriptor";
    return false;
  }
  PeerRegion allocated{};
  allocated.session_id = response.session_id;
  allocated.base = response.peer_base;
  allocated.length = response.peer_length;
  allocated.token_id = response.token_id;
  allocated.token_value = response.token_value;
  allocated.segment_key_bytes = response.segment_key_bytes;
  std::memcpy(allocated.eid.data(), response.peer_eid, allocated.eid.size());
  std::memcpy(allocated.segment_key.data(), response.segment_key, allocated.segment_key.size());
  *region = allocated;
  return true;
}

bool CreateChannelAt(const PeerLink& endpoint, const PeerRegion& region,
                     const uint8_t local_qp_key[kQpKeyBytes], uint32_t local_qp_key_bytes,
                     const uint8_t local_eid[kEidBytes], uint32_t peer_token_value,
                     uint64_t local_tp_handle, uint32_t local_psn,
                     std::array<uint8_t, kQpKeyBytes>* peer_qp_key, uint32_t* peer_qp_key_bytes,
                     uint64_t* peer_tp_handle, uint32_t* peer_psn, bool share_tp,
                     std::string* error) {
  if (local_qp_key == nullptr || local_eid == nullptr || peer_qp_key == nullptr ||
      peer_qp_key_bytes == nullptr || peer_tp_handle == nullptr || peer_psn == nullptr ||
      local_qp_key_bytes == 0U || local_qp_key_bytes > kQpKeyBytes ||
      std::all_of(local_eid, local_eid + kEidBytes,
                  [](uint8_t byte) { return byte == 0U; }) ||
      peer_token_value == 0U || local_tp_handle == 0U || local_psn == 0U) {
    SetError(error, "invalid peer channel request");
    return false;
  }
  CreateChannelRequest request{};
  request.header = MakeHeader(Command::kCreateChannel, sizeof(request));
  request.session_id = region.session_id;
  request.local_tp_handle = local_tp_handle;
  request.local_qp_key_bytes = local_qp_key_bytes;
  request.peer_token_value = peer_token_value;
  request.local_psn = local_psn;
  request.reserved = share_tp ? kChannelSharedTp : 0U;
  std::memcpy(request.local_eid, local_eid, sizeof(request.local_eid));
  std::memcpy(request.local_qp_key, local_qp_key, local_qp_key_bytes);
  CreateChannelResponse response{};
  if (!Exchange(endpoint, request, &response, error) ||
      !IsValidResponseHeader(response.header, Command::kCreateChannel, sizeof(response)) ||
      !ResultOk(static_cast<Result>(response.result), error) ||
      response.peer_qp_key_bytes == 0U || response.peer_qp_key_bytes > kQpKeyBytes ||
      response.peer_tp_handle == 0U || response.peer_psn == 0U) {
    if (error != nullptr && error->empty()) *error = "peer service returned an invalid channel descriptor";
    return false;
  }
  if (share_tp && (response.reserved & kChannelSharedTp) == 0U) {
    SetError(error, "CPU service did not acknowledge shared TP; rebuild and restart the CPU service");
    return false;
  }
  std::memcpy(peer_qp_key->data(), response.peer_qp_key, peer_qp_key->size());
  *peer_qp_key_bytes = response.peer_qp_key_bytes;
  *peer_tp_handle = response.peer_tp_handle;
  *peer_psn = response.peer_psn;
  return true;
}

bool ReleaseRegionAt(const std::string& endpoint, const PeerRegion& region, std::string* error) {
  ReleaseRegionRequest request{};
  request.header = MakeHeader(Command::kReleaseRegion, sizeof(request));
  request.session_id = region.session_id;
  StatusResponse response{};
  if (!Exchange(endpoint, request, &response, error) ||
      !IsValidResponseHeader(response.header, Command::kReleaseRegion, sizeof(response)) ||
      !ResultOk(static_cast<Result>(response.result), error)) {
    if (error != nullptr && error->empty()) *error = "CPU service returned an invalid release response";
    return false;
  }
  return true;
}

}  // namespace

struct PeerLink::ExecutionState {
  mutable std::mutex mutex;
  execution::Request create;
  execution::Response reply;
  std::vector<std::uint8_t> pending;
  std::uint64_t pending_id = 0;
  std::uint64_t pull_request = 0;
  std::uint32_t pull_candidate = 0;
  bool pull_ready = false;
  PeerRegion pull_binding;
  bool confirmed = false, closed = false;
};
namespace {
std::uint64_t ClientId() {
  const auto id = [] {
    std::random_device r;
    auto n = (std::uint64_t(r()) << 32) ^ r();
    return n ? n : 1;
  }();
  return id;
}
std::uint64_t NextRequest() {
  static std::atomic<std::uint64_t> next{1};
  return next.fetch_add(1);
}
Status SendExecution(const std::string& address, const execution::Request& q,
                     execution::Response* r) {
  std::string error;
  if (!execution::Exchange(address, q, r, &error)) return Status::CompletionUnknown(error);
  return Status::OK();
}
}  // namespace
Status PeerLink::AllocateCacheRegion(std::uint64_t bytes, std::uint64_t slot, std::uint64_t aux,
                                     TransportMode mode, PeerRegion* out) {
  if (!out) return Status::InvalidArgument("null region output");
  if (execution_ && execution_->closed) execution_.reset();
  if (!execution_) {
    execution::Request q;
    execution::Response caps;
    auto status = SendExecution(address_, q, &caps);
    if (!status.ok()) return status;
    if (!caps.status.ok()) return caps.status;
    const auto required = execution::kRequired |
                          (mode != TransportMode::Ubmem ? execution::kUrma : 0) |
                          (mode != TransportMode::Urma ? execution::kFabric : 0);
    if ((caps.capabilities & required) != required)
      return Status::NotImplemented("CPU service lacks Region/session capabilities");
    execution_ = std::make_shared<ExecutionState>();
    q.op = execution::Op::CreateRegion;
    q.client = ClientId();
    q.request = NextRequest();
    q.epoch = caps.epoch;
    q.logical_bytes = bytes;
    q.slot_size = slot;
    q.auxiliary_bytes = aux;
    q.mode = mode;
    execution_->create = q;
  }
  auto& q = execution_->create;
  if (q.logical_bytes != bytes || q.slot_size != slot || q.auxiliary_bytes != aux || q.mode != mode)
    return Status::InvalidState("unresolved allocation has different parameters; close first");
  auto s = PrepareExecution();
  if (s.ok()) *out = execution_->reply.binding;
  return s;
}
std::shared_ptr<PeerLink> PeerLink::ExecutionSession(const execution::Request& config,
                                                     const PeerCandidateInfo* candidate) {
  auto child = std::make_shared<PeerLink>(address_);
  child->parent_ = shared_from_this();
  child->execution_ = std::make_shared<ExecutionState>();
  auto& q = child->execution_->create;
  q = config;
  if (candidate) {
    q.candidate = candidate->candidate_id;
    q.payload.assign(candidate->eid.begin(), candidate->eid.end());
    child->selected_cpu_eid_ = candidate->eid;
  } else if (config.transport == Transport::Urma) {
    child->selected_cpu_eid_ = selected_cpu_eid_;
  }
  q.op = execution::Op::OpenSession;
  q.client = ClientId();
  q.request = NextRequest();
  if (execution_) {
    q.client = execution_->create.client;
    q.region = execution_->reply.region;
    q.epoch = execution_->reply.epoch;
  }
  return child;
}
Status PeerLink::PrepareCpuPullRegion(uint64_t base, uint32_t candidate, PeerRegion* out) {
  if (!out || !execution_ || parent_) return Status::InvalidArgument("CPU Pull requires an owned cache region");
  auto& state = *execution_;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.confirmed || state.closed || !state.reply.status.ok() ||
      state.reply.binding.base != base || state.create.mode == TransportMode::Ubmem)
    return Status::InvalidState("CPU Pull requires a ready URMA-enabled region");
  if (state.pull_request && state.pull_candidate != candidate)
    return Status::InvalidState("CPU Pull destination route is immutable");
  if (state.pull_ready) { *out = state.pull_binding; return Status::OK(); }
  if (!state.pull_request) {
    execution::Request caps_request;
    execution::Response caps;
    auto status = SendExecution(address_, caps_request, &caps);
    if (!status.ok()) return status;
    if (!caps.status.ok()) return caps.status;
    if (caps.epoch != state.reply.epoch) return Status::CompletionUnknown("CPU service epoch changed");
    if (!(caps.capabilities & execution::kCpuPull))
      return Status::NotImplemented("CPU service has no CPU Pull registration capability");
    state.pull_request = NextRequest(); state.pull_candidate = candidate;
  }
  auto q = state.create;
  q.op = execution::Op::PrepareCpuPullRegion;
  q.request = state.pull_request; q.candidate = candidate;
  q.region = state.reply.region; q.epoch = state.reply.epoch; q.session = 0;
  execution::Response response;
  auto status = SendExecution(address_, q, &response);
  if (!status.ok()) return status;
  if (!response.status.ok()) return response.status;
  if (response.state != execution::State::Ready || response.region != q.region || response.epoch != q.epoch ||
      response.binding.base != base || response.binding.length != state.reply.logical_bytes ||
      response.slot_size != state.reply.slot_size || !response.binding.token_value)
    return Status::CompletionUnknown("CPU Pull destination response identity mismatch");
  state.pull_binding = response.binding;
  state.pull_binding.epoch = response.epoch; state.pull_binding.region_id = response.region;
  state.pull_ready = true; *out = state.pull_binding;
  return Status::OK();
}
Status PeerLink::PrepareExecution() {
  if (!execution_) return Status::InvalidState("no execution identity");
  auto& state = *execution_;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.closed) return Status::InvalidState("execution closed");
  if (!state.confirmed) {
    auto s = SendExecution(address_, state.create, &state.reply);
    if (!s.ok()) return s;
    state.confirmed = true;
  }
  if (!state.reply.status.ok()) return state.reply.status;
  if (state.reply.state != execution::State::Ready)
    return Status::InvalidState("execution is not ready");
  const auto& b = state.reply.binding;
  if (state.reply.epoch != state.create.epoch || !state.reply.region || !b.base || !b.length ||
      b.length > UINT64_MAX - b.base ||
      (!parent_ && (state.reply.logical_bytes != state.create.logical_bytes ||
                    state.reply.slot_size != state.create.slot_size ||
                    state.create.auxiliary_bytes > UINT64_MAX - state.create.logical_bytes ||
                    b.length != state.create.logical_bytes + state.create.auxiliary_bytes)) ||
      (parent_ && (state.reply.region != state.create.region || !state.reply.session ||
                   state.reply.logical_bytes != parent_->execution_->reply.logical_bytes ||
                   state.reply.slot_size != parent_->execution_->reply.slot_size ||
                   b.length != parent_->execution_->reply.binding.length)))
    return Status::Internal("execution identity or binding differs from negotiated region");
  if (selected_cpu_eid_ && b.eid != *selected_cpu_eid_)
    return Status::InvalidState("registered CPU EID differs from the selected candidate");
  const auto canonical =
      parent_ ? parent_->execution_->reply.binding.base : state.reply.binding.base;
  regions_[canonical] = state.reply.binding;
  regions_[canonical].epoch = state.reply.epoch;
  regions_[canonical].region_id = state.reply.region;
  return Status::OK();
}
Status PeerLink::QueryExecution(execution::Response* out) const {
  if (!execution_ || !out) return Status::InvalidArgument("no execution identity or output");
  std::lock_guard<std::mutex> lock(execution_->mutex);
  auto q = execution_->create;
  q.op = execution::Op::Query;
  return SendExecution(address_, q, out);
}
Status PeerLink::ResolveExecutionControl() {
  if (!execution_ || !parent_) return Status::OK();
  auto& state = *execution_;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (!state.pending_id) return Status::OK();
  auto q = state.create;
  q.op = execution::Op::SessionCall;
  q.region = state.reply.region;
  q.epoch = state.reply.epoch;
  q.session = state.reply.session;
  q.request = state.pending_id;
  q.payload = state.pending;
  execution::Response response;
  auto s = SendExecution(address_, q, &response);
  if (!s.ok()) return s;
  // A confirmed rejection also resolves uncertainty; teardown can now proceed.
  state.pending_id = 0;
  state.pending.clear();
  return Status::OK();
}
Status PeerLink::CloseExecution() {
  if (!execution_) return Status::OK();
  auto& state = *execution_;
  std::lock_guard<std::mutex> lock(state.mutex);
  if (state.closed) return Status::OK();
  {
    std::lock_guard<std::mutex> mappings(mapping_mutex_);
    if (!ubmem_mappings_.empty()) return Status::Busy("Fabric mappings still retained");
  }
  // Re-send the immutable create on lost replies; the CPU returns its old record.
  if (!state.confirmed) {
    auto s = SendExecution(address_, state.create, &state.reply);
    if (!s.ok()) return s;
    state.confirmed = true;
  }
  auto q = state.create;
  q.region = state.reply.region;
  q.epoch = state.reply.epoch;
  q.session = state.reply.session;
  q.op = parent_ ? execution::Op::CloseSession : execution::Op::ReleaseRegion;
  execution::Response response;
  auto s = SendExecution(address_, q, &response);
  if (!s.ok()) return s;
  if (!response.status.ok()) return response.status;
  if (response.state != execution::State::Closed)
    return Status::InvalidState("close did not confirm resource retirement");
  state.closed = true;
  regions_.clear();
  return Status::OK();
}
bool PeerLink::Control(const void* input, std::size_t bytes, void* output, std::size_t capacity,
                       std::string* error) const {
  if (!execution_ || !parent_) {
    const int fd = Connect(address_, error);
    if (fd < 0) return false;
    const bool ok = WriteAll(fd, input, bytes, error) && ReadAll(fd, output, capacity, error);
    close(fd);
    return ok;
  }
  auto& state = *execution_;
  std::lock_guard<std::mutex> lock(state.mutex);
  auto fail = [&](const std::string& message) {
    SetError(error, message);
    return false;
  };
  if (!state.confirmed || state.closed || !state.reply.status.ok())
    return fail("execution session is not ready");
  const auto* data = static_cast<const std::uint8_t*>(input);
  std::vector<std::uint8_t> payload(data, data + bytes);
  if (state.pending_id && state.pending != payload)
    return fail("unresolved session command; close or retry the original operation");
  if (!state.pending_id) {
    state.pending_id = NextRequest();
    state.pending = payload;
  }
  auto q = state.create;
  q.op = execution::Op::SessionCall;
  q.region = state.reply.region;
  q.epoch = state.reply.epoch;
  q.session = state.reply.session;
  q.request = state.pending_id;
  q.payload = payload;
  execution::Response response;
  auto s = SendExecution(address_, q, &response);
  if (!s.ok()) return fail(s.message());
  state.pending_id = 0;
  state.pending.clear();
  if (!response.status.ok()) return fail(response.status.message());
  if (response.payload.size() != capacity) return fail("invalid session response size");
  std::memcpy(output, response.payload.data(), capacity);
  return true;
}

PeerLink::PeerLink(std::string address) : address_(std::move(address)) {}

Status PeerLink::GetCandidateCount(uint32_t* candidate_count, uint32_t* device_count,
                                   bool* device_filter_applied) const {
  std::string error;
  return GetCandidateCountAt(address_, candidate_count, device_count,
                             device_filter_applied, &error)
             ? Status::OK()
             : ToStatus("peer candidate discovery failed", std::move(error));
}

Status PeerLink::GetCacheCandidateCount(PeerCandidateTable* table) const {
  if (!table) return Status::InvalidArgument("missing CPU candidate table");
  std::string error;
  return GetCandidateCountAt(address_, &table->count, &table->device_count, &table->device_fixed,
                             &error, nullptr, Command::kGetCacheCandidateCount)
             ? Status::OK() : ToStatus("cache candidate discovery failed", std::move(error));
}
Status PeerLink::DiscoverCandidates(PeerCandidateTable* table, bool cache_scope) {
  if (!table) return Status::InvalidArgument("missing CPU candidate table");
  ConnectedPeer peer;
  std::string error;
  if (!GetCandidateCountAt(address_, &table->count, &table->device_count, &table->device_fixed,
                           &error, &peer, cache_scope ? Command::kGetCacheCandidateCount : Command::kGetCandidateCount))
    return ToStatus("CPU candidate discovery failed", std::move(error));
  table->local = peer.local;
  address_ = std::move(peer.endpoint);  // Keep metadata and execution on the same resolved address.
  return Status::OK();
}

namespace {
Status ParseDevices(const std::string& csv, std::vector<std::string>* devices) {
  if (csv.empty()) return Status::OK();
  std::size_t start = 0;
  for (;;) {
    const auto end = csv.find(',', start);
    auto name = csv.substr(start, end - start);
    const char* spaces = " \t\r\n";
    const auto first = name.find_first_not_of(spaces);
    if (first == std::string::npos)
      return Status::InvalidArgument("empty device in peer_ub_device");
    name = name.substr(first, name.find_last_not_of(spaces) - first + 1);
    if (std::find(devices->begin(), devices->end(), name) != devices->end())
      return Status::InvalidArgument("duplicate UB device in peer_ub_device: " + name);
    devices->push_back(std::move(name));
    if (end == std::string::npos) return Status::OK();
    start = end + 1;
  }
}

Status DefaultFeCount(uint32_t* count) {
  constexpr uint32_t kDefaultPeerFeCount = 2;
  *count = kDefaultPeerFeCount;
  const char* value = std::getenv("KVAF_DEFAULT_PEER_FE_COUNT");
  if (!value) return Status::OK();
  const char* end = value + std::strlen(value);
  const auto parsed = std::from_chars(value, end, *count);
  if (parsed.ec != std::errc{} || parsed.ptr != end || !*count)
    return Status::InvalidArgument("KVAF_DEFAULT_PEER_FE_COUNT must be a positive uint32");
  return Status::OK();
}
}  // namespace

Status PeerLink::SelectCpuFes(const PeerCandidateTable& table, const std::string& csv,
                              const std::optional<std::array<uint8_t, kEidBytes>>& pair_cpu_eid,
                              bool single_cpu_fe, CpuFeSelection* selected) const {
  if (!selected || !table.count) return Status::InvalidArgument("empty CPU candidate selection");
  std::vector<std::string> devices;
  auto status = ParseDevices(csv, &devices);
  if (!status.ok()) return status;
  if (devices.size() > 1 && (table.local || single_cpu_fe))
    return Status::InvalidArgument("multiple peer UB devices require a remote CPU and one local NPU EID");

  CpuFeSelection result;
  result.explicit_devices = !devices.empty();
  const bool automatic_multi = devices.empty() && !table.local && !table.device_fixed && !single_cpu_fe;
  const bool match_pair = devices.empty() && table.local && !table.device_fixed && pair_cpu_eid;
  result.requested_count = devices.empty() ? 1U : static_cast<uint32_t>(devices.size());
  if (automatic_multi) {
    status = DefaultFeCount(&result.requested_count);
    if (!status.ok()) return status;
  }
  // Slots preserve the caller's order even when the service enumerates in a different order.
  result.direct_get.resize(devices.size());
  std::size_t found = 0;
  for (uint32_t index = 0; index < table.count; ++index) {
    PeerCandidateInfo info;
    status = GetCandidateInfo(index, &info);
    if (!status.ok()) return status;
    if (info.device_id >= table.device_count) return Status::Internal("invalid CPU device index");
    if (!index) result.primary = info;
    if (!devices.empty()) {
      // Legacy single-device pin accepts the first advertised index, including a service pin.
      if (devices.size() > 1 && info.eid_index != 0) continue;
      const auto it = std::find(devices.begin(), devices.end(), info.device_name);
      if (it == devices.end()) continue;
      auto& slot = result.direct_get[it - devices.begin()];
      if (slot.device_name.empty()) {
        slot = std::move(info);
        ++found;
      }
      if (found == devices.size()) break;
    } else if (automatic_multi) {
      // Candidates are ordered by device and index by the CPU service.
      if (info.eid_index == 0 && (result.direct_get.empty() ||
                                  result.direct_get.back().device_id != info.device_id))
        result.direct_get.push_back(info);
      if (result.direct_get.size() == result.requested_count) break;
    } else if (!match_pair || info.eid == *pair_cpu_eid) {
      result.primary = info;
      break;
    }
  }
  for (std::size_t i = 0; i < devices.size(); ++i)
    if (result.direct_get[i].device_name.empty())
      return Status::NotFound("CPU service did not advertise requested UB device " + devices[i] +
                              (devices.size() > 1 ? " with eid_index=0" : ""));
  if (!automatic_multi && devices.empty()) result.direct_get.push_back(result.primary);
  if (result.explicit_devices) result.primary = result.direct_get.front();
  *selected = std::move(result);
  return Status::OK();
}

Status CpuFeSelection::ForDirectGet(uint64_t workers, std::vector<PeerCandidateInfo>* active) const {
  if (!active) return Status::InvalidArgument("missing Direct Get FE selection output");
  if (direct_get.empty()) return Status::NotFound("no advertised CPU UB device with eid_index=0");
  if (!workers || (explicit_devices && direct_get.size() > workers))
    return Status::InvalidArgument("Direct Get has " + std::to_string(workers) +
        " Jetty workers for " + std::to_string(direct_get.size()) + " peer UB devices");
  active->assign(direct_get.begin(), direct_get.begin() + std::min<uint64_t>(workers, direct_get.size()));
  return Status::OK();
}

Status PeerLink::SelectCpuCandidate(const PeerCandidateTable& table, const std::string& device,
                                    const std::optional<std::array<uint8_t, kEidBytes>>& pair_cpu_eid,
                                    PeerCandidateInfo* selected) {
  if (!selected) return Status::InvalidArgument("missing CPU candidate selection output");
  CpuFeSelection selection;
  auto status = SelectCpuFes(table, device, pair_cpu_eid, true, &selection);
  if (!status.ok()) return status;
  *selected = std::move(selection.primary);
  selected_cpu_eid_ = selected->eid;
  return Status::OK();
}

Status PeerLink::GetCandidateInfo(uint32_t candidate_id,
                                  PeerCandidateInfo* info, bool with_eid) const {
  std::string error;
  return GetCandidateInfoAt(address_, candidate_id, info, with_eid, &error)
             ? Status::OK()
             : ToStatus("peer candidate metadata query failed", std::move(error));
}

Status PeerLink::AllocateRegion(uint64_t ddr_bytes, uint64_t slot_size,
                                uint32_t candidate_id, PeerRegion* region) {
  std::string error;
  PeerRegion allocated{};
  if (!AllocateRegionAt(address_, ddr_bytes, slot_size, candidate_id, &allocated, &error)) {
    return ToStatus("peer DDR allocation failed", std::move(error));
  }
  regions_[allocated.base] = allocated;
  if (region != nullptr) *region = allocated;
  return Status::OK();
}

Status PeerLink::ReleaseRegion(uint64_t base) {
  {
    std::lock_guard<std::mutex> lock(mapping_mutex_);
    if (ubmem_mappings_.count(base)) return Status::Busy("UBmem Device VA must be unmapped before releasing CPU DDR");
  }
  const auto found = regions_.find(base);
  if (found == regions_.end()) return Status::NotFound("peer region was not found");
  if (execution_) return CloseExecution();
  std::string error;
  if (!ReleaseRegionAt(address_, found->second, &error)) {
    return ToStatus("peer DDR release failed", std::move(error));
  }
  const auto session_id = found->second.session_id;
  for (auto it = regions_.begin(); it != regions_.end();) {
    if (it->second.session_id == session_id) it = regions_.erase(it);
    else ++it;
  }
  return Status::OK();
}

Status PeerLink::RetainUbmemMapping(uint64_t base) {
  std::lock_guard<std::mutex> lock(mapping_mutex_);
  const auto found = regions_.find(base);
  if (found == regions_.end() || found->second.fabric_mapping_bytes == 0) {
    return Status::NotFound("UBmem mapping has no allocated Fabric region");
  }
  ++ubmem_mappings_[base];
  return Status::OK();
}

void PeerLink::ReleaseUbmemMapping(uint64_t base) {
  std::lock_guard<std::mutex> lock(mapping_mutex_);
  auto found = ubmem_mappings_.find(base);
  if (found != ubmem_mappings_.end() && --found->second == 0) ubmem_mappings_.erase(found);
}

Status PeerLink::AllocateUbmemRegion(uint64_t bytes, uint64_t slot_size, PeerRegion* region) {
  if (region == nullptr || bytes == 0 || slot_size == 0 || bytes % slot_size != 0) {
    return Status::InvalidArgument("invalid UBmem region request");
  }
  AllocateUbmemRegionRequest request{};
  request.header = MakeHeader(Command::kAllocateUbmemRegion, sizeof(request));
  request.ddr_bytes = bytes;
  request.slot_size = slot_size;
  AllocateUbmemRegionResponse response{};
  std::string error;
  if (!Exchange(*this, request, &response, &error)) {
    return ToStatus("UBmem allocation exchange failed", std::move(error));
  }
  if (!IsValidResponseHeader(response.header, Command::kAllocateUbmemRegion, sizeof(response))) {
    return Status::Internal("peer returned an invalid UBmem response");
  }
  if (response.result != static_cast<int32_t>(Result::kOk)) {
    response.error_message[sizeof(response.error_message) - 1] = '\0';
    return Status::Internal(std::string("CPU UBmem allocation failed: ") + response.error_message);
  }
  if (response.session_id == 0 || response.peer_base == 0 || response.peer_length != bytes ||
      response.peer_base > UINT64_MAX - bytes || response.mapping_bytes < bytes ||
      response.mapping_bytes % (2ULL * 1024 * 1024) != 0) {
    return Status::Internal("peer returned an invalid UBmem region");
  }
  PeerRegion allocated{};
  allocated.session_id = response.session_id;
  allocated.base = response.peer_base;
  allocated.length = response.peer_length;
  allocated.fabric_mapping_bytes = response.mapping_bytes;
  std::memcpy(allocated.fabric_handle.data(), response.fabric_handle, sizeof(response.fabric_handle));
  regions_[allocated.base] = allocated;
  *region = allocated;
  return Status::OK();
}

Status PeerLink::LookupRegion(uint64_t base, PeerRegion* region) const {
  if (region == nullptr) return Status::InvalidArgument("peer region output must not be null");
  const auto found = regions_.find(base);
  if (found == regions_.end()) return Status::NotFound("peer region was not found");
  *region = found->second;
  return Status::OK();
}

Status PeerLink::PrepareGatherWorkspace(uint64_t region_base, uint64_t bytes,
                                        PeerRegion* workspace) {
  if (!workspace || bytes < 128U || bytes % 64U)
    return Status::InvalidArgument("invalid Gather workspace request");
  PeerRegion source;
  auto status = LookupRegion(region_base, &source);
  if (!status.ok()) return status;
  PrepareGatherWorkspaceRequest request{};
  request.header = MakeHeader(Command::kPrepareGatherWorkspace, sizeof(request));
  request.session_id = source.session_id;
  request.bytes = bytes;
  AllocateRegionResponse response{};
  std::string error;
  if (!Exchange(*this, request, &response, &error) ||
      !IsValidResponseHeader(response.header, Command::kPrepareGatherWorkspace, sizeof(response)) ||
      !ResultOk(static_cast<Result>(response.result), &error))
    return ToStatus("peer Gather workspace preparation failed", std::move(error));
  if (response.session_id != source.session_id || !response.peer_base ||
      response.peer_length < bytes || response.peer_base > UINT64_MAX - response.peer_length ||
      !response.token_id || !response.token_value || !response.segment_key_bytes ||
      response.segment_key_bytes > kSegKeyBytes)
    return Status::Internal("peer returned an invalid Gather workspace");
  PeerRegion prepared{};
  prepared.session_id = response.session_id;
  prepared.base = response.peer_base;
  prepared.length = response.peer_length;
  prepared.token_id = response.token_id;
  prepared.token_value = response.token_value;
  prepared.segment_key_bytes = response.segment_key_bytes;
  prepared.epoch = source.epoch;
  prepared.region_id = source.region_id;
  std::memcpy(prepared.eid.data(), response.peer_eid, prepared.eid.size());
  std::memcpy(prepared.segment_key.data(), response.segment_key, prepared.segment_key_bytes);
  regions_[prepared.base] = prepared;
  *workspace = prepared;
  return Status::OK();
}

Status PeerLink::PrepareGatherWindow(uint64_t region_base, uint64_t budget, uint32_t token_bytes,
                                    uint32_t workers, PeerRegion* workspace, uint32_t* capacity) {
  if (!workspace || !capacity || !token_bytes || !workers || workers > kGatherMaxWorkers)
    return Status::InvalidArgument("invalid Gather window configuration");
  PeerRegion source;
  auto status = LookupRegion(region_base, &source);
  if (!status.ok()) return status;
  PrepareGatherWindowRequest request{};
  request.header = MakeHeader(Command::kPrepareGatherWindow, sizeof(request));
  request.session_id = source.session_id; request.budget = budget;
  request.token_bytes = token_bytes; request.workers = workers;
  PrepareGatherWindowResponse response{};
  std::string error;
  if (!Exchange(*this, request, &response, &error) ||
      !IsValidResponseHeader(response.header, Command::kPrepareGatherWindow, sizeof(response)) ||
      !ResultOk(static_cast<Result>(response.result), &error))
    return ToStatus("Gather DDR budget cannot prepare a window (minimum 4096 rows)", std::move(error));
  const auto& allocation = response.workspace;
  if (!ValidGatherWindow(token_bytes, workers, response.window_capacity) ||
      (!budget && response.window_capacity != kDefaultGetWindow) ||
      allocation.session_id != source.session_id || !allocation.peer_base || allocation.peer_base % 64U ||
      allocation.peer_length != GatherRemoteWorkspaceBytes(token_bytes, workers, response.window_capacity) ||
      allocation.peer_base > UINT64_MAX - allocation.peer_length ||
      response.backing_bytes < allocation.peer_length || (budget && response.backing_bytes > budget) ||
      !allocation.token_id || !allocation.token_value || !allocation.segment_key_bytes ||
      allocation.segment_key_bytes > kSegKeyBytes)
    return Status::Internal("peer returned an invalid Gather window");
  PeerRegion prepared{};
  prepared.session_id = allocation.session_id; prepared.base = allocation.peer_base;
  prepared.length = allocation.peer_length; prepared.token_id = allocation.token_id;
  prepared.token_value = allocation.token_value; prepared.segment_key_bytes = allocation.segment_key_bytes;
  prepared.epoch = source.epoch; prepared.region_id = source.region_id;
  std::memcpy(prepared.eid.data(), allocation.peer_eid, prepared.eid.size());
  std::memcpy(prepared.segment_key.data(), allocation.segment_key, prepared.segment_key_bytes);
  regions_[prepared.base] = prepared;
  *workspace = prepared; *capacity = response.window_capacity;
  return Status::OK();
}

Status PeerLink::CreateChannel(
    uint64_t region_base, const uint8_t local_qp_key[kQpKeyBytes],
    uint32_t local_qp_key_bytes, const uint8_t local_eid[kEidBytes],
    uint32_t peer_token_value, uint64_t local_tp_handle, uint32_t local_psn,
    std::array<uint8_t, kQpKeyBytes>* peer_qp_key,
    uint32_t* peer_qp_key_bytes, uint64_t* peer_tp_handle, uint32_t* peer_psn) {
  return CreateChannel(region_base, local_qp_key, local_qp_key_bytes, local_eid,
      peer_token_value, local_tp_handle, local_psn, peer_qp_key, peer_qp_key_bytes,
      peer_tp_handle, peer_psn, false);
}

Status PeerLink::CreateChannel(
    uint64_t region_base, const uint8_t local_qp_key[kQpKeyBytes],
    uint32_t local_qp_key_bytes, const uint8_t local_eid[kEidBytes],
    uint32_t peer_token_value, uint64_t local_tp_handle, uint32_t local_psn,
    std::array<uint8_t, kQpKeyBytes>* peer_qp_key,
    uint32_t* peer_qp_key_bytes, uint64_t* peer_tp_handle, uint32_t* peer_psn, bool share_tp) {
  const auto found = regions_.find(region_base);
  if (found == regions_.end()) return Status::NotFound("peer region was not found");
  std::string error;
  return CreateChannelAt(*this, found->second, local_qp_key, local_qp_key_bytes, local_eid,
                         peer_token_value, local_tp_handle, local_psn, peer_qp_key,
                         peer_qp_key_bytes, peer_tp_handle, peer_psn, share_tp, &error)
             ? Status::OK()
             : ToStatus("peer channel creation failed", std::move(error));
}

Status PeerLink::EnableDirectGather(uint64_t region_base) const {
  const auto found = regions_.find(region_base);
  if (found == regions_.end())
    return Status::NotFound("peer DDR region was not found");
  EnableDirectGatherRequest request{};
  request.header = MakeHeader(Command::kEnableDirectGather, sizeof(request));
  request.session_id = found->second.session_id;
  StatusResponse response{};
  std::string error;
  if (!Exchange(*this, request, &response, &error) ||
      !IsValidResponseHeader(response.header, Command::kEnableDirectGather, sizeof(response)) ||
      !ResultOk(static_cast<Result>(response.result), &error))
    return ToStatus("peer direct gather enable failed", std::move(error));
  return Status::OK();
}

Status PeerLink::EnableGatherLanes(uint64_t region_base, uint64_t control_address,
                                  uint32_t lanes) const {
  const auto found = regions_.find(region_base);
  if (found == regions_.end()) return Status::NotFound("peer DDR region was not found");
  EnableGatherLanesRequest request{};
  request.header = MakeHeader(Command::kEnableGatherLanes, sizeof(request));
  request.session_id = found->second.session_id;
  request.control_address = control_address;
  request.lanes = lanes;
  StatusResponse response{};
  std::string error;
  if (!Exchange(*this, request, &response, &error) ||
      !IsValidResponseHeader(response.header, Command::kEnableGatherLanes, sizeof(response)) ||
      !ResultOk(static_cast<Result>(response.result), &error))
    return ToStatus("peer independent gather enable failed", std::move(error));
  return Status::OK();
}

Status PeerLink::OpenCpuPushGet(uint64_t base, OpenCpuPushGetRequest request,
                                OpenCpuPushGetResponse* response) const {
  const auto found = regions_.find(base);
  if (found == regions_.end()) return Status::NotFound("peer DDR region was not found");
  request.header = MakeHeader(Command::kOpenCpuPushGet, sizeof(request));
  request.session_id = found->second.session_id;
  std::string error;
  if (!Exchange(*this, request, response, &error) ||
      !IsValidResponseHeader(response->header, Command::kOpenCpuPushGet, sizeof(*response)) ||
      !ResultOk(static_cast<Result>(response->result), &error))
    return ToStatus("CPU push setup failed", std::move(error));
  return Status::OK();
}

namespace {
Status PushSessionCommand(const PeerLink& address, uint64_t session, Command command) {
  CpuPushGetSessionRequest request{};
  request.header = MakeHeader(command, sizeof(request));
  request.session_id = session;
  StatusResponse response{};
  std::string error;
  if (!Exchange(address, request, &response, &error) ||
      !IsValidResponseHeader(response.header, command, sizeof(response)) ||
      !ResultOk(static_cast<Result>(response.result), &error))
    return ToStatus("CPU push session command failed", std::move(error));
  return Status::OK();
}
}  // namespace

Status PeerLink::StartCpuPushGet(uint64_t base) const {
  const auto found = regions_.find(base);
  if (found == regions_.end()) return Status::NotFound("peer DDR region was not found");
  return PushSessionCommand(*this, found->second.session_id, Command::kStartCpuPushGet);
}
Status PeerLink::StopCpuPushGet(uint64_t base) const {
  const auto found = regions_.find(base);
  // ReleaseRegion succeeds only after the remote producer has stopped.
  if (found == regions_.end()) return Status::OK();
  return PushSessionCommand(*this, found->second.session_id, Command::kStopCpuPushGet);
}
}  // namespace kvaf::peer
