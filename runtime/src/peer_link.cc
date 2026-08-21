#include "peer_link.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
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

template <typename Request, typename Response>
bool Exchange(const std::string& endpoint, const Request& request, Response* response,
              std::string* error) {
  if (response == nullptr) {
    SetError(error, "response must not be null");
    return false;
  }
  const int fd = Connect(endpoint, error);
  if (fd < 0) return false;
  const bool transferred = WriteAll(fd, &request, sizeof(request), error) &&
                           ReadAll(fd, response, sizeof(*response), error);
  close(fd);
  return transferred;
}

bool GetCandidateCountAt(const std::string& endpoint, uint32_t* candidate_count,
                         std::string* error) {
  if (candidate_count == nullptr) {
    SetError(error, "candidate count output must not be null");
    return false;
  }
  CandidateCountRequest request{};
  request.header = MakeHeader(Command::kGetCandidateCount, sizeof(request));
  CandidateCountResponse response{};
  if (!Exchange(endpoint, request, &response, error) ||
      !IsValidResponseHeader(response.header, Command::kGetCandidateCount, sizeof(response)) ||
      !ResultOk(static_cast<Result>(response.result), error) || response.candidate_count == 0U) {
    if (error != nullptr && error->empty()) *error = "peer service returned no usable UMDK/EID candidates";
    return false;
  }
  *candidate_count = response.candidate_count;
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

bool CreateChannelAt(const std::string& endpoint, const PeerRegion& region,
                     const uint8_t local_qp_key[kQpKeyBytes],
                     uint32_t local_qp_key_bytes, const uint8_t local_eid[kEidBytes],
                     uint32_t peer_token_value, uint64_t local_tp_handle,
                     uint32_t local_psn,
                     std::array<uint8_t, kQpKeyBytes>* peer_qp_key,
                     uint32_t* peer_qp_key_bytes, uint64_t* peer_tp_handle,
                     uint32_t* peer_psn, std::string* error) {
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

PeerLink::PeerLink(std::string address) : address_(std::move(address)) {}

Status PeerLink::GetCandidateCount(uint32_t* candidate_count) const {
  std::string error;
  return GetCandidateCountAt(address_, candidate_count, &error)
             ? Status::OK()
             : ToStatus("peer candidate discovery failed", std::move(error));
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
  const auto found = regions_.find(base);
  if (found == regions_.end()) return Status::NotFound("peer region was not found");
  std::string error;
  if (!ReleaseRegionAt(address_, found->second, &error)) {
    return ToStatus("peer DDR release failed", std::move(error));
  }
  regions_.erase(found);
  return Status::OK();
}

Status PeerLink::LookupRegion(uint64_t base, PeerRegion* region) const {
  if (region == nullptr) return Status::InvalidArgument("peer region output must not be null");
  const auto found = regions_.find(base);
  if (found == regions_.end()) return Status::NotFound("peer region was not found");
  *region = found->second;
  return Status::OK();
}

Status PeerLink::CreateChannel(
    uint64_t region_base, const uint8_t local_qp_key[kQpKeyBytes],
    uint32_t local_qp_key_bytes, const uint8_t local_eid[kEidBytes],
    uint32_t peer_token_value, uint64_t local_tp_handle, uint32_t local_psn,
    std::array<uint8_t, kQpKeyBytes>* peer_qp_key,
    uint32_t* peer_qp_key_bytes, uint64_t* peer_tp_handle, uint32_t* peer_psn) {
  const auto found = regions_.find(region_base);
  if (found == regions_.end()) return Status::NotFound("peer region was not found");
  std::string error;
  return CreateChannelAt(address_, found->second, local_qp_key, local_qp_key_bytes,
                         local_eid, peer_token_value, local_tp_handle, local_psn,
                         peer_qp_key, peer_qp_key_bytes, peer_tp_handle, peer_psn, &error)
             ? Status::OK()
             : ToStatus("peer channel creation failed", std::move(error));
}

}  // namespace kvaf::peer
