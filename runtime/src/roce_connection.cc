#include "roce_connection.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sys/random.h>

namespace kvaf::roce {
namespace {

constexpr unsigned kQueueDepth = 128;
constexpr std::uint32_t kMagic = 0x4b564350U;
constexpr std::uint32_t kVersion = 7;
constexpr std::uint32_t kAllocate = 1;
constexpr std::uint32_t kConnect = 2;
constexpr std::uint32_t kRelease = 3;
constexpr std::uint32_t kAllocateMulti = 4;
constexpr std::uint32_t kResponse = 0x8000;

std::string GidText(const Endpoint& endpoint) {
  char text[INET6_ADDRSTRLEN]{};
  const char* result = inet_ntop(AF_INET6, endpoint.gid.data(), text, sizeof(text));
  return result ? result : "<invalid>";
}

std::string VerbsError(int rc, int saved_errno) {
  const int error = rc == -1 && saved_errno ? saved_errno : (rc < 0 ? -rc : rc);
  return " rc=" + std::to_string(rc) + " errno=" + std::to_string(error) +
         " (" + std::strerror(error) + ")";
}

bool Transfer(int fd, void* data, std::size_t bytes, bool write) {
  auto* cursor = static_cast<char*>(data);
  while (bytes > 0) {
    const ssize_t count = write ? send(fd, cursor, bytes, MSG_NOSIGNAL)
                                : recv(fd, cursor, bytes, 0);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return false;
    cursor += count;
    bytes -= static_cast<std::size_t>(count);
  }
  return true;
}

void Encode(char* output, std::uint64_t value, unsigned bytes) {
  for (unsigned i = 0; i < bytes; ++i) {
    output[bytes - i - 1] = static_cast<char>(value >> (i * 8));
  }
}

std::uint64_t Decode(const char* input, unsigned bytes) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < bytes; ++i) {
    value = (value << 8) | static_cast<unsigned char>(input[i]);
  }
  return value;
}

bool Frame(int fd, std::uint32_t command, char* data, unsigned bytes, bool write) {
  char header[12]{};
  if (write) {
    Encode(header, kMagic, 4);
    Encode(header + 4, kVersion, 2);
    Encode(header + 6, command, 2);
    Encode(header + 8, sizeof(header) + bytes, 4);
  }
  if (!Transfer(fd, header, sizeof(header), write) ||
      Decode(header, 4) != kMagic || Decode(header + 4, 2) != kVersion ||
      Decode(header + 6, 2) != command ||
      Decode(header + 8, 4) != sizeof(header) + bytes) {
    return false;
  }
  return Transfer(fd, data, bytes, write);
}

}  // namespace

Connection::~Connection() {
  if (!Stop().ok()) return;
  if (cq_ != nullptr) (void)ibv_destroy_cq(cq_);
  if (owns_device_resources_) {
    if (pd_ != nullptr) (void)ibv_dealloc_pd(pd_);
    if (context_ != nullptr) (void)ibv_close_device(context_);
  }
}

Status Connection::Open(const RoceOptions& options) {
  if (options.device.empty() || options.port == 0 ||
      options.gid_index < 0 || options.gid_index > 255) {
    return Status::InvalidArgument("RoCE requires device, port and GID index");
  }

  int count = 0;
  ibv_device** devices = ibv_get_device_list(&count);
  if (devices == nullptr) return Status::Internal("ibv_get_device_list failed");
  for (int i = 0; i < count; ++i) {
    if (options.device == ibv_get_device_name(devices[i])) {
      context_ = ibv_open_device(devices[i]);
      break;
    }
  }
  ibv_free_device_list(devices);
  if (context_ == nullptr) return Status::NotFound("cannot open RDMA device " + options.device);

  device_name_ = options.device;
  port_ = options.port;
  gid_index_ = static_cast<std::uint8_t>(options.gid_index);
  ibv_port_attr port{};
  ibv_device_attr device{};
  ibv_gid gid{};
  if (ibv_query_port(context_, port_, &port) != 0 ||
      port.state != IBV_PORT_ACTIVE || port.link_layer != IBV_LINK_LAYER_ETHERNET ||
      gid_index_ >= port.gid_tbl_len ||
      ibv_query_gid(context_, port_, gid_index_, &gid) != 0 ||
      ibv_query_device(context_, &device) != 0) {
    return Status::InvalidArgument("RoCE device, active Ethernet port or GID is unavailable");
  }
  std::copy_n(gid.raw, local_.gid.size(), local_.gid.begin());
  const int read_depth = std::min({static_cast<int>(std::numeric_limits<std::uint8_t>::max()),
                                   device.max_qp_rd_atom, device.max_qp_init_rd_atom});
  const int queue_depth = std::min({static_cast<int>(kQueueDepth), device.max_qp_wr,
                                    device.max_cqe});
  if (read_depth <= 0 || queue_depth <= 0 || device.max_sge <= 0 ||
      port.active_mtu < IBV_MTU_256 || port.active_mtu > IBV_MTU_4096 ||
      std::all_of(local_.gid.begin(), local_.gid.end(), [](auto byte) { return byte == 0; })) {
    return Status::InvalidArgument("RoCE endpoint has no usable GID or queue capacity");
  }
  depth_ = static_cast<unsigned>(queue_depth);
  max_mr_size_ = device.max_mr_size;
  local_.read_depth = static_cast<unsigned>(read_depth);
  local_.mtu = port.active_mtu;

  pd_ = ibv_alloc_pd(context_);
  if (pd_ == nullptr) return Status::Internal("ibv_alloc_pd failed");
  return CreateQpResources();
}

Status Connection::OpenSibling(const Connection& owner) {
  if (owner.context_ == nullptr || owner.pd_ == nullptr || owner.qp_ == nullptr ||
      context_ != nullptr || qp_ != nullptr) {
    return Status::InvalidArgument("cannot create sibling QP from an unopened connection");
  }
  context_ = owner.context_;
  pd_ = owner.pd_;
  port_ = owner.port_;
  gid_index_ = owner.gid_index_;
  depth_ = owner.depth_;
  read_depth_ = owner.read_depth_;
  max_mr_size_ = owner.max_mr_size_;
  device_name_ = owner.device_name_;
  local_.gid = owner.local_.gid;
  local_.mtu = owner.local_.mtu;
  local_.read_depth = owner.local_.read_depth;
  owns_device_resources_ = false;
  return CreateQpResources();
}

Status Connection::CreateQpResources() {
  cq_ = ibv_create_cq(context_, static_cast<int>(depth_), nullptr, nullptr, 0);
  if (cq_ == nullptr) return Status::Internal("ibv_create_cq failed");
  ibv_qp_init_attr init{};
  init.send_cq = cq_;
  init.recv_cq = cq_;
  init.qp_type = IBV_QPT_RC;
  init.cap.max_send_wr = depth_;
  init.cap.max_recv_wr = 1;
  init.cap.max_send_sge = 1;
  init.cap.max_recv_sge = 1;
  qp_ = ibv_create_qp(pd_, &init);
  if (qp_ == nullptr) return Status::Internal("ibv_create_qp failed");
  depth_ = std::min(depth_, init.cap.max_send_wr);
  if (depth_ == 0) return Status::Internal("RoCE QP has no send capacity");
  local_.qpn = qp_->qp_num;
  if (getrandom(&local_.psn, sizeof(local_.psn), 0) != sizeof(local_.psn)) {
    return Status::Internal("cannot generate RoCE QP PSN");
  }
  local_.psn &= 0xffffffU;

  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = port_;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
  if (ibv_modify_qp(qp_, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                               IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
    return Status::Internal("RoCE QP INIT failed");
  }
  std::fprintf(stderr, "[KVAF_ROCE] device=%s port=%u gid_index=%u gid=%s qpn=%u mtu=%u read_depth=%u\n",
               device_name_.c_str(), port_, gid_index_, GidText(local_).c_str(), local_.qpn, local_.mtu,
               local_.read_depth);
  return Status::OK();
}

Status Connection::Connect(const Endpoint& remote) {
  if (remote.qpn == 0 || remote.qpn > 0xffffffU || remote.psn > 0xffffffU ||
      remote.mtu < IBV_MTU_256 || remote.mtu > IBV_MTU_4096 || remote.read_depth == 0 ||
      std::all_of(remote.gid.begin(), remote.gid.end(), [](auto byte) { return byte == 0; })) {
    return Status::InvalidArgument("invalid remote RoCE QP descriptor");
  }
  read_depth_ = std::min(local_.read_depth, remote.read_depth);
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = static_cast<ibv_mtu>(std::min(local_.mtu, remote.mtu));
  attr.dest_qp_num = remote.qpn;
  attr.rq_psn = remote.psn;
  attr.max_dest_rd_atomic = static_cast<std::uint8_t>(read_depth_);
  attr.min_rnr_timer = 12;
  attr.ah_attr.is_global = 1;
  attr.ah_attr.port_num = port_;
  std::copy(remote.gid.begin(), remote.gid.end(), attr.ah_attr.grh.dgid.raw);
  attr.ah_attr.grh.sgid_index = gid_index_;
  attr.ah_attr.grh.hop_limit = 64;
  errno = 0;
  const int rtr = ibv_modify_qp(qp_, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                           IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                           IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (rtr != 0) {
    const auto failure = VerbsError(rtr, errno);
    return Status::Internal("RoCE QP RTR failed" + failure +
                            " local_qpn=" + std::to_string(local_.qpn) +
                            " peer_qpn=" + std::to_string(remote.qpn) +
                            " peer_psn=" + std::to_string(remote.psn) +
                            " sgid=" + GidText(local_) + " dgid=" + GidText(remote) +
                            " port=" + std::to_string(port_) +
                            " gid_index=" + std::to_string(gid_index_) +
                            " path_mtu=" + std::to_string(attr.path_mtu) +
                            " max_dest_rd_atomic=" + std::to_string(read_depth_));
  }
  attr = {};
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = local_.psn;
  attr.max_rd_atomic = static_cast<std::uint8_t>(read_depth_);
  attr.timeout = 16;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  errno = 0;
  const int rts = ibv_modify_qp(qp_, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN |
                                           IBV_QP_MAX_QP_RD_ATOMIC | IBV_QP_TIMEOUT |
                                           IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY);
  if (rts != 0) {
    return Status::Internal("RoCE QP RTS failed" + VerbsError(rts, errno) +
                            " local_qpn=" + std::to_string(local_.qpn) +
                            " max_rd_atomic=" + std::to_string(read_depth_));
  }
  return Status::OK();
}

Status Connection::Stop() {
  if (qp_ == nullptr) return Status::OK();
  if (ibv_destroy_qp(qp_) != 0) return Status::Internal("ibv_destroy_qp failed");
  qp_ = nullptr;
  return Status::OK();
}

bool SendAllocation(int fd, std::uint64_t bytes, std::uint64_t slot_size) {
  char data[16];
  Encode(data, bytes, 8);
  Encode(data + 8, slot_size, 8);
  return Frame(fd, kAllocate, data, sizeof(data), true);
}

bool ReceiveAllocation(int fd, std::uint64_t* bytes, std::uint64_t* slot_size) {
  std::uint32_t qp_count = 0;
  return ReceiveAllocation(fd, bytes, slot_size, &qp_count);
}

bool SendAllocation(int fd, std::uint64_t bytes, std::uint64_t slot_size,
                    std::uint32_t qp_count) {
  char data[20];
  Encode(data, bytes, 8);
  Encode(data + 8, slot_size, 8);
  Encode(data + 16, qp_count, 4);
  return qp_count > 0 && Frame(fd, kAllocateMulti, data, sizeof(data), true);
}

bool ReceiveAllocation(int fd, std::uint64_t* bytes, std::uint64_t* slot_size,
                       std::uint32_t* qp_count) {
  char header[12]{};
  if (!Transfer(fd, header, sizeof(header), false) || Decode(header, 4) != kMagic ||
      Decode(header + 4, 2) != kVersion) return false;
  const auto command = Decode(header + 6, 2);
  const auto expected_bytes = command == kAllocate ? 16U : (command == kAllocateMulti ? 20U : 0U);
  if (expected_bytes == 0 || Decode(header + 8, 4) != sizeof(header) + expected_bytes) return false;
  char data[20]{};
  if (!Transfer(fd, data, expected_bytes, false)) return false;
  *bytes = Decode(data, 8);
  *slot_size = Decode(data + 8, 8);
  *qp_count = command == kAllocateMulti ? static_cast<std::uint32_t>(Decode(data + 16, 4)) : 1U;
  return *qp_count > 0;
}

bool SendRegion(int fd, const Region& region) {
  char data[20];
  Encode(data, region.base, 8);
  Encode(data + 8, region.bytes, 8);
  Encode(data + 16, region.rkey, 4);
  return Frame(fd, kAllocate | kResponse, data, sizeof(data), true);
}

bool ReceiveRegion(int fd, Region* region) {
  char data[20];
  if (!Frame(fd, kAllocate | kResponse, data, sizeof(data), false)) return false;
  *region = Region{Decode(data, 8), Decode(data + 8, 8),
                   static_cast<std::uint32_t>(Decode(data + 16, 4))};
  return true;
}

bool SendEndpoint(int fd, bool response, const Endpoint& endpoint) {
  char data[32];
  std::memcpy(data, endpoint.gid.data(), endpoint.gid.size());
  Encode(data + 16, endpoint.qpn, 4);
  Encode(data + 20, endpoint.psn, 4);
  Encode(data + 24, endpoint.mtu, 4);
  Encode(data + 28, endpoint.read_depth, 4);
  return Frame(fd, kConnect | (response ? kResponse : 0U), data, sizeof(data), true);
}

bool ReceiveEndpoint(int fd, bool response, Endpoint* endpoint) {
  char data[32];
  if (!Frame(fd, kConnect | (response ? kResponse : 0U), data, sizeof(data), false)) return false;
  std::memcpy(endpoint->gid.data(), data, endpoint->gid.size());
  endpoint->qpn = Decode(data + 16, 4);
  endpoint->psn = Decode(data + 20, 4);
  endpoint->mtu = Decode(data + 24, 4);
  endpoint->read_depth = Decode(data + 28, 4);
  return true;
}

bool SendRelease(int fd, bool response) {
  return Frame(fd, kRelease | (response ? kResponse : 0U), nullptr, 0, true);
}

bool ReceiveRelease(int fd, bool response) {
  return Frame(fd, kRelease | (response ? kResponse : 0U), nullptr, 0, false);
}

int ConnectControl(const std::string& address) {
  const auto separator = address.rfind(':');
  if (separator == std::string::npos || separator == 0 || separator + 1 == address.size()) {
    return -1;
  }
  std::string host = address.substr(0, separator);
  if (host.front() == '[' && host.back() == ']') host = host.substr(1, host.size() - 2);
  addrinfo hints{};
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* addresses = nullptr;
  if (getaddrinfo(host.c_str(), address.c_str() + separator + 1, &hints, &addresses) != 0) {
    return -1;
  }
  int fd = -1;
  for (auto* candidate = addresses; candidate != nullptr; candidate = candidate->ai_next) {
    fd = socket(candidate->ai_family, candidate->ai_socktype | SOCK_NONBLOCK,
                candidate->ai_protocol);
    if (fd < 0) continue;
    const int connected = connect(fd, candidate->ai_addr, candidate->ai_addrlen);
    pollfd pending{fd, POLLOUT, 0};
    int socket_error = 0;
    socklen_t size = sizeof(socket_error);
    const bool ready = connected == 0 ||
        (errno == EINPROGRESS && poll(&pending, 1, 5000) > 0 &&
         getsockopt(fd, SOL_SOCKET, SO_ERROR, &socket_error, &size) == 0 && socket_error == 0);
    if (ready && fcntl(fd, F_SETFL, 0) == 0 && SetControlTimeout(fd, true)) break;
    close(fd);
    fd = -1;
  }
  freeaddrinfo(addresses);
  return fd;
}

bool SetControlTimeout(int fd, bool read_timeout) {
  const timeval send_timeout{10, 0};
  const timeval receive_timeout{read_timeout ? 10 : 0, 0};
  const int keepalive = 1;
  return setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &send_timeout, sizeof(send_timeout)) == 0 &&
         setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &receive_timeout, sizeof(receive_timeout)) == 0 &&
         setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive)) == 0;
}

}  // namespace kvaf::roce
