#pragma once

#include <infiniband/verbs.h>

#include <array>
#include <cstdint>
#include <string>

#include "roce_options.h"
#include "status.h"

namespace kvaf::roce {

struct Endpoint {
  std::array<std::uint8_t, 16> gid{};
  std::uint32_t qpn{};
  std::uint32_t psn{};
  std::uint32_t mtu{};
  std::uint32_t read_depth{};
};

struct Region {
  std::uint64_t base{};
  std::uint64_t bytes{};
  std::uint32_t rkey{};
};

class Connection {
 public:
  Connection() = default;
  ~Connection();
  Connection(const Connection&) = delete;
  Connection& operator=(const Connection&) = delete;

  Status Open(const RoceOptions& options);
  // Create another QP/CQ using an existing connection's device context and PD.
  // This allows multiple QPs to share one MR registration.
  Status OpenSibling(const Connection& owner);
  Status Connect(const Endpoint& remote);
  // On success the provider has destroyed the QP and stopped its DMA.
  Status Stop();

  const Endpoint& local() const { return local_; }
  ibv_pd* pd() const { return pd_; }
  ibv_qp* qp() const { return qp_; }
  ibv_cq* cq() const { return cq_; }
  unsigned depth() const { return depth_; }
  unsigned read_depth() const { return read_depth_; }
  std::uint64_t max_mr_size() const { return max_mr_size_; }

 private:
  Status CreateQpResources();

  ibv_context* context_{};
  ibv_pd* pd_{};
  ibv_cq* cq_{};
  ibv_qp* qp_{};
  Endpoint local_{};
  std::uint8_t port_{};
  std::uint8_t gid_index_{};
  unsigned depth_{};
  unsigned read_depth_{};
  std::uint64_t max_mr_size_{};
  std::string device_name_;
  bool owns_device_resources_{true};
};

// Allocation messages support the original one-QP form and the multi-QP form.
bool ReceiveAllocation(int fd, std::uint64_t* bytes, std::uint64_t* slot_size);
bool ReceiveAllocation(int fd, std::uint64_t* bytes, std::uint64_t* slot_size,
                       std::uint32_t* qp_count);
bool SendAllocation(int fd, std::uint64_t bytes, std::uint64_t slot_size);
bool SendAllocation(int fd, std::uint64_t bytes, std::uint64_t slot_size,
                    std::uint32_t qp_count);
bool SendRegion(int fd, const Region& region);
bool ReceiveRegion(int fd, Region* region);
bool SendEndpoint(int fd, bool response, const Endpoint& endpoint);
bool ReceiveEndpoint(int fd, bool response, Endpoint* endpoint);
bool SendRelease(int fd, bool response);
bool ReceiveRelease(int fd, bool response);
int ConnectControl(const std::string& address);
bool SetControlTimeout(int fd, bool read_timeout);

}  // namespace kvaf::roce
