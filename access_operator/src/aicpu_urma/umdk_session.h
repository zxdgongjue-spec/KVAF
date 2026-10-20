#pragma once

#include <utility>
#include <vector>

#include "host.h"
#include "peer_link.h"
#include "umdk_gather.h"

namespace kvaf::aicpu_urma {

class UmdkSession {
 public:
  Status Ensure(peer::PeerLink& link, std::uint64_t region_base, const std::string& device,
                std::uint32_t eid_index, void* output, std::size_t output_bytes,
                std::uint32_t payload_bytes, std::uint32_t spans, std::uint32_t workers);
  Status Shutdown();
  Status Write(const std::vector<UmdkTransfer>& transfers, aclrtStream stream);
  std::uint64_t session() const { return args_.session; }
  std::uint64_t owner_pid() const { return args_.owner_pid; }

 private:
  Status Run(UmdkControlOp op, UmdkControlStatus& status, aclrtStream stream = nullptr);
  UmdkControlArgs args_{};
  aclrtStream stream_{nullptr};
  bool connected_{false};
  void* transfers_{nullptr};
  std::size_t transfers_capacity_{0};
  std::vector<std::pair<std::uint64_t, std::size_t>> registered_buffers_;
};

}  // namespace kvaf::aicpu_urma
