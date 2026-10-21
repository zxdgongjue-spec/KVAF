#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "peer_link.h"
#include "umdk/kernel_abi.h"

namespace kvaf::aicpu_urma {

struct UmdkSessionConfig {
  std::string kernel_json;
  std::string device;
  std::uint32_t eid_index{UINT32_MAX};
  std::uint32_t workers{1U};
  std::shared_ptr<peer::PeerLink> peer_link;
};

class UmdkSession {
 public:
  explicit UmdkSession(UmdkSessionConfig config) : config_(std::move(config)) {}
  Status Load();
  Status Ensure(std::uint64_t region_base, void* output, std::size_t output_bytes);
  Status Shutdown();
  using Launch = int32_t (*)(const UmdkControlArgs*, aclrtStream);
  Status WriteDescriptors(std::uint64_t descriptors, std::uint32_t count, aclrtStream stream,
                          Launch);
  const UmdkSessionConfig& config() const { return config_; }
  std::uint64_t session() const { return args_.session; }
  std::uint64_t owner_pid() const { return args_.owner_pid; }

 private:
  Status Run(UmdkControlOp op, UmdkControlStatus& status, aclrtStream stream = nullptr,
             Launch launch = nullptr);
  UmdkSessionConfig config_;
  UmdkControlArgs args_{};
  aclrtStream stream_{nullptr};
  bool connected_{false};
  std::vector<std::pair<std::uint64_t, std::size_t>> registered_buffers_;
};

}  // namespace kvaf::aicpu_urma
