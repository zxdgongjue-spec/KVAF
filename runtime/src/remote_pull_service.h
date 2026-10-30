#pragma once
#include "remote_pull_protocol.h"
#include "offload_service.h"

namespace kvaf::remote_pull {
class Service {
 public:
  using Describe = std::function<Status(std::uint64_t, RegionInfo*)>;
  using Attach = std::function<std::unique_ptr<offload::Backend>(const RegionInfo&)>;
  Service(Describe describe, Attach attach);
  ~Service();
  Response Dispatch(const Request&, const std::vector<Span>&);
  void Serve(int fd, const peer::MessageHeader&);
  bool RegionBusy(std::uint64_t id) const;
  Status Shutdown();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
Status Exchange(const std::string&, Request, const std::vector<Span>&, Response*);
}
