#pragma once
#include "remote_pull_protocol.h"
#include "offload_service.h"
#include <mutex>
#include "cpu_pull_timing.h"

namespace kvaf::remote_pull {
class Service {
 public:
  using Describe = std::function<Status(std::uint64_t, RegionInfo*)>;
  using Attach = std::function<std::unique_ptr<offload::Backend>(const RegionInfo&)>;
  // Share the CPU's region lock with KVCP; release it while waiting for READ.
  Service(Describe describe, Attach attach, std::mutex* control_mutex = nullptr);
  ~Service();
  Response Dispatch(const Request&, const std::vector<Span>&);
  void Serve(int fd, const peer::MessageHeader&);
  bool RegionBusy(std::uint64_t id) const;
  Status Shutdown();
 private:
  struct Impl;
  struct Trace;
  Response DispatchImpl(const Request&, const std::vector<Span>&, Trace*);
  std::unique_ptr<Impl> impl_;
};
Status Exchange(const std::string&, Request, const std::vector<Span>&, Response*);
// One owner per reader. A failed exchange closes the connection and never
// replays the request; the next recovery query may open a new connection.
class ControlChannel {
 public:
  ~ControlChannel();
  ControlChannel() = default;
  ControlChannel(const ControlChannel&) = delete;
  ControlChannel& operator=(const ControlChannel&) = delete;
  Status Exchange(const std::string&, Request, const std::vector<Span>&, Response*, CpuPullSubmitTiming* profile = nullptr);
  void Close();
 private:
  int fd_{-1};
  std::string address_;
};
}
