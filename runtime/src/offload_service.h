#pragma once
#include <functional>
#include <memory>
#include "offload_protocol.h"
#include "status.h"

namespace kvaf::offload {
class Backend {
 public:
  virtual ~Backend() = default;
  virtual Status Allocate(std::uint64_t bytes, Endpoint* local) = 0;
  virtual Status Open(const Endpoint& source, Endpoint* local) = 0;
  virtual Status Read(const std::vector<Span>& spans) = 0;
  // Startup only. A backend may retire a failed, completed probe before
  // returning a non-InDoubt error; unresolved DMA must remain InDoubt.
  virtual Status Probe(const std::vector<Span>& spans) { return Read(spans); }
  virtual Status ReleaseSource() = 0;
  virtual Status Close() = 0;
  virtual std::uint64_t Checksum(const std::vector<Span>& spans) const = 0;
};
// Control operations are serialized. Probe runs on the service thread;
// subsequent payload DMA runs on the sole worker of its region.
// One active task per region, with bounded memory and idempotent retry records.
class Service {
 public:
  using Factory = std::function<std::unique_ptr<Backend>(std::uint32_t)>;
  explicit Service(Factory factory);
  ~Service();
  Response Dispatch(const Request&, const std::vector<Span>&);
  void Serve(int fd, const peer::MessageHeader&);
  bool Busy() const;
  Status Shutdown();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
