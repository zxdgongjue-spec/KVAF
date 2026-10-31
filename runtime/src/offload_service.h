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
  // Shared-pool sources must not overwrite cache slots during link probing.
  virtual Status ProbeToScratch(std::uint64_t, std::uint64_t, std::uint64_t*) {
    return Status::NotImplemented("scratch probe is unavailable");
  }
  virtual Status ReleaseSource() = 0;
  virtual Status Close() = 0;
  virtual std::uint64_t Checksum(const std::vector<Span>& spans) const = 0;
};
}
