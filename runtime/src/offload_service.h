#pragma once
#include <functional>
#include <memory>
#include "offload_protocol.h"
#include "status.h"

namespace kvaf::offload {
class Backend {
 public:
  virtual ~Backend() = default;
  virtual Status Allocate(std::uint64_t bytes, MemoryDesc* local) = 0;
  virtual Status Open(const ConnectionDesc& peer, const MemoryDesc& source, ConnectionDesc* local) = 0;
  // One connection owns multiple imported source MRs. Selection never posts DMA
  // or recreates queues, and is forbidden while a batch is unresolved.
  virtual Status AddSource(std::uint64_t, const MemoryDesc&) {
    return Status::NotImplemented("multiple source MRs are unavailable");
  }
  virtual Status SelectSource(std::uint64_t id) {
    return id == 1 ? Status::OK() : Status::InvalidArgument("unknown source MR");
  }
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
