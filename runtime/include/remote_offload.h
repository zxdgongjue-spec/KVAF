#pragma once
#include <memory>
#include <string>
#include "remote_pull_protocol.h"
#include "types.h"
#include "status.h"
#include "connector_types.h"

namespace kvaf {
struct OffloadRegionTicket {
  std::string address;
  remote_pull::RegionInfo region{};
};
// Producer owns its HBM and exporter. One frozen batch per registered source.
// Keep input alive until Shutdown succeeds; never overwrite while frozen().
class CpuPullHbmSource {
 public:
  explicit CpuPullHbmSource(std::uint32_t device_id, std::string npu_eid = "");
  ~CpuPullHbmSource();
  Status Register(const OffloadRegionTicket&, ConstBufferView input, void* producer_stream);
  Status GetDescriptor(CpuPullSourceDesc*) const;
  Status Ready(ReadyToken*); // Application has already synchronized production.
  Status ReleaseBatch(const ReadyToken&);
  Status CancelBatch(const ReadyToken&);
  Status Shutdown();
  bool frozen() const;
  bool registered() const;
  std::uint32_t device_id() const;
  std::string selected_npu_eid() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
