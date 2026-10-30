#pragma once
#include <memory>
#include <string>
#include "remote_pull_protocol.h"
#include "remote_offload_types.h"
#include "types.h"
#include "status.h"

namespace kvaf {
struct OffloadRegionTicket {
  std::string address;
  remote_pull::RegionInfo region{};
};
struct HbmReadyBatch {
  RemoteSourceIdentity source;
  std::vector<std::uint64_t> input_offsets, lengths;
};
// Card B owns its HBM and exporter. One frozen batch per registered source.
// Keep input alive until Shutdown succeeds; never overwrite while frozen().
class HbmSource {
 public:
  explicit HbmSource(std::uint32_t device_id, std::string npu_eid = "");
  ~HbmSource();
  Status Register(const OffloadRegionTicket&, ConstBufferView input, void* producer_stream);
  Status Ready(const std::vector<std::uint64_t>& offsets, const std::vector<std::uint64_t>& lengths,
               void* producer_stream, HbmReadyBatch*);
  Status ReleaseBatch(); // Queries A's commit/abort acknowledgement; never resubmits READ.
  Status CancelBatch();  // Only succeeds before CPU submission.
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
