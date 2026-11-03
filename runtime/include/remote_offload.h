#pragma once
#include <array>
#include <memory>
#include <string>
#include "remote_pull_protocol.h"
#include "types.h"
#include "status.h"
#include "connector_types.h"

namespace kvaf {
// Producer-facing registration ticket; DDR allocation/layout stays on the reader.
struct OffloadRegionTicket {
  std::string address;
  std::uint64_t region_id{}, epoch{};
  std::uint32_t token{};
  std::array<std::uint8_t, 16> eid{};
};
// Producer owns a fixed set of HBM MRs and one shared connection group.
// One frozen batch selects one MR. Keep ALL allocations alive until Shutdown
// succeeds; never overwrite the selected tensor while frozen().
class CpuPullHbmSource {
 public:
  explicit CpuPullHbmSource(std::uint32_t device_id, std::string npu_eid = "");
  ~CpuPullHbmSource();
  // Completes registration, CPU linking and probes. No reader attachment is needed;
  // Ready carries the descriptor used for submission.
  Status Register(const OffloadRegionTicket&, ConstBufferView input, void* producer_stream);
  Status Register(const OffloadRegionTicket&, const std::vector<ConstBufferView>& inputs, void* producer_stream);
  Status Ready(ReadyToken*); // Application has already synchronized production.
  Status Ready(ConstBufferView input, ReadyToken*); // Select one of the registered tensors.
  Status ReleaseBatch(const ReadyToken&);
  Status CancelBatch(const ReadyToken&);
  Status Shutdown();
  bool frozen() const;
  bool registered() const;
  std::uint32_t device_id() const;
  std::string selected_npu_eid() const;
 private:
  Status GetDescriptor(CpuPullSourceDesc*) const;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
