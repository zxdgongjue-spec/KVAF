#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "status.h"
#include "types.h"
#include "connector_types.h"

namespace kvaf {
// Internal registration manifest. Single and multiple MRs use the same layout.
struct HbmReadTicket {
  std::uint64_t source_id{}, epoch{};
  std::uint32_t token{};
  std::vector<std::uint64_t> mr_bytes;
  NpuPullSourceDesc source() const { return {source_id, epoch}; }
  std::size_t mr_count() const { return mr_bytes.size(); }
  std::uint64_t bytes_for(std::uint64_t id) const {
    return !id || id > mr_bytes.size() ? 0 : mr_bytes[id-1];
  }
  bool matches(const NpuPullSourceDesc& s) const { return SameSource(source(), s); }
  bool same_registration(const HbmReadTicket& other) const {
    return matches(other.source()) && token == other.token && mr_bytes == other.mr_bytes;
  }
  bool valid() const {
    if (!source_id || !epoch || !token || mr_bytes.empty() || mr_bytes.size()>4096) return false;
    for (auto n:mr_bytes) if (!n) return false;
    return true;
  }
};
// Serializable receiver identity. This is exchanged once by application discovery;
// source/MR tickets are subsequently exchanged only by KVAF.
struct HbmReceiverTicket {
  std::string control_address;
  std::uint64_t receiver_id{}, receiver_epoch{};
  bool valid() const { return !control_address.empty() && receiver_id && receiver_epoch; }
};
// Opt-in diagnostics for one synchronous call. Host stages use steady_clock;
// kernel_event_us is the device event interval around the kernel on its stream
// and overlaps launch/synchronize (never add it to the Host phase sum).
struct HbmReadProfile {
  double total_us{}, validation_us{}, buffers_us{}, register_output_us{};
  double event_record_us{}, launch_us{}, synchronize_us{}, event_query_us{}, status_us{};
  double cleanup_us{}, complete_us{}, kernel_event_us{};
  bool kernel_event_valid{};
  std::int32_t event_error{};
};
// Independent from CpuPullHbmSource/CPU DDR. One reader, one frozen batch,
// and up to four target Jettys shared by 1..4096 MRs. Producer owns the tensors.
class NpuPullHbmSource {
 public:
  explicit NpuPullHbmSource(std::uint32_t device_id, std::string npu_eid = "");
  ~NpuPullHbmSource();
  NpuPullHbmSource(const NpuPullHbmSource&) = delete;
  NpuPullHbmSource& operator=(const NpuPullHbmSource&) = delete;
  // Blocking full initialization: TCP, all MR registrations, channel binding
  // and READ probes. Failure may retain resources; retry Shutdown for cleanup.
  Status Register(ConstBufferView, const HbmReceiverTicket&, void* producer_stream);
  Status Register(const std::vector<ConstBufferView>&, const HbmReceiverTicket&, void* producer_stream);
  Status Ready(ReadyToken*); // Single-MR convenience; publishes irrevocable read permission.
  Status Ready(ConstBufferView tensor, ReadyToken*);
  Status ReleaseBatch(const ReadyToken&);
  // Only releases a batch already ended by the reader. Ready is irrevocable
  // on the producer: the reader must fence late notifications before reuse.
  Status CancelBatch(const ReadyToken&);
  Status Shutdown(); // Requests reader detach; failure retains registration.
  bool frozen() const;
  bool connected() const;
  std::uint32_t device_id() const;
  std::string selected_npu_eid() const;
 private:
  Status FinishBatch(const ReadyToken&);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
