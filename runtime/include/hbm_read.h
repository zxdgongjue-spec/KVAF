#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "status.h"
#include "types.h"
#include "connector_types.h"

namespace kvaf {
struct HbmReadTicket {
  std::uint64_t source_id{}, epoch{}, bytes{};
  std::uint32_t token{};
  std::vector<std::uint64_t> mr_bytes;
  NpuPullSourceDesc source() const { return {source_id, bytes, epoch, mr_bytes}; }
  std::size_t mr_count() const { return mr_bytes.empty()?1:mr_bytes.size(); }
  std::uint64_t bytes_for(std::uint64_t id) const {
    return !id || id>mr_count()?0:mr_bytes.empty()?bytes:mr_bytes[id-1];
  }
  bool matches(const NpuPullSourceDesc& s) const {
    return source_id==s.source_id && epoch==s.source_epoch && bytes==s.source_bytes && mr_bytes==s.mr_bytes;
  }
  bool valid() const {
    if (!source_id || !epoch || !token || !bytes || mr_bytes.size()>4096) return false;
    if (!mr_bytes.empty() && (mr_bytes.size()<2 || mr_bytes.front()!=bytes)) return false;
    for (auto n:mr_bytes) if (!n) return false;
    return true;
  }
};
// Opt-in diagnostics for one synchronous call. Host stages use steady_clock;
// kernel_event_us is the device event interval around the kernel on its stream
// and overlaps launch/synchronize (never add it to the Host phase sum).
struct HbmReadProfile {
  double total_us{}, validation_us{}, acquire_us{}, buffers_us{}, register_output_us{};
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
  Status Register(ConstBufferView, void* producer_stream, HbmReadTicket*);
  Status Register(const std::vector<ConstBufferView>&, void* producer_stream, HbmReadTicket*);
  Status Ready(ReadyToken*); // Single-MR convenience only.
  Status Ready(ConstBufferView tensor, ReadyToken*);
  Status ReleaseBatch(const ReadyToken&);
  Status CancelBatch(const ReadyToken&);
  Status Listen(const std::string& address); // host:port; port 0 chooses a free port
  std::uint16_t port() const;
  Status Shutdown(); // Reader must detach first; failure retains registration.
  bool frozen() const;
  bool connected() const;
  std::uint32_t device_id() const;
  std::string selected_npu_eid() const;
 private:
  Status FinishBatch(const ReadyToken&, bool cancel);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
