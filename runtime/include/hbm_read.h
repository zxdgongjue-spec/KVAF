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
// and up to four target Jettys. The Tensor remains owned by its producer.
class NpuPullHbmSource {
 public:
  explicit NpuPullHbmSource(std::uint32_t device_id, std::string npu_eid = "");
  ~NpuPullHbmSource();
  NpuPullHbmSource(const NpuPullHbmSource&) = delete;
  NpuPullHbmSource& operator=(const NpuPullHbmSource&) = delete;
  Status Register(ConstBufferView, void* producer_stream, HbmReadTicket*);
  Status Ready(ReadyToken*);
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
