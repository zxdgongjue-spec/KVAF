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
struct HbmReadBatch {
  std::uint64_t source_id{}, epoch{}, sequence{};
  // Actual exported B addresses, not A-local pointers or lookup indices.
  std::vector<std::uint64_t> source_addresses, lengths;
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
  Status Ready(const std::vector<std::uint64_t>& offsets,
               const std::vector<std::uint64_t>& lengths, void* stream, HbmReadBatch*);
  Status ReleaseBatch();
  Status CancelBatch(); // Refused after the reader acquires this generation.
  Status Shutdown(); // Reader must detach first; failure retains registration.
  bool frozen() const;
  bool connected() const;
  std::uint32_t device_id() const;
  std::string selected_npu_eid() const;
 private:
  Status FinishBatch(const ReadyToken*, bool cancel);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
using HbmReadSource = NpuPullHbmSource;
}
