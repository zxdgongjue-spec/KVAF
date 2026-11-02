#pragma once
#include "hbm_read.h"
#include "runtime.h"

namespace kvaf {
// Owned by A's existing NpuRuntime. Independent from DDR metadata/operator.
class HbmReader {
 public:
  HbmReader(std::uint32_t device, std::uint32_t cores, std::uint32_t jettys,
            std::uint32_t budget, std::string kernel);
  ~HbmReader();
  Status Attach(const std::string&, const HbmReadTicket&, const std::string& local_eid);
  Status Read(const ReadyToken&, NpuInt64ArrayView src, NpuInt64ArrayView dst,
              NpuInt64ArrayView lengths, MutableBufferView output, void* stream,
              HbmReadProfile* profile = nullptr, bool validate_ranges = false);
  Status Resolve(const ReadyToken&);
  Status Detach(const HbmReadTicket&);
  Status Shutdown();
  bool pending() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
