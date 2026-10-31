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
  Status Attach(const std::string&, const HbmReadTicket&, const std::string& local_eid, LinkId*);
  Status Read(LinkId, const HbmReadBatch&, MutableBufferView output,
              const std::vector<std::uint64_t>& offsets, void* stream, bool identity_only = false);
  Status Read(LinkId, const ReadyToken&, NpuInt64ArrayView src, NpuInt64ArrayView dst,
              NpuInt64ArrayView lengths, MutableBufferView output, void* stream);
  Status Resolve(LinkId);
  Status Detach(LinkId);
  Status Shutdown();
  bool pending() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
