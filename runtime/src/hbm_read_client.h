#pragma once
#include "hbm_read.h"
#include "runtime.h"
#include "hbm_read_protocol.h"

namespace kvaf {
// Owned by A's existing NpuRuntime. Independent from DDR metadata/operator.
class HbmReader {
 public:
  HbmReader(std::uint32_t device, std::uint32_t cores, std::uint32_t jettys,
            std::uint32_t budget, std::string kernel, std::uint32_t max_inflight = 1024,
            std::uint32_t max_outputs = 64, bool diagnostics = false);
  ~HbmReader();
  Status Listen(const std::string& address, const std::string& advertise_address,
                const std::string& local_eid, HbmReceiverTicket*);
  Status Cancel(const ReadyToken&);
  Status Read(const ReadyToken&, NpuInt64ArrayView src, NpuInt64ArrayView dst,
              NpuInt64ArrayView lengths, MutableBufferView output, void* stream,
              HbmReadProfile* profile = nullptr, bool validate_ranges = false,
              bool synchronize = false, std::uint64_t graph_lease = 0,
              RawReadLifetime lifetime = {});
  Status Drain(const ReadyToken&);
  bool inflight() const;
  // A nonzero lease on failure must be passed to EndGraph for checked cleanup.
  Status BeginGraph(const ReadyToken&, NpuInt64ArrayView src, NpuInt64ArrayView dst,
                    NpuInt64ArrayView lengths, MutableBufferView output, void* stream,
                    std::uint64_t* lease, RawReadLifetime lifetime = {});
  Status EndGraph(std::uint64_t lease);
  Status Resolve(const ReadyToken&);
  Status Detach(const HbmReadTicket&);
  Status Shutdown();
  bool pending() const;
 private:
  Status Attach(std::shared_ptr<hbm_read::ControlChannel>, const HbmReadTicket&, const std::string&);
  void Serve();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
