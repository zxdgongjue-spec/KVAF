#pragma once

#include "connector_types.h"
#include "cpu_pull_timing.h"
#include "types.h"

namespace kvaf {
struct CpuPullProfile {
  double validation_us{}, prepare_put_us{}, dst_offsets_us{}, submit_wait_cpu_pull_us{};
  double wait_cpu_pull_us{}, commit_put_us{}, total_us{};
  CpuPullSubmitTiming submit;
};

// Independent CPU-initiated data plane. RuntimeCore owns the write transaction;
// this interface owns neither the cache allocator nor the ordinary dispatcher.
class CpuPullExecutor {
 public:
  virtual ~CpuPullExecutor() = default;
  virtual Status CanShutdown() const = 0;
  virtual Status Shutdown() = 0;
  virtual Status ValidateCpuSource(const CpuPullSourceDesc&) = 0;
  virtual Status SubmitCpuPull(const CpuPullRequest&) = 0;
  virtual Status SubmitPreparedCpuPullAndWait(const CpuPullRequest&, std::uint32_t,
                                             CpuPullState*, CpuPullSubmitTiming*) = 0;
  virtual Status QueryCpuPull(CpuPullState*) = 0;
  virtual Status CancelCpuPull(const ReadyToken&) = 0;
  virtual Status CompleteOffload(bool committed) = 0;
  virtual Status LastChecksum(std::uint64_t*) const = 0;
  virtual bool pending() const = 0;
};
}  // namespace kvaf
