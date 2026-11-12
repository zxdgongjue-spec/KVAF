#pragma once
#include <memory>
#include "access_operator.h"

namespace kvaf {
// Shared lifecycle with optional mode-specific timing. Resources, launch
// arguments and algorithms live in separate backends selected at initialization.
class SequentialBatchGet {
 public:
  virtual ~SequentialBatchGet() = default;
  virtual Status Initialize(const RemoteRegionDesc& region) = 0;
  virtual Status Execute(const BatchGetExecRequest& exec) = 0;
  virtual Status Drain() = 0;
  virtual Status Shutdown() = 0;
  virtual Status SetScatterProfiling(bool) {
    return Status::NotImplemented("scatter profiling requires cpu_push_scatter");
  }
  virtual Status CopyScatterProfiles(std::vector<ScatterBatchGetCoreProfile>*) {
    return Status::NotImplemented("scatter profiling requires cpu_push_scatter");
  }
  virtual Status SetGatherProfiling(bool) {
    return Status::NotImplemented("gather profiling requires gather_read mode");
  }
  virtual Status CopyGatherProfile(GatherBatchGetProfile*) const {
    return Status::NotImplemented("gather profiling requires gather_read mode");
  }
  virtual Status CheckGatherGraphReplay() {
    return Status::NotImplemented("gather graph requires gather_read mode");
  }
};
std::unique_ptr<SequentialBatchGet> CreateSequentialBatchGet(AccessOperatorConfig config);
}  // namespace kvaf
