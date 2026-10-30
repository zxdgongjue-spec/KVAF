#pragma once
#include <memory>
#include "access_operator.h"

namespace kvaf {
// Common lifecycle only; resources, launch arguments and algorithms live in
// separate backends. A Runtime selects its backend once during initialization.
class SequentialBatchGet {
 public:
  virtual ~SequentialBatchGet() = default;
  virtual Status Initialize(const RemoteRegionDesc& region) = 0;
  virtual Status Execute(const BatchGetExecRequest& exec) = 0;
  virtual Status Drain() = 0;
  virtual Status Shutdown() = 0;
  virtual Status SetGatherProfiling(bool) {
    return Status::NotImplemented("gather profiling requires gather_read mode");
  }
  virtual Status CopyGatherProfile(GatherBatchGetProfile*) const {
    return Status::NotImplemented("gather profiling requires gather_read mode");
  }
};
std::unique_ptr<SequentialBatchGet> CreateSequentialBatchGet(AccessOperatorConfig config);
}  // namespace kvaf
