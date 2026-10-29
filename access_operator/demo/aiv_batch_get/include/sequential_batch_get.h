#pragma once

#include <memory>
#include "access_operator.h"
#include "sequential_batch_get_abi.h"

extern "C" int32_t KvafSequentialBatchGetLoadKernel(const char* path);
extern "C" int32_t KvafSequentialBatchGetLaunch(
    const KvafSequentialBatchGetArgs* args, void* stream);

namespace kvaf {
class SequentialBatchGet {
 public:
  explicit SequentialBatchGet(AccessOperatorConfig config);
  ~SequentialBatchGet();
  Status Initialize(const RemoteRegionDesc& region);
  Status Execute(const BatchGetExecRequest& exec);
  Status Drain();
  Status Shutdown();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace kvaf
