#pragma once
#include <memory>
#include "access_operator.h"

namespace kvaf {
// Shared lifecycle with optional mode-specific timing. Resources, launch
// arguments and algorithms live in separate backends selected at initialization.
class SequentialBatchGet : public AccessOperator {
 public:
  Status ExecuteBatchPut(const BatchPutExecRequest&) final {
    return Status::NotImplemented("sequential provider is Get-only");
  }
};
std::unique_ptr<SequentialBatchGet> CreateSequentialBatchGet(AccessOperatorConfig config);
}  // namespace kvaf
