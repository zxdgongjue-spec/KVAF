#pragma once
#include <memory>
#include "access_operator.h"

namespace kvaf {
// Get-only adapter for the two sequential URMA AIV algorithms.
class SequentialBatchGet : public AccessOperator {
 public:
  Status ExecuteBatchPut(const BatchPutExecRequest&) final {
    return Status::NotImplemented("sequential provider is Get-only");
  }
};
std::unique_ptr<SequentialBatchGet> CreateSequentialBatchGet(GetAlgorithm algorithm, AccessOperatorConfig config);
}  // namespace kvaf
