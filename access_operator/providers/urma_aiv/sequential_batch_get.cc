#include "scatter_batch_get.h"
#include "gather_batch_get.h"
#include <utility>

namespace kvaf {
std::unique_ptr<SequentialBatchGet> CreateSequentialBatchGet(GetAlgorithm algorithm, AccessOperatorConfig config) {
  if (algorithm == GetAlgorithm::Scatter)
    return CreateScatterBatchGet(std::move(config));
  if (algorithm == GetAlgorithm::Gather)
    return CreateGatherBatchGet(std::move(config));
  return nullptr;
}
}  // namespace kvaf
