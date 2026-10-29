#include "scatter_batch_get.h"
#include "gather_batch_get.h"
#include <utility>

namespace kvaf {
std::unique_ptr<SequentialBatchGet> CreateSequentialBatchGet(AccessOperatorConfig config) {
  if (config.aiv_batch_get_mode == "cpu_push_scatter")
    return CreateScatterBatchGet(std::move(config));
  if (config.aiv_batch_get_mode == "gather_read")
    return CreateGatherBatchGet(std::move(config));
  return nullptr;
}
}  // namespace kvaf
