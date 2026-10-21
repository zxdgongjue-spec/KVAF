#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "umdk/kernel_loader.h"
namespace kvaf::aicpu_urma {
int32_t LaunchBatchGet(const UmdkBatchGetArgs* args, aclrtStream stream) {
  if (!args) return -1;
  if (LaunchUmdkKernel("KvafAicpuGatherRead", args, sizeof(*args), stream)) return -1;
  return args->gather.worker_count <= 1U
             ? 0
             : LaunchUmdkKernel("KvafAicpuGatherScatter", args, sizeof(*args), stream,
                                args->gather.worker_count);
}
}  // namespace kvaf::aicpu_urma
