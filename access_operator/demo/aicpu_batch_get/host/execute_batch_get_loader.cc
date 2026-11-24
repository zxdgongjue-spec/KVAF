#include "aicpu_batch_get/include/kvaf_aicpu_batch_get.h"
#include "aicpu_urma/kernel_loader.h"
namespace kvaf::aicpu_urma {
int32_t LaunchBatchGet(const UmdkBatchGetArgs* args, aclrtStream stream) {
  if (!args) return -1;
  return LaunchUmdkKernel("KvafAicpuGatherReadV2", args, sizeof(*args), stream,
                          args->gather.worker_count);
}
int32_t LaunchBatchGetRanges(const UmdkBatchGetRangesArgs* args, aclrtStream stream) {
  if (!args) return -1;
  return LaunchUmdkKernel("KvafAicpuGatherReadRangesV1", args, sizeof(*args), stream,
                          args->gather.worker_count);
}
}  // namespace kvaf::aicpu_urma
namespace kvaf::cpu_push {
int32_t LaunchCpuPushScatter(const KernelArgs* args, aclrtStream stream) {
  if (!args) return -1;
  return aicpu_urma::LaunchUmdkKernel("KvafAicpuCpuPushScatterV2", args, sizeof(*args), stream,
                                    args->workers);
}
}  // namespace kvaf::cpu_push
